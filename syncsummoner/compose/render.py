"""Offline multi-pass rendering: burn timecode, play out, capture, align, crop, composite.

Every pass is a real-time capture. Frames self-identify through a croppable
gray-code edge strip, which is what makes alignment possible without reopening
the capture stream. Program is fixed per pass; loading it blacks the output out.
"""

from __future__ import annotations

import queue
import subprocess
import threading
import time
from contextlib import ExitStack
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

import numpy as np
from pyvmancer.const import ROUTE_HDMI_IN

from syncsummoner.device.profile import ProgramProfile
from syncsummoner.device.stored import captures
from syncsummoner.device.recorder import (
    BlankTakeError,
    TakeReport,
    await_transmitting,
    require_input,
    require_route,
    require_signal,
    rig_state,
)
from syncsummoner.compose import timecode
from syncsummoner.compose.cut import Cut, cut_plan
from syncsummoner.compose.master import DEFAULT_CRF, X264
from syncsummoner.compose.planner import plan_automation
from syncsummoner.compose.score import Layer, Score
from syncsummoner.compose.timecode import source_stream  # pylint: disable=unused-import
from syncsummoner.compose.vocabulary import Automation
from syncsummoner.compose.lock import (  # pylint: disable=unused-import
    LockError,
    LostTimecodeError,
    attribute_take,
    lock_path,
    picked_path,
    save_lock,
    take_start,
)
from syncsummoner.probe.patterns import burn_timecode  # pylint: disable=unused-import
from syncsummoner.progress import LOG, human, meter, stage


class UnknownProgramError(RuntimeError):
    """A named program cannot be rendered, so no pass should be started."""


class LeadInError(RuntimeError):
    """The take's program had not finished loading when the clip's first frame played."""


class UnsafeOutputError(RuntimeError):
    """Output tripped a hard safety veto and mitigation was declined."""


class RigFaultError(RuntimeError):
    """A program showed no picture and the Passthru control on the same rig showed none either."""


class MissingProfileError(RuntimeError):
    """A layer's program has no profile, so its pass would play no automation at all."""


def require_profiles(score: Score, profiles: Mapping[str, ProgramProfile]) -> None:
    """Refuse a score with a layer whose program has no profile, naming it and where profiles were read."""
    missing = sorted({layer.program for layer in score.layers} - set(profiles))
    if missing:
        where = getattr(profiles, "directory", None)
        raise MissingProfileError(
            f"no profile for {', '.join(missing)} in {where if where is not None else 'the profiles given'}: "
            "a layer without one would render no automation; fit one with `syncsummoner probe run`"
        )


class Offered(tuple):
    """Program names a device listed, with the authors it enumerated; None on firmware without authors."""

    authors: tuple[str, ...] | None

    def __new__(cls, names: Sequence[str], authors: Sequence[str] | None = None) -> "Offered":
        offered = super().__new__(cls, names)
        offered.authors = None if authors is None else tuple(authors)
        return offered


PASSTHRU = "Passthru"
#: Passes reloaded and retried after showing no picture while the Passthru control did.
PASS_RETRIES = 1
#: Program names shown from a listing that lacked a requested one.
LISTING_SAMPLE = 5
#: A pass sleeps in ticks no longer than this, so its progress moves while it waits.
DRIVE_TICK_S = 1.0
#: Playout writes the Pi's framebuffer and capture reads the card, so a session is
#: whatever both ends already run at; 1080p30 is this rig, measured.
SESSION_FORMATS = {"720p60": (1280, 720, 60.0), "1080p30": (1920, 1080, 30.0), "ntsc": (720, 480, 29.97)}
#: Perceptual lead on top of the device latency: visuals may run early by far more than they may
#: run late, so arrival is placed a little ahead (design 4.4).
EARLY_BIAS_S = 0.015


@dataclass(frozen=True)
class RenderConfig:
    """Session constants and the measured timing budget for one render."""

    width: int = 1280
    height: int = 720
    fps: float = 60.0
    #: The output timing the device must drive, as it names it; the raster above is its size.
    timing: str = "720p60"
    strip_px: int = 8
    bits: int = 16
    #: Automation is shown from the first frame starting at or after its time less this bias.
    early_bias_s: float = EARLY_BIAS_S
    #: The card's budget to deliver a probe once the transmitter drives the session timing.
    probe_timeout_s: float = 40.0
    #: The input the session feeds the device, verified after every load; None skips the check.
    source: str | None = "hdmi"
    #: Vid Route Mode's settings value, verified before the run and after every load; None skips it.
    route: int | None = ROUTE_HDMI_IN
    #: Length of each probe recorded while a load settles, the Passthru control's included.
    probe_s: float = 2.0
    #: pyvmancer's opt-in stall guard for device waits; None waits for as long as the device is on USB.
    stall_s: float | None = None
    #: Recorded past the clip's last frame, so the card's last captures of it are inside the take.
    tail_s: float = 1.0
    #: Where the liveness probe is recorded before a take begins.
    probe_path: str = "/tmp/syncsummoner-probe.mkv"
    #: ssh target driving playout and the HDMI link; None takes the device layer's default.
    source_host: str | None = None

    @classmethod
    def for_format(cls, name: str, **kw: Any) -> "RenderConfig":
        """Config for a named session format; format is a session constant, never a parameter."""
        width, height, fps = SESSION_FORMATS[name]
        return cls(width=width, height=height, fps=fps, timing=name, **kw)


@dataclass
class Rig:
    """The real-time endpoints a pass drives, injectable so render stays testable.

    ``link`` is the source HDMI link, held down for every program load command; None
    leaves it alone, which only a rig without link control should do.
    """

    session: Any
    capture: Any
    playout: Any
    link: Any = None
    transport: Any = None


def crop_strip(frame: np.ndarray, *, strip_px: int = 8) -> np.ndarray:
    """Remove the timecode strip before anything else looks at the frame."""
    return frame[: frame.shape[0] - strip_px]


def write_timecoded(
    source: Any,
    out: str | Path,
    *,
    config: RenderConfig,
    seconds: float | None = None,
    start: float = 0.0,
    lead: int,
    tail: int,
    cache: str | Path | None = None,
) -> int:
    """:func:`.timecode.write_timecoded` at the session's raster."""
    return timecode.write_timecoded(
        source, out, config=config, seconds=seconds, start=start, lead=lead, tail=tail, cache=cache
    )


def _wait_until(target: float, *, clock, sleep, tick_s: float, report=None) -> None:
    """Sleep to a wall-clock instant in ticks, so a long wait still reports progress.

    Each tick is timed against the clock rather than accumulated, so subdividing
    a wait cannot drift; a clock that does not advance ends the wait rather than
    spinning on it.
    """
    while True:
        now = clock()
        left = target - now
        if left <= 0:
            return
        sleep(min(left, tick_s))
        if clock() <= now:
            return
        if report is not None:
            report(clock())


def play_take(
    rig: Rig,
    recorder: Any,
    out: str | Path,
    *,
    program: str | None,
    seconds: float,
    config: RenderConfig,
    clip: timecode.ClipLayout,
) -> float:
    """Record one locked take of the whole ``clip`` with no automation, ``program`` loaded in its lead-in.

    Passthru carries the lead-in and comes back for the lead-out; the lock record is saved beside the
    take. Returns when the load ended; a load that overran the lead-in or lead-out is a :class:`LeadInError`.
    """
    lead, count, tail = clip.lead, int(round(seconds * config.fps)), clip.tail
    lead_s = lead / config.fps
    with stage("passthru", program=PASSTHRU):
        rig.session.load_program(PASSTHRU, link=rig.link)
        require_route(rig.transport, program=PASSTHRU, route=config.route)
        await_transmitting(rig.transport, program=PASSTHRU, timing=config.timing)
    with stage("take", program=program or PASSTHRU, seconds=f"{lead_s + seconds:.1f}") as said:
        with recorder.recording(out):
            with rig.playout.playing(fps=config.fps) as player:
                began = time.monotonic()
                if program is not None:
                    rig.session.load_program(program, link=rig.link)
                    require_route(rig.transport, program=program, route=config.route)
                    rig.session.set_params(rig.session.working_point(rig.transport.program_info()))
                    await_transmitting(rig.transport, program=program, timing=config.timing)
                    require_input(rig.transport, program=program, source=config.source)
                loaded = time.monotonic()
                said["load_s"] = round(loaded - began, 2)
                if loaded - began >= lead_s:
                    raise LeadInError(
                        f"{program}: loading took {loaded - began:.1f}s, past the {lead_s:.1f}s lead-in; "
                        "timecode the clip with a longer one"
                    )
                end = began + lead_s + seconds
                _wait_until(end, clock=time.monotonic, sleep=time.sleep, tick_s=DRIVE_TICK_S)
                if program is not None:
                    rig.session.load_program(PASSTHRU, link=rig.link)
                    await_transmitting(rig.transport, program=PASSTHRU, timing=config.timing)
                    if time.monotonic() - end >= tail / config.fps:
                        raise LeadInError(
                            f"{program}: Passthru loaded past the lead-out; timecode a longer one"
                        )
                end += tail / config.fps + config.tail_s
                _wait_until(end, clock=time.monotonic, sleep=time.sleep, tick_s=DRIVE_TICK_S)
                hold = int(player.status().get("hold", 1))
        save_lock(out, rig.playout.flips(), hold=hold, lead=lead, count=count, tail=tail, wet=bool(program))
    return loaded


class TakeFaultError(RuntimeError):
    """The capture driver lost frames during a take, so some clip frames have no capture; recapture it."""


def _attempt(
    rig: Rig,
    recorder: Any,
    out: str | Path,
    *,
    score: Score,
    autos: Mapping,
    config: RenderConfig,
    clip: str | Path,
    layout: timecode.ClipLayout,
) -> tuple[str, TakeReport]:
    """Record the score's pass with every write frame-locked, then prove it; returns what was judged.

    The take's evidence (:func:`~syncsummoner.compose.locktake.take_evidence`) is saved beside
    it, and a take not proven is reported; frames the capture driver lost are a :class:`TakeFaultError`.
    """
    from syncsummoner.compose import locktake, session  # pylint: disable=import-outside-toplevel

    layer = min(score.layers, key=lambda item: item.index)
    program = layer.program
    signal = {"program": program, "timing": config.timing}
    plan = locktake.clip_plan(autos.get(layer.index, Automation.empty()), score.duration, config)

    def load(name: str) -> Any:
        rig.session.load_program(name, link=rig.link)
        require_route(rig.transport, program=name, route=config.route)
        await_transmitting(rig.transport, program=name, timing=config.timing)
        require_input(rig.transport, program=name, source=config.source)
        return rig.transport.program_info()

    session.locked_take(
        rig,
        recorder,
        out,
        program=program,
        seconds=score.duration,
        config=config,
        plan=plan,
        working=rig.session.working_point,
        load=load,
        clip=layout,
    )
    rig.transport.verify_cc_map()
    require_route(rig.transport, program=program, route=config.route)
    require_signal(rig.transport, when="as the pass ended", **signal)
    from syncsummoner.compose.judge import judge_take  # pylint: disable=import-outside-toplevel

    try:
        judged = judge_take(out, clip=clip, config=config, program=program)
    except LockError as err:
        from syncsummoner.device.v4l2rec import read_log  # pylint: disable=import-outside-toplevel

        lost = int(np.clip(np.diff(read_log(out)["sequence"]) - 1, 0, None).sum())
        if lost:
            raise TakeFaultError(f"{program}: the capture driver lost {lost} frames: {err}") from err
        raise
    proof = judged.proof
    locktake.add_schedule(Path(out).parent, proof.schedule())
    for line in proof.lines():
        LOG.info("%s", line)
    if not proof.ok:
        LOG.warning(
            "%s: the take's parameters are not all proven; see %s", program, locktake.evidence_path(out)
        )
    return "take", judged.report


def verify_dark(
    rig: Rig, recorder: Any, *, program: str, judged: str, report: TakeReport, config: RenderConfig
) -> None:
    """Run the Passthru control before a dark program is judged; :class:`RigFaultError` when it fails too.

    Passthru carries the playing clip, so it showing no picture, or a wrong Vid Route Mode,
    blames the rig: the error names the route, the input and the transmitter.
    """
    from syncsummoner.probe.harvest import control

    with rig.playout.playing(fps=config.fps):
        said = control(rig.session, recorder, rig.transport, config=config, link=rig.link)
    if said:
        raise RigFaultError(
            f"{program}: the {judged} shows no picture ({report}), and the Passthru control failed: {said}; "
            f"{rig_state(rig.transport)}"
        )


def render_played(
    score: Score,
    source: Any,
    out: str | Path,
    *,
    profiles: Mapping[str, ProgramProfile],
    rig: Rig | None = None,
    config: RenderConfig | None = None,
    scratch: str | Path = "timecoded.mkv",
    prepared: bool = False,
    start: float = 0.0,
    recorder: Any = None,
) -> TakeReport:
    """Render one pass, its program loaded once: the rig plays, the card records, the host only drives.

    A layer with no profile is refused; each take is :func:`_attempt`, and one with no picture is a
    program verdict only after :func:`verify_dark` passes, then retried :data:`PASS_RETRIES` times.
    """
    layers = sorted(score.layers, key=lambda item: item.index)
    if not layers:
        raise ValueError("score has no layers to render")
    config = RenderConfig() if config is None else config
    program = layers[0].program
    require_profiles(score, profiles)
    layout = ensure_clip(
        source,
        scratch,
        program,
        config=config,
        directory=Path(out).parent,
        seconds=score.duration,
        start=start,
        prepared=prepared,
    )
    owned = rig is None
    rig = open_rig(config, capture=False) if owned else rig
    if recorder is None:
        from syncsummoner.device.recorder import Recorder

        recorder = Recorder(width=config.width, height=config.height, fps=config.fps)
    autos = plan_automation(score, profiles, fps=config.fps)
    try:
        require_route(rig.transport, program=program, route=config.route)
        with stage("upload", clip=str(scratch)) as sent:
            sent["bytes"] = rig.playout.upload(str(scratch))
        for attempt in range(PASS_RETRIES + 1):
            judged, report = _attempt(
                rig, recorder, out, score=score, autos=autos, config=config, clip=scratch, layout=layout
            )
            if report.mostly_picture:
                return report
            verify_dark(rig, recorder, program=program, judged=judged, report=report, config=config)
            LOG.warning(
                "%s: the %s shows no picture (%s) while Passthru does; %s",
                program,
                judged,
                report,
                (
                    f"reloading for retry {attempt + 1} of {PASS_RETRIES}"
                    if attempt < PASS_RETRIES
                    else "no retry left"
                ),
            )
        raise BlankTakeError(
            f"{program}: the program outputs no picture ({report}), while Passthru on the same rig does"
        )
    finally:
        if owned:
            if rig.capture is not None:
                rig.capture.close()
            rig.transport.close()


def device_programs() -> Offered:
    """Program names the attached device offers, and its authors.

    The device layer is imported here, so compose needs no hardware.
    """
    from syncsummoner.device.transport import Transport

    transport = Transport.open()
    try:
        return Offered(transport.programs(), transport.authors())
    finally:
        transport.close()


def describe_offer(offered: Sequence[str], missing: Sequence[str] = ()) -> str:
    """A listing as its count, the authors enumerated, a sample, and any name matching a missing one's.

    A name matches when its plain part, after any ``author/``, is the missing name's.
    """
    names = list(offered)
    authors = getattr(offered, "authors", None)
    said = (
        f"{len(names)} programs from authors {', '.join(authors) if authors is not None else 'unreported'}"
        f", e.g. {', '.join(names[:LISTING_SAMPLE]) or 'none'}"
    )
    plain = {name: name.rpartition("/")[2] for name in names}
    near = [
        f"{want} as {', '.join(found)}"
        for want in missing
        if (found := [n for n, p in plain.items() if p == want.rpartition("/")[2] and n != want])
    ]
    return said + (f"; listed {'; '.join(near)}" if near else "")


def unknown_programs(programs: Sequence[str], available: Sequence[str] | None) -> list[str]:
    """Named programs the device does not offer, in the order they were asked for.

    A program is loaded only when its turn comes and a pass costs minutes, so a name
    that can never load is worth catching before the first one. Firmware moves under
    an archive: a profile set fitted on an older release can name a program the
    device has since dropped. ``None`` checks nothing, for a caller with no device.
    """
    if available is None:
        return []
    offered = set(available)
    return [name for name in dict.fromkeys(programs) if name not in offered]


def ensure_clip(
    source: Any,
    scratch: str | Path,
    program: str,
    *,
    config: RenderConfig,
    directory: str | Path,
    seconds: float,
    start: float = 0.0,
    prepared: bool = False,
) -> timecode.ClipLayout:
    """The timecoded clip at ``scratch``, its lead-in and lead-out long enough for ``program``'s session.

    One too short is timecoded again, never shorter than it was; a ``prepared`` one is only checked.
    """
    from syncsummoner.compose import locktake, session  # pylint: disable=import-outside-toplevel

    dry = locktake.dry_state(program, locktake.declared_bypass) is not None
    lead, tail = session.session_frames(locktake.LeadIn(), config.fps, dry=dry, directory=directory)
    try:
        have = timecode.clip_layout(scratch) if Path(scratch).is_file() else None
    except timecode.LayoutError:
        have = None
    if have is not None and have.lead >= lead and have.tail >= tail:
        return have
    if prepared:
        raise timecode.LayoutError(
            f"{scratch} holds {have} and {program}'s session needs a {lead}-frame lead-in "
            f"and a {tail}-frame lead-out"
        )
    write_timecoded(
        source,
        scratch,
        config=config,
        seconds=seconds,
        start=start,
        lead=max(lead, have.lead if have else 0),
        tail=max(tail, have.tail if have else 0),
    )
    return timecode.clip_layout(scratch)


#: 4:2:0 at full range, what the card's MJPEG decodes to, so a splice converts no level.
SPLICE_PIX_FMT = "yuvj420p"


class SpliceError(RuntimeError):
    """A clip frame cannot be taken from the take its cut names."""


def frame_owners(cuts: Sequence[Cut], fps: float) -> np.ndarray:
    """Per clip frame, the index of the cut showing it; frame ``n`` sits at ``n / fps``."""
    if not cuts:
        raise ValueError("nothing to splice")
    edges = np.rint(np.array([(cut.start, cut.end) for cut in cuts], dtype=np.float64) * fps).astype(np.int64)
    owner = np.full(int(edges[:, 1].max()), -1, dtype=np.int64)
    for index, (a, b) in enumerate(edges):
        owner[max(a, 0) : b] = index
    gap = np.flatnonzero(owner < 0)
    if gap.size:
        raise SpliceError(f"no cut covers {gap[0] / fps:.3f}s")
    return owner


def splice_indices(
    cuts: Sequence[Cut], picked: Mapping[str, np.ndarray], fps: float
) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    """Per clip frame, the program showing it, and per program the capture indices it is taken at.

    ``picked`` is per program what :attr:`~syncsummoner.compose.takecheck.Trace.picked` holds:
    the capture showing each clip frame, -1 where none does. A take is decoded once, forward,
    so its captures must rise with the clip frames they show.
    """
    names = list(dict.fromkeys(cut.program for cut in cuts))
    owner = np.array([names.index(cut.program) for cut in cuts], dtype=np.int64)[frame_owners(cuts, fps)]
    wanted = {}
    for index, name in enumerate(names):
        frames = np.flatnonzero(owner == index)
        known = np.full(owner.size, -1, dtype=np.int64)
        given = np.asarray(picked.get(name, ()), dtype=np.int64)[: owner.size]
        known[: given.size] = given
        at = known[frames]
        missing = frames[at < 0]
        if missing.size:
            run = missing[: int(np.argmax(np.r_[np.diff(missing) != 1, True])) + 1]
            raise SpliceError(
                f"{name}: no capture shows clip frames {run[0] / fps:.3f}-{(run[-1] + 1) / fps:.3f}s"
            )
        back = np.flatnonzero(np.diff(at) <= 0)
        if back.size:
            raise SpliceError(
                f"{name}: the capture showing {frames[back[0] + 1] / fps:.3f}s comes before the last"
            )
        wanted[name] = at
    return owner, {name: wanted[name] for name in names}


def _frame_bytes(width: int, height: int) -> int:
    """One :data:`SPLICE_PIX_FMT` frame: full-size luma and two half-size chroma planes."""
    return width * height + 2 * ((width + 1) // 2) * ((height + 1) // 2)


def _drain(stream, into: list[bytes]) -> None:
    with stream:
        into.extend(stream)


class _Take:
    """One take decoded forward on its own process, handing over only the captures asked for.

    The hand-over queue holds one frame, so a take waits once it is a frame ahead of the cut.
    """

    def __init__(self, path: str | Path, wanted: np.ndarray, *, config: RenderConfig, ffmpeg: str):
        self.path, self.wanted = str(path), wanted
        self.size = _frame_bytes(config.width, config.height)
        self.source = ExitStack()
        source = self.source.enter_context(captures(self.path, ffmpeg=ffmpeg))
        argv = [ffmpeg, "-nostdin", "-hide_banner", "-loglevel", "error", *source, "-map", "0:v:0"]
        argv += ["-fps_mode", "passthrough", "-s", f"{config.width}x{config.height}"]
        argv += ["-pix_fmt", SPLICE_PIX_FMT, "-f", "rawvideo", "pipe:1"]
        self.proc = subprocess.Popen(  # pylint: disable=consider-using-with
            argv, stdin=subprocess.DEVNULL, stdout=subprocess.PIPE, stderr=subprocess.PIPE
        )
        self.said: list[bytes] = []
        self.frames: queue.Queue = queue.Queue(maxsize=1)
        self.threads = [
            threading.Thread(target=_drain, args=(self.proc.stderr, self.said), daemon=True),
            threading.Thread(target=self._pull, daemon=True),
        ]
        for thread in self.threads:
            thread.start()

    def _pull(self) -> None:
        buffer = bytearray(self.size)
        at = 0
        try:
            for want in self.wanted:
                while at <= want:
                    if self.proc.stdout.readinto(buffer) != self.size:
                        raise EOFError(f"capture {at}")
                    at += 1
                self.frames.put(bytes(buffer))
        except (EOFError, OSError, ValueError) as err:
            self.frames.put(err)
        finally:
            self.proc.kill()

    def next(self) -> bytes:
        """The next capture asked for, or why the take ran out before it."""
        item = self.frames.get()
        if isinstance(item, BaseException):
            self.proc.wait()
            said = b"".join(self.said).decode("utf-8", "replace").strip().splitlines()
            raise SpliceError(f"{self.path} ended before {item}" + (f": {said[-1]}" if said else ""))
        return item

    def close(self) -> None:
        """Stop decoding, free a hand-over left waiting, and reap the process and its feed."""
        self.proc.kill()
        try:
            self.frames.get_nowait()
        except queue.Empty:
            pass
        self.proc.wait()
        self.source.close()


def splice(
    cuts: Sequence[Cut],
    takes: Mapping[str, str | Path],
    picked: Mapping[str, np.ndarray],
    out: str | Path,
    *,
    config: RenderConfig,
    ffmpeg: str = "ffmpeg",
) -> int:
    """Write one frame per clip frame, each the capture its cut's take shows that frame in.

    Every take is decoded once, forward, at full size, all at once; the cut takes each frame
    from whichever take shows it, so no seek rounds a splice and no lead-in is guessed. The
    picture is written at ``config.fps``, the clip's rate, with the master's encoder; the frame
    count is returned.
    """
    owner, wanted = splice_indices(cuts, picked, config.fps)
    names = list(wanted)
    argv = [ffmpeg, "-nostdin", "-hide_banner", "-loglevel", "error", "-y", "-f", "rawvideo"]
    argv += ["-pix_fmt", SPLICE_PIX_FMT, "-s", f"{config.width}x{config.height}", "-r", f"{config.fps:g}"]
    argv += ["-i", "pipe:0", *X264, "-crf", str(DEFAULT_CRF), "-pix_fmt", SPLICE_PIX_FMT, str(out)]
    readers = [_Take(takes[name], wanted[name], config=config, ffmpeg=ffmpeg) for name in names]
    encoder = subprocess.Popen(  # pylint: disable=consider-using-with
        argv, stdin=subprocess.PIPE, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE
    )
    said: list[bytes] = []
    drain = threading.Thread(target=_drain, args=(encoder.stderr, said), daemon=True)
    drain.start()
    written = 0
    try:
        with meter(owner.size / config.fps, desc="splicing") as progress:
            for index in owner:
                encoder.stdin.write(readers[index].next())
                written += 1
                progress.to(written / config.fps)
    except BrokenPipeError:
        pass
    finally:
        for reader in readers:
            reader.close()
        encoder.stdin.close()
        encoder.wait()
        drain.join()
    if encoder.returncode or written < owner.size:
        text = b"".join(said).decode("utf-8", "replace").strip().splitlines()
        raise SpliceError(f"encoding {out} failed: {text[-1] if text else 'no output'}")
    LOG.info("spliced %d frames from %d takes into %s", written, len(names), out)
    return written


def render_cuts(
    score: Score,
    source: Any,
    out: str | Path,
    *,
    profiles: Mapping[str, ProgramProfile],
    programs: Sequence[str] = (),
    config: RenderConfig | None = None,
    scratch: str | Path = "timecoded.mkv",
    prepared: bool = False,
    start: float = 0.0,
    takes: str | Path = ".",
    available: Sequence[str] | Callable[[], Sequence[str]] | None = None,
    pass_render: Callable[..., Any] | None = None,
    reuse: bool = False,
) -> list[Cut]:
    """Render one pass per program and splice between them on the score's sections.

    A program the score already evolved a layer for is driven by that layer;
    anything else falls back to the first. Each pass timecodes the source as its session
    needs, never shorter than before. The cuts are :func:`~syncsummoner.compose.cut.cut_plan`'s: the
    plan the score was composed against, else ``programs`` in rotation over its sections,
    where naming more than there are sections leaves the surplus unrendered, which is
    said rather than silently done. With ``reuse``, a program whose take is
    already in ``takes`` is measured and cut from that take without a pass, and
    ``available`` (a list, or a callable asked only when a pass is due) is
    consulted only for the programs still to render. Returns the plan that was cut.
    """
    from syncsummoner.probe.store import slug

    config = RenderConfig() if config is None else config
    plan = cut_plan(score, programs)
    programs = list(programs) or list(dict.fromkeys(cut.program for cut in plan))
    ordered = list(dict.fromkeys(cut.program for cut in plan))
    paths = {p: str(Path(takes) / f"take-{slug(p)}.mkv") for p in ordered}
    fresh = [p for p in ordered if not (reuse and Path(paths[p]).is_file())]
    pending = [p for p in programs if p in fresh or p not in paths]
    lister = available if fresh and callable(available) else None
    listings = [lister()] if lister is not None else [available]
    unknown = unknown_programs(pending, listings[-1] if fresh else None)
    if unknown and lister is not None:
        LOG.warning(
            "the device does not offer %s in %s; listing again",
            ", ".join(unknown),
            describe_offer(listings[-1]),
        )
        listings.append(lister())
        unknown = unknown_programs(pending, listings[-1])
        LOG.info("second listing: %s", describe_offer(listings[-1], unknown))
    if unknown:
        said = " then ".join(describe_offer(listing, unknown) for listing in listings)
        raise UnknownProgramError(f"the device does not offer {', '.join(unknown)}; it listed {said}")
    run = render_played if pass_render is None else pass_render
    evolved = {layer.program: layer for layer in score.layers}
    Path(takes).mkdir(parents=True, exist_ok=True)
    unused = [p for p in programs if p not in paths]
    if unused:
        LOG.warning(
            "%d sections for %d programs, so %s never comes up; compose a longer span for more",
            len(plan),
            len(programs),
            ", ".join(unused),
        )
    LOG.info(
        "%d cuts over %d programs: %d passes of %s, one load each, %s of capture",
        len(plan),
        len(ordered),
        len(fresh),
        human(score.duration),
        human(len(fresh) * score.duration),
    )
    _passes(
        ordered,
        fresh,
        paths,
        lambda program: run(
            replace(score, layers=[_layer(score, evolved, program)]),
            source,
            paths[program],
            profiles=profiles,
            config=config,
            scratch=scratch,
            prepared=prepared,
            start=start,
        ),
    )
    plan, picked = checked_plan(score, plan, paths, scratch, config=config)
    with stage("splice", cuts=len(plan), out=str(out)):
        splice(
            plan,
            paths,
            picked,
            out,
            config=replace(config, fps=float(score.fps)),
            ffmpeg=getattr(config, "ffmpeg", "ffmpeg"),
        )
    return plan


def _layer(score: Score, evolved: Mapping[str, Layer], program: str) -> Layer:
    """The one layer a program's pass plays: its own evolved layer's gestures, else the score's first."""
    source_layer = evolved.get(program) or (score.layers[0] if score.layers else None)
    return Layer(
        index=0, program=program, gestures=list(source_layer.gestures) if source_layer is not None else []
    )


def _passes(
    ordered: Sequence[str], fresh: Sequence[str], paths: Mapping[str, str], run: Callable[[str], Any]
) -> None:
    """Render each fresh program's pass in order; one whose capture lost frames is recaptured once."""
    waiting, faulted = list(ordered), set()
    with meter(len(ordered), desc="passes", unit="program") as progress:
        while waiting:
            program = waiting.pop(0)
            if program not in fresh:
                LOG.info("%s: reusing take %s", program, paths[program])
            else:
                try:
                    report = run(program)
                except TakeFaultError as err:
                    if program in faulted:
                        raise
                    LOG.warning("%s; recapturing it", err)
                    faulted.add(program)
                    waiting.insert(0, program)
                    continue
                if report is not None and not report.usable:
                    raise BlankTakeError(f"{program}: {report}")
            progress.to(len(ordered) - len(waiting))


def checked_plan(
    score: Score,
    plan: Sequence[Cut],
    takes: Mapping[str, str | Path],
    clip: str | Path,
    *,
    config: RenderConfig,
) -> tuple[list[Cut], dict[str, np.ndarray]]:
    """Judge every take against the clip it was fed, covering dead spans or failing the render.

    Every take is attributed exactly from its lock record (:func:`attribute_take`); returns
    the covered plan and, per take, the capture showing each clip frame, which :func:`splice` cuts by.
    """
    from syncsummoner.compose.takecheck import DeadTakeError, check_takes
    from syncsummoner.probe import decoded  # pylint: disable=import-outside-toplevel

    ffmpeg, spec = getattr(config, "ffmpeg", "ffmpeg"), decoded.Spec.of(config)
    features = {name: decoded.features(take, spec, ffmpeg=ffmpeg) for name, take in takes.items()}
    attributions = {
        name: attribute_take(take, config=config, features=features[name]).picked
        for name, take in takes.items()
    }
    with stage("check", takes=len(takes)):
        report = check_takes(
            score,
            plan,
            takes,
            clip,
            strip_px=config.strip_px,
            bits=config.bits,
            attributions=attributions,
            features=features,
            source=decoded.features(clip, spec, ffmpeg=ffmpeg),
            ffmpeg=ffmpeg,
        )
    for line in report.lines():
        LOG.info("%s", line)
    if report.failed:
        raise DeadTakeError(
            "dead picture no take can cover: " + "; ".join(str(span) for span in report.failed)
        )
    return report.cuts, {name: t.picked for name, t in report.traces.items()}


def open_rig(config: RenderConfig, *, capture: bool = True) -> Rig:
    """Build the real rig from the device layer, capture open and ready to grab.

    A recorded pass wants no capture: the recorder opens the card itself. A device wait seeing nothing for
    longer than ``stall_s``, else the longest load on the rig's record, fails; with neither it refuses.
    """
    from syncsummoner.device.loadtimes import stall_guard  # pylint: disable=import-outside-toplevel

    from syncsummoner.device import capture as capture_mod
    from syncsummoner.device import link as link_mod
    from syncsummoner.device import playout as playout_mod
    from syncsummoner.device import session as session_mod
    from syncsummoner.device import transport as transport_mod

    transport = transport_mod.Transport.open(stall=stall_guard(config.stall_s))
    host = () if config.source_host is None else (config.source_host,)
    return Rig(
        transport=transport,
        session=session_mod.Session(transport),
        capture=(
            capture_mod.Capture(width=config.width, height=config.height, fps=int(config.fps)).open()
            if capture
            else None
        ),
        playout=playout_mod.ClipPlayer(*host, width=config.width, height=config.height),
        link=link_mod.Link(*host),
    )


def enforce_safety(frames: np.ndarray, *, fps: float, mitigate: bool = True) -> np.ndarray:
    """Apply the photosensitive-seizure veto to a finished pass.

    A hard constraint, not a score. Offending windows are temporally low-passed
    rather than discarded, so a take is repaired instead of lost.
    """
    from syncsummoner import aesthetics

    risk = aesthetics.flash_risk(frames, fps=fps)
    if risk.safe:
        return frames
    if not mitigate:
        raise UnsafeOutputError(
            f"{risk.flashes_per_s:.1f} flashes/s over {risk.area_frac:.0%} of frame "
            f"in windows {list(risk.windows)}"
        )
    return aesthetics.mitigate_flashes(frames, risk, fps=fps)

"""Offline multi-pass rendering: burn timecode, play out, capture, align, crop, composite.

Every pass is a real-time capture. Frames self-identify through a croppable
gray-code edge strip, which is what makes alignment possible without reopening
the capture stream. Program is fixed per pass; loading it blacks the output out.
"""

from __future__ import annotations

import subprocess
import time
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, Callable, Iterator, Mapping, Sequence

import numpy as np

from syncsummoner.device.profile import PARAM_MAX, ProgramProfile
from syncsummoner.compose.planner import plan_automation
from syncsummoner.compose.score import Layer, Score
from syncsummoner.compose.vocabulary import Automation


class UnsafeOutputError(RuntimeError):
    """Output tripped a hard safety veto and mitigation was declined."""


class BlankTakeError(RuntimeError):
    """The device never returned a picture, so the take would have been black."""


@dataclass(frozen=True)
class TakeReport:
    """What a finished take actually holds, so a blank one is caught here and not on playback."""

    frames: int
    distinct: int
    blank: int
    luma: float

    @property
    def usable(self) -> bool:
        """Whether the take carries picture at all."""
        return bool(self.frames) and self.blank < self.frames // 2 and self.distinct > 1

    def __str__(self) -> str:
        return (
            f"{self.frames} frames, {self.distinct} distinct, "
            f"{self.blank} blank, mean luma {self.luma:.3f}"
            f"{'' if self.usable else '  UNUSABLE'}"
        )


#: Playout writes the Pi's framebuffer and capture reads the card, so a session is
#: whatever both ends already run at; 1080p30 is this rig, measured.
SESSION_FORMATS = {"720p60": (1280, 720, 60.0), "1080p30": (1920, 1080, 30.0), "ntsc": (720, 480, 29.97)}
EARLY_BIAS_S = 0.015
#: Mean level below which a captured frame carries no picture; the archive uses the same idea.
BLANK_LEVEL = 0.02


@dataclass(frozen=True)
class RenderConfig:
    """Session constants and the measured timing budget for one render."""

    width: int = 1280
    height: int = 720
    fps: float = 60.0
    strip_px: int = 8
    bits: int = 16
    latency_s: float = 0.0
    early_bias_s: float = EARLY_BIAS_S
    cc_budget_hz: float = 200.0
    #: Budget for the picture to return after a load; relock was measured at 12 to 19 seconds.
    settle_s: float = 40.0
    #: Where the liveness probe is recorded before a take begins.
    probe_path: str = "/tmp/syncsummoner-probe.mkv"
    #: ssh target driving playout and the HDMI link; None takes the device layer's default.
    source_host: str | None = None

    @classmethod
    def for_format(cls, name: str, **kw: Any) -> "RenderConfig":
        """Config for a named session format; format is a session constant, never a parameter."""
        width, height, fps = SESSION_FORMATS[name]
        return cls(width=width, height=height, fps=fps, **kw)


@dataclass
class Rig:
    """The real-time endpoints a pass drives, injectable so render stays testable.

    ``link`` is the source HDMI link, held down across every program change; None
    leaves it alone, which only a rig without link control should do.
    """

    session: Any
    capture: Any
    playout: Any
    link: Any = None
    transport: Any = None


def burn_timecode(frame: np.ndarray, index: int, *, bits: int = 16, strip_px: int = 8) -> np.ndarray:
    """Write a gray-coded frame index into a croppable edge strip, behind a constant marker cell."""
    out = np.array(frame, dtype=np.float32, copy=True)
    code = int(index) ^ (int(index) >> 1)
    cells = np.concatenate(([1.0], ((code >> np.arange(bits - 1, -1, -1)) & 1).astype(np.float32)))
    edges = np.round(np.linspace(0, out.shape[1], cells.size + 1)).astype(int)
    out[:strip_px] = np.repeat(cells, np.diff(edges))[None, :, None]
    return out


def read_timecode(
    frame: np.ndarray, *, bits: int = 16, strip_px: int = 8, min_contrast: float = 0.25
) -> int | None:
    """Recover the frame index from the edge strip; ``None`` when the strip is absent or washed out."""
    strip = np.asarray(frame[:strip_px], dtype=np.float64).mean(axis=(0, 2))
    edges = np.round(np.linspace(0, strip.size, bits + 2)).astype(int)
    cells = np.add.reduceat(strip, edges[:-1]) / np.maximum(np.diff(edges), 1)
    if np.ptp(cells) < min_contrast or cells[0] < cells.mean():
        return None
    binary = np.bitwise_xor.accumulate((cells[1:] > cells.mean()).astype(np.int64))
    return int((binary * (1 << np.arange(bits - 1, -1, -1))).sum())


def crop_strip(frame: np.ndarray, *, strip_px: int = 8) -> np.ndarray:
    """Remove the timecode strip before anything else looks at the frame."""
    return frame[strip_px:]


def schedule(auto: Automation, *, latency_s: float = 0.0, early_bias_s: float = EARLY_BIAS_S) -> Automation:
    """Shift automation earlier by the measured latency plus a 10-20 ms bias.

    Audio leading video is detected at roughly 30-50 ms while the reverse
    tolerates over 100 ms, so the error budget is asymmetric: visuals must never
    run late. Gestures already end on their anchor, so arrival lands on the beat.
    """
    return auto.shift(-(latency_s + early_bias_s))


def write_timecoded(
    source: Any, out: str | Path, *, config: RenderConfig, seconds: float | None = None
) -> int:
    """Write the source at session geometry with each frame's index burnt in.

    A clip the playout can identify frame by frame is what lets the source be
    played from the other end, where it can run at rate.
    """
    total, frames = source_stream(source, config, seconds=seconds)
    argv = [
        getattr(config, "ffmpeg", "ffmpeg"),
        "-loglevel",
        "error",
        "-y",
        "-f",
        "rawvideo",
        "-pix_fmt",
        "rgb24",
        "-s",
        f"{config.width}x{config.height}",
        "-r",
        str(config.fps),
        "-i",
        "pipe:0",
        "-c:v",
        "libx264",
        "-crf",
        "14",
        "-preset",
        "veryfast",
        "-pix_fmt",
        "yuv420p",
        str(out),
    ]
    proc = subprocess.Popen(  # pylint: disable=consider-using-with
        argv, stdin=subprocess.PIPE, stdout=subprocess.DEVNULL
    )
    written = 0
    try:
        for index, frame in enumerate(frames):
            stamped = burn_timecode(frame, index, bits=config.bits, strip_px=config.strip_px)
            proc.stdin.write(np.ascontiguousarray((np.clip(stamped, 0, 1) * 255).astype(np.uint8)).data)
            written = index + 1
    finally:
        proc.stdin.close()
        proc.wait()
    return written or total


def drive(rig: Rig, auto: Automation, *, duration: float, clock=time.monotonic, sleep=time.sleep) -> int:
    """Write the automation on a wall clock while the recorder captures.

    The host does nothing per frame: it wakes when a parameter is due, writes it,
    and sleeps again, so the pass is paced by the score rather than by a loop.
    """
    order = np.argsort(auto.times, kind="stable")
    times, indices, values = auto.times[order], auto.indices[order], auto.values[order]
    start, cursor, written = clock(), 0, 0
    while cursor < times.size:
        wait = start + float(times[cursor]) - clock()
        if wait > 0:
            sleep(wait)
        due = max(int(np.searchsorted(times, clock() - start, side="right")), cursor + 1)
        rig.session.set_params(
            {int(k): float(v) / PARAM_MAX for k, v in zip(indices[cursor:due], values[cursor:due])}
        )
        written += due - cursor
        cursor = due
    remaining = start + duration - clock()
    if remaining > 0:
        sleep(remaining)
    return written


def inspect_take(path: str | Path, *, ffmpeg: str = "ffmpeg", samples: int = 240) -> TakeReport:
    """Measure a finished take, so a blank one is caught here and not on playback."""
    import hashlib

    argv = [
        ffmpeg,
        "-loglevel",
        "error",
        "-i",
        str(path),
        "-vf",
        f"scale=160:-2,fps=source_fps/{max(1, samples // 60)}",
        "-f",
        "rawvideo",
        "-pix_fmt",
        "gray",
        "pipe:1",
    ]
    done = subprocess.run(argv, check=False, capture_output=True)
    data = done.stdout
    if not data:
        return TakeReport(frames=0, distinct=0, blank=0, luma=0.0)
    stride = 160 * 90
    frames = [data[i : i + stride] for i in range(0, len(data) - stride + 1, stride)]
    levels = [float(np.frombuffer(f, np.uint8).mean()) / 255.0 for f in frames]
    return TakeReport(
        frames=len(frames),
        distinct=len({hashlib.blake2b(f, digest_size=8).digest() for f in frames}),
        blank=sum(1 for lvl in levels if lvl < BLANK_LEVEL),
        luma=float(np.mean(levels)) if levels else 0.0,
    )


def settle(
    recorder: Any,
    config: RenderConfig,
    *,
    program: str,
    probe_s: float = 2.0,
    clock=time.monotonic,
    sleep=time.sleep,
) -> TakeReport:
    """Wait for the picture to come back after a load, by recording a short probe.

    Liveness is judged the same way a take is, because judging it any other way
    is how a working rig gets called faulted.
    """
    deadline = clock() + config.settle_s
    probe = Path(str(config.probe_path))
    report = TakeReport(frames=0, distinct=0, blank=0, luma=0.0)
    while clock() < deadline:
        with recorder.recording(probe, seconds=probe_s):
            sleep(probe_s)
        report = inspect_take(probe, ffmpeg=getattr(config, "ffmpeg", "ffmpeg"))
        if report.usable:
            probe.unlink(missing_ok=True)
            return report
    probe.unlink(missing_ok=True)
    raise BlankTakeError(f"{program}: no picture within {config.settle_s}s of the load ({report})")


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
    recorder: Any = None,
) -> TakeReport:
    """Render one pass: the rig plays, ffmpeg records, the host only drives.

    Nothing on the host touches a frame while the rig is running, which is what
    a real-time capture needs and what a per-frame loop could never give it.
    """
    layers = sorted(score.layers, key=lambda item: item.index)
    if not layers:
        raise ValueError("score has no layers to render")
    config = RenderConfig() if config is None else config
    owned = rig is None
    rig = open_rig(config, player=True, capture=False) if owned else rig
    if not prepared:
        write_timecoded(source, scratch, config=config, seconds=score.duration)
    if recorder is None:
        from syncsummoner.device.recorder import Recorder

        recorder = Recorder(width=config.width, height=config.height, fps=config.fps)
    autos = plan_automation(score, profiles, fps=config.fps, cc_budget_hz=config.cc_budget_hz)
    auto = schedule(
        autos.get(layers[0].index, Automation.empty()),
        latency_s=config.latency_s,
        early_bias_s=config.early_bias_s,
    )
    try:
        rig.playout.upload(str(scratch))
        with rig.playout.playing(fps=config.fps):
            rig.session.load_program(layers[0].program, link=rig.link)
            rig.session.set_params(rig.session.working_point(rig.transport.program_info()))
            settle(recorder, config, program=layers[0].program)
            with recorder.recording(out, seconds=score.duration):
                drive(rig, auto, duration=score.duration)
    finally:
        if owned and rig.capture is not None:
            rig.capture.close()
    return inspect_take(out, ffmpeg=getattr(config, "ffmpeg", "ffmpeg"))


@dataclass(frozen=True)
class Cut:
    """One span of the timeline and the program that renders it."""

    start: float
    end: float
    program: str

    @property
    def duration(self) -> float:
        """Span length in seconds."""
        return max(0.0, self.end - self.start)


def cut_plan(score: Score, programs: Sequence[str]) -> list[Cut]:
    """Assign programs to the score's sections in rotation.

    Cutting inside a pass is not open to us: a program change blacks this device
    out for seconds, so each program renders the whole take and the spans are
    taken from those afterwards. Sections are where the music already changes.
    """
    if not programs:
        raise ValueError("cutting needs at least one program")
    spans = [(s.start, s.end) for s in score.sections] or [(0.0, score.duration)]
    return [Cut(start, end, programs[i % len(programs)]) for i, (start, end) in enumerate(spans)]


def assemble(
    cuts: Sequence[Cut], takes: Mapping[str, str | Path], out: str | Path, *, ffmpeg: str = "ffmpeg"
) -> None:
    """Splice each cut's span out of its program's take, in order.

    Every take covers the whole timeline, so a span is the same span in each; the
    cut is a choice of which one to show, not a re-timing.
    """
    if not cuts:
        raise ValueError("nothing to assemble")
    argv = [ffmpeg, "-loglevel", "error", "-y"]
    for cut in cuts:
        argv += ["-ss", f"{cut.start:.3f}", "-to", f"{cut.end:.3f}", "-i", str(takes[cut.program])]
    joins = "".join(f"[{i}:v]" for i in range(len(cuts)))
    argv += ["-filter_complex", f"{joins}concat=n={len(cuts)}:v=1:a=0[v]", "-map", "[v]", str(out)]
    done = subprocess.run(argv, check=False, capture_output=True)
    if done.returncode:
        raise RuntimeError(f"assembling the cut failed: {done.stderr.decode('utf-8', 'replace')[:200]}")


def render_cuts(
    score: Score,
    source: Any,
    out: str | Path,
    *,
    profiles: Mapping[str, ProgramProfile],
    programs: Sequence[str],
    config: RenderConfig | None = None,
    scratch: str | Path = "timecoded.mkv",
    prepared: bool = False,
    takes: str | Path = ".",
    pass_render: Callable[..., Any] | None = None,
) -> list[Cut]:
    """Render one pass per program and cut between them on the score's sections.

    Returns the plan that was cut, so a caller can report what came from where.
    """
    config = RenderConfig() if config is None else config
    plan = cut_plan(score, programs)
    run = render_played if pass_render is None else pass_render
    paths: dict[str, str] = {}
    for program in dict.fromkeys(cut.program for cut in plan):
        layer = Layer(
            index=0, program=program, gestures=list(score.layers[0].gestures) if score.layers else []
        )
        take = str(Path(takes) / f"take-{program.replace(' ', '_')}.mkv")
        report = run(
            replace(score, layers=[layer]),
            source,
            take,
            profiles=profiles,
            config=config,
            scratch=scratch,
            prepared=prepared,
        )
        if report is not None and not report.usable:
            raise BlankTakeError(f"{program}: {report}")
        paths[program] = take
    assemble(plan, paths, out, ffmpeg=getattr(config, "ffmpeg", "ffmpeg"))
    return plan


def open_rig(config: RenderConfig, *, player: bool = False, capture: bool = True) -> Rig:
    """Build the real rig from the device layer, capture open and ready to grab.

    Imported here so compose never needs hardware. A recorded pass wants no
    capture at all: ffmpeg opens the card itself, and a handle held here would
    keep it from doing so.
    """
    from syncsummoner.device import capture as capture_mod
    from syncsummoner.device import link as link_mod
    from syncsummoner.device import playout as playout_mod
    from syncsummoner.device import session as session_mod
    from syncsummoner.device import transport as transport_mod

    transport = transport_mod.Transport.open()
    host = () if config.source_host is None else (config.source_host,)
    return Rig(
        transport=transport,
        session=session_mod.Session(transport, cc_budget_hz=config.cc_budget_hz),
        capture=(
            capture_mod.Capture(width=config.width, height=config.height, fps=int(config.fps)).open()
            if capture
            else None
        ),
        playout=(playout_mod.ClipPlayer if player else playout_mod.Playout)(
            *host, width=config.width, height=config.height
        ),
        link=link_mod.Link(*host),
    )


def source_stream(source: Any, config: RenderConfig, *, seconds: float | None = None):
    """Yield ``(total, frames)``: how many session frames a take is, and them one at a time.

    A long take is never held: 180s at 1080p30 is 134GB as a stack, and the pass
    only ever looks at one frame.
    """
    from syncsummoner.compose.features import read_frames

    probe, rate = [], config.fps
    for rate, frame in read_frames(source):
        probe.append(frame)
        break
    if not probe:
        return 0, iter(())
    src_fps = rate if rate > 0 else config.fps
    total = int(round((seconds or 0.0) * config.fps)) if seconds else 0

    def frames() -> Iterator[np.ndarray]:
        held, index = None, 0
        source_index = -1
        for _, frame in read_frames(source):
            source_index += 1
            held = frame
            while index * src_fps / config.fps <= source_index:
                yield _conform(np.asarray(held, dtype=np.float32)[None], config)[0]
                index += 1
                if total and index >= total:
                    return

    return total, frames()


def _conform(stack: np.ndarray, config: RenderConfig) -> np.ndarray:
    """Fit the source into the session raster, padding rather than stretching it.

    Playout takes the session geometry and nothing else, and the aspect a source
    was shot at is not the rig's to change: it is centred, and the rest is black.
    """
    # pylint: disable=no-member
    import cv2

    if not stack.shape[0] or stack.shape[1:3] == (config.height, config.width):
        return stack
    height, width = stack.shape[1:3]
    scale = min(config.width / width, config.height / height)
    fit = (max(1, round(width * scale)), max(1, round(height * scale)))
    left, top = (config.width - fit[0]) // 2, (config.height - fit[1]) // 2
    out = np.zeros((stack.shape[0], config.height, config.width, stack.shape[3]), dtype=np.float32)
    for i, frame in enumerate(stack):
        out[i, top : top + fit[1], left : left + fit[0]] = cv2.resize(
            frame, fit, interpolation=cv2.INTER_AREA
        )
    return out


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

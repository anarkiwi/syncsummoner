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
from syncsummoner.device.recorder import BlankTakeError, TakeReport, inspect_take, settle
from syncsummoner.compose.planner import plan_automation
from syncsummoner.compose.score import Layer, Score
from syncsummoner.compose.vocabulary import Automation
from syncsummoner.progress import LOG, human, stage, track


class UnsafeOutputError(RuntimeError):
    """Output tripped a hard safety veto and mitigation was declined."""


#: Relock after a load was measured at 12 to 19s, against a settle budget that is
#: a timeout; half the budget is what a pass is expected to wait, for estimates.
EXPECTED_RELOCK_FRAC = 0.5
#: Playout writes the Pi's framebuffer and capture reads the card, so a session is
#: whatever both ends already run at; 1080p30 is this rig, measured.
SESSION_FORMATS = {"720p60": (1280, 720, 60.0), "1080p30": (1920, 1080, 30.0), "ntsc": (720, 480, 29.97)}
EARLY_BIAS_S = 0.015


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
    #: Recorded before the clip is started, so the picture's own first frame is inside the take.
    lead_s: float = 3.0
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
    """Write a gray-coded frame index into a croppable bottom edge strip, behind a marker cell."""
    out = np.array(frame, dtype=np.float32, copy=True)
    code = int(index) ^ (int(index) >> 1)
    cells = np.concatenate(([1.0], ((code >> np.arange(bits - 1, -1, -1)) & 1).astype(np.float32)))
    edges = np.round(np.linspace(0, out.shape[1], cells.size + 1)).astype(int)
    out[out.shape[0] - strip_px :] = np.repeat(cells, np.diff(edges))[None, :, None]
    return out


def read_timecode(
    frame: np.ndarray, *, bits: int = 16, strip_px: int = 8, min_contrast: float = 0.25
) -> int | None:
    """Recover the frame index from the edge strip; ``None`` when the strip is absent or washed out."""
    strip = np.asarray(frame[frame.shape[0] - strip_px :], dtype=np.float64).mean(axis=(0, 2))
    edges = np.round(np.linspace(0, strip.size, bits + 2)).astype(int)
    cells = np.add.reduceat(strip, edges[:-1]) / np.maximum(np.diff(edges), 1)
    if np.ptp(cells) < min_contrast or cells[0] < cells.mean():
        return None
    binary = np.bitwise_xor.accumulate((cells[1:] > cells.mean()).astype(np.int64))
    return int((binary * (1 << np.arange(bits - 1, -1, -1))).sum())


def picture_start(path: str | Path, *, config: RenderConfig, search_s: float | None = None) -> float:
    """Seconds into a take at which the played clip's first frame lands.

    Lead-in is not blank: it holds whatever the framebuffer last showed, which is
    the previous play's final frame and carries that frame's own timecode. Played
    at rate, every fresh frame has the same lag between where it sits in the take
    and the frame it says it is, so the lag they agree on is the answer and one
    misread strip cannot move it. The card returns frames at its own rate, so
    they are resampled to the session's before an index means a time.
    """
    search = config.lead_s + 2.0 if search_s is None else search_s
    argv = [
        getattr(config, "ffmpeg", "ffmpeg"),
        "-loglevel",
        "error",
        "-i",
        str(path),
        "-t",
        f"{search:.3f}",
        "-vf",
        f"fps={config.fps},crop=iw:{config.strip_px}:0:ih-{config.strip_px}",
        "-f",
        "rawvideo",
        "-pix_fmt",
        "rgb24",
        "pipe:1",
    ]
    data = subprocess.run(argv, check=False, capture_output=True).stdout
    stride = config.width * config.strip_px * 3
    strips = np.frombuffer(data[: len(data) - len(data) % stride], np.uint8)
    if not strips.size:
        return 0.0
    strips = strips.reshape(-1, config.strip_px, config.width, 3).astype(np.float32) / 255.0
    codes = [read_timecode(s, bits=config.bits, strip_px=config.strip_px) for s in strips]
    lags = np.array([i - c for i, c in enumerate(codes) if c is not None], dtype=np.int64)
    if not lags.size:
        return 0.0
    values, counts = np.unique(lags, return_counts=True)
    return max(0.0, float(values[counts.argmax()]) / config.fps)


def crop_strip(frame: np.ndarray, *, strip_px: int = 8) -> np.ndarray:
    """Remove the timecode strip before anything else looks at the frame."""
    return frame[: frame.shape[0] - strip_px]


def schedule(auto: Automation, *, latency_s: float = 0.0, early_bias_s: float = EARLY_BIAS_S) -> Automation:
    """Shift automation earlier by the measured latency plus a 10-20 ms bias.

    Audio leading video is detected at roughly 30-50 ms while the reverse
    tolerates over 100 ms, so the error budget is asymmetric: visuals must never
    run late. Gestures already end on their anchor, so arrival lands on the beat.
    """
    return auto.shift(-(latency_s + early_bias_s))


def write_timecoded(
    source: Any,
    out: str | Path,
    *,
    config: RenderConfig,
    seconds: float | None = None,
    start: float = 0.0,
) -> int:
    """Write the source at session geometry with each frame's index burnt in.

    A clip the playout can identify frame by frame is what lets the source be
    played from the other end, where it can run at rate. ``start`` skips into it,
    which is how an excerpt gets the footage it was composed against. An encoder
    that dies takes its own diagnosis with it, so that is what is raised rather
    than the broken pipe writing to it produces.
    """
    total, frames = source_stream(source, config, seconds=seconds, start=start)
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
        argv, stdin=subprocess.PIPE, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE
    )
    written = 0
    try:
        for index, frame in enumerate(track(frames, desc="timecoding", total=total or None, unit="frame")):
            stamped = burn_timecode(frame, index, bits=config.bits, strip_px=config.strip_px)
            try:
                proc.stdin.write(np.ascontiguousarray((np.clip(stamped, 0, 1) * 255).astype(np.uint8)).data)
            except BrokenPipeError:
                break
            written = index + 1
    finally:
        proc.stdin.close()
        said = (proc.stderr.read() or b"").decode("utf-8", "replace").strip()
        proc.stderr.close()
        proc.wait()
    if proc.returncode:
        raise RuntimeError(f"timecoding {out} failed: {said.splitlines()[-1] if said else 'no output'}")
    LOG.info("timecoded %d frames into %s at %dx%d", written or total, out, config.width, config.height)
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
    """Render one pass: the rig plays, ffmpeg records, the host only drives.

    Nothing on the host touches a frame while the rig is running, which is what
    a real-time capture needs and what a per-frame loop could never give it. The
    clip is played twice: once as the moving picture the load's liveness probe
    needs, then again, inside the recording, for the take itself. The take opens
    on lead-in the picture has not reached yet; :func:`picture_start` says where
    it does.
    """
    layers = sorted(score.layers, key=lambda item: item.index)
    if not layers:
        raise ValueError("score has no layers to render")
    config = RenderConfig() if config is None else config
    owned = rig is None
    rig = open_rig(config, player=True, capture=False) if owned else rig
    if not prepared:
        write_timecoded(source, scratch, config=config, seconds=score.duration, start=start)
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
        with stage("upload", clip=str(scratch)) as sent:
            sent["bytes"] = rig.playout.upload(str(scratch))
        with rig.playout.playing(fps=config.fps):
            with stage("load", program=layers[0].program):
                rig.session.load_program(layers[0].program, link=rig.link)
                rig.session.set_params(rig.session.working_point(rig.transport.program_info()))
                settle(
                    recorder,
                    program=layers[0].program,
                    timeout_s=config.settle_s,
                    probe_path=config.probe_path,
                )
        with stage("pass", program=layers[0].program, seconds=f"{score.duration:.1f}") as written:
            with recorder.recording(out, seconds=score.duration + config.lead_s):
                with rig.playout.playing(fps=config.fps):
                    written["writes"] = drive(rig, auto, duration=score.duration)
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
    cuts: Sequence[Cut],
    takes: Mapping[str, str | Path],
    out: str | Path,
    *,
    ffmpeg: str = "ffmpeg",
    starts: Mapping[str, float] | None = None,
) -> None:
    """Splice each cut's span out of its program's take, in order.

    Every take covers the whole timeline, so a span is the same span in each; the
    cut is a choice of which one to show, not a re-timing. ``starts`` carries
    where each take's picture begins, since a pass opens on lead-in.
    """
    if not cuts:
        raise ValueError("nothing to assemble")
    offset = dict(starts or {})
    argv = [ffmpeg, "-loglevel", "error", "-y"]
    for cut in cuts:
        head = offset.get(cut.program, 0.0)
        argv += [
            "-ss",
            f"{head + cut.start:.3f}",
            "-to",
            f"{head + cut.end:.3f}",
            "-i",
            str(takes[cut.program]),
        ]
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
    start: float = 0.0,
    takes: str | Path = ".",
    pass_render: Callable[..., Any] | None = None,
) -> list[Cut]:
    """Render one pass per program and cut between them on the score's sections.

    A program the score already evolved a layer for is driven by that layer;
    anything else falls back to the first. The timecoded source is built once and
    replayed by every pass. Programs are assigned to sections in rotation, so
    naming more than there are sections leaves the surplus unrendered, which is
    said rather than silently done. Returns the plan that was cut.
    """
    config = RenderConfig() if config is None else config
    plan = cut_plan(score, programs)
    run = render_played if pass_render is None else pass_render
    evolved = {layer.program: layer for layer in score.layers}
    Path(takes).mkdir(parents=True, exist_ok=True)
    unused = [p for p in programs if p not in {cut.program for cut in plan}]
    if unused:
        LOG.warning(
            "%d sections for %d programs, so %s never comes up; compose a longer span for more",
            len(plan),
            len(programs),
            ", ".join(unused),
        )
    passes = len(dict.fromkeys(cut.program for cut in plan))
    LOG.info(
        "%d cuts over %d programs: %d passes of %s plus a relock each, about %s of rig time",
        len(plan),
        passes,
        passes,
        human(score.duration),
        human(passes * (score.duration + config.settle_s * EXPECTED_RELOCK_FRAC)),
    )
    if not prepared:
        write_timecoded(source, scratch, config=config, seconds=score.duration, start=start)
        prepared = True
    paths: dict[str, str] = {}
    starts: dict[str, float] = {}
    ordered = list(dict.fromkeys(cut.program for cut in plan))
    for program in track(ordered, desc="passes", total=len(ordered), unit="program"):
        source_layer = evolved.get(program) or (score.layers[0] if score.layers else None)
        layer = Layer(
            index=0,
            program=program,
            gestures=list(source_layer.gestures) if source_layer is not None else [],
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
        starts[program] = picture_start(take, config=config)
        LOG.info("%s: picture starts %.2fs into the take", program, starts[program])
    with stage("assemble", cuts=len(plan), out=str(out)):
        assemble(plan, paths, out, ffmpeg=getattr(config, "ffmpeg", "ffmpeg"), starts=starts)
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


def source_stream(source: Any, config: RenderConfig, *, seconds: float | None = None, start: float = 0.0):
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
    skip = max(0, int(round(start * src_fps)))

    def frames() -> Iterator[np.ndarray]:
        held, index = None, 0
        source_index = -1
        for _, frame in read_frames(source):
            source_index += 1
            if source_index < skip:
                continue
            held = frame
            while index * src_fps / config.fps + skip <= source_index:
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

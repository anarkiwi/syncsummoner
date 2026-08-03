"""Offline multi-pass rendering: burn timecode, play out, capture, align, crop, composite.

Every pass is a real-time capture. Frames self-identify through a croppable
gray-code edge strip, which is what makes alignment possible without reopening
the capture stream. Program is fixed per pass; loading it blacks the output out.
"""

from __future__ import annotations

import subprocess
from contextlib import nullcontext
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


def align(captured: Mapping[int, np.ndarray], n_frames: int, shape: tuple[int, ...]) -> np.ndarray:
    """Order captured frames by their decoded timecode, holding the last good frame over gaps."""
    out = np.zeros((n_frames,) + tuple(shape), dtype=np.float32)
    have = np.zeros(n_frames, dtype=bool)
    for k, frame in captured.items():
        out[k] = frame
        have[k] = True
    if not have.any():
        return out
    idx = np.maximum.accumulate(np.where(have, np.arange(n_frames), 0))
    idx[: int(np.argmax(have))] = int(np.argmax(have))
    return out[idx]


class FrameSink:
    """Writes a pass out as it is captured, so a take is never held whole.

    Frames arrive by index and leave in order, the last good one held over a gap.
    Safety runs on a window rather than the finished take: flash risk is a rate,
    so a window is where it can be judged at all.
    """

    def __init__(self, write: Callable[[np.ndarray], None], *, fps: float, window: int = 60):
        self.write = write
        self.fps = float(fps)
        self.window = max(2, int(window))
        self.pending: dict[int, np.ndarray] = {}
        self.next = 0
        self.held: np.ndarray | None = None
        self.buffer: list[np.ndarray] = []
        self.emitted = 0
        self.blank = 0
        self.levels = 0.0
        self._seen: set[bytes] = set()

    @property
    def report(self) -> "TakeReport":
        """What has gone out so far, which is what a caller should check before shipping it."""
        return TakeReport(
            frames=self.emitted,
            distinct=len(self._seen),
            blank=self.blank,
            luma=self.levels / self.emitted if self.emitted else 0.0,
        )

    def add(self, index: int, frame: np.ndarray) -> None:
        """Take one captured frame, emitting whatever its arrival completes."""
        if index >= self.next:
            self.pending[index] = frame
        self._drain(len(self.pending) > self.window)

    def _drain(self, forced: bool) -> None:
        while self.pending and (forced or self.next in self.pending):
            if self.next in self.pending:
                self.held = self.pending.pop(self.next)
            elif self.held is None:
                self.held = self.pending[min(self.pending)]
            self._emit(self.held)
            self.next += 1
            forced = len(self.pending) > self.window

    def _emit(self, frame: np.ndarray) -> None:
        level = float(np.asarray(frame[::16, ::16]).mean())
        self.emitted += 1
        self.levels += level
        self.blank += level < BLANK_LEVEL
        self._seen.add(np.ascontiguousarray(frame[::32, ::32]).tobytes())
        self.buffer.append(frame)
        if len(self.buffer) >= self.window:
            self._flush()

    def _flush(self) -> None:
        if not self.buffer:
            return
        for frame in enforce_safety(np.stack(self.buffer), fps=self.fps):
            self.write(frame)
        self.buffer.clear()

    def close(self, total: int) -> None:
        """Emit everything left, holding the last frame out to ``total``."""
        self._drain(True)
        while self.next < total and self.held is not None:
            self._emit(self.held)
            self.next += 1
        self._flush()


def play_pass_stream(
    rig: Rig,
    frames: "Iterator[np.ndarray]",
    auto: Automation,
    *,
    program: str,
    config: RenderConfig,
    sink: FrameSink,
    total: int,
) -> None:
    """One real-time pass that never holds the take: each frame is written as it lands."""
    rig.session.load_program(program, link=rig.link)
    order = np.argsort(auto.times, kind="stable")
    times, indices, values = auto.times[order], auto.indices[order], auto.values[order]
    cursor, lag, stamped = 0, 0, 0
    streaming = getattr(rig.playout, "streaming", None)
    with streaming() if streaming is not None else nullcontext():
        _drive(rig, frames, config, times, indices, values, cursor, lag, stamped, sink, total)
    sink.close(total)


def _drive(rig, frames, config, times, indices, values, cursor, lag, stamped, sink, total):
    """The frame loop of a streaming pass: show, grab, place."""
    for i, frame in enumerate(frames):
        due = int(np.searchsorted(times, i / config.fps, side="right"))
        if due > cursor:
            rig.session.set_params(
                {int(k): float(v) / PARAM_MAX for k, v in zip(indices[cursor:due], values[cursor:due])}
            )
            cursor = due
        rig.playout.show(burn_timecode(frame, i, bits=config.bits, strip_px=config.strip_px))
        got = rig.capture.read()
        if got is None:
            continue
        tc = read_timecode(got, bits=config.bits, strip_px=config.strip_px)
        if tc is not None and 0 <= tc < total:
            lag, stamped = (lag * stamped + (i - tc)) // (stamped + 1), stamped + 1
        sink.add(
            tc if tc is not None and 0 <= tc < total else i - lag, crop_strip(got, strip_px=config.strip_px)
        )
    sink.close(total)


def play_pass(
    rig: Rig, frames: np.ndarray, auto: Automation, *, program: str, config: RenderConfig
) -> np.ndarray:
    """Run one real-time pass: load the program once, play out, drive CC, capture and place.

    A frame whose strip does not decode is still a frame the instrument made, so
    it is placed by arrival less the capture lag rather than discarded: a program
    that eats its own timecode used to render as nothing at all.
    """
    rig.session.load_program(program, link=rig.link)
    order = np.argsort(auto.times, kind="stable")
    times, indices, values = auto.times[order], auto.indices[order], auto.values[order]
    arrived: dict[int, np.ndarray] = {}
    stamps: dict[int, int] = {}
    cursor = 0
    shape: tuple[int, ...] = (max(1, frames.shape[1] - config.strip_px),) + tuple(frames.shape[2:])
    for i in range(frames.shape[0]):
        due = int(np.searchsorted(times, i / config.fps, side="right"))
        if due > cursor:
            rig.session.set_params(
                {int(k): float(v) / PARAM_MAX for k, v in zip(indices[cursor:due], values[cursor:due])}
            )
            cursor = due
        rig.playout.show(burn_timecode(frames[i], i, bits=config.bits, strip_px=config.strip_px))
        got = rig.capture.read()
        if got is None:
            continue
        arrived[i] = crop_strip(got, strip_px=config.strip_px)
        shape = arrived[i].shape
        tc = read_timecode(got, bits=config.bits, strip_px=config.strip_px)
        if tc is not None and 0 <= tc < frames.shape[0]:
            stamps[i] = tc
    return align(_placed(arrived, stamps, frames.shape[0]), frames.shape[0], shape)


def _placed(
    arrived: Mapping[int, np.ndarray], stamps: Mapping[int, int], n_frames: int
) -> dict[int, np.ndarray]:
    """Index every captured frame, by its own stamp where it has one and by arrival otherwise.

    The lag between showing a frame and capturing it is measured from the frames
    that did decode, so an undecodable strip costs alignment accuracy rather than
    the whole take.
    """
    lags = [arrival - stamp for arrival, stamp in stamps.items()]
    lag = int(round(float(np.median(lags)))) if lags else 0
    placed: dict[int, np.ndarray] = {}
    for arrival, frame in arrived.items():
        index = stamps.get(arrival, arrival - lag)
        if 0 <= index < n_frames:
            placed[index] = frame
    return placed


def composite(passes: Sequence[np.ndarray], *, mode: str = "screen") -> np.ndarray:
    """Combine per-pass frame stacks offline; layering is where depth beyond one program comes from."""
    shapes = np.array([p.shape[:3] for p in passes]).min(axis=0)
    stack = np.stack([np.asarray(p[: shapes[0], : shapes[1], : shapes[2]], dtype=np.float32) for p in passes])
    if mode == "mean":
        return stack.mean(axis=0)
    if mode == "max":
        return stack.max(axis=0)
    return 1.0 - np.prod(1.0 - np.clip(stack, 0.0, 1.0), axis=0)


def write_video(path: str | Path, frames: np.ndarray, fps: float) -> None:
    """Write an RGB float32 frame stack; OpenCV's BGR convention stops at this boundary."""
    # pylint: disable=no-member
    import cv2

    h, w = frames.shape[1:3]
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"FFV1"), fps, (w, h))
    for frame in frames:
        writer.write(cv2.cvtColor((np.clip(frame, 0, 1) * 255).astype(np.uint8), cv2.COLOR_RGB2BGR))
    writer.release()


class RawSink:
    """Writes frames verbatim, so the capture loop pays for no encoder.

    Lossless 1080p costs hundreds of percent of CPU to encode, which starves the
    loop that is meant to be grabbing at session rate: the card then hands back
    the same buffer and the take fills with repeats. Raw bytes to an NVMe cost
    almost nothing, and the encode is a job for afterwards.
    """

    def __init__(self, path: str | Path, fps: float):
        self.path = str(path)
        self.fps = float(fps)
        self.shape: tuple[int, ...] = ()
        self._handle: Any = None

    def write(self, frame: np.ndarray) -> None:
        """Append one frame as raw uint8."""
        if self._handle is None:
            self.shape = frame.shape
            self._handle = open(self.path, "wb")  # pylint: disable=consider-using-with
        self._handle.write(np.ascontiguousarray((np.clip(frame, 0, 1) * 255).astype(np.uint8)).data)

    def close(self) -> None:
        """Finish the file."""
        if self._handle is not None:
            self._handle.close()
            self._handle = None

    def encode(self, out: str | Path, *, ffmpeg: str = "ffmpeg", crf: int = 16) -> None:
        """Turn the raw take into a video, once the rig is no longer waiting on us."""
        if not self.shape:
            raise ValueError(f"nothing was written to {self.path}")
        height, width = self.shape[:2]
        done = subprocess.run(
            [
                ffmpeg,
                "-loglevel",
                "error",
                "-y",
                "-f",
                "rawvideo",
                "-pix_fmt",
                "rgb24",
                "-s",
                f"{width}x{height}",
                "-r",
                str(self.fps),
                "-i",
                self.path,
                "-c:v",
                "libx264",
                "-crf",
                str(crf),
                "-preset",
                "veryfast",
                "-pix_fmt",
                "yuv420p",
                str(out),
            ],
            check=False,
            capture_output=True,
        )
        if done.returncode:
            raise RuntimeError(f"encoding {self.path} failed: {done.stderr.decode('utf-8','replace')[:200]}")


class VideoSink:
    """An encoder that takes frames one at a time, opened once the first one arrives."""

    def __init__(self, path: str | Path, fps: float, *, ffmpeg: str = "ffmpeg"):
        self.path = str(path)
        self.fps = float(fps)
        self.ffmpeg = ffmpeg
        self._writer: Any = None

    def write(self, frame: np.ndarray) -> None:
        """Hand one RGB float32 frame to the encoder, which runs in its own process.

        Encoding in the capture loop cannot keep up: OpenCV's FFV1 measured 189ms
        a frame, so a pass sampled one frame in six and held the rest.
        """
        if self._writer is None:
            self._writer = self._open(frame.shape[1], frame.shape[0])
        try:
            self._writer.stdin.write(np.ascontiguousarray((np.clip(frame, 0, 1) * 255).astype(np.uint8)).data)
        except (BrokenPipeError, OSError) as exc:
            raise RuntimeError(f"the encoder stopped taking frames for {self.path}") from exc

    def _open(self, width: int, height: int) -> Any:
        """Start ffmpeg reading raw frames on stdin, as the frame archive does."""
        return subprocess.Popen(
            [
                self.ffmpeg,
                "-loglevel",
                "error",
                "-y",
                "-f",
                "rawvideo",
                "-pix_fmt",
                "rgb24",
                "-s",
                f"{width}x{height}",
                "-r",
                str(self.fps),
                "-i",
                "pipe:0",
                "-c:v",
                "ffv1",
                "-level",
                "3",
                "-slices",
                "4",
                "-slicecrc",
                "1",
                self.path,
            ],
            stdin=subprocess.PIPE,
            stdout=subprocess.DEVNULL,
        )

    def close(self) -> None:
        """Finish the file, if anything was ever written to it."""
        if self._writer is None:
            return
        writer, self._writer = self._writer, None
        try:
            writer.stdin.close()
        except (BrokenPipeError, OSError):
            pass
        writer.wait()


def write_timecoded(
    source: Any, out: str | Path, *, config: RenderConfig, seconds: float | None = None
) -> int:
    """Write the source at session geometry with each frame's index burnt in.

    A clip the playout can identify frame by frame is what lets the source be
    played from the other end, where it can run at rate.
    """
    total, frames = source_stream(source, config, seconds=seconds)
    video = VideoSink(out, config.fps)
    written = 0
    try:
        for index, frame in enumerate(frames):
            video.write(burn_timecode(frame, index, bits=config.bits, strip_px=config.strip_px))
            written = index + 1
    finally:
        video.close()
    return written or total


def capture_pass(
    rig: Rig,
    auto: Automation,
    *,
    program: str,
    config: RenderConfig,
    sink: FrameSink,
    total: int,
    timeout_s: float = 30.0,
    settle_s: float = 40.0,
) -> int:
    """Capture a pass whose source plays itself, driving automation from what arrives.

    Position comes from the frame the card hands over, so the schedule follows the
    playback rather than a clock the host keeps on its own. A load blacks the
    output out for seconds, so the take does not start until the picture is back.
    """
    rig.session.load_program(program, link=rig.link)
    if not rig.capture.wait_for_content(timeout_s=settle_s):
        raise BlankTakeError(f"{program}: no moving picture within {settle_s}s of the load")
    order = np.argsort(auto.times, kind="stable")
    times, indices, values = auto.times[order], auto.indices[order], auto.values[order]
    cursor, seen, idle, lag = 0, 0, 0.0, 0
    while seen < total and idle < timeout_s:
        got = rig.capture.read()
        if got is None:
            idle += 1.0 / config.fps
            continue
        idle = 0.0
        stamp = read_timecode(got, bits=config.bits, strip_px=config.strip_px)
        index = stamp if stamp is not None and 0 <= stamp < total else seen - lag
        if stamp is not None and 0 <= stamp < total:
            lag = seen - stamp
        due = int(np.searchsorted(times, index / config.fps, side="right"))
        if due > cursor:
            rig.session.set_params(
                {int(k): float(v) / PARAM_MAX for k, v in zip(indices[cursor:due], values[cursor:due])}
            )
            cursor = due
        sink.add(index, crop_strip(got, strip_px=config.strip_px))
        seen += 1
    sink.close(total)
    return seen


def render_stream(
    score: Score,
    source: Any,
    out: str | Path,
    *,
    profiles: Mapping[str, ProgramProfile],
    rig: Rig | None = None,
    config: RenderConfig | None = None,
    sink: FrameSink | None = None,
) -> None:
    """Render one pass straight to ``out``, holding neither the source nor the take.

    The multi-pass path composites, so it needs both in hand; a single pass does
    not, and a full length take is hundreds of gigabytes it would rather not hold.
    """
    layers = sorted(score.layers, key=lambda item: item.index)
    if not layers:
        raise ValueError("score has no layers to render")
    config = RenderConfig() if config is None else config
    owned = rig is None
    rig = open_rig(config) if owned else rig
    video = VideoSink(out, config.fps)
    try:
        autos = plan_automation(score, profiles, fps=config.fps, cc_budget_hz=config.cc_budget_hz)
        auto = schedule(
            autos.get(layers[0].index, Automation.empty()),
            latency_s=config.latency_s,
            early_bias_s=config.early_bias_s,
        )
        total, frames = source_stream(source, config, seconds=score.duration)
        play_pass_stream(
            rig,
            frames,
            auto,
            program=layers[0].program,
            config=config,
            sink=FrameSink(video.write, fps=config.fps) if sink is None else sink,
            total=total,
        )
    finally:
        video.close()
        if owned:
            rig.capture.close()


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
    raw: str | Path | None = None,
) -> "TakeReport":
    """Render a pass the playout plays for itself, at rate.

    Pushing frames caps a pass at a few a second, which samples the instrument's
    own motion far too slowly to be the performance it is meant to record. The
    source goes over once, timecoded, and the host only drives and captures.
    ``raw`` writes the take verbatim and encodes it afterwards, so the encoder
    never competes with the capture loop for the CPU.
    """
    layers = sorted(score.layers, key=lambda item: item.index)
    if not layers:
        raise ValueError("score has no layers to render")
    config = RenderConfig() if config is None else config
    owned = rig is None
    rig = open_rig(config, player=True) if owned else rig
    sink_of: Any = RawSink(raw, config.fps) if raw else VideoSink(out, config.fps)
    total = (
        int(round(score.duration * config.fps))
        if prepared
        else write_timecoded(source, scratch, config=config, seconds=score.duration)
    )
    try:
        autos = plan_automation(score, profiles, fps=config.fps, cc_budget_hz=config.cc_budget_hz)
        auto = schedule(
            autos.get(layers[0].index, Automation.empty()),
            latency_s=config.latency_s,
            early_bias_s=config.early_bias_s,
        )
        rig.playout.upload(str(scratch))
        sink = FrameSink(sink_of.write, fps=config.fps)
        with rig.playout.playing(fps=config.fps):
            capture_pass(
                rig,
                auto,
                program=layers[0].program,
                config=config,
                sink=sink,
                total=total,
            )
        report = sink.report
    finally:
        sink_of.close()
        if owned:
            rig.capture.close()
    if raw:
        sink_of.encode(out, ffmpeg=getattr(config, "ffmpeg", "ffmpeg"))
    return report


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


def open_rig(config: RenderConfig, *, player: bool = False) -> Rig:
    """Build the real rig from the device layer, capture open and ready to grab.

    Imported here so compose never needs hardware. The capture is opened rather
    than merely constructed: a pass grabs frame by frame and never enters it as a
    context, so an unopened handle only fails once the first frame is due.
    """
    from syncsummoner.device import capture as capture_mod
    from syncsummoner.device import link as link_mod
    from syncsummoner.device import playout as playout_mod
    from syncsummoner.device import session as session_mod
    from syncsummoner.device import transport as transport_mod

    transport = transport_mod.Transport.open()
    host = () if config.source_host is None else (config.source_host,)
    return Rig(
        session=session_mod.Session(transport, cc_budget_hz=config.cc_budget_hz),
        capture=capture_mod.Capture(width=config.width, height=config.height, fps=int(config.fps)).open(),
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


def _source_frames(source: Any, config: RenderConfig, *, seconds: float | None = None) -> np.ndarray:
    """Source frames at the session rate, decoding only the span a pass will use.

    A pass consumes one source frame per session frame, so a clip at its own rate
    plays at the wrong speed; frames are resampled by index rather than blended,
    which repeats or drops whole frames exactly as playout would.
    """
    if isinstance(source, np.ndarray):
        stack, fps = source.astype(np.float32), config.fps
    else:
        from syncsummoner.compose.features import read_frames

        frames, fps, budget = [], config.fps, None
        for rate, frame in read_frames(source):
            fps = rate
            budget = None if seconds is None else max(1, int(round(seconds * rate)))
            frames.append(frame)
            if budget is not None and len(frames) >= budget:
                break
        stack = np.stack(frames).astype(np.float32) if frames else np.zeros((0, 1, 1, 3), np.float32)
    if stack.shape[0] and fps > 0 and abs(fps - config.fps) >= 1e-6:
        wanted = max(1, int(round(stack.shape[0] * config.fps / fps)))
        stack = stack[np.minimum((np.arange(wanted) * fps / config.fps).astype(int), stack.shape[0] - 1)]
    return _conform(stack, config)


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


def _passes(
    score: Score,
    frames: np.ndarray,
    profiles: Mapping[str, ProgramProfile],
    rig: Rig,
    config: RenderConfig,
    n_passes: int,
) -> list[np.ndarray]:
    """Run each layer in turn, re-feeding the previous pass so program stays the outer loop."""
    autos = plan_automation(score, profiles, fps=config.fps, cc_budget_hz=config.cc_budget_hz)
    results = []
    current = frames
    for layer in sorted(score.layers, key=lambda item: item.index)[:n_passes]:
        auto = schedule(
            autos.get(layer.index, Automation.empty()),
            latency_s=config.latency_s,
            early_bias_s=config.early_bias_s,
        )
        current = play_pass(rig, current, auto, program=layer.program, config=config)
        results.append(current)
    return results


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


def render(
    score: Score,
    source: Any,
    out: str | Path,
    *,
    passes: int = 1,
    profiles: Mapping[str, ProgramProfile] | None = None,
    rig: Rig | None = None,
    config: RenderConfig | None = None,
    sink: Callable[[str | Path, np.ndarray, float], None] | None = None,
    mode: str = "screen",
) -> None:
    """Render the full score to ``out`` as one real-time capture pass per layer."""
    if profiles is None:
        raise ValueError("render requires the measured profiles the score was planned against")
    config = RenderConfig() if config is None else config
    owned = rig is None
    rig = open_rig(config) if owned else rig
    try:
        frames = _source_frames(source, config, seconds=score.duration)
        results = _passes(score, frames, profiles, rig, config, passes)
        if not results:
            raise ValueError("score has no layers to render")
        final = composite(results, mode=mode) if len(results) > 1 else results[0]
        (sink or write_video)(out, enforce_safety(final, fps=config.fps), config.fps)
    finally:
        if owned:
            rig.capture.close()


def audition(
    score: Score,
    source: Any,
    *,
    seconds: float = 30.0,
    scale: float = 0.25,
    passes: int = 1,
    profiles: Mapping[str, ProgramProfile] | None = None,
    rig: Rig | None = None,
    config: RenderConfig | None = None,
    mode: str = "screen",
) -> np.ndarray:
    """Render a short downscaled excerpt for planner iteration, returning the frames."""
    if profiles is None:
        raise ValueError("audition requires the measured profiles the score was planned against")
    config = RenderConfig() if config is None else config
    owned = rig is None
    rig = open_rig(config) if owned else rig
    try:
        frames = _source_frames(source, config, seconds=seconds)[: max(1, int(round(seconds * config.fps)))]
        step = max(1, int(round(1.0 / max(scale, 1e-6))))
        frames = np.ascontiguousarray(frames[:, ::step, ::step])
        results = _passes(score, frames, profiles, rig, config, passes)
        return composite(results, mode=mode) if len(results) > 1 else results[0]
    finally:
        if owned:
            rig.capture.close()

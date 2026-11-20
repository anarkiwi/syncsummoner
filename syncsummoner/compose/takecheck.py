"""Check captured takes against the source frames they were fed, before anything is mastered.

Every capture is attributed to the clip frame it shows, so each bar of each take is judged
by what it kept of that bar's picture and motion: a flat bar is blank and a frozen bar static,
either is dead, and a cut that shows one is covered from a take alive there or the render fails.
Whether the output's motion follows the source's is no part of being alive. A bar is also unusable
where the pictures the take would show there fail the general or red flash test
(:func:`~syncsummoner.aesthetics.series_risk`), read from the decode's calibration-size colour, and is
covered the same way. The finished picture is screened again as a whole (:func:`gate`), since a splice
can join two safe takes into a flash.
"""

from __future__ import annotations

import enum
import subprocess
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Callable, Mapping, Sequence

import numpy as np

from syncsummoner import aesthetics
from syncsummoner.compose.chroma import exchanged
from syncsummoner.compose.cut import Cut
from syncsummoner.compose.score import Score
from syncsummoner.device.recorder import held
from syncsummoner.device.stored import captures
from syncsummoner.probe import decoded
from syncsummoner.probe.chroma import to_rgb
from syncsummoner.probe.decoded import Features
from syncsummoner.probe.footage import (
    ClockModel,
    Recording,
    attribute,
    clock_model,
    motion_model,
    recording_of,
    source_series,
    unit_frames,
)
from syncsummoner.progress import LOG, meter, track

__all__ = [
    "CheckReport",
    "DeadSpan",
    "DeadTakeError",
    "Fault",
    "Trace",
    "UnsafeOutputError",
    "Verdict",
    "bar_edges",
    "check_takes",
    "cover",
    "flashing",
    "gate",
    "judge",
    "measure",
    "picks",
    "screen",
    "trace",
    "unsafe_spans",
]


class DeadTakeError(RuntimeError):
    """A cut shows a dead span that no other program's take is alive across."""


class UnsafeOutputError(RuntimeError):
    """A finished picture fails the flash test."""


class Fault(enum.Enum):
    """Why a bar of output cannot be shown: blank, static, or harmful flashing."""

    FLAT = "flat"
    FROZEN = "frozen"
    FLASHING = "flashing"


@dataclass(frozen=True)
class Trace:
    """One recording read against its clip: per clip frame, the capture that shows it and its SI and TI.

    ``picked`` indexes ``frames`` and is -1 where no capture of that clip frame was kept;
    ``si`` and ``ti`` are NaN there. ``flashing`` marks the clip frames completing a flash in a second
    that fails the flash test, and ``flash`` is that test's whole verdict (:func:`flashing`); both are
    None where the take's colour was not screened.
    """

    picked: np.ndarray
    frames: np.ndarray
    si: np.ndarray
    ti: np.ndarray
    model: ClockModel | None
    how: str
    flashing: np.ndarray | None = None
    flash: aesthetics.FlashRisk | None = None

    @property
    def start(self) -> float | None:
        """Seconds into the recording at which the clip's first frame went up, if known."""
        return None if self.model is None else self.model.origin - self.model.interval / 2.0


def picks(shown: np.ndarray, count: int) -> np.ndarray:
    """Per clip frame, the capture :func:`~syncsummoner.probe.footage.attribute` gave it, or -1."""
    picked = np.full(count, -1, dtype=np.int64)
    seen = np.flatnonzero((shown >= 0) & (shown < count))
    picked[shown[seen]] = seen
    return picked


def measure(frames: np.ndarray, picked: np.ndarray, model: ClockModel | None, how: str) -> Trace:
    """SI at the capture attributed to each clip frame, TI against the frame before's."""
    picked = np.asarray(picked, dtype=np.int64)
    count = picked.size
    si, ti = np.full(count, np.nan), np.full(count, np.nan)
    unit = np.float32(255.0)
    for index in np.flatnonzero(picked >= 0):
        frame = frames[picked[index]].astype(np.float32) / unit
        si[index] = aesthetics.spatial_information(frame)
        if index and picked[index - 1] >= 0:
            ti[index] = aesthetics.temporal_information(
                frames[picked[index - 1]].astype(np.float32) / unit, frame
            )
    return Trace(picked, frames, si, ti, model, how)


def trace(
    recording: Recording,
    source_si: np.ndarray,
    source_ti: np.ndarray,
    *,
    fps: float,
    bits: int = 16,
    model: ClockModel | None = None,
    picked: np.ndarray | None = None,
) -> Trace:
    """Attribute a recording's captures to clip frames and measure the output at each.

    A given ``picked`` (per clip frame, the capture index showing it, -1 where none does) is
    the attribution as it is, and a given ``model`` is the clock as it is. Otherwise the strip's
    clock stands unless the motion's :func:`_explains` the output better by more than
    :func:`~syncsummoner.aesthetics.critical_z`, chance among the lags searched, as a scrambled
    strip loses.
    """
    count = int(np.size(source_ti))
    times, frames = recording.times, recording.frames
    if picked is not None:
        given = np.full(count, -1, dtype=np.int64)
        given[: min(count, np.size(picked))] = np.asarray(picked, dtype=np.int64)[:count]
        given[given >= frames.shape[0]] = -1
        return measure(frames, given, None, "given")
    if model is not None:
        return measure(frames, picks(attribute(times, model, frames=count), count), model, "given")
    found = {"strip": clock_model(times, recording.codes, fps=fps, bits=bits, frames=count)}
    if frames.shape[0] > 1:
        _, per_capture = aesthetics.information(unit_frames(frames))
        found["motion"] = motion_model(times, per_capture, source_ti, fps=fps)
    traces = [
        measure(frames, picks(attribute(times, m, frames=count), count), m, how)
        for how, m in found.items()
        if m
    ]
    if not traces:
        return measure(frames, np.full(count, -1, dtype=np.int64), None, "none")
    return max(traces, key=lambda t: _explains(t, source_si, source_ti))


def _explains(out: Trace, source_si: np.ndarray, source_ti: np.ndarray) -> float:
    """How far a clock makes the output's motion follow the source's, over the frames that move.

    Frames the output holds still follow any clock equally, so only those changing by more than
    quantisation noise decide between clocks.
    """
    moving = np.nan_to_num(out.ti) > aesthetics.quantisation_ti()
    blank = np.where(moving, 0.0, np.nan)
    return aesthetics.survival_stats(source_si + blank, source_ti + blank, out.si + blank, out.ti + blank).z


def screen(
    colour: np.ndarray,
    shown: np.ndarray,
    *,
    fps: float,
    full: bool = True,
    space: str = "unknown",
    swap: bool = False,
) -> aesthetics.FlashRisk:
    """The flash tests on the pictures ``shown``, indices into planar YUV ``colour``, in the order shown.

    ``full`` and ``space`` are the planes' range and matrix
    (:func:`~syncsummoner.probe.decoded.calibration_colour`) and ``swap`` exchanges the chroma pair,
    as a take that came back exchanged is spliced. The calibration size is an area scaling, which keeps
    every share of the screen the tests count, and is under :func:`~syncsummoner.aesthetics.flash_risk`'s
    own, so the tests see what they would at full size.
    """
    order = np.array([0, 2, 1] if swap else [0, 1, 2])
    chunks = (
        to_rgb(colour[shown[i : i + decoded.CHUNK]][:, order], full=full, space=space)
        for i in range(0, shown.size, decoded.CHUNK)
    )
    return aesthetics.series_risk(*aesthetics.flash_series(chunks), fps=fps)


def flashing(
    colour: np.ndarray,
    picked: np.ndarray,
    *,
    fps: float,
    full: bool = True,
    space: str = "unknown",
    swap: bool = False,
) -> tuple[np.ndarray, aesthetics.FlashRisk]:
    """The take's :func:`screen` on the clip frames ``picked`` shows, and per clip frame whether the
    change to it from the frame before completes a transition inside a second that fails, so a bar
    holding none is not marked for its neighbour's flashes.

    A clip frame no capture shows holds the one before, as a display would; frames before the
    first capture are not shown at all.
    """
    picked = np.asarray(picked, dtype=np.int64)
    mask = np.zeros(picked.size, dtype=bool)
    seen = np.flatnonzero(picked >= 0)
    if seen.size < 2:
        return mask, aesthetics.FlashRisk(0.0, 0.0, 0.0, (), True)
    held_at = np.maximum.accumulate(np.where(picked >= 0, np.arange(picked.size), -1))[seen[0] :]
    risk = screen(colour, picked[held_at], fps=fps, full=full, space=space, swap=swap)
    mask[seen[0] + np.array(risk.flashes, dtype=np.int64)] = True
    return mask, risk


def unsafe_spans(risk: aesthetics.FlashRisk, fps: float) -> str:
    """The seconds a :class:`~syncsummoner.aesthetics.FlashRisk` fails over, and its peak rates."""
    spans = ", ".join(f"{a / fps:.2f}-{b / fps:.2f}s" for a, b in risk.windows)
    return f"{risk.flashes_per_s:.1f} flashes/s, {risk.red_flashes_per_s:.1f} red flashes/s, at {spans}"


def gate(path: str | Path, *, strip_px: int = 0, ffmpeg: str = "ffmpeg") -> aesthetics.FlashRisk:
    """Screen a finished picture whole, every frame in turn, and raise :class:`UnsafeOutputError` naming
    the seconds that fail.

    Decoded straight to the calibration-size colour the takes are screened at, ``strip_px`` rows off the
    bottom, at the stream's own rate.
    """
    spec = decoded.Spec(strip_px=strip_px)
    full, space = decoded.calibration_colour(path, spec, ffmpeg=ffmpeg)
    rate, duration = _rate(path, ffmpeg)
    width, height = spec.calibration
    with captures(path, ffmpeg=ffmpeg) as source:
        argv = [ffmpeg, "-hide_banner", "-nostdin", "-loglevel", "error", *source, "-map", "0:v:0"]
        argv += ["-vf", spec.colour(), "-fps_mode", "passthrough", "-f", "rawvideo", "pipe:1"]
        proc = subprocess.Popen(  # pylint: disable=consider-using-with
            argv, stdin=subprocess.DEVNULL, stdout=subprocess.PIPE, stderr=subprocess.PIPE
        )
        try:
            with meter(duration, desc=f"screening {Path(path).name}") as progress:
                pictures = _pictures(proc.stdout, (3, height, width), lambda n: progress.to(n / rate))
                found = (to_rgb(frames, full=full, space=space) for frames in pictures)
                risk = aesthetics.series_risk(*aesthetics.flash_series(found), fps=rate)
            said = proc.stderr.read()
        except BaseException:
            proc.kill()
            raise
        finally:
            proc.wait()
    if proc.returncode:
        raise RuntimeError(f"screening {path} failed: {said.decode('utf-8', 'replace')[-200:]}")
    if not risk.safe:
        raise UnsafeOutputError(f"{path} flashes harmfully: {unsafe_spans(risk, rate)}")
    return risk


def _pictures(stream, shape: tuple[int, int, int], done: Callable[[int], None]):
    """8-bit pictures of ``shape`` read from ``stream`` a :data:`~syncsummoner.probe.decoded.CHUNK` at a
    time, ``done`` told how many so far."""
    size, count = int(np.prod(shape)), 0
    while block := stream.read(size * decoded.CHUNK):
        frames = np.frombuffer(block, np.uint8, len(block) // size * size).reshape(-1, *shape)
        count += frames.shape[0]
        done(count)
        yield frames


def _rate(path: str | Path, ffmpeg: str) -> tuple[float, float]:
    """A picture's frame rate and duration, from its container."""
    argv = [decoded.ffprobe_for(ffmpeg), "-v", "error", "-select_streams", "v:0"]
    argv += ["-show_entries", "stream=avg_frame_rate:format=duration", "-of", "default=nw=1:nk=1", str(path)]
    said = subprocess.run(argv, check=True, capture_output=True, text=True).stdout.split()
    num, _, den = said[0].partition("/")
    return float(num) / float(den or 1), float(said[1])


def _screen_cmd(args) -> int:
    """Screen finished pictures for harmful flashing, as a render's last gate does; 1 if any fails."""
    failed = 0
    for path in args.videos:
        try:
            risk = gate(path, strip_px=args.strip_px, ffmpeg=args.ffmpeg)
        except UnsafeOutputError as err:
            LOG.error("%s", err)
            failed = 1
            continue
        LOG.info("%s: safe, peak %.1f flashes/s, %.1f red", path, risk.flashes_per_s, risk.red_flashes_per_s)
    return failed


def add_parser(sub) -> None:
    """The ``screen`` command."""
    screen_p = sub.add_parser(
        "screen", help="screen finished pictures for harmful flashing, as a render does"
    )
    screen_p.add_argument("videos", nargs="+")
    screen_p.add_argument("--strip-px", type=int, default=0, help="rows off the bottom left out, a strip")
    screen_p.add_argument("--ffmpeg", default="ffmpeg")
    screen_p.set_defaults(func=_screen_cmd)


def bar_edges(
    duration: float, length: float, fps: float, *, downbeats: Sequence[float] = (), cuts: Sequence[Cut] = ()
) -> np.ndarray:
    """Clip-frame boundaries of the bars every take is judged on, up to ``duration``: the planner's.

    Bars run between ``downbeats`` where there are any, else every ``length`` seconds from zero, and
    split at every cut, so each cut is judged on the frames it shows alone.
    """
    marks = (
        np.asarray(downbeats, dtype=np.float64) if len(downbeats) > 1 else np.arange(0.0, duration, length)
    )
    marks = np.r_[0.0, marks, [c.start for c in cuts], [c.end for c in cuts], duration]
    return np.unique(np.rint(marks[(marks >= 0.0) & (marks <= duration)] * fps).astype(np.int64))


@dataclass(frozen=True)
class Verdict:
    """One bar of one take: its span in seconds, what it kept, and why it is dead if it is."""

    start: float
    end: float
    survival: aesthetics.Survival
    fault: Fault | None


def fault(out: Trace, a: int, b: int, kept: aesthetics.Survival) -> Fault | None:
    """Why clip frames ``a:b`` of a trace cannot be shown, or None while they can.

    Flashing: a frame after the first marked in :attr:`Trace.flashing`, so the transition is between two
    of the bar's own frames; one across a bar's edge belongs to neither, since covering either changes
    it and :func:`gate` judges what the splice made of it. Judged first, since a flash between plain
    fields is also flat. Flat: not mostly frames whose luma spans more than one 8-bit step, or SI
    under the source's :func:`~syncsummoner.aesthetics.live_floor`; frozen: TI under it, or
    :func:`~syncsummoner.device.recorder.held` unable to tell the bar from one picture under
    capture noise while its motion does not follow the source either, since a held picture's TI
    is that noise and independent of the source's.
    """
    if out.flashing is not None and out.flashing[a + 1 : b].any():
        return Fault.FLASHING
    picked = out.picked[a:b]
    frames = out.frames[picked[picked >= 0]].astype(np.float32)
    structured = int((np.ptp(frames.reshape(len(frames), -1), axis=1) > 1).sum()) if len(frames) else 0
    if 2 * structured <= len(frames) or kept.output_si < aesthetics.live_floor(
        kept.source_si, aesthetics.quantisation_si()
    ):
        return Fault.FLAT
    if (held(frames)[1] and not kept.following) or kept.output_ti < aesthetics.live_floor(
        kept.source_ti, aesthetics.quantisation_ti()
    ):
        return Fault.FROZEN
    return None


def judge(
    out: Trace, source_si: np.ndarray, source_ti: np.ndarray, edges: Sequence[int], *, fps: float
) -> list[Verdict]:
    """Every bar of a trace, judged on what it kept of the same bar of the source."""
    verdicts = []
    for a, b in zip(edges[:-1], edges[1:]):
        kept = aesthetics.survival_stats(source_si[a:b], source_ti[a:b], out.si[a:b], out.ti[a:b])
        verdicts.append(Verdict(a / fps, b / fps, kept, fault(out, a, b, kept)))
    return verdicts


@dataclass(frozen=True)
class DeadSpan:
    """A dead run inside one cut, and the program whose take covers it, or None where none can."""

    program: str
    start: float
    end: float
    fault: Fault
    cover: str | None
    si_ratio: float = float("nan")
    ti_ratio: float = float("nan")

    def __str__(self) -> str:
        fix = f"covered by {self.cover}" if self.cover else "no take is alive there"
        kept = f"keeps SI {self.si_ratio:.0%} TI {self.ti_ratio:.0%}"
        return f"{self.program}: {self.start:7.2f}-{self.end:7.2f}s {self.fault.value} ({kept}), {fix}"


@dataclass
class CheckReport:
    """The cut plan after covering, every dead span found, and the takes' traces."""

    cuts: list[Cut]
    spans: list[DeadSpan] = field(default_factory=list)
    traces: dict[str, Trace] = field(default_factory=dict)

    @property
    def failed(self) -> list[DeadSpan]:
        """Dead spans no take could cover."""
        return [span for span in self.spans if span.cover is None]

    def lines(self) -> list[str]:
        """One line per take's alignment and peak flash rates, then one per dead span."""
        said = [
            (
                f"{name}: never aligned"
                if t.how == "none"
                else f"{name}: aligned by {t.how}"
                + (f", picture starts {t.start:.3f}s" if t.start is not None else "")
            )
            + (
                ""
                if t.flash is None
                else f", peak {t.flash.flashes_per_s:.1f} flashes/s, {t.flash.red_flashes_per_s:.1f} red"
            )
            for name, t in self.traces.items()
        ]
        return said + [str(span) for span in self.spans]


def _runs(dead: np.ndarray) -> list[tuple[int, int]]:
    edges = np.flatnonzero(np.diff(np.r_[0, dead.astype(np.int8), 0]))
    return list(zip(edges[::2], edges[1::2]))


def cover(
    cuts: Sequence[Cut],
    verdicts: Mapping[str, Sequence[Verdict]],
    *,
    destroy: Callable[[float], bool] = lambda _t: False,
) -> tuple[list[Cut], list[DeadSpan]]:
    """Replace every dead run of every cut with the take alive across it that follows best.

    Every take is judged on the same bars, each in the cut its middle falls in, as :func:`bar_edges`
    splits them at every cut on the frame grid. A run reaching a cut's edge takes that edge itself, so
    no sliver of the cut is left between the edge and its bar's frame. Destruction is budgeted at one
    bar at a time, so a lone blank or static bar inside a destroy-marked section stays; any other dead
    run, and any flashing one, is covered.
    """
    grid = next(iter(verdicts.values()), [])
    middles = np.array([0.5 * (v.start + v.end) for v in grid])
    out: list[Cut] = []
    spans: list[DeadSpan] = []
    for cut in cuts:
        index = np.flatnonzero((middles >= cut.start) & (middles < cut.end))
        dead = np.array([verdicts[cut.program][i].fault is not None for i in index], dtype=bool)
        cursor = cut.start
        for a, b in _runs(dead):
            run = index[a:b]
            lone = run.size == 1 and verdicts[cut.program][run[0]].fault is not Fault.FLASHING
            if lone and destroy(0.5 * (grid[run[0]].start + grid[run[0]].end)):
                continue
            start = cut.start if a == 0 else grid[run[0]].start
            end = cut.end if b == index.size else grid[run[-1]].end
            alive = {
                name: float(np.mean([found[i].survival.z for i in run]))
                for name, found in verdicts.items()
                if name != cut.program and all(found[i].fault is None for i in run)
            }
            best = max(alive, key=alive.get) if alive else None
            mine = [verdicts[cut.program][i].survival for i in run]
            spans.append(
                DeadSpan(
                    cut.program,
                    start,
                    end,
                    verdicts[cut.program][run[0]].fault,
                    best,
                    float(np.nanmean([k.si_ratio for k in mine])),
                    float(np.nanmean([k.ti_ratio for k in mine])),
                )
            )
            if best is None:
                continue
            if start > cursor:
                out.append(replace(cut, start=cursor, end=start))
            out.append(Cut(start, end, best))
            cursor = end
        if cut.end > cursor:
            out.append(replace(cut, start=cursor))
    return out, spans


def check_takes(
    score: Score,
    cuts: Sequence[Cut],
    takes: Mapping[str, str | Path],
    clip: str | Path,
    *,
    strip_px: int,
    bits: int = 16,
    bar: float | None = None,
    models: Mapping[str, ClockModel] | None = None,
    attributions: Mapping[str, np.ndarray] | None = None,
    features: Mapping[str, Features] | None = None,
    source: Features | None = None,
    ffmpeg: str = "ffmpeg",
) -> CheckReport:
    """Trace every take against the clip, judge every bar, and cover the cut plan.

    ``attributions`` gives a recording's per-clip-frame capture indices outright, with the
    meaning of :attr:`Trace.picked`, and ``models`` fixes a recording's clock, as a master's identity
    clock is; any other is fitted. ``features`` and ``source`` are decodes already made of takes and clip.
    Each take is screened for flashing (:func:`flashing`) on the clip frames its trace shows, in the
    channel order it is spliced in.
    """
    fps = float(score.fps)
    with_bar = float(bar or score.meta.get("bar") or 4 * 60.0 / score.bpm)
    LOG.info("reading the played clip %s", clip)
    source_si, source_ti = source_series(clip if source is None else source, strip_px=strip_px, ffmpeg=ffmpeg)
    if source_si.size < 2:
        raise DeadTakeError(f"cannot judge any take: {clip} decoded to {source_si.size} frames")
    edges = bar_edges(
        min(score.duration, source_si.size / fps),
        with_bar,
        fps,
        downbeats=score.meta.get("downbeats", ()),
        cuts=cuts,
    )
    fixed, given, decodes = dict(models or {}), dict(attributions or {}), dict(features or {})

    spec = decoded.Spec(strip_px=strip_px, bits=bits)

    def read(name: str) -> tuple[str, Trace]:
        found = decodes[name] if name in decodes else decoded.features(takes[name], spec, ffmpeg=ffmpeg)
        out = trace(
            recording_of(found),
            source_si,
            source_ti,
            fps=fps,
            bits=bits,
            model=fixed.get(name),
            picked=given.get(name),
        )
        full, space = decoded.calibration_colour(takes[name], spec, ffmpeg=ffmpeg)
        mask, risk = flashing(
            found.small, out.picked, fps=fps, full=full, space=space, swap=exchanged(takes[name])
        )
        return name, replace(out, flashing=mask, flash=risk)

    with ThreadPoolExecutor(max_workers=max(1, len(takes))) as pool:
        traces = dict(track(pool.map(read, list(takes)), desc="takes", total=len(takes), unit="take"))
    verdicts = {name: judge(t, source_si, source_ti, edges, fps=fps) for name, t in traces.items()}
    mask = score.destroy_mask
    covered, spans = cover(cuts, verdicts, destroy=lambda t: bool(mask(np.array([t]))[0]))
    return CheckReport(cuts=covered, spans=spans, traces=traces)

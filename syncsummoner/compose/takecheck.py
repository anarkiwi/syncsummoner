"""Check captured takes against the source frames they were fed, before anything is mastered.

Every capture is attributed to the clip frame it shows, so each bar of each take is judged
by how much of that bar's picture and motion it kept: flat, frozen and unfollowed bars are
dead, and a cut that shows one is covered from a take alive there or the render fails.
"""

from __future__ import annotations

import enum
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Callable, Mapping, Sequence

import numpy as np

from syncsummoner import aesthetics
from syncsummoner.compose.cut import Cut
from syncsummoner.compose.score import Score
from syncsummoner.device.recorder import held
from syncsummoner.probe.decoded import Features
from syncsummoner.probe.footage import (
    ClockModel,
    Recording,
    attribute,
    clock_model,
    decode_recording,
    motion_model,
    recording_of,
    source_series,
    unit_frames,
)
from syncsummoner.progress import LOG, track

__all__ = [
    "CheckReport",
    "DeadSpan",
    "DeadTakeError",
    "Fault",
    "Trace",
    "Verdict",
    "bar_edges",
    "check_takes",
    "cover",
    "judge",
    "measure",
    "picks",
    "read_recording",
    "trace",
]


class DeadTakeError(RuntimeError):
    """A cut shows a dead span that no other program's take is alive across."""


class Fault(enum.Enum):
    """Why a bar of output carries no live picture."""

    FLAT = "flat"
    FROZEN = "frozen"
    UNFOLLOWED = "unfollowed"


@dataclass(frozen=True)
class Trace:
    """One recording read against its clip: per clip frame, the capture that shows it and its SI and TI.

    ``picked`` indexes ``frames`` and is -1 where no capture of that clip frame was kept;
    ``si`` and ``ti`` are NaN there.
    """

    picked: np.ndarray
    frames: np.ndarray
    si: np.ndarray
    ti: np.ndarray
    model: ClockModel | None
    how: str

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


def read_recording(
    path: str | Path,
    source_si: np.ndarray,
    source_ti: np.ndarray,
    *,
    fps: float,
    strip_px: int,
    bits: int = 16,
    model: ClockModel | None = None,
    picked: np.ndarray | None = None,
    ffmpeg: str = "ffmpeg",
) -> Trace:
    """Decode a take or a master and trace it against the clip's TI."""
    recording = decode_recording(path, strip_px=strip_px, bits=bits, ffmpeg=ffmpeg)
    return trace(recording, source_si, source_ti, fps=fps, bits=bits, model=model, picked=picked)


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


def fault(out: Trace, a: int, b: int, kept: aesthetics.Survival, source_ti: np.ndarray) -> Fault | None:
    """Why clip frames ``a:b`` of a trace are dead, or None while they live.

    Flat: not mostly frames whose luma spans more than one 8-bit step, or SI under the source's
    :func:`~syncsummoner.aesthetics.live_floor`; frozen: TI under it, or
    :func:`~syncsummoner.device.recorder.held` unable to tell the bar from one picture under
    capture noise while its motion does not follow the source either, since a held picture's TI
    is that noise and independent of the source's; unfollowed: motion not following a source TI
    that varies beyond quantisation.
    """
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
    window = source_ti[a:b]
    varied = bool(np.isfinite(window).any() and np.nanstd(window) > aesthetics.quantisation_ti())
    if kept.frames > 3 and varied and not kept.following:
        return Fault.UNFOLLOWED
    return None


def judge(
    out: Trace, source_si: np.ndarray, source_ti: np.ndarray, edges: Sequence[int], *, fps: float
) -> list[Verdict]:
    """Every bar of a trace, judged on what it kept of the same bar of the source."""
    verdicts = []
    for a, b in zip(edges[:-1], edges[1:]):
        kept = aesthetics.survival_stats(source_si[a:b], source_ti[a:b], out.si[a:b], out.ti[a:b])
        verdicts.append(Verdict(a / fps, b / fps, kept, fault(out, a, b, kept, source_ti)))
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
        """One line per take's alignment, then one per dead span."""
        said = [
            (
                f"{name}: never aligned"
                if t.how == "none"
                else f"{name}: aligned by {t.how}"
                + (f", picture starts {t.start:.3f}s" if t.start is not None else "")
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
    splits them at every cut. Destruction is budgeted at one bar at a time, so a lone dead bar
    inside a destroy-marked section stays; any other dead run is covered.
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
            if run.size == 1 and destroy(0.5 * (grid[run[0]].start + grid[run[0]].end)):
                continue
            start, end = max(cut.start, grid[run[0]].start), min(cut.end, grid[run[-1]].end)
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

    def read(name: str) -> tuple[str, Trace]:
        if name in decodes:
            return name, trace(
                recording_of(decodes[name]),
                source_si,
                source_ti,
                fps=fps,
                bits=bits,
                model=fixed.get(name),
                picked=given.get(name),
            )
        return name, read_recording(
            takes[name],
            source_si,
            source_ti,
            fps=fps,
            strip_px=strip_px,
            bits=bits,
            model=fixed.get(name),
            picked=given.get(name),
            ffmpeg=ffmpeg,
        )

    with ThreadPoolExecutor(max_workers=max(1, len(takes))) as pool:
        traces = dict(track(pool.map(read, list(takes)), desc="takes", total=len(takes), unit="take"))
    verdicts = {name: judge(t, source_si, source_ti, edges, fps=fps) for name, t in traces.items()}
    mask = score.destroy_mask
    covered, spans = cover(cuts, verdicts, destroy=lambda t: bool(mask(np.array([t]))[0]))
    return CheckReport(cuts=covered, spans=spans, traces=traces)

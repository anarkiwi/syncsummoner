"""Natural footage as a stimulus: which source frame every captured frame shows.

The player shows frame ``n`` of a timecoded clip at ``origin + n * interval`` on its own
clock, so identity is a line fitted through the captures whose strip still reads, unwrapped
over the loop; a capture is attributed by its time, so a program that mangles the strip loses none.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
from scipy import sparse
from scipy.optimize import linprog

from syncsummoner import aesthetics
from syncsummoner.device.recorder import INSPECT_SIZE
from syncsummoner.probe import decoded

__all__ = [
    "ClockModel",
    "Recording",
    "attribute",
    "clock_model",
    "decode_recording",
    "motion_model",
    "recording_of",
    "source_series",
    "unit_frames",
]


@dataclass(frozen=True)
class ClockModel:
    """When the player showed each clip frame, in the capture clock's seconds.

    ``origin`` is the centre of frame 0's display interval and ``interval`` the fitted frame
    period; ``period`` is the loop length in frames, or None for a clip played once.
    """

    origin: float
    interval: float
    period: int | None = None
    support: int = 0

    @classmethod
    def identity(cls, fps: float) -> "ClockModel":
        """The clock of a recording whose frame ``n``, stamped ``n / fps``, is clip frame ``n``."""
        return cls(origin=0.0, interval=1.0 / float(fps))

    def index(self, times: np.ndarray) -> np.ndarray:
        """Clip frame on show at each capture time, wrapped over the loop."""
        n = np.rint((np.asarray(times, dtype=np.float64) - self.origin) / self.interval).astype(np.int64)
        return n % self.period if self.period else n

    def centre(self, index: np.ndarray) -> np.ndarray:
        """Centre of each unwrapped frame's display interval."""
        return self.origin + self.interval * np.asarray(index, dtype=np.float64)


def _agreeing(times: np.ndarray, codes: np.ndarray, interval: float, bits: int, period: int | None):
    """Unwrapped codes, and which captures agree with the modal lag in runs no misread could make.

    A misread lands on a lag with odds ``2**-bits``, so ``k`` successive frames agree by chance with
    odds ``2**-(bits * (k - 1))``: runs over more than ``1 + log2(n) / bits`` distinct frames keep the
    expected chance runs under one, and a held strip, one code at a growing lag, spans one frame.
    """
    span = period * interval if period else np.inf
    phase = np.mod(times - codes * interval, span) if period else times - codes * interval
    bins = np.rint(phase / (interval / 2.0)).astype(np.int64)
    values, counts = np.unique(bins, return_counts=True)
    mode = values[counts.argmax()] * interval / 2.0
    off = phase - mode
    if period:
        off = (off + span / 2.0) % span - span / 2.0
        codes = codes + period * np.rint(((times - mode) / interval - codes) / period)
    hit = np.abs(off) <= interval / 2.0 + interval / 4.0
    edges = np.flatnonzero(np.diff(np.r_[0, hit.astype(np.int8), 0]))
    keep = np.zeros(hit.size, dtype=bool)
    least = 1.0 + np.log2(max(2, hit.size)) / bits
    for start, stop in zip(edges[::2], edges[1::2]):
        if np.unique(codes[start:stop]).size > least:
            keep[start:stop] = True
    return codes, keep


#: HiGHS's own primal feasibility tolerance, below which a slack is zero.
SLACK_TOL = 1e-7


def _bounds(times: np.ndarray, codes: np.ndarray) -> tuple:
    """Rows ``s0 + c I - t <= 0`` and ``t - s0 - (c + 1) I <= 0``: each capture inside its frame."""
    ones = np.ones_like(codes)
    return np.concatenate([np.c_[ones, codes], np.c_[-ones, -(codes + 1)]]), np.concatenate([times, -times])


def _l1(times: np.ndarray, codes: np.ndarray, interval: float) -> tuple[float, float, np.ndarray] | None:
    """Display start and period that leave the least total time outside the frames captured.

    Least absolute violation is robust where a least-squares line is not: a torn or misread
    strip costs only its own slack. Returns the line and which captures it fits exactly.
    """
    n = codes.size
    rows, rhs = _bounds(times, codes)
    slack = sparse.vstack([-sparse.identity(n), -sparse.identity(n)])
    matrix = sparse.hstack([sparse.csr_matrix(rows), slack]).tocsr()
    bounds = [(None, None), (0.5 * interval, 1.5 * interval)] + [(0, None)] * n
    found = linprog(np.r_[0.0, 0.0, np.ones(n)], A_ub=matrix, b_ub=rhs, bounds=bounds, method="highs")
    if not found.success:
        return None
    return float(found.x[0]), float(found.x[1]), found.x[2:] <= SLACK_TOL


def _margin(times: np.ndarray, codes: np.ndarray, interval: float) -> tuple[float, float] | None:
    """The line that keeps every capture inside its frame with the most room either side.

    Maximising the least margin centres the line in what the captures allow, which a fit of
    the captures themselves would tilt by the slow beat between card and player phase.
    """
    rows, rhs = _bounds(times, codes)
    matrix = np.c_[rows, np.ones(rhs.size)]
    bounds = [(None, None), (0.5 * interval, 1.5 * interval), (None, None)]
    found = linprog([0.0, 0.0, -1.0], A_ub=matrix, b_ub=rhs, bounds=bounds, method="highs")
    if not found.success or found.x[2] < -SLACK_TOL:
        return None
    return float(found.x[0]), float(found.x[1])


def _fit(times: np.ndarray, codes: np.ndarray, interval: float) -> tuple[float, float, int] | None:
    """An :func:`_l1` line settled by :func:`_margin` over the captures it fits exactly."""
    found = _l1(times, codes, interval) if codes.size >= 2 else None
    if found is None:
        return None
    start, step, exact = found
    start, step = _margin(times[exact], codes[exact], interval) or (start, step)
    return start, step, int(exact.sum())


def clock_model(
    times: np.ndarray,
    codes: np.ndarray,
    *,
    fps: float,
    bits: int = 16,
    period: int | None = None,
    frames: int | None = None,
) -> ClockModel | None:
    """Fit when each clip frame was shown from captures whose strip reads, or None if none agree.

    ``codes`` holds each capture's strip reading, negative where illegible; a code past the clip's
    ``frames`` or loop ``period`` is a misread. The agreeing runs are fitted by :func:`_fit`, then
    every legible capture that line places in its own frame; a clip played once must overlap the take.
    """
    times = np.asarray(times, dtype=np.float64)
    codes = np.asarray(codes, dtype=np.int64)
    length = period or frames
    legible = (codes >= 0) & ((codes < length) if length else True)
    if legible.sum() < 2:
        return None
    interval = 1.0 / float(fps)
    t = times[legible]
    c, keep = _agreeing(t, codes[legible].astype(np.float64), interval, bits, period)
    line = _fit(t[keep], c[keep], interval)
    if line is None:
        return None
    on = c == np.floor((t - line[0]) / line[1])
    start, step, support = _fit(t[on], c[on], interval) or line
    if frames and not period and not (start < times[-1] and start + frames * step > times[0]):
        return None
    return ClockModel(start + step / 2.0, step, period, support)


def attribute(times: np.ndarray, model: ClockModel, *, frames: int | None = None) -> np.ndarray:
    """Per capture, the clip frame it shows, or -1 where another capture of that frame sat nearer its centre.

    Captures arrive at the card's rate, not the player's, so a frame may be captured twice or not
    at all; the one nearest the centre of its display interval is furthest from either switch.
    """
    times = np.asarray(times, dtype=np.float64)
    unwrapped = np.rint((times - model.origin) / model.interval).astype(np.int64)
    offset = np.abs(times - model.centre(unwrapped))
    order = np.lexsort((offset, unwrapped))
    first = np.r_[True, np.diff(unwrapped[order]) != 0]
    out = np.full(times.size, -1, dtype=np.int64)
    out[order[first]] = model.index(times[order[first]])
    if frames is not None and not model.period:
        out[(out < 0) | (out >= frames)] = -1
    return out


def _ranks(values: np.ndarray) -> np.ndarray:
    """Centred ranks, the scale a rank correlation is taken on."""
    ranks = np.argsort(np.argsort(values, kind="stable"), kind="stable").astype(np.float64)
    return ranks - ranks.mean()


def motion_model(
    times: np.ndarray,
    output_ti: np.ndarray,
    source_ti: np.ndarray,
    *,
    fps: float,
    period: int | None = None,
) -> ClockModel | None:
    """Fit the clock from motion alone, for a recording whose strip never reads.

    ``output_ti`` (each capture's TI against the one before) is summed over grids at the clip's rate
    and rank-correlated with the source's TI at every whole-frame lag, circularly over a loop, ranks
    so no single cut decides; grid phases step by half the capture spacing, as finely as captures resolve.
    """
    times = np.asarray(times, dtype=np.float64)
    interval = 1.0 / float(fps)
    count = int((times[-1] - times[0]) / interval) if times.size > 1 else 0
    if count < 4 or np.size(source_ti) < 4:
        return None
    total = np.cumsum(np.nan_to_num(np.asarray(output_ti, dtype=np.float64)))
    x = _ranks(np.nan_to_num(np.asarray(source_ti, dtype=np.float64)))
    spacing = float(np.median(np.diff(times)))
    best = (-np.inf, 0.0)
    for phase in np.arange(0.0, interval, spacing / 2.0 if spacing > 0 else interval):
        grid = times[0] + phase + interval * np.arange(count)
        slot = np.clip(np.searchsorted(times, grid, side="right") - 1, 0, times.size - 1)
        y = np.diff(np.r_[total[slot[0]], total[slot]])
        if period:
            y = np.bincount(np.arange(y.size) % period, weights=y, minlength=period)[:period]
        y = _ranks(y)
        size = x.size if period else x.size + y.size
        score = np.fft.irfft(np.fft.rfft(y, size) * np.conj(np.fft.rfft(x, size)), size)
        shift = int(np.argmax(score))
        found = score[shift] / (np.linalg.norm(y) * np.linalg.norm(x) or 1.0)
        if found > best[0]:
            lag = shift if period or shift < y.size else shift - size
            best = (found, float(grid[0] + interval * lag))
    return ClockModel(origin=best[1], interval=interval, period=period, support=0)


@dataclass(frozen=True)
class Recording:
    """Every frame of one recording at survival size, with its capture time and strip reading."""

    times: np.ndarray
    codes: np.ndarray
    frames: np.ndarray


def decode_recording(
    path: str | Path,
    *,
    strip_px: int,
    bits: int,
    size: tuple[int, int] = INSPECT_SIZE,
    ffmpeg: str = "ffmpeg",
) -> Recording:
    """A recording as :class:`Recording`: luma area-scaled to ``size``, the strip read at full width.

    Read from the recording's :func:`~syncsummoner.probe.decoded.features`, decoded once and cached.
    """
    found = decoded.features(path, decoded.Spec(strip_px=strip_px, bits=bits, survival=size), ffmpeg=ffmpeg)
    return recording_of(found)


def recording_of(found: decoded.Features) -> Recording:
    """The :class:`Recording` a recording's features hold."""
    return Recording(times=found.times, codes=found.codes, frames=found.luma)


def source_series(
    clip: str | Path | decoded.Features,
    *,
    strip_px: int,
    size: tuple[int, int] = INSPECT_SIZE,
    ffmpeg: str = "ffmpeg",
) -> tuple[np.ndarray, np.ndarray]:
    """SI and TI of every frame of the clip that was played, in clip order, at survival size.

    ``clip`` is a path or its decoded features. A locked take's lead-in, numbered below zero and
    so wrapped to the strip's top half, is skipped.
    """
    if isinstance(clip, decoded.Features):
        recording = recording_of(clip)
    else:
        recording = decode_recording(clip, strip_px=strip_px, bits=16, size=size, ffmpeg=ffmpeg)
    clip_codes = (recording.codes >= 0) & (recording.codes < 1 << 15)
    lead = int(np.argmax(clip_codes)) if np.any(recording.codes >= 1 << 15) else 0
    return aesthetics.information(unit_frames(recording.frames[lead:]))


def source_frame(shown: np.ndarray, lead: int) -> np.ndarray:
    """The :func:`source_series` frame a capture shows, from the played frame it names, lead-in counted.

    The one conversion between the two numberings: rows name played frames, the series starts at clip frame 0.
    """
    return np.asarray(shown, dtype=np.int64) - int(lead)


def unit_frames(frames: np.ndarray):
    """Stored 8-bit frames one at a time in the unit range the metrics take."""
    return (frame.astype(np.float32) / np.float32(255.0) for frame in frames)

"""The device's frame boundaries on the host's capture clock: one straight line per gap-free run.

Card and device run on their own crystals, so frame ``n`` starts at ``origin + n * period``. A
capture starting a frame brackets that start between the capture before and itself, within the
capture log's jitter; the line is the least-squares fit inside the polygon every bracket admits.
"""

from __future__ import annotations

from bisect import bisect_left, insort
from dataclasses import dataclass, field

import numpy as np

__all__ = ["Line", "LineError", "Polygon", "capture_line", "extend", "fit", "segments"]

#: A capture interval past the midpoint between one capture period and two is a lost capture or an outage.
GAP_PERIODS = 1.5


class LineError(RuntimeError):
    """The brackets admit no straight frame line; ``miss`` is how far the nearest one passes a bracket."""

    def __init__(self, message: str, miss: float = float("inf")):
        super().__init__(message)
        self.miss = miss


def segments(t_ns: np.ndarray, sequence: np.ndarray) -> np.ndarray:
    """Each capture's gap-free run: a lost sequence number or a long interval starts the next."""
    dt = np.diff(np.asarray(t_ns, dtype=np.float64))
    if not dt.size:
        return np.zeros(np.size(t_ns), dtype=np.int64)
    breaks = (dt > GAP_PERIODS * np.median(dt)) | (np.diff(sequence) != 1)
    return np.r_[0, np.cumsum(breaks)].astype(np.int64)


def capture_line(t_ns: np.ndarray, sequence: np.ndarray, seg: np.ndarray) -> tuple[float, float]:
    """The card's capture period and the timestamps' jitter: the largest residual off its line.

    One crystal drives every run, so the slope is common and each run has its own intercept.
    """
    t = np.asarray(t_ns, dtype=np.float64)
    s = np.asarray(sequence, dtype=np.float64)
    counts = np.maximum(np.bincount(seg), 1)
    x = s - (np.bincount(seg, s) / counts)[seg]
    y = t - (np.bincount(seg, t) / counts)[seg]
    xx = float(x @ x)
    if xx == 0:
        return float("nan"), 0.0
    period = float(x @ y) / xx
    return period, float(np.abs(y - period * x).max())


@dataclass
class Polygon:
    """Lines ``b(n) = u + (n - ref) * period`` meeting every bracket so far, as vertices ``(u, period)``."""

    ref: int
    vertices: np.ndarray

    @classmethod
    def through(cls, n: tuple[int, int], lo: np.ndarray, hi: np.ndarray) -> "Polygon":
        """The lines through two brackets at frames ``n[0] < n[1]``: a parallelogram."""
        span = float(n[1] - n[0])
        corners = [(lo[0], lo[1]), (hi[0], lo[1]), (hi[0], hi[1]), (lo[0], hi[1])]
        return cls(n[0], np.array([(a, (b - a) / span) for a, b in corners]))

    @classmethod
    def within(cls, n: int, lo: float, hi: float, period: tuple[float, float]) -> "Polygon":
        """The lines through one bracket with a period in the given range: a rectangle."""
        return cls(n, np.array([(lo, period[0]), (hi, period[0]), (hi, period[1]), (lo, period[1])]))

    @property
    def periods(self) -> tuple[float, float]:
        """The least and greatest period any line has."""
        return float(self.vertices[:, 1].min()), float(self.vertices[:, 1].max())

    @property
    def empty(self) -> bool:
        """Whether no line is left."""
        return not self.vertices.size

    def at(self, n: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """The least and greatest boundary time any line gives each frame."""
        offset = np.asarray(n, dtype=np.float64) - self.ref
        b = self.vertices[:, 0] + np.multiply.outer(offset, self.vertices[:, 1])
        return b.min(axis=-1), b.max(axis=-1)

    def meet(self, n: int, lo: float, hi: float) -> float:
        """Clip to a bracket, widened just enough to meet the polygon; returns how far it was widened."""
        least, most = (float(x[0]) for x in self.at(np.array([n])))
        before = self.vertices
        self.clip(n, min(lo, most), max(hi, least))
        if self.empty:
            self.vertices = before
        return max(lo - most, least - hi, 0.0)

    def clip(self, n: int, lo: float, hi: float) -> None:
        """Keep only the lines whose frame ``n`` boundary lies in ``[lo, hi]``."""
        for sign, bound in ((1.0, hi), (-1.0, -lo)):
            self._half(sign, sign * (n - self.ref), bound)

    def _half(self, a: float, b: float, c: float) -> None:
        v = self.vertices
        if not v.size:
            return
        side = v @ (a, b) - c
        inside = side <= 0
        if inside.all():
            return
        after, s_after = np.roll(v, -1, axis=0), np.roll(side, -1)
        cross = side * s_after < 0
        with np.errstate(invalid="ignore", divide="ignore"):
            cut = v + (side / (side - s_after))[:, None] * (after - v)
        both = np.stack([v, cut], axis=1).reshape(-1, 2)
        self.vertices = both[np.stack([inside, cross], axis=1).ravel()]

    def least_squares(self, n: np.ndarray, mid: np.ndarray) -> tuple[float, float]:
        """The line in the polygon nearest the midpoints in least squares, as ``(u, period)``.

        The free fit when the polygon holds it, else the best point on the polygon's boundary.
        """
        a = np.c_[np.ones(n.size), np.asarray(n, dtype=np.float64) - self.ref]
        h, g = a.T @ a, a.T @ mid
        v = self.vertices
        if v.shape[0] > 2 and np.linalg.matrix_rank(h) == 2:
            free = np.linalg.solve(h, g)
            if _inside(v, free):
                return float(free[0]), float(free[1])
        d = np.roll(v, -1, axis=0) - v
        curv = np.einsum("ij,jk,ik->i", d, h, d)
        slope = np.einsum("ij,ij->i", d, g - v @ h)
        with np.errstate(invalid="ignore", divide="ignore"):
            s = np.clip(np.where(curv > 0, slope / curv, 0.0), 0.0, 1.0)
        p = v + s[:, None] * d
        cost = np.einsum("ij,jk,ik->i", p, h, p) - 2 * p @ g
        best = p[int(np.argmin(cost))]
        return float(best[0]), float(best[1])


def _inside(v: np.ndarray, x: np.ndarray) -> bool:
    d = np.roll(v, -1, axis=0) - v
    cross = d[:, 0] * (x[1] - v[:, 1]) - d[:, 1] * (x[0] - v[:, 0])
    return bool((cross >= 0).all() or (cross <= 0).all())


@dataclass(frozen=True)
class Line:
    """One run's frame line: frame ``n`` starts at ``origin + n * period`` on the capture clock.

    ``frame`` names each bracket's start, -1 for a stray no line meets within the jitter; ``tied``
    starts fit two frames and took the one hiding no frame from the neighbour they were named from.
    ``slack`` is the most a bracket was widened, ``residual`` the furthest a named one passes the line,
    and ``rejected`` the named starts whose group no line through the largest met.
    """

    origin: float
    period: float
    slack: float
    residual: float
    frame: np.ndarray
    polygon: Polygon
    tied: int = 0
    mid: np.ndarray = field(default_factory=lambda: np.zeros(0))
    rejected: int = 0

    @property
    def strays(self) -> np.ndarray:
        """Brackets no admitted line meets: bytes that changed with no frame starting."""
        return np.flatnonzero(self.frame < 0)

    def frame_of(self, t_ns: np.ndarray) -> np.ndarray:
        """The frame each time falls in."""
        return np.floor((np.asarray(t_ns, dtype=np.float64) - self.origin) / self.period).astype(np.int64)

    def margin(self, t_ns: np.ndarray) -> np.ndarray:
        """How far each time lies from the nearest boundary any admitted line puts there.

        Zero where a boundary may fall on it, so the frame it shows is uncertain.
        """
        t = np.asarray(t_ns, dtype=np.float64)
        lo, hi = self.polygon.at(np.rint((t - self.origin) / self.period))
        return np.maximum(np.maximum(lo - t, t - hi), 0.0)


def fit(
    lo: np.ndarray,
    hi: np.ndarray,
    frame: np.ndarray,
    jitter: float,
    *,
    period: tuple[float, float] | None = None,
    group: np.ndarray | None = None,
    resolution: float = 0.0,
) -> Line:
    """The frame line through starts bracketed by ``(lo, hi]`` in capture order, ``frame >= 0`` named already.

    Named brackets clip the polygon a ``group`` at a time, the largest first; a later group missing it
    past ``jitter`` is unnamed again, the first a :class:`LineError`. The rest are named outward, each by
    the one frame the polygon admits between its named neighbours'; one missing it by less than
    ``jitter`` is widened to meet it, by more a stray, -1. A start lies a ``resolution`` past ``lo``.
    """
    lo, hi = np.asarray(lo, dtype=np.float64) + resolution, np.asarray(hi, dtype=np.float64)
    frame = np.asarray(frame, dtype=np.int64).copy()
    named = np.flatnonzero(frame >= 0)
    group = np.zeros(frame.size, dtype=np.int64) if group is None else np.asarray(group, dtype=np.int64)
    if not named.size:
        if period is None or not frame.size:
            raise LineError("no bracket is named, so nothing fixes the frame line")
        frame[0], named = 0, np.array([0])
    ids, sizes = np.unique(group[named], return_counts=True)
    order_groups = ids[np.argsort(-sizes, kind="stable")]
    first = named[group[named] == order_groups[0]]
    ends = first[[np.argmin(frame[first]), np.argmax(frame[first])]]
    if frame[ends[0]] != frame[ends[1]]:
        poly = Polygon.through((int(frame[ends[0]]), int(frame[ends[1]])), lo[ends], hi[ends])
    elif period is not None:
        poly = Polygon.within(int(frame[ends[0]]), float(lo[ends[0]]), float(hi[ends[0]]), period)
    else:
        raise LineError("the named brackets are all one frame's, so nothing fixes the period")
    worst, rejected = 0.0, 0
    for index, g in enumerate(order_groups):
        members = named[group[named] == g]
        before, misses = poly.vertices.copy(), [
            poly.meet(int(frame[i]), float(lo[i]), float(hi[i])) for i in members
        ]
        if max(misses) <= jitter:
            worst = max(worst, *misses)
            continue
        if not index:
            miss = max(misses)
            raise LineError(
                f"frame {frame[members[int(np.argmax(misses))]]}'s start misses every line through those "
                f"before it by {miss / 1e6:.3f} ms, past the capture jitter of {jitter / 1e6:.3f} ms",
                miss=miss,
            )
        poly.vertices, frame[members], rejected = before, -1, rejected + members.size
    named = np.flatnonzero(frame >= 0)
    rest = np.setdiff1d(np.arange(frame.size), named)
    if rest.size:
        place = np.searchsorted(named, rest)
        left, right = named[np.maximum(place - 1, 0)], named[np.minimum(place, named.size - 1)]
        rest = rest[np.argsort(np.minimum(np.abs(rest - left), np.abs(rest - right)), kind="stable")]
    order, tied = sorted(int(i) for i in named), 0
    for i in rest:
        at = bisect_left(order, int(i))
        low = int(frame[order[at - 1]]) if at else None
        high = int(frame[order[at]]) if at < len(order) else None
        nearer_low = at and (at == len(order) or i - order[at - 1] <= order[at] - i)
        frame[i], tie, miss = _name(poly, float(lo[i]), float(hi[i]), (low, high), bool(nearer_low), jitter)
        if frame[i] >= 0:
            tied, worst = tied + tie, max(worst, miss)
            insort(order, int(i))
    return _line(poly, frame, 0.5 * (lo + hi), worst, tied, lo, hi, rejected)


def extend(line: Line, lo: np.ndarray, hi: np.ndarray, jitter: float, resolution: float = 0.0) -> Line:
    """``line`` carried on over brackets after all it named in capture order, named as :func:`fit` does."""
    lo = np.asarray(lo, dtype=np.float64) + resolution
    poly = Polygon(line.polygon.ref, line.polygon.vertices.copy())
    frame = np.full(np.size(lo), -1, dtype=np.int64)
    low = int(line.frame.max()) if line.frame.size else None
    worst, tied = line.slack, line.tied
    for i, (a, b) in enumerate(zip(np.asarray(lo, dtype=np.float64), np.asarray(hi, dtype=np.float64))):
        frame[i], tie, miss = _name(poly, float(a), float(b), (low, None), True, jitter)
        if frame[i] >= 0:
            low, tied, worst = int(frame[i]), tied + tie, max(worst, miss)
    mid = np.r_[line.mid, 0.5 * (np.asarray(lo, dtype=np.float64) + np.asarray(hi, dtype=np.float64))]
    return _line(poly, np.r_[line.frame, frame], mid, worst, tied, rejected=line.rejected)


def _name(
    poly: Polygon, lo: float, hi: float, between: tuple[int | None, int | None], from_low: bool, jitter: float
) -> tuple[int, bool, float]:
    """``(frame, tied, widened)`` for one start: the frame the polygon admits strictly ``between`` two.

    Two that fit take the one nearer the neighbour it is named from; none that fits within ``jitter``
    of widening is a stray, -1. The polygon is clipped to the bracket named.
    """
    low, high = between
    u, p = poly.vertices.mean(axis=0)
    guess = poly.ref + int(np.rint((0.5 * (lo + hi) - u) / p))
    near = np.arange(guess - 1, guess + 2)
    near = near[((low is None) | (near > (low or 0))) & ((high is None) | (near < (high or 0)))]
    least, most = poly.at(near)
    gap = np.maximum(np.maximum(lo - most, least - hi), 0.0)
    fits = near[(most > lo) & (least <= hi)]
    if not fits.size:
        if not near.size or gap.min() > jitter:
            return -1, False, 0.0
        fits = near[gap == gap.min()]
    side = low if from_low and low is not None else (high if high is not None else low)
    chosen = int(fits[np.argmin(np.abs(fits - side))]) if side is not None else int(fits[0])
    return chosen, fits.size > 1, poly.meet(chosen, lo, hi)


def _line(
    poly: Polygon,
    frame: np.ndarray,
    mid: np.ndarray,
    worst: float,
    tied: int,
    lo: np.ndarray | None = None,
    hi: np.ndarray | None = None,
    rejected: int = 0,
) -> Line:
    """The least-squares line in ``poly`` through the named brackets' midpoints."""
    ok = frame >= 0
    u, p = poly.least_squares(frame[ok], mid[ok])
    origin = u - poly.ref * p
    residual = worst
    if lo is not None and hi is not None and ok.any():
        at = origin + frame[ok] * p
        residual = max(float(np.max(np.maximum(lo[ok] - at, at - hi[ok]))), worst)
    return Line(
        origin=origin,
        period=p,
        slack=worst,
        residual=residual,
        frame=frame,
        polygon=poly,
        tied=int(tied),
        mid=mid,
        rejected=rejected,
    )

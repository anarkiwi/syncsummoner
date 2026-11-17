"""Numbering a processed run between two Passthru runs by the capture clock alone.

Frame ``n`` of a run starts at ``t0 + n * T + latency`` on the capture clock, one ``T`` a take and one
latency a run, changing only where the card relocks. A run whose picture a program processed carries no
timecode, so its numbering is every whole shift that keeps its latency within half a frame of both ends'.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.optimize import linprog

__all__ = ["Brackets", "candidates", "latency"]

#: Nanoseconds to milliseconds, so the feasibility problem is posed in well-scaled units.
_MS = 1e6


@dataclass(frozen=True)
class Brackets:
    """Frame starts in ``(lo, hi]`` on the capture clock, of frames ``n`` in some numbering."""

    lo: np.ndarray
    hi: np.ndarray
    n: np.ndarray

    @classmethod
    def of(cls, lo: np.ndarray, hi: np.ndarray, n: np.ndarray) -> "Brackets":
        """The brackets of named frames only, ``n >= 0``."""
        ok = np.asarray(n) >= 0
        return cls(
            np.asarray(lo, np.float64)[ok], np.asarray(hi, np.float64)[ok], np.asarray(n, np.float64)[ok]
        )


def latency(b: Brackets, period: float, shift: int = 0) -> tuple[float, float]:
    """The run latency the brackets admit at ``period``, frames numbered ``n + shift``, ``t0`` folded in."""
    n = b.n + shift
    return float(np.max(b.lo - n * period)), float(np.min(b.hi - n * period))


def _rows(b: Brackets, shift: int, column: int, width: int, ref: float, t0: float) -> tuple:
    """``lo - n T <= v <= hi - n T`` for variable ``column``, ``T = t0 + d``: inequality rows and bounds."""
    n = b.n + shift
    lo, hi = (b.lo - ref - n * t0) / _MS, (b.hi - ref - n * t0) / _MS
    a = np.zeros((2 * n.size, width))
    a[: n.size, 0], a[: n.size, column] = -n, -1.0
    a[n.size :, 0], a[n.size :, column] = n, 1.0
    return a, np.r_[-lo, hi]


def _feasible(ends: list[Brackets], mid: Brackets, shift: int, periods: tuple[float, float]) -> bool:
    """Whether some period in range puts the mid latency, shifted by ``shift``, near every end's."""
    t0 = 0.5 * (periods[0] + periods[1])
    ref = float(mid.lo[0])
    width = 2 + len(ends)
    blocks = [_rows(mid, shift, 1, width, ref, t0)]
    blocks += [_rows(end, 0, 2 + i, width, ref, t0) for i, end in enumerate(ends)]
    near = []
    for i in range(len(ends)):
        for sign in (1.0, -1.0):
            row = np.zeros(width)
            row[0], row[1], row[2 + i] = -0.5, sign, -sign
            near.append(row)
    a = np.vstack([b[0] for b in blocks] + [np.array(near)])
    rhs = np.r_[np.concatenate([b[1] for b in blocks]), np.full(len(near), 0.5 * t0 / _MS)]
    bounds = [((periods[0] - t0) / _MS, (periods[1] - t0) / _MS)] + [(None, None)] * (width - 1)
    return linprog(np.zeros(width), A_ub=a, b_ub=rhs, bounds=bounds, method="highs").status == 0


def candidates(ends: list[Brackets], mid: Brackets, periods: tuple[float, float]) -> list[int]:
    """Every shift ``m`` numbering the mid run's frame ``j`` as ``j + m`` within half a frame of every end.

    Only shifts whose latency intervals at the period range's ends can come within half a frame of the
    first end's are tried; each is then decided over the whole range.
    """
    reach = []
    for t in periods:
        own, first = latency(mid, t), latency(ends[0], t)
        reach += [(own[0] - first[1] - 0.5 * t) / t, (own[1] - first[0] + 0.5 * t) / t]
    lo, hi = int(np.floor(min(reach))), int(np.ceil(max(reach)))
    return [m for m in range(lo, hi + 1) if _feasible(ends, mid, m, periods)]

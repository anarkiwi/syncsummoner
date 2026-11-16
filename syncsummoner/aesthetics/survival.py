"""How much of a source's picture and motion an output keeps, frame against exact frame.

Spatial and temporal information are ITU-T P.910's SI and TI: the standard deviation of the
Sobel gradient magnitude of luma, and of the luma difference between consecutive frames.
"""

# pylint: disable=no-member  ; cv2 is a compiled extension pylint cannot introspect

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Iterable, Sequence

import cv2
import numpy as np
from scipy.stats import norm

from syncsummoner.aesthetics.levels import luma_709

#: The conventional three-sigma bound: a 0.13% one-sided chance of calling independent motion followed.
SIGMA = 3.0
#: One 8-bit code value in unit range: the coarsest step any frame here was stored at.
QUANTUM = 1.0 / 255.0
#: Frames identity may lead picture: an unsynced framebuffer scans out torn, its bottom edge one frame newer.
LEADS = (0, 1)


def _plane(frame: np.ndarray) -> np.ndarray:
    """Luma of an RGB frame, or a single-channel frame as it is, in float32."""
    arr = np.asarray(frame, dtype=np.float32)
    return luma_709(arr) if arr.ndim == 3 else arr


def spatial_information(frame: np.ndarray) -> float:
    """P.910 SI of one frame: the standard deviation of its Sobel gradient magnitude."""
    plane = _plane(frame)
    return float(np.hypot(cv2.Sobel(plane, cv2.CV_32F, 1, 0), cv2.Sobel(plane, cv2.CV_32F, 0, 1)).std())


def temporal_information(prev: np.ndarray, curr: np.ndarray) -> float:
    """P.910 TI between two frames: the standard deviation of their luma difference."""
    return float((_plane(curr) - _plane(prev)).std())


def information(frames: Iterable[np.ndarray]) -> tuple[np.ndarray, np.ndarray]:
    """SI of every frame and TI of every frame against the one before; the first TI is NaN."""
    si, ti, prev = [], [np.nan], None
    for frame in frames:
        plane = _plane(frame)
        si.append(spatial_information(plane))
        if prev is not None:
            ti.append(temporal_information(prev, plane))
        prev = plane
    return np.asarray(si, dtype=np.float64), np.asarray(ti[: len(si)], dtype=np.float64)


def _lag1(x: np.ndarray) -> float:
    """Lag-one autocorrelation, zero where the series carries no variance."""
    d = x - x.mean()
    den = float(d @ d)
    return float(d[:-1] @ d[1:] / den) if den > 0 and d.size > 2 else 0.0


def effective_samples(x: np.ndarray, y: np.ndarray) -> float:
    """Independent samples behind a correlation of two autocorrelated series (Bretherton et al. 1999).

    Each series is taken as first-order autoregressive, giving ``n (1 - a b) / (1 + a b)``
    for lag-one autocorrelations ``a`` and ``b``.
    """
    a, b = _lag1(np.asarray(x, np.float64)), _lag1(np.asarray(y, np.float64))
    return float(np.size(x) * (1.0 - a * b) / (1.0 + a * b))


def critical_z(leads: int = len(LEADS), sigma: float = SIGMA) -> float:
    """Fisher z score a correlation must reach, Bonferroni-corrected for the leads tried."""
    return float(norm.isf(norm.sf(sigma) / max(1, int(leads))))


def follow_z(r: float, n_eff: float) -> float:
    """Fisher z score of a correlation against independence, at an effective sample count."""
    if not np.isfinite(r) or n_eff <= 3.0:
        return 0.0
    return float(math.atanh(float(np.clip(r, -1.0 + 1e-12, 1.0 - 1e-12))) * math.sqrt(n_eff - 3.0))


def _pearson(x: np.ndarray, y: np.ndarray) -> float:
    dx, dy = x - x.mean(), y - y.mean()
    den = math.sqrt(float(dx @ dx) * float(dy @ dy))
    return float(dx @ dy / den) if den > 0 else 0.0


@dataclass(frozen=True)
class Survival:
    """What an output kept of its source over one span of source frames.

    ``follow`` correlates the output's TI with the source's at the best of :data:`LEADS`,
    ``lead`` is that lead and ``z`` its Fisher score; the means cover frames both sides measured.
    """

    frames: int
    source_si: float
    source_ti: float
    output_si: float
    output_ti: float
    follow: float
    lead: int
    z: float

    @property
    def si_ratio(self) -> float:
        """Share of the source's spatial information the output keeps."""
        return self.output_si / self.source_si if self.source_si > 0 else float("nan")

    @property
    def ti_ratio(self) -> float:
        """Share of the source's temporal information the output keeps."""
        return self.output_ti / self.source_ti if self.source_ti > 0 else float("nan")

    @property
    def following(self) -> bool:
        """Whether the output's motion follows the source's beyond chance."""
        return self.z > critical_z()


def quantisation_si(quantum: float = QUANTUM) -> float:
    """SI of quantisation noise alone, whose Sobel magnitude is near enough Rayleigh.

    Each Sobel axis sums weights whose squares total 12, so the magnitude is Rayleigh with scale
    ``q``, whose standard deviation is ``q * sqrt((4 - pi) / 2)``.
    """
    return float(quantum * math.sqrt((4.0 - math.pi) / 2.0))


def quantisation_ti(quantum: float = QUANTUM) -> float:
    """TI of quantisation noise alone: the difference of two i.i.d. q²/12 fields has std ``q / sqrt(6)``."""
    return float(quantum / math.sqrt(6.0))


def live_floor(source: float, noise: float) -> float:
    """Least information an output can show and still read as keeping the source's.

    Perceived magnitude is logarithmic in stimulus (Weber-Fechner), so the boundary between the
    source's own information and quantisation noise alone is their geometric mean.
    """
    return float(math.sqrt(max(source, 0.0) * max(noise, 0.0)))


def survival_stats(
    source_si: np.ndarray,
    source_ti: np.ndarray,
    output_si: np.ndarray,
    output_ti: np.ndarray,
    *,
    leads: Sequence[int] = LEADS,
) -> Survival:
    """Survival of one span, the four series indexed by source frame and NaN where unmeasured.

    At lead ``k`` the output attributed to source frame ``n`` shows frame ``n - k``, so it is
    compared with the source ``k`` frames earlier; the lead that follows best is kept.
    """
    series = [np.asarray(v, dtype=np.float64) for v in (source_si, source_ti, output_si, output_ti)]
    best = None
    for lead in leads:
        s_si, s_ti = (v[: v.size - lead] for v in series[:2])
        o_si, o_ti = (v[lead:] for v in series[2:])
        both = np.isfinite(s_si) & np.isfinite(o_si)
        moved = np.isfinite(s_ti) & np.isfinite(o_ti)
        r = _pearson(s_ti[moved], o_ti[moved]) if moved.sum() > 2 else 0.0
        z = follow_z(r, effective_samples(s_ti[moved], o_ti[moved])) if moved.sum() > 2 else 0.0
        found = Survival(
            frames=int(moved.sum()),
            source_si=float(s_si[both].mean()) if both.any() else 0.0,
            source_ti=float(s_ti[moved].mean()) if moved.any() else 0.0,
            output_si=float(o_si[both].mean()) if both.any() else 0.0,
            output_ti=float(o_ti[moved].mean()) if moved.any() else 0.0,
            follow=r,
            lead=int(lead),
            z=z,
        )
        if best is None or found.z > best.z:
            best = found
    return best

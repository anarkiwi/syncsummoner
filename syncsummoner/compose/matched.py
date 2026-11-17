"""Matched-control statistics: each frame's change against step-free frames of like source motion.

Scores are robust departures from the nearest controls' median; p values are empirical; a
parameter's steps combine by Fisher's method, shrunk for serial dependence (docs/evidence.md).
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Mapping

import numpy as np
from scipy.stats import chi2, norm

from syncsummoner.aesthetics.survival import SIGMA


@dataclass(frozen=True)
class Regime:
    """Each clip frame's gating regime as a code, and the bits of it each parameter's own switch sets.

    A step is tested against the step-free frames of its regime, its own switch aside.
    """

    code: np.ndarray
    own: Mapping[int, int] = field(default_factory=dict)

    def of(self, param: int, frames: np.ndarray) -> np.ndarray:
        """The regime of ``frames`` as seen by a step of ``param``."""
        return self.code[frames] & ~self.own.get(param, 0)


def frame_change(stack: np.ndarray) -> np.ndarray:
    """Mean absolute difference of every frame from the one before, all planes; NaN for the first."""
    flat = np.asarray(stack, dtype=np.float32).reshape(len(stack), -1)
    return np.r_[np.nan, np.abs(np.diff(flat, axis=0)).mean(axis=1)]


def matched_change(
    d_out: np.ndarray, d_src: np.ndarray, frames: np.ndarray, controls: np.ndarray, quantum: float
) -> tuple[np.ndarray, np.ndarray]:
    """Each frame's output change less the median of the controls nearest it in source change, and their MAD.

    The ``sqrt(n)`` nearest controls (Loftsgaarden and Quesenberry), the frame itself left out;
    ``quantum``, the measure's resolution, stands in for a MAD of zero.
    """
    k = max(1, int(round(math.sqrt(controls.size))))
    order = np.argsort(d_src[controls], kind="stable")
    ranked, values = controls[order], d_src[controls][order]
    residual, spread = np.full(frames.size, np.nan), np.full(frames.size, np.nan)
    for i, frame in enumerate(frames):
        at = int(np.searchsorted(values, d_src[frame]))
        lo, hi = max(0, at - k - 1), min(values.size, at + k + 1)
        window = ranked[lo:hi]
        window = window[window != frame]
        near = window[np.argsort(np.abs(d_src[window] - d_src[frame]), kind="stable")[:k]]
        ref = d_out[near]
        centre = float(np.median(ref))
        residual[i] = d_out[frame] - centre
        spread[i] = max(1.4826 * float(np.median(np.abs(ref - centre))), quantum)
    return residual, spread


def matched_scores(
    d_out: np.ndarray, d_src: np.ndarray, frames: np.ndarray, controls: np.ndarray, quantum: float
) -> np.ndarray:
    """Each frame's output change against the controls nearest it in source change, in robust units."""
    residual, spread = matched_change(d_out, d_src, frames, controls, quantum)
    return residual / spread


def tail_p(scores: np.ndarray, null: np.ndarray) -> np.ndarray:
    """One-sided p of each score in the empirical distribution of the null scores (mid-p)."""
    ordered = np.sort(null)
    above = null.size - np.searchsorted(ordered, scores, side="left")
    ties = np.searchsorted(ordered, scores, side="right") - np.searchsorted(ordered, scores, side="left")
    return (above - 0.5 * ties + 0.5) / (null.size + 1)


def fisher_z(p: np.ndarray) -> tuple[float, float]:
    """Fisher's combination of one-sided p values as a z score, and the independent samples behind it.

    Serial dependence shrinks the statistic and its degrees of freedom by ``n_eff / n`` (Brown).
    """
    if not p.size:
        return 0.0, 0.0
    terms = -2.0 * np.log(p)
    n_eff = effective(terms)
    stat = float(terms.sum()) * n_eff / p.size
    return float(norm.isf(chi2.sf(stat, 2.0 * n_eff))), n_eff


def effective(u: np.ndarray) -> float:
    """Independent samples behind a mean of a first-order autoregressive series."""
    if u.size < 3:
        return float(u.size)
    if np.ptp(u) == 0:
        return float(u.size)
    d = u - u.mean()
    rho = float(d[:-1] @ d[1:] / float(d @ d))
    return float(np.clip(u.size * (1 - rho) / (1 + rho), 1.0, u.size))


def critical(tests: int, sigma: float = SIGMA) -> float:
    """The one-sided z a test must pass for a family-wise error of the ``sigma`` tail over ``tests``."""
    return float(norm.isf(norm.sf(sigma) / max(1, int(tests))))

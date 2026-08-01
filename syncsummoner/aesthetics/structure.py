"""Whether a transform remaps values in place or re-addresses the picture.

A pointwise transform sets each output sample from the co-located input sample
alone, so a single curve explains it. Anything that displaces, tiles, delays or
re-samples breaks that correspondence however smooth the result looks.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

#: Luma weights matching the rest of the package.
LUMA = np.array([0.2126, 0.7152, 0.0722], dtype=np.float32)
DEFAULT_BINS = 32


@dataclass(frozen=True)
class PointwiseFit:
    """How well a co-located value curve explains one frame."""

    r2: float
    curve: np.ndarray
    support: float


def luma(frame: np.ndarray) -> np.ndarray:
    """Rec.709 luma of an RGB frame."""
    return frame @ LUMA


def pointwise_fit(source: np.ndarray, output: np.ndarray, *, bins: int = DEFAULT_BINS) -> PointwiseFit:
    """Fraction of output variance explained by co-located source luma.

    The conditional mean of output given binned input is the best pointwise
    predictor there is, so what it leaves unexplained is what no value curve
    can account for.
    """
    x = np.clip(luma(source), 0.0, 1.0).ravel()
    y = luma(output).ravel()
    if x.size != y.size or y.size == 0:
        raise ValueError(f"source and output must match in size, got {x.size} and {y.size}")
    total = float(y.var())
    index = np.minimum((x * bins).astype(np.intp), bins - 1)
    counts = np.bincount(index, minlength=bins).astype(np.float64)
    sums = np.bincount(index, weights=y, minlength=bins)
    squares = np.bincount(index, weights=y * y, minlength=bins)
    live = counts > 0
    curve = np.divide(sums, counts, out=np.zeros(bins), where=live)
    if total <= 0:
        return PointwiseFit(1.0, curve, float(live.mean()))
    within = float((squares[live] - sums[live] ** 2 / counts[live]).sum() / y.size)
    return PointwiseFit(float(np.clip(1.0 - within / total, 0.0, 1.0)), curve, float(live.mean()))


def pointwise_r2(source: np.ndarray, output: np.ndarray, *, bins: int = DEFAULT_BINS) -> float:
    """Scalar form of :func:`pointwise_fit`, high when a value curve explains the output."""
    return pointwise_fit(source, output, bins=bins).r2

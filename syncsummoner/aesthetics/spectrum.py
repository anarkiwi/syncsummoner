"""Radially averaged power spectrum: 1/f slope and fractal dimension."""

from dataclasses import dataclass

import numpy as np
from scipy.signal.windows import hann

from syncsummoner.aesthetics.levels import luma


@dataclass(frozen=True)
class SpectralStats:
    """Log-log radial power fit; natural scenes sit near slope -1.0..-1.4."""

    slope: float
    r2: float
    fractal_dimension: float


def radial_power(frame: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Radially averaged power spectrum of frame luma, as (cycles/pixel, power)."""
    gray = luma(frame)
    h, w = gray.shape
    window = np.outer(hann(h, sym=False), hann(w, sym=False)).astype(np.float32)
    power = np.abs(np.fft.fft2((gray - gray.mean()) * window)) ** 2
    radius = np.hypot(*np.meshgrid(np.fft.fftfreq(w), np.fft.fftfreq(h)))
    n_bins = min(h, w) // 2
    if n_bins < 4:
        return np.empty(0), np.empty(0)
    index = np.floor(radius * 2.0 * n_bins).astype(np.intp)
    keep = (index >= 1) & (index <= n_bins)
    counts = np.bincount(index[keep], minlength=n_bins + 1)
    sums = np.bincount(index[keep], weights=power[keep], minlength=n_bins + 1)
    valid = counts > 0
    freq = (np.arange(n_bins + 1)[valid] + 0.5) / (2.0 * n_bins)
    return freq, sums[valid] / counts[valid]


def spectral_stats(frame: np.ndarray) -> SpectralStats:
    """Slope and fractal dimension from a linear fit of log power against log frequency."""
    freq, power = radial_power(frame)
    finite = power > 0.0
    if finite.sum() < 3:
        return SpectralStats(slope=0.0, r2=0.0, fractal_dimension=1.0)
    x = np.log10(freq[finite])
    y = np.log10(power[finite])
    slope, intercept = np.polyfit(x, y, 1)
    residual = float(np.sum(np.square(y - (slope * x + intercept))))
    total = float(np.sum(np.square(y - y.mean())))
    r2 = 1.0 - residual / total if total > 0.0 else 0.0
    return SpectralStats(
        slope=float(slope),
        r2=float(np.clip(r2, 0.0, 1.0)),
        fractal_dimension=float(np.clip((7.0 + 2.0 * slope) / 2.0, 1.0, 2.0)),
    )

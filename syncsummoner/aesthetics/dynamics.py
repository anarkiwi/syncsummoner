"""Temporal periodicity, winding number and stability classification.

Every threshold derives from the white-noise null at level ``alpha``: the
autocorrelation bound is Bonferroni corrected over the lags searched, line
detection uses Fisher's g and comb coverage uses the Beta order statistic.
"""

import enum
from dataclasses import dataclass
from fractions import Fraction

import numpy as np
from scipy.signal import correlate, find_peaks, periodogram
from scipy.stats import beta, norm

STATIC_TOLERANCE = 1e-6
ALPHA = 0.05
MAX_LOCK_ORDER = 4


class StabilityClass(enum.Enum):
    """Qualitative regime of a temporal series."""

    STATIC = "static"
    PERIODIC = "periodic"
    QUASIPERIODIC = "quasiperiodic"
    CHAOTIC = "chaotic"


@dataclass(frozen=True)
class DynamicsResult:
    """Periodicity, winding number and stability class of a scalar series."""

    period_frames: float | None
    periodicity_strength: float
    winding_number: float | None
    stability: StabilityClass


def _as_series(series: np.ndarray) -> np.ndarray:
    arr = np.asarray(series, dtype=np.float64).ravel()
    if arr.size == 0 or not np.all(np.isfinite(arr)):
        raise ValueError("series must be non-empty and finite")
    return arr


def autocorrelation(series: np.ndarray) -> np.ndarray:
    """Normalized autocorrelation of a series at non-negative lags."""
    arr = _as_series(series)
    arr = arr - arr.mean()
    acf = correlate(arr, arr, mode="full", method="fft")[arr.size - 1 :]
    return acf / acf[0] if acf[0] > 0.0 else np.zeros_like(acf)


def _refine_peak(values: np.ndarray, index: int) -> float:
    """Parabolic sub-sample refinement of a local maximum."""
    if index <= 0 or index >= values.size - 1:
        return float(index)
    left, mid, right = values[index - 1 : index + 2]
    denom = left - 2.0 * mid + right
    return float(index) if denom == 0.0 else float(index) - 0.5 * (right - left) / denom


def acf_critical(n_samples: int, n_lags: int, alpha: float) -> float:
    """Autocorrelation significance bound under the white-noise null, Bonferroni corrected."""
    return float(norm.isf(alpha / (2.0 * max(n_lags, 1))) / np.sqrt(n_samples))


def dominant_period(series: np.ndarray, *, alpha: float = ALPHA) -> tuple[float | None, float]:
    """Autocorrelation period in frames and its peak strength in [0, 1]."""
    acf = autocorrelation(series)
    limit = acf.size // 2
    if limit < 2:
        return None, 0.0
    peaks, _ = find_peaks(acf[: limit + 2])
    peaks = peaks[peaks <= limit]
    if peaks.size == 0:
        return None, 0.0
    best = int(peaks[np.argmax(acf[peaks])])
    strength = float(np.clip(acf[best], 0.0, 1.0))
    if strength < acf_critical(acf.size, limit, alpha):
        return None, strength
    return _refine_peak(acf, best), strength


def fisher_g_pvalue(g: float, m: int) -> float:
    """Union bound on Fisher's g statistic for ``m`` periodogram ordinates."""
    if m < 2 or g <= 0.0:
        return 1.0
    return float(np.clip(m * (1.0 - min(g, 1.0)) ** (m - 1), 0.0, 1.0))


def fisher_g_critical(m: int, alpha: float) -> float:
    """Line-detection threshold: the g at which the union-bound p-value reaches ``alpha``."""
    return 1.0 - (alpha / m) ** (1.0 / (m - 1)) if m >= 2 else 1.0


def harmonic_bins(n_bins: int, fundamental: float, bin_width: float) -> np.ndarray:
    """Periodogram bins covering the fundamental and its harmonics, one lobe wide."""
    harmonics = np.arange(1, max(int(n_bins * bin_width / fundamental), 1) + 1) * fundamental
    centres = np.round(harmonics / bin_width).astype(np.intp) - 1
    return np.unique(np.clip(centres[:, None] + np.arange(-1, 2), 0, n_bins - 1))


def _classify(series: np.ndarray, *, period: float, alpha: float, max_lock_order: int) -> StabilityClass:
    """Chaotic unless the harmonic comb of ``period`` explains the spectrum.

    Two significant lines whose ratio is not a rational of order at most
    ``max_lock_order``, at the available frequency resolution, read as quasiperiodic.
    """
    freq, power = periodogram(series, window="hann", detrend="constant")
    freq, power = freq[1:], power[1:]
    n_bins = power.size
    total = float(power.sum())
    if n_bins < 3 or total <= 0.0:
        return StabilityClass.CHAOTIC
    comb = harmonic_bins(n_bins, 1.0 / period, float(freq[0]))
    explained = float(power[comb].sum()) / total
    if comb.size < n_bins and explained <= beta.ppf(1.0 - alpha, comb.size, n_bins - comb.size):
        return StabilityClass.CHAOTIC
    peaks, _ = find_peaks(power)
    lines = peaks[power[peaks] >= fisher_g_critical(n_bins, alpha) * total]
    if lines.size < 2:
        return StabilityClass.PERIODIC
    low, high = np.sort(freq[lines[np.argsort(power[lines])[::-1][:2]]])
    lock = Fraction(float(high / low)).limit_denominator(max_lock_order)
    detune = abs(float(high) - float(lock) * float(low))
    return StabilityClass.PERIODIC if detune <= float(freq[0]) else StabilityClass.QUASIPERIODIC


def winding_number(series: np.ndarray, *, fps: float, alpha: float = ALPHA) -> float | None:
    """Cycles per frame of the series' dominant periodicity, relative to ``fps``."""
    period, _ = dominant_period(series, alpha=alpha)
    if period is None or period <= 0.0:
        return None
    return float((fps / period) / fps)


def analyze_dynamics(
    series: np.ndarray,
    *,
    fps: float,
    static_tolerance: float = STATIC_TOLERANCE,
    alpha: float = ALPHA,
    max_lock_order: int = MAX_LOCK_ORDER,
) -> DynamicsResult:
    """Classify a scalar series and report its periodicity and winding number."""
    arr = _as_series(series)
    if float(np.ptp(arr)) <= static_tolerance:
        return DynamicsResult(None, 0.0, None, StabilityClass.STATIC)
    period, strength = dominant_period(arr, alpha=alpha)
    if period is None:
        return DynamicsResult(None, strength, None, StabilityClass.CHAOTIC)
    stability = _classify(arr, period=period, alpha=alpha, max_lock_order=max_lock_order)
    if stability is StabilityClass.CHAOTIC:
        return DynamicsResult(period, strength, None, stability)
    return DynamicsResult(period, strength, float((fps / period) / fps), stability)

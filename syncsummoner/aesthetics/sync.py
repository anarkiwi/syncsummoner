"""Audio/visual alignment by normalized cross-correlation.

Tolerance is asymmetric in perception (audio lag > 100 ms passes, audio lead is
caught at 30-50 ms), so the sign of ``lag_s`` is load bearing: positive means the
visual series lags the audio series.
"""

from dataclasses import dataclass

import numpy as np
from scipy.signal import correlate, correlation_lags


@dataclass(frozen=True)
class SyncResult:
    """Peak cross-correlation between a visual and an audio series."""

    lag_s: float
    correlation: float


def _resample(series: np.ndarray, *, src_fps: float, dst_fps: float) -> np.ndarray:
    arr = np.asarray(series, dtype=np.float64).ravel()
    if arr.size < 2 or src_fps == dst_fps:
        return arr
    duration = (arr.size - 1) / src_fps
    n_out = int(round(duration * dst_fps)) + 1
    return np.interp(np.arange(n_out) / dst_fps, np.arange(arr.size) / src_fps, arr)


def _standardize(arr: np.ndarray) -> np.ndarray:
    centred = arr - arr.mean()
    scale = float(np.sqrt(np.mean(np.square(centred))))
    return centred / scale if scale > 0.0 else centred


def av_correlation(
    visual: np.ndarray,
    audio_strength: np.ndarray,
    *,
    visual_fps: float,
    audio_fps: float,
    max_lag_s: float = 0.5,
) -> SyncResult:
    """Best lag and correlation between visual and audio strength series."""
    rate = max(float(visual_fps), float(audio_fps))
    vis = _standardize(_resample(visual, src_fps=visual_fps, dst_fps=rate))
    aud = _standardize(_resample(audio_strength, src_fps=audio_fps, dst_fps=rate))
    if vis.size < 2 or aud.size < 2:
        raise ValueError("both series need at least two samples")
    xcorr = correlate(vis, aud, mode="full", method="fft") / min(vis.size, aud.size)
    lags = correlation_lags(vis.size, aud.size, mode="full")
    window = np.abs(lags) <= max(1, int(round(max_lag_s * rate)))
    peak = int(np.argmax(xcorr[window]))
    lag = float(lags[window][peak])
    inside = xcorr[window]
    if 0 < peak < inside.size - 1:
        left, mid, right = inside[peak - 1 : peak + 2]
        denom = left - 2.0 * mid + right
        if denom != 0.0:
            lag -= 0.5 * (right - left) / denom
    return SyncResult(lag_s=lag / rate, correlation=float(np.clip(inside[peak], -1.0, 1.0)))

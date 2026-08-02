"""Hard safety constraints on rendered output.

These are vetoes, not scores: they override any fitness value. The flash test is
a relative-luminance proxy for a Harding assessment, which needs absolute
cd/m^2 and display characterisation this library does not have.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.ndimage import uniform_filter1d

from syncsummoner.aesthetics.levels import LUMA_709, downscale

SRGB_KNEE = 0.04045


@dataclass(frozen=True)
class FlashRisk:
    """Worst-case flash statistics for a clip, and the windows responsible."""

    flashes_per_s: float
    area_frac: float
    red_flashes_per_s: float
    windows: tuple[tuple[int, int], ...]
    safe: bool


def relative_luminance(frames: np.ndarray) -> np.ndarray:
    """WCAG relative luminance of an RGB frame stack, shape ``(T, H, W)``."""
    lin = np.where(frames <= SRGB_KNEE, frames / 12.92, ((frames + 0.055) / 1.055) ** 2.4)
    return (lin * LUMA_709).sum(-1).astype(np.float32)


def saturated_red(frames: np.ndarray, *, fraction: float = 0.8) -> np.ndarray:
    """Mask of pixels whose red component dominates, shape ``(T, H, W)``."""
    total = frames.sum(-1)
    with np.errstate(divide="ignore", invalid="ignore"):
        ratio = np.where(total > 0, frames[..., 0] / np.maximum(total, 1e-6), 0.0)
    return (ratio >= fraction) & (frames[..., 0] >= 0.5)


def _downsample(frames: np.ndarray, max_side: int) -> np.ndarray:
    """Shrink a frame stack so area statistics stay cheap on long clips."""
    small = [downscale(f, max_side=max_side) for f in frames]
    if small[0].shape == frames.shape[1:]:
        return frames.astype(np.float32, copy=False)
    return np.stack(small).astype(np.float32)


def _windowed_rate(events: np.ndarray, window: int) -> np.ndarray:
    """Per-pixel event count in a trailing window of ``window`` frames."""
    if window <= 1 or events.shape[0] <= window:
        return events.sum(0, keepdims=True)
    cumulative = np.cumsum(events, axis=0, dtype=np.float32)
    return cumulative[window:] - cumulative[:-window]


def flash_risk(
    frames: np.ndarray,
    *,
    fps: float,
    luminance_delta: float = 0.1,
    dark_max: float = 0.8,
    area_threshold: float = 0.25,
    rate_limit: float = 3.0,
    max_side: int = 128,
) -> FlashRisk:
    """Screen a clip for photosensitive-seizure risk.

    Flags opposing luminance transitions exceeding ``luminance_delta`` where the
    darker state is below ``dark_max``, occurring faster than ``rate_limit`` per
    second over more than ``area_threshold`` of the frame.
    """
    if frames.shape[0] < 2:
        return FlashRisk(0.0, 0.0, 0.0, (), True)
    small = _downsample(frames, max_side)
    lum = relative_luminance(small)
    delta = np.diff(lum, axis=0)
    darker = np.minimum(lum[:-1], lum[1:])
    transition = (np.abs(delta) >= luminance_delta) & (darker < dark_max)
    red = saturated_red(small)
    red_transition = transition & (red[:-1] != red[1:])

    window = max(1, int(round(fps)))
    pairs = _windowed_rate(transition.astype(np.float32), window) / 2.0
    red_pairs = _windowed_rate(red_transition.astype(np.float32), window) / 2.0
    offending = pairs > rate_limit
    area = offending.mean(axis=(1, 2))
    unsafe = area > area_threshold

    return FlashRisk(
        flashes_per_s=float(pairs.max(initial=0.0)),
        area_frac=float(area.max(initial=0.0)),
        red_flashes_per_s=float(red_pairs.max(initial=0.0)),
        windows=_runs(unsafe, window),
        safe=not bool(unsafe.any()),
    )


def _runs(flags: np.ndarray, window: int) -> tuple[tuple[int, int], ...]:
    """Contiguous True runs of ``flags``, widened to the frames they cover."""
    if not flags.any():
        return ()
    padded = np.concatenate(([False], flags, [False]))
    edges = np.flatnonzero(padded[1:] != padded[:-1])
    return tuple((int(a), int(b) + window) for a, b in zip(edges[::2], edges[1::2]))


def dead_output(frames: np.ndarray, *, fps: float, min_seconds: float = 2.0, tol: float = 0.02) -> bool:
    """True when the clip holds full black or full white for ``min_seconds``."""
    run_len = max(1, int(round(fps * min_seconds)))
    if frames.shape[0] < run_len:
        return False
    mean = frames.reshape(frames.shape[0], -1).mean(1)
    spread = frames.reshape(frames.shape[0], -1).std(1)
    flat = (spread <= tol) & ((mean <= tol) | (mean >= 1.0 - tol))
    return bool(_windowed_rate(flat[:, None, None].astype(np.float32), run_len).max(initial=0.0) >= run_len)


def mitigate_flashes(
    frames: np.ndarray, risk: FlashRisk, *, fps: float, rate_limit: float = 3.0
) -> np.ndarray:
    """Temporally low-pass the offending windows instead of discarding the take.

    The filter length places the moving average's first null at ``rate_limit``,
    so everything the veto objects to lands in the stopband.
    """
    if risk.safe:
        return frames
    taps = max(3, int(round(fps / max(rate_limit, 1e-3))) | 1)
    out = frames.copy()
    for start, stop in risk.windows:
        lo, hi = max(0, start), min(frames.shape[0], stop + 1)
        if hi - lo < 2:
            continue
        out[lo:hi] = uniform_filter1d(frames[lo:hi], taps, axis=0, mode="nearest")
    return out

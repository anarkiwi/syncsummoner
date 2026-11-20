"""Hard safety constraints on rendered output.

Vetoes, not scores. The general and red flash tests of ITU-R BT.1702 and WCAG 2.3.1 as EA's IRIS reads
them, in relative luminance, which stands in for cd/m^2 without a display characterisation.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.ndimage import uniform_filter1d

from syncsummoner.aesthetics.levels import LUMA_709, downscale

SRGB_KNEE = 0.04045
#: A general flash's least luminance change, and the brightest its darker state may be (WCAG 2.x).
LUMINANCE_DELTA, DARK_MAX = 0.1, 0.8
#: A red flash's least change in ``RED_GAIN * (R - G - B)`` of saturated red, red share ``RED_FRACTION``.
RED_DELTA, RED_GAIN, RED_FRACTION = 20.0, 320.0, 0.8
#: More flashes than this in any one second fail, over at least this share of the screen (BT.1702).
RATE_LIMIT, AREA_LIMIT = 3.0, 0.25


@dataclass(frozen=True)
class FlashRisk:
    """Worst-case flash statistics for a clip, and the windows responsible."""

    flashes_per_s: float
    area_frac: float
    red_flashes_per_s: float
    windows: tuple[tuple[int, int], ...]
    safe: bool
    red_area_frac: float = 0.0
    general_safe: bool = True
    red_safe: bool = True


def linear_rgb(frames: np.ndarray) -> np.ndarray:
    """sRGB values in unit range, linearised."""
    return np.where(frames <= SRGB_KNEE, frames / 12.92, ((frames + 0.055) / 1.055) ** 2.4)


def relative_luminance(frames: np.ndarray) -> np.ndarray:
    """WCAG relative luminance of an RGB frame stack, shape ``(T, H, W)``."""
    return (linear_rgb(frames) * LUMA_709).sum(-1).astype(np.float32)


def saturated_red(frames: np.ndarray, *, fraction: float = RED_FRACTION) -> np.ndarray:
    """Mask of pixels whose linear red share is at least ``fraction``, shape ``(T, H, W)``."""
    lin = linear_rgb(frames)
    total = lin.sum(-1)
    return (lin[..., 0] >= fraction * total) & (total > 0)


def red_quantity(frames: np.ndarray) -> np.ndarray:
    """WCAG's ``RED_GAIN * max(0, R - G - B)`` of linear RGB where the red is saturated, else zero."""
    lin = linear_rgb(frames)
    red = RED_GAIN * np.maximum(lin[..., 0] - lin[..., 1] - lin[..., 2], 0.0)
    return np.where(saturated_red(frames), red, 0.0).astype(np.float32)


def _downsample(frames: np.ndarray, max_side: int) -> np.ndarray:
    """Shrink a frame stack so area statistics stay cheap on long clips."""
    small = [downscale(f, max_side=max_side) for f in frames]
    if small[0].shape == frames.shape[1:]:
        return frames.astype(np.float32, copy=False)
    return np.stack(small).astype(np.float32)


def transitions(
    planes: np.ndarray, *, delta: float, fps: float, dark_max: float = np.inf, area: float = AREA_LIMIT
) -> tuple[np.ndarray, np.ndarray]:
    """Which frames complete a flash transition, and the share of the screen changing at each.

    A frame's change is the screen mean's, counted where at least ``area`` of the screen changed. Same-signed
    changes over at most a second accumulate; a run becomes one transition when its total first reaches
    ``delta`` with the darker of the two frames' means below ``dark_max``.
    """
    x = np.asarray(planes, dtype=np.float32).reshape(planes.shape[0], -1)
    mean = x.mean(axis=1)
    changed = (np.diff(x, axis=0) != 0).mean(axis=1)
    step = np.where(changed >= area, np.diff(mean), 0.0)
    window = max(1, int(round(fps)))
    events = np.zeros(x.shape[0], dtype=bool)
    run: list[float] = []
    for t, d in enumerate(step, 1):
        before = float(np.sum(run))
        onward = bool(run) and (d == 0 or np.sign(d) == np.sign(before))
        run = (run[-(window - 1) :] if window > 1 else []) + [float(d)] if onward else [float(d)]
        total = float(np.sum(run))
        counted = onward and abs(before) >= delta
        events[t] = (not counted) and abs(total) >= delta and min(mean[t - 1], mean[t]) < dark_max
    return events, np.r_[0.0, changed]


def _flashing(
    events: np.ndarray, area: np.ndarray, window: int, rate_limit: float
) -> tuple[float, float, np.ndarray]:
    """Peak flashes a second, peak changing area, and which windows hold more than ``rate_limit`` flashes."""
    count = np.convolve(events.astype(np.float64), np.ones(min(window, events.size)), mode="valid")
    pairs = count / 2.0
    return float(pairs.max(initial=0.0)), float(area.max(initial=0.0)), pairs > rate_limit


def flash_risk(
    frames: np.ndarray,
    *,
    fps: float,
    luminance_delta: float = LUMINANCE_DELTA,
    dark_max: float = DARK_MAX,
    area_threshold: float = AREA_LIMIT,
    rate_limit: float = RATE_LIMIT,
    max_side: int = 128,
) -> FlashRisk:
    """Screen a clip for photosensitive-seizure risk by the general and red flash tests.

    A flash is a pair of opposing transitions (:func:`transitions`); a clip fails where any second holds
    more than ``rate_limit`` of them.
    """
    if frames.shape[0] < 2:
        return FlashRisk(0.0, 0.0, 0.0, (), True)
    small = _downsample(frames, max_side)
    window = max(1, int(round(fps)))
    general = transitions(
        relative_luminance(small), delta=luminance_delta, fps=fps, dark_max=dark_max, area=area_threshold
    )
    red = transitions(red_quantity(small), delta=RED_DELTA, fps=fps, area=area_threshold)
    rate, area, unsafe = _flashing(*general, window, rate_limit)
    red_rate, red_area, red_unsafe = _flashing(*red, window, rate_limit)
    return FlashRisk(
        flashes_per_s=rate,
        area_frac=area,
        red_flashes_per_s=red_rate,
        windows=_runs(unsafe | red_unsafe, min(window, frames.shape[0])),
        safe=not bool(unsafe.any() or red_unsafe.any()),
        red_area_frac=red_area,
        general_safe=not bool(unsafe.any()),
        red_safe=not bool(red_unsafe.any()),
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
    return bool(np.convolve(flat, np.ones(run_len), mode="valid").max(initial=0.0) >= run_len)


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

"""Safety vetoes: flash risk, dead output, mitigation."""

# pylint: disable=missing-function-docstring

import numpy as np
import pytest

from syncsummoner.aesthetics import safety

FPS = 30.0
SHAPE = (48, 64)


def solid(value, n, shape=SHAPE):
    return np.full((n,) + shape + (3,), value, dtype=np.float32)


def alternating(n, *, hz, fps=FPS, lo=0.0, hi=1.0, area=1.0, shape=SHAPE):
    """Full-frame square-wave flash at ``hz``, over ``area`` of the frame."""
    frames = solid(lo, n, shape)
    period = max(1, int(round(fps / (2 * hz))))
    rows = int(round(shape[0] * area))
    for t in range(n):
        if (t // period) % 2:
            frames[t, :rows] = hi
    return frames


def test_relative_luminance_matches_wcag_anchors():
    lum = safety.relative_luminance(np.stack([solid(0.0, 1)[0], solid(1.0, 1)[0]]))
    assert lum[0].max() == pytest.approx(0.0, abs=1e-6)
    assert lum[1].min() == pytest.approx(1.0, abs=1e-3)


def test_static_clip_is_safe():
    risk = safety.flash_risk(solid(0.5, 60), fps=FPS)
    assert risk.safe and risk.flashes_per_s == 0.0 and not risk.windows


def test_slow_flash_is_safe():
    risk = safety.flash_risk(alternating(90, hz=2.0), fps=FPS)
    assert risk.safe


def test_fast_full_field_flash_is_unsafe():
    risk = safety.flash_risk(alternating(90, hz=7.5), fps=FPS)
    assert not risk.safe
    assert risk.flashes_per_s > 3.0
    assert risk.area_frac > 0.25
    assert risk.windows


def test_fast_flash_over_small_area_is_safe():
    risk = safety.flash_risk(alternating(90, hz=7.5, area=0.1), fps=FPS)
    assert risk.safe and risk.area_frac <= 0.25


def test_low_contrast_flash_is_safe():
    risk = safety.flash_risk(alternating(90, hz=7.5, lo=0.5, hi=0.55), fps=FPS)
    assert risk.safe


def test_bright_flash_is_exempt_when_darker_state_is_bright():
    risk = safety.flash_risk(alternating(90, hz=7.5, lo=0.95, hi=1.0), fps=FPS)
    assert risk.safe


def test_saturated_red_mask_selects_red_only():
    frames = np.zeros((1,) + SHAPE + (3,), np.float32)
    frames[0, :, :16, 0] = 1.0
    frames[0, :, 16:32, 1] = 1.0
    mask = safety.saturated_red(frames)
    assert mask[0, :, :16].all() and not mask[0, :, 16:].any()


def test_red_flash_is_counted():
    frames = solid(0.0, 90)
    for t in range(0, 90, 2):
        frames[t, :, :, 0] = 1.0
    risk = safety.flash_risk(frames, fps=FPS)
    assert risk.red_flashes_per_s > 3.0


def test_short_clip_is_safe():
    assert safety.flash_risk(solid(0.5, 1), fps=FPS).safe


def test_dead_output_detects_sustained_black_and_white():
    assert safety.dead_output(solid(0.0, 90), fps=FPS)
    assert safety.dead_output(solid(1.0, 90), fps=FPS)


def test_dead_output_ignores_content_and_short_clips():
    rng = np.random.default_rng(0)
    noise = rng.random((90,) + SHAPE + (3,)).astype(np.float32)
    assert not safety.dead_output(noise, fps=FPS)
    assert not safety.dead_output(solid(0.0, 4), fps=FPS)


def test_mitigation_makes_an_unsafe_clip_safe():
    frames = alternating(90, hz=7.5)
    risk = safety.flash_risk(frames, fps=FPS)
    assert not risk.safe
    fixed = safety.mitigate_flashes(frames, risk, fps=FPS)
    assert fixed.shape == frames.shape
    assert safety.flash_risk(fixed, fps=FPS).safe


def test_mitigation_leaves_safe_clips_untouched():
    frames = solid(0.5, 30)
    risk = safety.flash_risk(frames, fps=FPS)
    assert safety.mitigate_flashes(frames, risk, fps=FPS) is frames


def test_downsample_preserves_area_fraction():
    big = alternating(90, hz=7.5, area=1.0, shape=(480, 640))
    assert not safety.flash_risk(big, fps=FPS, max_side=64).safe

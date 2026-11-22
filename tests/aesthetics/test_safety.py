"""Safety vetoes: flash risk, dead output, mitigation."""

# pylint: disable=missing-function-docstring

import numpy as np
import pytest
from scipy.ndimage import gaussian_filter

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


def test_a_fast_red_flash_fails_the_red_test():
    frames = solid(0.0, 60)
    frames[::4, :, :, 0] = 1.0
    frames[1::4, :, :, 0] = 1.0
    risk = safety.flash_risk(frames, fps=FPS)
    assert not risk.red_safe and not risk.safe and risk.red_area_frac == pytest.approx(1.0)


def test_a_gradual_swing_accumulates_into_one_transition():
    ramp = np.r_[np.linspace(0.0, 0.3, 4), np.linspace(0.3, 0.0, 4)[1:]]
    events, _ = safety.transitions(ramp[:, None, None], delta=0.1, fps=FPS)
    assert events.sum() == 2


def test_a_change_over_too_little_of_the_screen_is_no_transition():
    frames = solid(0.0, 8)
    frames[1::2, :4] = 1.0
    events, area = safety.transitions(safety.relative_luminance(frames), delta=0.1, fps=FPS)
    assert not events.any() and area.max() < 0.25


def moving_texture(n, *, shape=SHAPE, seed=1):
    """A smooth random texture of mid luminance and low-amplitude noise drifting a pixel a frame."""
    rng = np.random.default_rng(seed)
    field = gaussian_filter(rng.random((shape[0], shape[1] + n)), 4.0)
    field = 0.5 + (field - field.mean()) / (field.max() - field.min()) * 0.4
    frames = np.stack([field[:, t : t + shape[1]] for t in range(n)])
    frames += rng.normal(0.0, 0.01, frames.shape)
    return np.repeat(np.clip(frames, 0.0, 1.0)[..., None], 3, axis=-1).astype(np.float32)


def test_a_moving_texture_is_not_a_flash():
    risk = safety.flash_risk(moving_texture(90), fps=FPS)
    assert risk.safe and risk.flashes_per_s == 0.0 and risk.area_frac < 0.25


def test_a_small_region_flipping_over_a_moving_texture_is_not_a_flash():
    frames = moving_texture(90)
    frames[1::2, 16:32, 20:44] = 1.0
    frames[::2, 16:32, 20:44] = 0.0
    risk = safety.flash_risk(frames, fps=FPS)
    assert risk.safe and risk.flashes_per_s == 0.0 and 0.0 < risk.area_frac <= 0.25


def test_a_full_screen_alternation_over_a_moving_texture_is_a_flash():
    frames = moving_texture(90)
    frames[1::2] = np.clip(frames[1::2] + 0.5, 0.0, 1.0)
    risk = safety.flash_risk(frames, fps=FPS)
    assert not risk.general_safe and risk.flashes_per_s == pytest.approx(15.0) and risk.area_frac > 0.9


@pytest.mark.parametrize(
    "pixels, flagged", [(SHAPE[0] * SHAPE[1] // 4 + 1, True), (SHAPE[0] * SHAPE[1] // 4, False)]
)
def test_a_flash_fails_over_more_than_a_quarter_of_the_screen_alone(pixels, flagged):
    frames = solid(0.0, 90)
    region = np.zeros(SHAPE[0] * SHAPE[1], dtype=bool)
    region[:pixels] = True
    frames[1::2, region.reshape(SHAPE)] = 1.0
    assert safety.flash_risk(frames, fps=FPS).general_safe is not flagged


def test_opposing_halves_flash_though_the_screen_mean_holds():
    frames = solid(0.0, 90)
    frames[::2, : SHAPE[0] // 2] = 1.0
    frames[1::2, SHAPE[0] // 2 :] = 1.0
    risk = safety.flash_risk(frames, fps=FPS)
    assert not risk.general_safe and risk.flashes_per_s == pytest.approx(15.0)


def test_a_red_region_flashing_over_a_moving_texture_is_a_red_flash():
    frames = moving_texture(90)
    frames[1::2, : SHAPE[0] // 2] = [1.0, 0.0, 0.0]
    risk = safety.flash_risk(frames, fps=FPS)
    assert not risk.red_safe and risk.red_flashes_per_s == pytest.approx(15.0)


def test_a_clip_measured_in_chunks_is_measured_as_one():
    frames = np.r_[alternating(45, hz=7.5), alternating(45, hz=2.0, lo=0.3)]
    frames[50:60, :, :, 0] = 1.0
    whole = safety.flash_series([frames], fps=FPS)
    split = safety.flash_series([frames[:7], frames[7:7], frames[7:50], frames[50:]], fps=FPS)
    for one, other in zip(whole, split):
        assert np.array_equal(one.events, other.events) and np.array_equal(one.area, other.area)
    assert safety.series_risk(*split, fps=FPS) == safety.flash_risk(frames, fps=FPS)
    assert safety.series_risk(*safety.flash_series([], fps=FPS), fps=FPS).safe


def test_the_frames_to_blame_are_the_transitions_in_the_failing_seconds_alone():
    frames = np.r_[solid(0.5, 30), alternating(30, hz=5.0), solid(0.0, 30), alternating(30, hz=1.0)]
    risk = safety.flash_risk(frames, fps=FPS)
    ((start, stop),) = risk.windows
    assert not risk.safe and start < 30 and stop > 61
    assert min(risk.flashes) == 30 and max(risk.flashes) == 60, "the last flash ends on the black after it"
    assert not safety.flash_risk(frames[:30], fps=FPS).flashes

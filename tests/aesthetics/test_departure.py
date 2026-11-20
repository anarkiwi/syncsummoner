"""Effect intensity and safety: SSIM, CIEDE2000, blank, static and flashing outputs."""

# pylint: disable=missing-function-docstring

import numpy as np
import pytest
from scipy.ndimage import gaussian_filter

from syncsummoner.aesthetics import departure as E

SHAPE = (180, 192)
FPS = 30.0


def texture(n=8, seed=0, shape=SHAPE):
    """A drifting band-limited grey texture, ``(n, H, W, 3)`` in unit range."""
    rng = np.random.default_rng(seed)
    field = gaussian_filter(rng.standard_normal((shape[0], shape[1] + 4 * n)), 2.0)
    field = 0.5 + 0.15 * field / field.std()
    frames = np.stack([field[:, 4 * t : 4 * t + shape[1]] for t in range(n)])
    return np.repeat(np.clip(frames, 0, 1)[..., None], 3, axis=-1).astype(np.float32)


def test_ssim_is_one_for_identity_and_falls_with_blur():
    x = texture()
    assert E.ssim(x, x) == pytest.approx(np.ones(len(x)))
    mild = gaussian_filter(x, (0, 1, 1, 0))
    strong = gaussian_filter(x, (0, 4, 4, 0))
    assert 1.0 > E.ssim(x, mild).mean() > E.ssim(x, strong).mean() > 0.0


def test_ssim_of_unrelated_pictures_is_near_zero_and_of_an_inversion_negative():
    assert abs(E.ssim(texture(seed=1), texture(seed=2)).mean()) < 0.2
    x = texture()
    assert E.ssim(x, 1.0 - x).mean() < -0.5


def test_ssim_accepts_luma_and_8_bit():
    x = (texture() * 255).astype(np.uint8)
    assert E.ssim(x[..., 0], x[..., 0]) == pytest.approx(np.ones(len(x)))


def test_ssim_refuses_mismatched_or_small_frames():
    with pytest.raises(ValueError, match="matching"):
        E.ssim(texture(), texture()[:, :-2])
    with pytest.raises(ValueError, match="window"):
        E.ssim(texture(shape=(8, 200)), texture(shape=(8, 200)))


def test_delta_e_spans_black_to_white_and_sees_colour():
    black, white = np.zeros((1, 4, 4, 3)), np.ones((1, 4, 4, 3))
    assert E.delta_e(black, white) == pytest.approx([E.DELTA_E_SPAN], abs=1e-3)
    grey = np.full((1, 4, 4, 3), 0.5)
    tinted = grey.copy()
    tinted[..., 0] = 0.6
    assert 0 < E.delta_e(grey, tinted)[0] < 20
    with pytest.raises(ValueError, match="matching"):
        E.delta_e(grey, grey[..., 0])


def test_passthrough_is_safe_and_has_no_intensity():
    x = texture(32)
    found = E.effect(x, x, fps=FPS)
    assert found.safe and found.intensity == pytest.approx(0.0, abs=1e-6)
    assert found.as_dict()["safe"] and found.frames == 32


def test_a_flat_output_is_blank_and_static():
    x = texture(32)
    found = E.effect(x, np.full_like(x, 0.5), fps=FPS)
    assert found.blank and found.static and not found.safe
    assert found.intensity > 0.4


def test_a_frozen_output_is_static_but_not_blank():
    x = texture(32)
    found = E.effect(x, np.repeat(x[:1], len(x), axis=0), fps=FPS)
    assert found.static and not found.blank


def test_a_full_screen_strobe_is_flashing():
    x = texture(32)
    strobe = np.where((np.arange(32) % 4 < 2)[:, None, None, None], x, np.ones_like(x))
    found = E.effect(x, strobe.astype(np.float32), fps=FPS)
    assert found.flashing and found.flash_rate > 3 and not found.safe


def test_a_recolour_ranks_by_colour_and_a_restructure_by_structure():
    x = texture(8)
    tinted = x.copy()
    tinted[..., 1:] *= 0.3
    recolour = E.effect(x, tinted, fps=FPS)
    assert recolour.intensity == pytest.approx(recolour.colour / E.DELTA_E_SPAN)
    scrambled = E.effect(x, texture(8, seed=5), fps=FPS)
    assert scrambled.intensity == pytest.approx(scrambled.structure)


def test_a_single_frame_has_no_motion_to_judge():
    x = texture(1)
    found = E.effect(x, x, fps=FPS)
    assert np.isnan(found.source_ti) and not found.static

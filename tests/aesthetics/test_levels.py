"""Luma / chroma statistics and passthrough distance."""

import numpy as np
import pytest

from syncsummoner.aesthetics.levels import chroma, level_stats, luma, passthrough_distance
from tests.aesthetics import grating, texture, to_rgb


def test_luma_matches_bt601_weights():
    """Pure primaries carry the BT.601 luma coefficients."""
    frame = np.zeros((2, 2, 3), dtype=np.float32)
    frame[..., 1] = 1.0
    assert luma(frame) == pytest.approx(0.587, abs=1e-6)


def test_grey_frames_have_no_chroma_or_colour():
    """A neutral grey frame is achromatic and not colourful."""
    stats = level_stats(np.full((8, 8, 3), 0.5, dtype=np.float32))
    assert stats.chroma_mean == pytest.approx(0.0, abs=1e-5)
    assert stats.colourfulness == pytest.approx(0.0, abs=1e-5)
    assert (stats.luma_mean, stats.luma_std) == pytest.approx((0.5, 0.0), abs=1e-5)


def test_saturated_frame_is_clipped_and_illegal():
    """A black/white checkerboard is entirely clipped and outside legal range."""
    frame = to_rgb(np.indices((8, 8)).sum(axis=0) % 2)
    stats = level_stats(frame)
    assert stats.clip_frac == 1.0 and stats.illegal_frac == 1.0


def test_legal_ramp_is_neither_clipped_nor_illegal():
    """A frame inside 16..235 reports zero clipping and zero illegal samples."""
    ramp = np.linspace(0.1, 0.9, 64, dtype=np.float32).reshape(8, 8)
    stats = level_stats(to_rgb(ramp))
    assert stats.clip_frac == 0.0 and stats.illegal_frac == 0.0


def test_colourfulness_orders_grey_below_saturated():
    """Hasler-Susstrunk colourfulness increases with chroma spread."""
    grey = level_stats(np.full((8, 8, 3), 0.5, dtype=np.float32))
    rng = np.random.default_rng(0)
    colour = level_stats(rng.random((8, 8, 3), dtype=np.float32))
    assert colour.colourfulness > grey.colourfulness
    assert chroma(rng.random((8, 8, 3), dtype=np.float32)).mean() > 0.0


def test_passthrough_distance_is_zero_for_identity():
    """Identical frames are at distance zero, different frames are not."""
    rng = np.random.default_rng(2)
    frame = texture(rng)
    assert passthrough_distance(frame, frame) == 0.0
    assert passthrough_distance(frame, grating()) > 0.0


def test_passthrough_distance_resizes_mismatched_output():
    """Output frames of a different size are resized to the source geometry."""
    rng = np.random.default_rng(4)
    frame = texture(rng, size=64)
    assert passthrough_distance(frame, texture(rng, size=32)) > 0.0


@pytest.mark.parametrize("shape", [(8, 8), (8, 8, 4)])
def test_non_rgb_input_is_rejected(shape):
    """Only (H, W, 3) frames are accepted at the boundary."""
    with pytest.raises(ValueError):
        level_stats(np.zeros(shape, dtype=np.float32))

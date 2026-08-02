"""Pointwise fit: does a value curve explain the output, or was the picture re-addressed?"""

# pylint: disable=missing-function-docstring

import numpy as np
import pytest

from syncsummoner.aesthetics import structure


def ramp(shape=(64, 96)):
    y = np.linspace(0.0, 0.4, shape[0], dtype=np.float32)[:, None]
    x = np.linspace(0.0, 0.6, shape[1], dtype=np.float32)[None, :]
    return np.repeat((y + x)[:, :, None], 3, axis=2)


def test_identity_is_fully_pointwise():
    source = ramp()
    assert structure.pointwise_r2(source, source) > 0.99


@pytest.mark.parametrize("curve", [lambda v: v**2, lambda v: 1.0 - v, lambda v: np.floor(v * 4) / 4])
def test_value_remapping_stays_pointwise(curve):
    """Gamma, inversion and posterisation are analog-style: a curve explains them."""
    source = ramp()
    assert structure.pointwise_r2(source, curve(source).astype(np.float32)) > 0.98


def test_displacement_breaks_the_correspondence():
    """Rolling the picture is digital-style; no value curve can explain it."""
    source = ramp()
    assert structure.pointwise_r2(source, np.roll(source, 23, axis=1)) < 0.6


def test_tiling_breaks_the_correspondence():
    source = ramp()
    tiled = np.tile(source[:32, :48], (2, 2, 1)).astype(np.float32)
    assert structure.pointwise_r2(source, tiled) < 0.8


def test_noise_is_not_explained():
    rng = np.random.default_rng(0)
    source = ramp()
    assert structure.pointwise_r2(source, rng.random(source.shape).astype(np.float32)) < 0.2


def test_flat_output_is_trivially_explained():
    source = ramp()
    flat = np.full_like(source, 0.5)
    assert structure.pointwise_r2(source, flat) == 1.0


def test_fit_reports_curve_and_support():
    source = ramp()
    fit = structure.pointwise_fit(source, (source**2).astype(np.float32), bins=16)
    assert fit.curve.shape == (16,)
    assert 0.0 < fit.support <= 1.0
    assert np.all(np.diff(fit.curve[fit.curve > 0]) >= -1e-6), "a monotone curve must stay monotone"


def test_mismatched_sizes_raise():
    with pytest.raises(ValueError, match="match in size"):
        structure.pointwise_r2(ramp(), ramp((32, 32)))

"""Gabor bank construction and channel energy."""

import numpy as np
import pytest

from syncsummoner.aesthetics.channels import GAMMA, ChannelEnergy, gabor_bank, gabor_energy
from tests.aesthetics import grating, texture


def test_bank_shape_and_normalization():
    """Kernels are zero-mean, unit-norm and shaped (scales, orientations, k, k)."""
    bank = gabor_bank(n_orientations=4, n_scales=5, ksize=31)
    assert bank.shape == (5, 4, 31, 31) and bank.dtype == np.float32
    flat = bank.reshape(-1, 31 * 31)
    assert np.allclose(flat.mean(axis=1), 0.0, atol=1e-6)
    assert np.allclose(np.linalg.norm(flat, axis=1), 1.0, atol=1e-5)


def test_bank_geometry_matches_declared_bandwidths():
    """Envelope aspect ratio follows the one-octave, 30 degree channel model."""
    assert 0.75 < GAMMA < 0.85


@pytest.mark.parametrize("bad", [{"ksize": 4}, {"ksize": 5}, {"n_scales": 0}])
def test_bank_rejects_bad_geometry(bad):
    """Even, tiny or empty bank geometries raise."""
    with pytest.raises(ValueError):
        gabor_bank(**bad)


def test_energy_is_a_normalized_distribution():
    """Channel energy sums to 1 and reports a peak cell."""
    result = gabor_energy(grating())
    assert isinstance(result, ChannelEnergy)
    assert result.energy.shape == (5, 4)
    assert result.energy.sum() == pytest.approx(1.0, abs=1e-5)
    assert result.peak == tuple(np.unravel_index(int(np.argmax(result.energy)), result.energy.shape))


@pytest.mark.parametrize("angle,expected", [(0.0, 0), (np.pi / 2, 2)])
def test_energy_peaks_at_the_grating_orientation(angle, expected):
    """A grating puts its energy in the channel orthogonal to its bars."""
    assert gabor_energy(grating(angle=angle)).peak[1] == expected


def test_grating_is_more_concentrated_than_texture():
    """Single-channel content reads as mud; broadband texture does not."""
    rng = np.random.default_rng(0)
    assert gabor_energy(grating()).concentration > gabor_energy(texture(rng)).concentration


def test_flat_frame_spreads_energy_uniformly():
    """A frame with no structure yields a uniform, minimally concentrated distribution."""
    result = gabor_energy(np.full((32, 32, 3), 0.5, dtype=np.float32))
    assert np.allclose(result.energy, 1.0 / result.energy.size)
    assert result.concentration == pytest.approx(0.0, abs=1e-6)


def test_explicit_bank_is_used():
    """A caller-supplied bank overrides the cached default."""
    bank = gabor_bank(n_orientations=2, n_scales=3, ksize=15)
    assert gabor_energy(grating(), bank=bank).energy.shape == (3, 2)


def test_single_cell_bank_is_fully_concentrated():
    """A one-cell bank is degenerate and reports concentration 1.0."""
    bank = gabor_bank(n_orientations=1, n_scales=1, ksize=15)
    assert gabor_energy(grating(), bank=bank).concentration == 1.0

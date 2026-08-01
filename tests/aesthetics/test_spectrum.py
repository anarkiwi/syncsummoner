"""Radial power spectrum slope and fractal dimension."""

import numpy as np
import pytest

from syncsummoner.aesthetics.spectrum import radial_power, spectral_stats
from tests.aesthetics import to_rgb


def power_law_frame(rng, size=128, beta=2.0):
    """Synthetic frame whose power spectrum follows f**-beta."""
    freq = np.hypot(*np.meshgrid(np.fft.fftfreq(size), np.fft.fftfreq(size)))
    freq[0, 0] = 1.0
    spectrum = np.fft.fft2(rng.standard_normal((size, size))) * freq ** (-beta / 2.0)
    image = np.real(np.fft.ifft2(spectrum))
    return to_rgb((image - image.min()) / np.ptp(image))


@pytest.mark.parametrize("beta", [1.0, 2.0, 3.0])
def test_slope_recovers_the_synthesized_power_law(beta):
    """The log-log fit returns the exponent used to synthesize the frame."""
    stats = spectral_stats(power_law_frame(np.random.default_rng(3), beta=beta))
    assert stats.slope == pytest.approx(-beta, abs=0.2)
    assert stats.r2 > 0.9


def test_white_noise_is_flat_and_maximally_rough():
    """White noise has near-zero slope, so fractal dimension clamps at 2.0."""
    rng = np.random.default_rng(1)
    stats = spectral_stats(rng.random((64, 64, 3), dtype=np.float32))
    assert abs(stats.slope) < 0.2
    assert stats.fractal_dimension == 2.0


def test_fractal_dimension_follows_the_slope_relation():
    """Fractal dimension is (7 + 2*slope)/2 inside the unclamped band."""
    stats = spectral_stats(power_law_frame(np.random.default_rng(5), beta=2.4))
    assert stats.fractal_dimension == pytest.approx((7.0 + 2.0 * stats.slope) / 2.0, abs=1e-6)
    assert 1.0 <= stats.fractal_dimension <= 2.0


def test_constant_frame_is_degenerate():
    """A frame with no spectral content returns a null fit."""
    stats = spectral_stats(np.full((32, 32, 3), 0.25, dtype=np.float32))
    assert (stats.slope, stats.r2, stats.fractal_dimension) == (0.0, 0.0, 1.0)


def test_tiny_frame_has_no_radial_bins():
    """Frames smaller than the minimum radial binning return an empty spectrum."""
    freq, power = radial_power(np.zeros((6, 6, 3), dtype=np.float32))
    assert freq.size == 0 and power.size == 0


def test_non_square_frames_are_supported():
    """Radial averaging handles anisotropic frame shapes."""
    rng = np.random.default_rng(7)
    freq, power = radial_power(rng.random((32, 96, 3), dtype=np.float32))
    assert freq.size == power.size > 0 and freq.max() <= 0.55

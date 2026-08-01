"""Gabor channel energy over a scale x orientation bank.

Bank geometry follows the visual-channel model in the design appendix: one-octave
spatial-frequency bandwidth and ~30 degree orientation bandwidth, which fixes the
Gabor envelope aspect ratio rather than leaving it to be tuned.
"""

# cv2 is a C extension; pylint cannot introspect its members.
# pylint: disable=no-member

from dataclasses import dataclass
from functools import lru_cache

import cv2
import numpy as np

from syncsummoner.aesthetics.levels import luma

BANDWIDTH_OCTAVES = 1.0
ORIENTATION_BANDWIDTH_DEG = 30.0
NYQUIST_WAVELENGTH = 2.0
SUPPORT_SIGMAS = 3.0
DEFAULT_KSIZE = 31

_RATIO = 2.0**BANDWIDTH_OCTAVES
# sigma along the modulation axis, per unit wavelength, for the given octave bandwidth.
SIGMA_PER_WAVELENGTH = float(np.sqrt(np.log(2.0) / 2.0) * (_RATIO + 1.0) / (_RATIO - 1.0) / np.pi)
# sigma across the modulation axis, per unit wavelength, for the given orientation bandwidth.
SIGMA_PERP_PER_WAVELENGTH = float(
    np.sqrt(2.0 * np.log(2.0)) / (2.0 * np.pi * np.tan(np.deg2rad(ORIENTATION_BANDWIDTH_DEG) / 2.0))
)
GAMMA = SIGMA_PER_WAVELENGTH / SIGMA_PERP_PER_WAVELENGTH


@dataclass(frozen=True)
class ChannelEnergy:
    """Normalized energy across the Gabor bank plus its concentration."""

    energy: np.ndarray
    concentration: float
    peak: tuple[int, int]


def bank_wavelengths(n_scales: int, ksize: int) -> np.ndarray:
    """Geometric wavelengths from the Nyquist limit to the largest the kernel supports."""
    lam_max = ksize / (2.0 * SUPPORT_SIGMAS * SIGMA_PERP_PER_WAVELENGTH)
    if lam_max < NYQUIST_WAVELENGTH:
        raise ValueError(f"ksize={ksize} cannot support a Nyquist-limited Gabor kernel")
    return np.geomspace(NYQUIST_WAVELENGTH, lam_max, n_scales)


def gabor_bank(*, n_orientations: int = 4, n_scales: int = 5, ksize: int = DEFAULT_KSIZE) -> np.ndarray:
    """Zero-mean, unit-norm Gabor kernels, ``(n_scales, n_orientations, ksize, ksize)``."""
    if min(n_orientations, n_scales) < 1 or ksize < 3 or ksize % 2 == 0:
        raise ValueError("need n_orientations >= 1, n_scales >= 1 and odd ksize >= 3")
    lambdas = bank_wavelengths(n_scales, ksize)
    thetas = np.arange(n_orientations) * np.pi / n_orientations
    bank = np.empty((n_scales, n_orientations, ksize, ksize), dtype=np.float32)
    for s, lam in enumerate(lambdas):
        sigma = SIGMA_PER_WAVELENGTH * float(lam)
        for o, theta in enumerate(thetas):
            kern = cv2.getGaborKernel(
                (ksize, ksize), sigma, float(theta), float(lam), GAMMA, 0.0, ktype=cv2.CV_32F
            )
            kern -= kern.mean()
            bank[s, o] = kern / max(float(np.linalg.norm(kern)), np.finfo(np.float32).tiny)
    return bank


@lru_cache(maxsize=8)
def _cached_bank(n_orientations: int, n_scales: int, ksize: int) -> np.ndarray:
    bank = gabor_bank(n_orientations=n_orientations, n_scales=n_scales, ksize=ksize)
    bank.flags.writeable = False
    return bank


def gabor_energy(
    frame: np.ndarray, *, bank: np.ndarray | None = None, n_orientations: int = 4, n_scales: int = 5
) -> ChannelEnergy:
    """Energy of frame luma in each Gabor channel, normalized to sum to 1."""
    if bank is None:
        bank = _cached_bank(n_orientations, n_scales, DEFAULT_KSIZE)
    gray = luma(frame)
    gray = gray - gray.mean()
    energy = np.array(
        [[float(np.mean(np.square(cv2.filter2D(gray, -1, k)))) for k in row] for row in bank],
        dtype=np.float32,
    )
    total = float(energy.sum())
    energy = energy / total if total > 0.0 else np.full(energy.shape, 1.0 / energy.size, np.float32)
    cells = energy.size
    herfindahl = float(np.sum(np.square(energy, dtype=np.float64)))
    concentration = (herfindahl * cells - 1.0) / (cells - 1.0) if cells > 1 else 1.0
    peak = np.unravel_index(int(np.argmax(energy)), energy.shape)
    return ChannelEnergy(energy.astype(np.float32), float(concentration), (int(peak[0]), int(peak[1])))

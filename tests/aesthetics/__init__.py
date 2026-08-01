"""Synthetic fixtures for the aesthetics tests; no files, hardware or network."""

# cv2 is a C extension; pylint cannot introspect its members.
# pylint: disable=no-member

import cv2
import numpy as np


def to_rgb(gray: np.ndarray) -> np.ndarray:
    """Replicate a ``(H, W)`` grey image into an ``(H, W, 3) float32`` RGB frame."""
    return np.repeat(np.asarray(gray, dtype=np.float32)[:, :, None], 3, axis=2)


def grating(size: int = 64, period: float = 8.0, angle: float = 0.0) -> np.ndarray:
    """Sine grating frame at the given wavelength (px) and orientation (radians)."""
    yy, xx = np.mgrid[0:size, 0:size]
    phase = (xx * np.cos(angle) + yy * np.sin(angle)) * 2.0 * np.pi / period
    return to_rgb(0.5 + 0.4 * np.sin(phase))


def texture(rng: np.random.Generator, size: int = 64, sigma: float = 2.0) -> np.ndarray:
    """Band-limited random texture frame, values spanning [0, 1]."""
    blurred = cv2.GaussianBlur(rng.random((size, size), dtype=np.float32), (0, 0), sigma)
    return to_rgb((blurred - blurred.min()) / np.ptp(blurred))


def drifting(frame: np.ndarray, n_frames: int, shift: int = 8) -> np.ndarray:
    """Frame stack translating horizontally by ``shift`` px per frame."""
    return np.stack([np.roll(frame, i * shift, axis=1) for i in range(n_frames)])

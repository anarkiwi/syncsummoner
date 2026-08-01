"""Luma / chroma level statistics and distance from passthrough."""

# cv2 is a C extension; pylint cannot introspect its members.
# pylint: disable=no-member

from dataclasses import dataclass

import cv2
import numpy as np

# BT.601 RGB -> YUV. Rows: luma, U (blue difference), V (red difference).
YUV_MATRIX = np.array(
    [[0.299, 0.587, 0.114], [-0.14713, -0.28886, 0.436], [0.615, -0.51499, -0.10001]],
    dtype=np.float32,
)
LEGAL_LO = 16.0 / 255.0
LEGAL_HI = 235.0 / 255.0


@dataclass(frozen=True)
class LevelStats:
    """Per-frame luma, chroma, legality and colourfulness summary."""

    luma_mean: float
    luma_std: float
    chroma_mean: float
    chroma_std: float
    clip_frac: float
    illegal_frac: float
    colourfulness: float


def as_frame(frame: np.ndarray) -> np.ndarray:
    """Validate and coerce a boundary frame to ``(H, W, 3) float32`` RGB."""
    arr = np.asarray(frame, dtype=np.float32)
    if arr.ndim != 3 or arr.shape[2] != 3:
        raise ValueError(f"expected an (H, W, 3) RGB frame, got shape {arr.shape}")
    return arr


def luma(frame: np.ndarray) -> np.ndarray:
    """BT.601 luma of an RGB frame, ``(H, W) float32``."""
    return as_frame(frame) @ YUV_MATRIX[0]


def chroma(frame: np.ndarray) -> np.ndarray:
    """BT.601 chroma magnitude ``sqrt(u^2 + v^2)``, ``(H, W) float32``."""
    uv = as_frame(frame) @ YUV_MATRIX[1:].T
    return np.hypot(uv[..., 0], uv[..., 1])


def level_stats(frame: np.ndarray) -> LevelStats:
    """Luma/chroma moments, clipping, broadcast-illegal fraction and colourfulness."""
    arr = as_frame(frame)
    y = arr @ YUV_MATRIX[0]
    cmag = chroma(arr)
    rgb = arr.reshape(-1, 3)
    rg = rgb[:, 0] - rgb[:, 1]
    yb = 0.5 * (rgb[:, 0] + rgb[:, 1]) - rgb[:, 2]
    return LevelStats(
        luma_mean=float(y.mean()),
        luma_std=float(y.std()),
        chroma_mean=float(cmag.mean()),
        chroma_std=float(cmag.std()),
        clip_frac=float(np.mean((arr <= 0.0) | (arr >= 1.0))),
        illegal_frac=float(np.mean((arr < LEGAL_LO) | (arr > LEGAL_HI))),
        colourfulness=float(np.hypot(rg.std(), yb.std()) + 0.3 * np.hypot(rg.mean(), yb.mean())),
    )


def passthrough_distance(source: np.ndarray, output: np.ndarray) -> float:
    """RMS difference between an input frame and the device output; 0.0 == identical."""
    src = as_frame(source)
    out = as_frame(output)
    if out.shape != src.shape:
        out = cv2.resize(out, (src.shape[1], src.shape[0]), interpolation=cv2.INTER_AREA)
    return float(np.sqrt(np.mean(np.square(src - out))))

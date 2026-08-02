"""Level statistics, and the frame helpers every other metric module shares.

Both luma bases live here so the split is visible in one place: BT.601 for the
level statistics, Rec.709 for the perceptual metrics. They are different
matrices, deliberately.
"""

# pylint: disable=no-member  ; cv2 is a compiled extension pylint cannot introspect

from dataclasses import dataclass

import cv2
import numpy as np

# BT.601 RGB -> YUV. Rows: luma, U (blue difference), V (red difference).
YUV_MATRIX = np.array(
    [[0.299, 0.587, 0.114], [-0.14713, -0.28886, 0.436], [0.615, -0.51499, -0.10001]],
    dtype=np.float32,
)
#: Rec.709 luma weights, for RGB in [0, 1].
LUMA_709 = np.array([0.2126, 0.7152, 0.0722], dtype=np.float32)
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


def luma_709(frames: np.ndarray) -> np.ndarray:
    """Rec.709 luma of an RGB frame or frame stack, weighted over the last axis."""
    return frames @ LUMA_709


def downscale(frame: np.ndarray, *, max_width: int | None = None, max_side: int | None = None):
    """Area-average a frame down to a width or longest-side cap, never enlarging.

    Area statistics are scale-free and their cost is not, so the metrics run on
    the small frame; INTER_AREA is the resampling that preserves an area fraction.
    """
    height, width = frame.shape[:2]
    scale = min(
        max_width / width if max_width else 1.0,
        max_side / max(height, width) if max_side else 1.0,
    )
    if scale >= 1.0:
        return frame
    size = (max(1, round(width * scale)), max(1, round(height * scale)))
    return cv2.resize(frame, size, interpolation=cv2.INTER_AREA)


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

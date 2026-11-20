"""Which channel order a capture shows a played picture in, from planar YUV of both.

Each of :data:`ORDERS` is applied to the picture played and correlated with the capture plane by plane: luma
tells red and blue apart, the chroma pair then whether it was exchanged. :attr:`ChromaOrder.ok` judges the
chroma pair against the picture as played, which an exchange flips whatever luma's margin.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from syncsummoner.device.recorder import SIGMA

__all__ = [
    "AS_PLAYED",
    "ORDERS",
    "SWAP",
    "TO_YUV",
    "ChromaOrder",
    "order_of",
    "to_rgb",
    "to_yuv",
    "variants",
]

#: The ffmpeg filter that puts an exchanged chroma pair back.
SWAP = "swapuv"

ORDERS = ("as played", "chroma exchanged", "red and blue exchanged", "both exchanged")
AS_PLAYED = ORDERS[0]
#: BT.601 RGB to YUV, the matrix ffmpeg converts untagged 8-bit video with.
TO_YUV = np.array([[0.299, 0.587, 0.114], [-0.168736, -0.331264, 0.5], [0.5, -0.418688, -0.081312]])
CHROMA_ZERO = np.array([0.0, 128.0, 128.0])


@dataclass(frozen=True)
class ChromaOrder:
    """The order captures show a played picture in, each order's mean Y, U and V correlation, frames read, and
    ``margin``: the chroma pair's correlation as played less exchanged, in standard errors over the frames."""

    order: str
    scores: dict[str, list[float]]
    frames: int
    margin: float

    @property
    def ok(self) -> bool:
        """Whether the chroma pair follows the picture as played rather than exchanged."""
        return bool(self.margin > 0)

    @property
    def judged(self) -> bool:
        """Whether the margin clears :data:`~syncsummoner.device.recorder.SIGMA` either way."""
        return bool(abs(self.margin) >= SIGMA)

    def as_dict(self) -> dict[str, Any]:
        """The verdict as stored in a take's attempt record."""
        return {
            "order": self.order,
            "ok": self.ok,
            "judged": self.judged,
            "margin": round(float(self.margin), 2),
            "frames": self.frames,
            "scores": self.scores,
        }


def variants(yuv: np.ndarray) -> np.ndarray:
    """Each of :data:`ORDERS` applied to 8-bit planar ``(..., 3, h, w)`` YUV, stacked on a new first axis."""
    planes = np.moveaxis(np.asarray(yuv, dtype=np.float64), -3, -1) - CHROMA_ZERO
    rgb = planes @ np.linalg.inv(TO_YUV).T
    swapped = rgb[..., ::-1] @ TO_YUV.T
    exchange = [0, 2, 1]
    found = np.stack([planes, planes[..., exchange], swapped, swapped[..., exchange]])
    return np.moveaxis(found, -1, -3)


def _correlation(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Pearson correlation over the last two axes; 0 where either side is flat."""
    a = a - a.mean(axis=(-2, -1), keepdims=True)
    b = b - b.mean(axis=(-2, -1), keepdims=True)
    den = np.sqrt((a * a).sum(axis=(-2, -1)) * (b * b).sum(axis=(-2, -1)))
    return np.divide((a * b).sum(axis=(-2, -1)), den, out=np.zeros_like(den), where=den > 0)


def standard_errors(diff: np.ndarray) -> float:
    """The mean of per-frame differences in standard errors; one frame, or no spread, goes by its sign."""
    mean = float(diff.mean())
    spread = float(diff.std(ddof=1)) / np.sqrt(diff.size) if diff.size > 1 else 0.0
    return mean / spread if spread > 0 else float(np.sign(mean)) * np.inf if mean else 0.0


def order_of(captured: np.ndarray, shown: np.ndarray) -> ChromaOrder:
    """The order planar YUV ``captured`` shows ``shown`` in, frame for frame, each ``(n, 3, h, w)``."""
    if np.asarray(captured).shape[0] == 0:
        raise ValueError("no captures to judge the channel order by")
    got = np.asarray(captured, dtype=np.float64)
    per_frame = _correlation(got[None], variants(shown))
    scores = per_frame.mean(axis=1)
    swapped = int(scores[2, 0] > scores[0, 0])
    exchanged = int(scores[2 * swapped + 1, 1:].mean() > scores[2 * swapped, 1:].mean())
    named = {o: [round(float(v), 4) for v in s] for o, s in zip(ORDERS, scores)}
    margin = standard_errors(per_frame[0, :, 1:].mean(axis=1) - per_frame[1, :, 1:].mean(axis=1))
    return ChromaOrder(ORDERS[2 * swapped + exchanged], named, len(got), margin)


def to_yuv(rgb: np.ndarray) -> np.ndarray:
    """RGB in ``[0, 1]``, ``(..., h, w, 3)``, as 8-bit-scaled planar ``(..., 3, h, w)`` YUV."""
    return np.moveaxis(np.asarray(rgb, dtype=np.float64) * 255.0 @ TO_YUV.T + CHROMA_ZERO, -1, -3)


#: Luma weights of red and blue by the matrix a stream is tagged with; any other is read as ffmpeg reads it.
MATRICES = {"bt709": (0.2126, 0.0722)}
BT601 = (0.299, 0.114)


def to_rgb(yuv: np.ndarray, *, full: bool = True, space: str = "unknown") -> np.ndarray:
    """8-bit planar ``(..., 3, h, w)`` YUV as RGB ``(..., h, w, 3)`` in ``[0, 1]``, as a decoder shows it.

    ``full`` is the planes' range, else studio swing; ``space`` the matrix they are tagged with, ffmpeg's
    BT.601 when it names none other.
    """
    red, blue = MATRICES.get(space, BT601)
    planes = np.moveaxis(np.asarray(yuv, dtype=np.float32), -3, -1) - np.float32(128.0)
    scale = np.array([255.0, 255.0, 255.0] if full else [219.0, 224.0, 224.0], dtype=np.float32)
    planes[..., 0] += np.float32(128.0 if full else 112.0)
    y, cb, cr = np.moveaxis(planes / scale, -1, 0)
    r = y + np.float32(2.0 * (1.0 - red)) * cr
    b = y + np.float32(2.0 * (1.0 - blue)) * cb
    g = (y - np.float32(red) * r - np.float32(blue) * b) / np.float32(1.0 - red - blue)
    return np.clip(np.stack([r, g, b], axis=-1), 0.0, 1.0)

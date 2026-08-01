"""Frame-difference energy and dense optical flow statistics."""

# cv2 is a C extension; pylint cannot introspect its members.
# pylint: disable=no-member

from dataclasses import dataclass

import cv2
import numpy as np

from syncsummoner.aesthetics.levels import luma

# OpenCV reference parameters for Farneback dense flow.
FARNEBACK = {
    "pyr_scale": 0.5,
    "levels": 3,
    "winsize": 15,
    "iterations": 3,
    "poly_n": 5,
    "poly_sigma": 1.2,
    "flags": 0,
}


@dataclass(frozen=True)
class MotionStats:
    """Temporal change between two frames; coherence 1.0 == rigid translation."""

    framediff_energy: float
    flow_magnitude: float
    flow_coherence: float


def optical_flow(prev: np.ndarray, curr: np.ndarray) -> np.ndarray:
    """Dense Farneback flow between two frames, ``(H, W, 2) float32`` px/frame."""
    grays = [np.clip(luma(f) * 255.0, 0.0, 255.0).astype(np.uint8) for f in (prev, curr)]
    return cv2.calcOpticalFlowFarneback(grays[0], grays[1], None, **FARNEBACK)


def motion_stats(prev: np.ndarray, curr: np.ndarray) -> MotionStats:
    """Frame-difference energy plus magnitude and coherence of dense optical flow."""
    prev_y = luma(prev)
    curr_y = luma(curr)
    if prev_y.shape != curr_y.shape:
        raise ValueError(f"frame shape mismatch: {prev_y.shape} vs {curr_y.shape}")
    flow = optical_flow(prev, curr)
    magnitude = np.hypot(flow[..., 0], flow[..., 1])
    mean_magnitude = float(magnitude.mean())
    resultant = float(np.hypot(*flow.reshape(-1, 2).mean(axis=0)))
    return MotionStats(
        framediff_energy=float(np.mean(np.square(curr_y - prev_y))),
        flow_magnitude=mean_magnitude,
        flow_coherence=float(resultant / mean_magnitude) if mean_magnitude > 0.0 else 0.0,
    )

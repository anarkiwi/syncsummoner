"""Decode shim: the only module in this package that touches the filesystem.

Kept out of every other module so the metrics stay pure; the rest of the project
does not import it. Converts OpenCV's BGR uint8 to the RGB float32 boundary type.
"""

# cv2 is a C extension; pylint cannot introspect its members.
# pylint: disable=no-member

import cv2
import numpy as np

DEFAULT_FPS = 30.0


def read_clip(
    path: str, *, max_frames: int | None = None, stride: int = 1, max_width: int | None = None
) -> tuple[np.ndarray, float]:
    """Decode a video to an ``(T, H, W, 3) float32`` RGB stack and its effective fps."""
    if stride < 1:
        raise ValueError("stride must be >= 1")
    capture = cv2.VideoCapture(str(path))
    try:
        if not capture.isOpened():
            raise FileNotFoundError(f"cannot open video: {path}")
        fps = float(capture.get(cv2.CAP_PROP_FPS))
        frames = list(_iter_frames(capture, max_frames, stride, max_width))
    finally:
        capture.release()
    if not frames:
        raise ValueError(f"no frames decoded from {path}")
    if not np.isfinite(fps) or fps <= 0.0:
        fps = DEFAULT_FPS
    return np.stack(frames), fps / stride


def _iter_frames(capture, max_frames: int | None, stride: int, max_width: int | None):
    kept = 0
    index = 0
    while max_frames is None or kept < max_frames:
        ok, bgr = capture.read()
        if not ok:
            return
        if index % stride == 0:
            frame = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
            if max_width is not None and frame.shape[1] > max_width:
                height = max(1, round(frame.shape[0] * max_width / frame.shape[1]))
                frame = cv2.resize(frame, (max_width, height), interpolation=cv2.INTER_AREA)
            kept += 1
            yield np.asarray(frame, dtype=np.float32) / 255.0
        index += 1

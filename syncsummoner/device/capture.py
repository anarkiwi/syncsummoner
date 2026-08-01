"""Long-lived V4L2 capture session.

Lock costs ~3.2 s on a timing change and ~0.5 s on reopen, so the stream is
opened once for a whole sweep and never per sample.
"""

from __future__ import annotations

import time
from typing import Any, Callable

import cv2
import numpy as np

# pylint: disable=no-member  ; cv2 is a compiled extension pylint cannot introspect

#: Rec.709 luma weights, applied to RGB float frames.
LUMA = np.array([0.2126, 0.7152, 0.0722], dtype=np.float32)
#: HSV saturation 60/255, the threshold the no-signal splash was measured against.
CHROMA_LEVEL = 60.0 / 255.0


class CaptureError(RuntimeError):
    """The capture device could not be opened or configured."""


class Capture:
    """RGB float32 frames from a V4L2 capture card, held open for the session.

    ``read`` returns ``(H, W, 3)`` in ``[0, 1]``; OpenCV's BGR is converted at
    this boundary and never leaves it.
    """

    def __init__(
        self,
        device: str | int = "/dev/video0",
        *,
        width: int = 720,
        height: int = 576,
        fps: int = 50,
        fourcc: str = "YUYV",
        sleep: Callable[[float], None] = time.sleep,
        clock: Callable[[], float] = time.monotonic,
    ):
        self.device = device
        self.width = int(width)
        self.height = int(height)
        self.fps = int(fps)
        self.fourcc = fourcc
        self._sleep = sleep
        self._clock = clock
        self._cap: Any = None

    def open(self) -> "Capture":
        """Open and configure the stream, raising when the card does not come up."""
        if self._cap is not None:
            return self
        cap = cv2.VideoCapture(self.device, cv2.CAP_V4L2)
        if not cap.isOpened():
            raise CaptureError(f"cannot open {self.device!r} with the V4L2 backend")
        for prop, value in (
            (cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*self.fourcc)),
            (cv2.CAP_PROP_FRAME_WIDTH, self.width),
            (cv2.CAP_PROP_FRAME_HEIGHT, self.height),
            (cv2.CAP_PROP_FPS, self.fps),
            (cv2.CAP_PROP_CONVERT_RGB, 1),
        ):
            cap.set(prop, value)
        self._cap = cap
        return self

    def read(self) -> np.ndarray | None:
        """One frame as RGB float32 in ``[0, 1]``, or None when the grab failed."""
        if self._cap is None:
            raise CaptureError("capture is not open")
        ok, frame = self._cap.read()
        if not ok or frame is None:
            return None
        return np.ascontiguousarray(frame[:, :, ::-1], dtype=np.float32) / np.float32(255.0)

    def chroma_fraction(self, frame: np.ndarray, *, level: float = CHROMA_LEVEL) -> float:
        """Fraction of pixels carrying meaningful chroma (HSV saturation above ``level``)."""
        peak = frame.max(axis=2)
        span = peak - frame.min(axis=2)
        return float(np.count_nonzero(span > level * np.maximum(peak, 1e-6)) / span.size)

    def is_no_signal(
        self,
        frame: np.ndarray,
        *,
        max_chroma_frac: float = 0.01,
        min_bright_frac: float = 0.02,
        max_mid_frac: float = 0.10,
        dark: float = 0.15,
        bright: float = 0.85,
    ) -> bool:
        """True for the card's synthesized "No Signal" splash rather than real content.

        The splash is achromatic and bilevel while far from uniformly black;
        variance-based liveness tests score it as content, which silently
        corrupts a sweep.
        """
        if frame is None:
            return True
        if self.chroma_fraction(frame) > max_chroma_frac:
            return False
        luma = frame @ LUMA
        bright_frac = float(np.count_nonzero(luma >= bright) / luma.size)
        mid_frac = float(np.count_nonzero((luma > dark) & (luma < bright)) / luma.size)
        return bright_frac >= min_bright_frac and mid_frac <= max_mid_frac

    def wait_for_lock(self, timeout_s: float = 10.0, *, poll_s: float = 0.05) -> bool:
        """Block until a frame arrives that is neither a failed grab nor the splash."""
        self.open()
        deadline = self._clock() + timeout_s
        while True:
            frame = self.read()
            if frame is not None and not self.is_no_signal(frame):
                return True
            if self._clock() >= deadline:
                return False
            self._sleep(poll_s)

    def wait_for_content(self, timeout_s: float = 15.0, *, run: int = 10, min_motion: float = 1e-4) -> bool:
        """Block until frames are both past the splash and actually moving.

        A program change blacks the output out and the card needs seconds to
        re-lock, so a fixed dwell samples the splash. The drop this rig is prone
        to freezes the output instead, which only the motion test catches.
        """
        self.open()
        deadline = self._clock() + timeout_s
        recent: list[np.ndarray] = []
        while True:
            frame = self.read()
            if frame is None or self.is_no_signal(frame):
                recent.clear()
            else:
                recent.append(frame)
                if len(recent) >= run:
                    moving = np.abs(np.diff(np.stack(recent[-run:]), axis=0)).mean()
                    if moving > min_motion:
                        return True
                    recent = recent[-(run - 1) :]
            if self._clock() >= deadline:
                return False

    def close(self) -> None:
        """Release the capture device."""
        if self._cap is not None:
            self._cap.release()
            self._cap = None

    def __enter__(self) -> "Capture":
        return self.open()

    def __exit__(self, *exc: Any) -> bool:
        self.close()
        return False

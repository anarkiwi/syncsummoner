"""Long-lived V4L2 capture session.

Locking is slow next to any per-sample dwell (docs/hardware.md), so the stream
is opened once for a whole sweep and never per sample. The card advertises YUYV
and delivers YVYU, so every frame is decoded with the chroma pair exchanged.
"""

from __future__ import annotations

import time
from itertools import islice
from typing import Any, Callable, Iterator

import cv2
import numpy as np

# pylint: disable=no-member  ; cv2 is a compiled extension pylint cannot introspect

#: Samples the uniform-field test reads: a strided view of a uniform field is
#: uniform, so a decimated frame answers it at a bounded cost per frame.
STRUCTURE_SAMPLES = 50_000
#: One step of the 8-bit samples every frame is decoded from. A uniform field is
#: one DC level per block, which JPEG and resampling reproduce exactly, so its
#: spatial range is zero up to the rounding of a single step.
QUANTUM = 1.0 / 255.0
#: The one format V4L2 enumerates, and what it must be asked for; the bytes are YVYU.
NATIVE_FOURCC = "YUYV"


class CaptureError(RuntimeError):
    """The capture device could not be opened or configured."""


def yvyu_to_bgr(frame: np.ndarray) -> np.ndarray:
    """Packed native ``(H, W, 2)`` to uint8 BGR, reading the chroma pair as YVYU.

    The chroma pair arrives exchanged whatever format the card is asked for: its
    own ``Colorbars`` decodes to SMPTE order only once the pair is put back.
    """
    return cv2.cvtColor(frame, cv2.COLOR_YUV2BGR_YVYU)


def structured(frame: np.ndarray) -> bool:
    """Whether a frame holds spatial structure rather than one uniform field.

    A dead output is a constant, black or not, which survives capture exactly, so
    any channel ranging over more than one whole :data:`QUANTUM` is picture however dark it
    is. Brightness says nothing: dark footage carries full-range structure.
    """
    flat = frame.reshape(-1, frame.shape[2] if frame.ndim == 3 else 1)
    return bool(np.rint(np.ptp(flat, axis=0).max() / QUANTUM) > 1)


def bgr_to_rgb(frame: np.ndarray) -> np.ndarray:
    """uint8 BGR to the RGB float32 in ``[0, 1]`` the analyzer works in."""
    return np.ascontiguousarray(frame[:, :, ::-1], dtype=np.float32) / np.float32(255.0)


class Capture:
    """RGB float32 frames from a V4L2 capture card, held open for the session.

    ``read`` returns ``(H, W, 3)`` in ``[0, 1]``. Recording a pass is the
    recorder's job; only one process may hold the card at a time.
    """

    def __init__(
        self,
        device: str | int = "/dev/video0",
        *,
        width: int = 720,
        height: int = 576,
        fps: int = 50,
        fourcc: str = NATIVE_FOURCC,
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
            (cv2.CAP_PROP_CONVERT_RGB, 0),
        ):
            cap.set(prop, value)
        self._cap = cap
        return self

    def _grab(self) -> np.ndarray | None:
        """The one call into OpenCV; every other read is a conversion of this."""
        if self._cap is None:
            raise CaptureError("capture is not open")
        ok, frame = self._cap.read()
        return frame if ok and frame is not None else None

    def read_raw(self) -> np.ndarray | None:
        """One frame as uint8 BGR, or None on a failed grab.

        The 4:2:2 upsample runs here in both modes, so both return the same array
        for the same native buffer.
        """
        frame = self._grab()
        return None if frame is None else yvyu_to_bgr(frame)

    def read(self) -> np.ndarray | None:
        """One frame as RGB float32 in ``[0, 1]``, or None when the grab failed."""
        frame = self.read_raw()
        return None if frame is None else bgr_to_rgb(frame)

    @staticmethod
    def _decimate(frame: np.ndarray, samples: int = STRUCTURE_SAMPLES) -> np.ndarray:
        """Strided view holding about ``samples`` pixels."""
        pixels = frame.shape[0] * frame.shape[1]
        step = max(1, int(np.sqrt(pixels / max(samples, 1))))
        return frame[::step, ::step]

    def _stream(self, timeout_s: float, *, poll_s: float = 0.0) -> Iterator[np.ndarray | None]:
        """Yield every read until ``timeout_s`` elapses, None where the grab failed.

        Bounded on purpose: a read-until-good loop hangs forever on a card that
        never delivers.
        """
        self.open()
        deadline = self._clock() + timeout_s
        while True:
            frame = self.read()
            yield frame
            if self._clock() >= deadline:
                return
            if poll_s:
                self._sleep(poll_s)

    def wait_for_lock(self, timeout_s: float = 10.0, *, poll_s: float = 0.05) -> bool:
        """Block until a frame arrives carrying picture, not a grab failure.

        A dead output is a uniform field, which is neither, so spatial structure
        is the test: a still stimulus has it even when nothing is moving. A
        strided view keeps a uniform field uniform, so the decimated frame answers.
        """
        return any(
            frame is not None and structured(self._decimate(frame))
            for frame in self._stream(timeout_s, poll_s=poll_s)
        )

    def wait_for_content(self, timeout_s: float = 15.0, *, run: int = 10, min_motion: float = 1e-4) -> bool:
        """Block until frames are both arriving and actually moving.

        A program change blacks the output out and the card needs seconds to
        re-lock, so a fixed dwell samples the blackout. The drop this rig is prone
        to freezes the output instead, which only the motion test catches.
        """
        recent: list[np.ndarray] = []
        for frame in self._stream(timeout_s):
            if frame is None:
                recent.clear()
                continue
            recent.append(frame)
            if len(recent) >= run:
                if np.abs(np.diff(np.stack(recent[-run:]), axis=0)).mean() > min_motion:
                    return True
                recent = recent[-(run - 1) :]
        return False

    def frames(self, count: int, *, timeout_s: float = 10.0, settle: int = 0) -> list[np.ndarray]:
        """Collect ``count`` usable frames, or fewer if the budget runs out."""
        self.open()
        for _ in range(max(0, settle)):
            self.read()
        usable = (f for f in self._stream(timeout_s) if f is not None)
        return list(islice(usable, max(0, count)))

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

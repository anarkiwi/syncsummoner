"""Long-lived V4L2 capture session.

Lock costs ~3.2 s on a timing change and ~0.5 s on reopen, so the stream is
opened once for a whole sweep and never per sample. The card advertises YUYV
and delivers YVYU, so every frame is decoded with the chroma pair exchanged.
"""

from __future__ import annotations

import time
from itertools import islice
from typing import Any, Callable, Iterator

import cv2
import numpy as np

# pylint: disable=no-member  ; cv2 is a compiled extension pylint cannot introspect

#: HSV saturation 60/255, the threshold the no-signal splash was measured against.
CHROMA_LEVEL = 60.0 / 255.0
#: Samples the splash test estimates its area fractions from. They are fractions,
#: so a decimated sample answers them; every pixel costs 3x a 30fps frame budget.
SPLASH_SAMPLES = 50_000
#: Spatial variation below this is a dead output, however unlike the splash it looks.
MIN_SIGNAL_STD = 0.005
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
    def _decimate(frame: np.ndarray, samples: int = SPLASH_SAMPLES) -> np.ndarray:
        """Strided view holding about ``samples`` pixels, for area statistics."""
        pixels = frame.shape[0] * frame.shape[1]
        step = max(1, int(np.sqrt(pixels / max(samples, 1))))
        return frame[::step, ::step]

    def chroma_fraction(
        self, frame: np.ndarray, *, level: float = CHROMA_LEVEL, samples: int = SPLASH_SAMPLES
    ) -> float:
        """Fraction of pixels carrying meaningful chroma (HSV saturation above ``level``).

        Estimated from a strided sample: it is an area fraction, and reading
        every pixel of a 1080p frame costs three times a 30fps frame budget.
        """
        view = self._decimate(frame, samples)
        peak = view.max(axis=2)
        span = peak - view.min(axis=2)
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
        from syncsummoner.aesthetics.levels import luma_709  # keeps scipy off the device import path

        if frame is None:
            return True
        if self.chroma_fraction(frame) > max_chroma_frac:
            return False
        luma = luma_709(self._decimate(frame))
        bright_frac = float(np.count_nonzero(luma >= bright) / luma.size)
        mid_frac = float(np.count_nonzero((luma > dark) & (luma < bright)) / luma.size)
        return bright_frac >= min_bright_frac and mid_frac <= max_mid_frac

    def _stream(self, timeout_s: float, *, poll_s: float = 0.0) -> Iterator[np.ndarray | None]:
        """Yield every read until ``timeout_s`` elapses, None where it was unusable.

        Bounded on purpose: a read-until-good loop hangs forever whenever the
        splash test rejects everything, which is how a stimulus that happens to
        look achromatic stalls a whole sweep.
        """
        self.open()
        deadline = self._clock() + timeout_s
        while True:
            frame = self.read()
            yield None if frame is None or self.is_no_signal(frame) else frame
            if self._clock() >= deadline:
                return
            if poll_s:
                self._sleep(poll_s)

    def wait_for_lock(
        self, timeout_s: float = 10.0, *, poll_s: float = 0.05, min_std: float = MIN_SIGNAL_STD
    ) -> bool:
        """Block until a frame arrives carrying picture, not a grab failure or the splash.

        A dead output is black, which is neither, so spatial variance is the
        test: a still stimulus has it even when nothing is moving.
        """
        return any(
            frame is not None and float(self._decimate(frame).std()) >= min_std
            for frame in self._stream(timeout_s, poll_s=poll_s)
        )

    def wait_for_content(self, timeout_s: float = 15.0, *, run: int = 10, min_motion: float = 1e-4) -> bool:
        """Block until frames are both past the splash and actually moving.

        A program change blacks the output out and the card needs seconds to
        re-lock, so a fixed dwell samples the splash. The drop this rig is prone
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

"""Stimulus playout: RGB frames to a remote framebuffer feeding the analog input.

A Raspberry Pi drives ``/dev/fb0`` into the device's video input. A frame is
displayed by writing its raw BGR565 bytes there; geometry follows the session
format.
"""

from __future__ import annotations

import subprocess
import time
from typing import Callable, Iterable

import numpy as np

DEFAULT_HOST = "pi@videopi"
DEFAULT_FB = "/dev/fb0"
#: 565 quantisation levels per channel, for an RGB-ordered source frame.
LEVELS = np.array([31, 63, 31], dtype=np.float32)
#: Measured word layout is BGR565: blue occupies the high bits, not red.
SHIFTS = np.array([0, 5, 11], dtype=np.uint16)
LIMITED_BLACK, LIMITED_WHITE = 16.0 / 255.0, 235.0 / 255.0

Runner = Callable[[str, bytes], None]


def to_fb565(frame: np.ndarray, *, limited_range: bool = True) -> bytes:
    """Pack an RGB float32 frame in ``[0, 1]`` into the framebuffer's BGR565 words.

    ``limited_range`` maps to studio swing, measured at unity gain end to end;
    full range clips above ~92% against the chain's limited-range expectation.
    """
    if frame.ndim != 3 or frame.shape[2] != 3:
        raise ValueError(f"frame must be (H, W, 3), got {frame.shape}")
    values = np.clip(frame, 0.0, 1.0)
    if limited_range:
        values = LIMITED_BLACK + values * (LIMITED_WHITE - LIMITED_BLACK)
    quantised = np.rint(values * LEVELS).astype(np.uint16)
    words = np.bitwise_or.reduce(quantised << SHIFTS, axis=2)
    return words.astype("<u2").tobytes()


def ssh_runner(host: str = DEFAULT_HOST, *, timeout: float = 30.0) -> Runner:
    """Return a runner that pipes bytes into a command over BatchMode ssh."""

    def run(command: str, data: bytes) -> None:
        subprocess.run(
            ["ssh", "-o", "BatchMode=yes", host, command],
            input=data,
            check=True,
            timeout=timeout,
        )

    return run


class Playout:
    """Pushes stimulus frames to the Pi framebuffer.

    ``runner`` is injected so the whole class is exercisable with no Pi and no
    network.
    """

    def __init__(
        self,
        host: str = DEFAULT_HOST,
        *,
        width: int = 1920,
        height: int = 1080,
        framebuffer: str = DEFAULT_FB,
        runner: Runner | None = None,
        sleep: Callable[[float], None] = time.sleep,
        clock: Callable[[], float] = time.monotonic,
    ):
        self.width = int(width)
        self.height = int(height)
        self.framebuffer = framebuffer
        self._runner = ssh_runner(host) if runner is None else runner
        self._sleep = sleep
        self._clock = clock

    @property
    def frame_bytes(self) -> int:
        """Size of one framebuffer write."""
        return self.width * self.height * 2

    def encode(self, frame: np.ndarray) -> bytes:
        """Pack one frame, checking it matches the framebuffer geometry."""
        if frame.shape[:2] != (self.height, self.width):
            raise ValueError(f"frame must be ({self.height}, {self.width}), got {frame.shape[:2]}")
        return to_fb565(frame)

    def show(self, frame: np.ndarray) -> None:
        """Display one still frame."""
        self._runner(f"cat > {self.framebuffer}", self.encode(frame))

    def play(self, frames: Iterable[np.ndarray], *, fps: float = 60.0) -> int:
        """Display a sequence, paced against a monotonic clock; returns frames shown."""
        if fps <= 0:
            raise ValueError(f"fps must be positive, got {fps}")
        interval = 1.0 / fps
        origin = self._clock()
        count = 0
        for count, frame in enumerate(frames, start=1):
            self.show(frame)
            delay = origin + count * interval - self._clock()
            if delay > 0:
                self._sleep(delay)
        return count

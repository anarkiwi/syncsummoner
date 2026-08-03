"""Capture recorded by ffmpeg, so nothing on the host competes with the rig.

Reading frames into the host per frame cannot hold a session's rate: measured, a
Python loop managed 7.7 frames a second with a lossless encoder in the same
process and 25 without it, against 59.8 for ffmpeg reading the card in MJPEG.
"""

from __future__ import annotations

import subprocess
import time
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Callable, Iterator

__all__ = ["MJPEG", "RAW_422", "Recorder", "RecorderError"]

#: Compressed by the card; the only mode measured above session rate on this hardware.
MJPEG = ("mjpeg", "copy")
#: The card's own samples, for an archive that must not lose a bit.
RAW_422 = ("yuyv422", "rawvideo")


class RecorderError(RuntimeError):
    """The recording did not start, or did not produce a file."""


class Recorder:
    """Records the capture card to a file for the length of a pass.

    ``mode`` picks what crosses the wire and what is stored: ``MJPEG`` for a take
    that only has to look right, ``RAW_422`` for the card's own bytes.
    """

    def __init__(
        self,
        device: str = "/dev/video0",
        *,
        width: int = 1920,
        height: int = 1080,
        fps: float = 30.0,
        mode: tuple[str, str] = MJPEG,
        ffmpeg: str = "ffmpeg",
        popen: Callable[..., Any] = subprocess.Popen,
        sleep: Callable[[float], None] = time.sleep,
    ):
        self.device = device
        self.width = int(width)
        self.height = int(height)
        self.fps = float(fps)
        self.mode = mode
        self.ffmpeg = ffmpeg
        self._popen = popen
        self._sleep = sleep

    def command(self, path: str | Path, *, seconds: float | None = None) -> list[str]:
        """Recorder argv: the card in, one file out, no filtering in between."""
        fmt, codec = self.mode
        argv = [
            self.ffmpeg,
            "-loglevel",
            "error",
            "-y",
            "-f",
            "v4l2",
            "-input_format",
            fmt,
            "-video_size",
            f"{self.width}x{self.height}",
            "-framerate",
            str(int(self.fps)),
            "-i",
            self.device,
        ]
        if seconds is not None:
            argv += ["-t", f"{seconds:.3f}"]
        return argv + ["-c:v", codec, str(path)]

    @contextmanager
    def recording(
        self, path: str | Path, *, seconds: float | None = None, settle_s: float = 1.5
    ) -> Iterator[Any]:
        """Record for the duration of the block, yielding once it is running."""
        proc = self._popen(
            self.command(path, seconds=seconds),
            stdin=subprocess.PIPE,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        self._sleep(settle_s)
        if proc.poll() is not None:
            raise RecorderError(f"the recorder exited immediately for {self.device}")
        try:
            yield proc
        finally:
            self.stop(proc)
        written = Path(path)
        if not written.exists() or not written.stat().st_size:
            raise RecorderError(f"the recorder wrote nothing to {path}")

    def stop(self, proc: Any, *, timeout_s: float = 20.0) -> int:
        """Ask ffmpeg to finish the file, and wait for it to do so."""
        try:
            proc.stdin.write(b"q")
            proc.stdin.flush()
            proc.stdin.close()
        except (BrokenPipeError, OSError, ValueError, AttributeError):
            pass
        try:
            return int(proc.wait(timeout=timeout_s) or 0)
        except subprocess.TimeoutExpired:
            proc.kill()
            return int(proc.wait() or 0)

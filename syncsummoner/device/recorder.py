"""Capture recorded by ffmpeg, so nothing on the host competes with the rig.

Reading frames into the host per frame cannot hold a session's rate: measured, a
Python loop managed 7.7 frames a second with a lossless encoder in the same
process and 25 without it, against 59.8 for ffmpeg reading the card in MJPEG.
"""

from __future__ import annotations

import hashlib
import subprocess
import tempfile
import time
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Iterator

import numpy as np

__all__ = [
    "BLANK_LEVEL",
    "FFV1",
    "MJPEG",
    "RAW_422",
    "BlankTakeError",
    "Recorder",
    "RecorderError",
    "TakeReport",
    "inspect_take",
    "settle",
]

#: Compressed by the card; the only mode measured above session rate on this hardware.
MJPEG = ("mjpeg", "-c:v", "copy")
#: The card's own samples, unencoded, for a scratch pass that is read once.
RAW_422 = ("yuyv422", "-c:v", "rawvideo")
#: Lossless intra-only at session rate, with the exchanged chroma pair put back.
FFV1 = (
    "yuyv422",
    "-vf",
    "swapuv",
    "-c:v",
    "ffv1",
    "-level",
    "3",
    "-coder",
    "1",
    "-context",
    "1",
    "-slices",
    "4",
    "-slicecrc",
    "1",
    "-pix_fmt",
    "yuv422p",
)
#: Mean level below which a captured frame carries no picture.
BLANK_LEVEL = 0.02
#: Width every take is judged at; a blank frame is blank at any scale.
INSPECT_WIDTH = 160


class RecorderError(RuntimeError):
    """The recording did not start, or did not produce a file."""


def _diagnostic(errors: Any, *, limit: int = 400) -> str:
    """The tail of what ffmpeg said, for an error that has to be actionable."""
    try:
        errors.seek(0)
        said = errors.read().decode("utf-8", "replace").strip()
    except (OSError, ValueError, AttributeError):
        return "no diagnostic"
    return said[-limit:] if said else "no diagnostic"


class BlankTakeError(RuntimeError):
    """The device never returned a picture, so the take would have been black."""


@dataclass(frozen=True)
class TakeReport:
    """What a finished take actually holds, so a blank one is caught here and not on playback."""

    frames: int
    distinct: int
    blank: int
    luma: float

    @property
    def usable(self) -> bool:
        """Whether the take carries picture at all."""
        return bool(self.frames) and self.blank < self.frames // 2 and self.distinct > 1

    def __str__(self) -> str:
        return (
            f"{self.frames} frames, {self.distinct} distinct, "
            f"{self.blank} blank, mean luma {self.luma:.3f}"
            f"{'' if self.usable else '  UNUSABLE'}"
        )


class Recorder:
    """Records the capture card to a file for the length of a pass.

    ``mode`` is the card format to ask for followed by the output arguments to
    store it with: :data:`MJPEG` for a take that only has to look right,
    :data:`FFV1` for an archive that must not lose a sample.
    """

    def __init__(
        self,
        device: str = "/dev/video0",
        *,
        width: int = 1920,
        height: int = 1080,
        fps: float = 30.0,
        mode: tuple[str, ...] = MJPEG,
        copyts: bool = False,
        ffmpeg: str = "ffmpeg",
        ffprobe: str = "ffprobe",
        popen: Callable[..., Any] = subprocess.Popen,
        run: Callable[..., Any] = subprocess.run,
        sleep: Callable[[float], None] = time.sleep,
    ):
        self.device = device
        self.width = int(width)
        self.height = int(height)
        self.fps = float(fps)
        self.mode = tuple(mode)
        self.copyts = bool(copyts)
        self.ffmpeg = ffmpeg
        self.ffprobe = ffprobe
        self._popen = popen
        self._run = run
        self._sleep = sleep

    def command(self, path: str | Path, *, seconds: float | None = None) -> list[str]:
        """Recorder argv: the card in, one file out, whatever ``mode`` stores it as.

        ``-t`` is unavailable under ``copyts``, where it is read against the
        card's own clock and would stop the recording before it started.
        """
        argv = [
            self.ffmpeg,
            "-loglevel",
            "error",
            "-y",
            "-f",
            "v4l2",
            "-input_format",
            self.mode[0],
            "-video_size",
            f"{self.width}x{self.height}",
            "-framerate",
            str(int(self.fps)),
            "-i",
            self.device,
        ]
        if self.copyts:
            argv.append("-copyts")
        elif seconds is not None:
            argv += ["-t", f"{seconds:.3f}"]
        return argv + list(self.mode[1:]) + [str(path)]

    @contextmanager
    def recording(
        self, path: str | Path, *, seconds: float | None = None, settle_s: float = 1.5
    ) -> Iterator[Any]:
        """Record for the duration of the block, yielding once it is running.

        ffmpeg's diagnostics are kept rather than discarded: a recorder that dies
        silently is a fault nobody can tell from a rig that has gone dark.
        """
        errors = tempfile.TemporaryFile()
        try:
            proc = self._popen(
                self.command(path, seconds=seconds),
                stdin=subprocess.PIPE,
                stdout=subprocess.DEVNULL,
                stderr=errors,
            )
            self._sleep(settle_s)
            if proc.poll() is not None:
                raise RecorderError(
                    f"the recorder exited {proc.returncode} for {self.device}: {_diagnostic(errors)}"
                )
            try:
                yield proc
            finally:
                self.stop(proc)
            written = Path(path)
            if not written.exists() or not written.stat().st_size:
                raise RecorderError(f"the recorder wrote nothing to {path}: {_diagnostic(errors)}")
        finally:
            errors.close()

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

    def timestamps(self, path: str | Path) -> np.ndarray:
        """When the card delivered each recorded frame, in stream order.

        Under ``copyts`` the stored presentation times are the card's own
        ``CLOCK_MONOTONIC`` capture times, which is what :func:`time.monotonic`
        reads, so the host's schedule and the recording share one timeline.
        """
        done = self._run(
            [
                self.ffprobe,
                "-v",
                "error",
                "-select_streams",
                "v",
                "-show_entries",
                "packet=pts_time",
                "-of",
                "csv=p=0",
                str(path),
            ],
            capture_output=True,
            check=False,
        )
        out = done.stdout
        text = out.decode("utf-8", "replace") if isinstance(out, bytes) else (out or "")
        return np.array([float(v) for v in text.split() if v], dtype=np.float64)


def inspect_take(path: str | Path, *, ffmpeg: str = "ffmpeg", samples: int = 240) -> TakeReport:
    """Measure a finished take, so a blank one is caught here and not on playback."""
    height = INSPECT_WIDTH * 9 // 16
    argv = [
        ffmpeg,
        "-loglevel",
        "error",
        "-i",
        str(path),
        "-vf",
        f"scale={INSPECT_WIDTH}:-2,fps=source_fps/{max(1, samples // 60)}",
        "-f",
        "rawvideo",
        "-pix_fmt",
        "gray",
        "pipe:1",
    ]
    data = subprocess.run(argv, check=False, capture_output=True).stdout
    stride = INSPECT_WIDTH * height
    frames = [data[i : i + stride] for i in range(0, len(data) - stride + 1, stride)]
    if not frames:
        return TakeReport(frames=0, distinct=0, blank=0, luma=0.0)
    levels = np.frombuffer(b"".join(frames), np.uint8).reshape(len(frames), stride).mean(axis=1) / 255.0
    return TakeReport(
        frames=len(frames),
        distinct=len({hashlib.blake2b(f, digest_size=8).digest() for f in frames}),
        blank=int((levels < BLANK_LEVEL).sum()),
        luma=float(levels.mean()),
    )


def settle(
    recorder: Recorder,
    *,
    program: str,
    timeout_s: float,
    probe_path: str | Path,
    probe_s: float = 2.0,
    clock: Callable[[], float] = time.monotonic,
    sleep: Callable[[float], None] = time.sleep,
) -> TakeReport:
    """Wait for the picture to come back after a load, by recording a short probe.

    Liveness is judged the same way a take is, because judging it any other way
    is how a working rig gets called faulted.
    """
    deadline = clock() + timeout_s
    probe = Path(str(probe_path))
    report = TakeReport(frames=0, distinct=0, blank=0, luma=0.0)
    try:
        while clock() < deadline:
            with recorder.recording(probe, seconds=probe_s):
                sleep(probe_s)
            report = inspect_take(probe, ffmpeg=recorder.ffmpeg)
            if report.usable:
                return report
    finally:
        probe.unlink(missing_ok=True)
    raise BlankTakeError(f"{program}: no picture within {timeout_s}s of the load ({report})")

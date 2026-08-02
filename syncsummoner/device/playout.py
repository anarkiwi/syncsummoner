"""Stimulus playout: RGB frames to a remote framebuffer feeding the analog input.

A Raspberry Pi drives ``/dev/fb0`` into the device's video input. A frame is
displayed by writing its raw BGR565 bytes there; an invariant loop is instead
uploaded once and cycled by the Pi, so the host pushes no frame per pass.
"""

from __future__ import annotations

import gzip
import time
from contextlib import contextmanager
from typing import Callable, Iterable, Iterator

import numpy as np

from .host import DEFAULT_HOST, DEFAULT_TIMEOUT, Runner, ssh_runner

DEFAULT_FB = "/dev/fb0"
#: 565 quantisation levels per channel, for an RGB-ordered source frame.
LEVELS = np.array([31, 63, 31], dtype=np.float32)
#: Measured word layout is BGR565: blue occupies the high bits, not red.
SHIFTS = np.array([0, 5, 11], dtype=np.uint16)
LIMITED_BLACK, LIMITED_WHITE = 16.0 / 255.0, 235.0 / 255.0
#: tmpfs, so the Pi's own reads of the loop run at memory speed.
DEFAULT_LOOP_DIR = "/dev/shm/syncsummoner"
#: Capture runs at 30 fps and needs a clean frame per loop position; caller tunes.
DEFAULT_LOOP_FPS = 12.0
#: Sized for one compressed loop upload, which the per-command default cannot cover.
UPLOAD_TIMEOUT = 300.0
#: Measured ~10x on 565 codeframes, so the upload is bounded by the link, not the CPU.
GZIP_LEVEL = 1
RUNNING, STOPPED = "RUNNING", "STOPPED"

#: The Pi-side player: frames are read into memory once, then paced into the framebuffer.
PLAYER_SOURCE = '''"""Cycle raw framebuffer frames from tmpfs, paced against a monotonic clock.

A signal ends the loop only between frames and a short write is resumed, so the
framebuffer is never left holding half of one frame and half of another.
"""

import glob
import os
import signal
import sys
import time

directory, framebuffer, fps = sys.argv[1], sys.argv[2], float(sys.argv[3])
frames = [open(path, "rb").read() for path in sorted(glob.glob(directory + "/frame.*"))]
running = [True]
signal.signal(signal.SIGTERM, lambda *_: running.clear())
interval, shown, origin = 1.0 / fps, 0, time.monotonic()
fd = os.open(framebuffer, os.O_WRONLY | os.O_CREAT, 0o644)
while frames and running:
    view = memoryview(frames[shown % len(frames)])
    os.lseek(fd, 0, os.SEEK_SET)
    while view:
        view = view[os.write(fd, view) :]
    shown += 1
    delay = origin + shown * interval - time.monotonic()
    if delay > 0:
        time.sleep(delay)
os.close(fd)
'''


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
        ssh_timeout_s: float = DEFAULT_TIMEOUT,
        runner: Runner | None = None,
        sleep: Callable[[float], None] = time.sleep,
        clock: Callable[[], float] = time.monotonic,
    ):
        self.width = int(width)
        self.height = int(height)
        self.framebuffer = framebuffer
        self._runner = ssh_runner(host, timeout=ssh_timeout_s) if runner is None else runner
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


class PlayoutError(RuntimeError):
    """The Pi-side player did not reach the state it was driven to."""


class LoopPlayer(Playout):
    """Cycles an uploaded loop on the Pi itself, so the host pushes nothing per pass.

    The loop is invariant across programs and setpoints, so it is uploaded once
    into tmpfs and cycled by a detached process addressed by pidfile; pattern
    matching on ``ps`` is never used to find it.
    """

    def __init__(
        self,
        host: str = DEFAULT_HOST,
        *,
        directory: str = DEFAULT_LOOP_DIR,
        ssh_timeout_s: float = UPLOAD_TIMEOUT,
        timeout_s: float = 10.0,
        poll_s: float = 0.25,
        **kwargs,
    ):
        super().__init__(host, ssh_timeout_s=ssh_timeout_s, **kwargs)
        if poll_s <= 0:
            raise ValueError(f"poll_s must be positive, got {poll_s}")
        self.directory = directory.rstrip("/")
        self.timeout_s = float(timeout_s)
        self.poll_s = float(poll_s)

    @property
    def pid_path(self) -> str:
        """Pidfile the player records its own pid in."""
        return f"{self.directory}/player.pid"

    @property
    def log_path(self) -> str:
        """Where the detached player's output lands."""
        return f"{self.directory}/player.log"

    @property
    def script_path(self) -> str:
        """The player program on the Pi."""
        return f"{self.directory}/player.py"

    def upload(self, frames: Iterable[np.ndarray]) -> int:
        """Send the loop, compressed, into tmpfs as one raw frame per file.

        Returns the bytes put on the wire; the loop is invariant, so this is paid
        once for a whole run rather than once per pass.
        """
        blob = b"".join(self.encode(frame) for frame in frames)
        count = len(blob) // self.frame_bytes
        if count < 1:
            raise ValueError("loop must contain at least one frame")
        payload = gzip.compress(blob, GZIP_LEVEL)
        digits = max(3, len(str(count - 1)))
        self._runner(
            f"mkdir -p {self.directory} && rm -f {self.directory}/frame.* && "
            f"gzip -dc | split -b {self.frame_bytes} -d -a {digits} - {self.directory}/frame.",
            payload,
        )
        return len(payload)

    def frame_count(self) -> int:
        """How many loop frames are currently uploaded."""
        out = str(self._runner(f"ls {self.directory}/frame.* 2>/dev/null | wc -l") or "").strip()
        return int(out) if out.isdigit() else 0

    def pid(self) -> int | None:
        """Pid in the pidfile, or None when the player has none recorded."""
        out = str(self._runner(f"cat {self.pid_path} 2>/dev/null || true") or "").strip()
        return int(out) if out.isdigit() else None

    def _alive(self, pid: int) -> bool:
        """Liveness by signal 0, which names one process and cannot match another."""
        return self._runner(f"kill -0 {pid} 2>/dev/null && echo {RUNNING} || echo {STOPPED}") == RUNNING

    def is_running(self) -> bool:
        """Whether the process named by the pidfile is alive."""
        pid = self.pid()
        return pid is not None and self._alive(pid)

    def log(self) -> str:
        """Tail of the player's output, which is where a failed start explains itself."""
        return self._runner(f"tail -n 5 {self.log_path} 2>/dev/null || true")

    def _wait(self, running: bool, what: str) -> None:
        """Poll until liveness matches ``running``, or raise once the timeout passes."""
        deadline = self._clock() + self.timeout_s
        while self.is_running() != running:
            if self._clock() >= deadline:
                raise PlayoutError(f"{what} after {self.timeout_s}s: {self.log()}")
            self._sleep(self.poll_s)

    def start(self, *, fps: float = DEFAULT_LOOP_FPS) -> int:
        """Restart the detached player at ``fps`` per loop frame; returns its pid.

        The pid is recorded by the shell that execs the player, so the pidfile
        holds the player's own pid and survives the ssh session ending.
        """
        if fps <= 0:
            raise ValueError(f"fps must be positive, got {fps}")
        self.stop()
        self._runner(f"mkdir -p {self.directory} && cat > {self.script_path}", PLAYER_SOURCE.encode())
        self._runner(
            f"setsid sh -c 'echo $$ > {self.pid_path}; "
            f"exec python3 {self.script_path} {self.directory} {self.framebuffer} {float(fps)}' "
            f"< /dev/null > {self.log_path} 2>&1 &"
        )
        self._wait(True, f"player did not start at {fps} fps")
        return self.pid()

    def stop(self) -> bool:
        """Stop the player, leaving its last frame standing; True if it was running."""
        pid = self.pid()
        running = pid is not None and self._alive(pid)
        if running:
            self._runner(f"kill {pid} 2>/dev/null || true")
            self._wait(False, f"player pid {pid} did not exit")
        if pid is not None:
            self._runner(f"rm -f {self.pid_path}")
        return running

    @contextmanager
    def playing(self, *, fps: float = DEFAULT_LOOP_FPS) -> Iterator["LoopPlayer"]:
        """Hold the loop running for the block, stopping it on the way out or on error."""
        self.start(fps=fps)
        try:
            yield self
        finally:
            self.stop()

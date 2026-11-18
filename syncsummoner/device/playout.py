"""Stimulus playout: a vsync-locked DRM/KMS player on the Pi feeding the device's input.

The Pi runs :mod:`syncsummoner.device.kmsplay`, which flips one frame per vblank and logs any
flip that misses its vblank, flagged :data:`~syncsummoner.device.kmsplay.MISSED`. The host
uploads the stimulus once, starts and stops the player by its pidfile, and reads back its status
and its log of every flip.
"""

from __future__ import annotations

import base64
import gzip
import hashlib
import json
import time
from contextlib import contextmanager
from fractions import Fraction
from pathlib import Path
from typing import Any, Callable, Iterable, Iterator, Sequence

import numpy as np

from syncsummoner.progress import LOG

from .host import DEFAULT_HOST, DEFAULT_TIMEOUT, Runner, ssh_runner
from .kmsplay import MISSED, RECORD, SEGMENT

DEFAULT_FB = "/dev/fb0"
#: 565 quantisation levels per channel, for an RGB-ordered source frame.
LEVELS = np.array([31, 63, 31], dtype=np.float32)
#: Measured word layout is BGR565: blue occupies the high bits, not red.
SHIFTS = np.array([0, 5, 11], dtype=np.uint16)
LIMITED_BLACK, LIMITED_WHITE = 16.0 / 255.0, 235.0 / 255.0
#: tmpfs, so the Pi's own reads of the stimulus run at memory speed.
DEFAULT_DIR = "/dev/shm/syncsummoner"
#: A loop frame is held for a whole number of vblanks: three at the 30 Hz session.
DEFAULT_LOOP_FPS = 10.0
#: Sized for one compressed loop or clip upload, which the per-command default cannot cover.
UPLOAD_TIMEOUT = 300.0
#: Measured ~10x on 565 codeframes, so the upload is bounded by the link, not the CPU.
GZIP_LEVEL = 1
RUNNING, STOPPED = "RUNNING", "STOPPED"
#: :data:`~syncsummoner.device.kmsplay.RECORD` as a numpy record.
FLIP = np.dtype([("v", "<i8"), ("seq", "<i8"), ("frame", "<i8"), ("t_ns", "<i8"), ("flags", "<i8")])
assert FLIP.itemsize == RECORD.size
#: The pacing thread's ``SCHED_FIFO`` ceiling, raised for the player's own shell only.
RTPRIO_LIMIT = 20
#: Status states the player settles in once started.
SETTLED = ("playing", "failed", "stopped")


def file_digest(path: str | Path, chunk: int = 1 << 22) -> str:
    """SHA-256 of a file, read in chunks."""
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        while block := handle.read(chunk):
            digest.update(block)
    return digest.hexdigest()


class PlayoutError(RuntimeError):
    """The Pi-side player faulted, or did not reach the state it was driven to."""


def to_fb565(frame: np.ndarray, *, limited_range: bool = True) -> bytes:
    """Pack an RGB float32 frame in ``[0, 1]`` into the display's BGR565 words.

    ``limited_range`` maps to studio swing, measured at unity gain end to end; full range
    clips above ~92% against the chain's limited-range expectation.
    """
    if frame.ndim != 3 or frame.shape[2] != 3:
        raise ValueError(f"frame must be (H, W, 3), got {frame.shape}")
    values = np.clip(frame, 0.0, 1.0)
    if limited_range:
        values = LIMITED_BLACK + values * (LIMITED_WHITE - LIMITED_BLACK)
    quantised = np.rint(values * LEVELS).astype(np.uint16)
    words = np.bitwise_or.reduce(quantised << SHIFTS, axis=2)
    return words.astype("<u2").tobytes()


def rate(fps: float | str | Fraction) -> Fraction:
    """A frame rate as an exact fraction, NTSC's 1000/1001 rates included.

    A float within half an NTSC offset of ``n * 1000/1001`` is taken to be that rate.
    """
    if isinstance(fps, (str, Fraction)):
        return Fraction(fps)
    whole = round(fps * 1.001)
    ntsc = Fraction(whole * 1000, 1001)
    return ntsc if 2 * abs(float(ntsc) - fps) < float(whole - ntsc) else Fraction(fps).limit_denominator()


def player_source() -> bytes:
    """The Pi-side player program."""
    return Path(__file__).with_name("kmsplay.py").read_bytes()


def _late(flips: np.ndarray) -> np.ndarray:
    """Indices of the flips in a segment flagged :data:`MISSED` or off the vblank after the one before."""
    flags, v, seq = flips["flags"], flips["v"], flips["seq"]
    late = 1 + np.flatnonzero(((flags[1:] & MISSED) != 0) | (np.diff(seq) != 1) | (np.diff(v) != 1))
    return late[(flags[late] & SEGMENT) == 0]


def misses(flips: np.ndarray, hold: int) -> np.ndarray:
    """Per late flip, the first and last played frame due on a vblank it left showing another, ``(n, 2)``.

    Every vblank from the one after the flip before's to the late flip's own showed a frame other than
    the one due, so every frame due on them, and only those, may be mis-shown.
    """
    late = _late(flips)
    return np.c_[(flips["v"][late - 1] + 1) // hold, flips["v"][late] // hold].astype(np.int64)


def flip_summary(flips: np.ndarray, hold: int, spans: Sequence[tuple[int, int]] | None = None) -> dict:
    """Counts a flip log is judged by: flips, segments, late flips and frames shown out of cadence.

    ``spans`` are the played frames ``a..b-1`` the judgement rests on, all by default: a late flip counts
    when a frame it may have mis-shown is in one, any other flip when the frame due on it is.
    """
    lo, hi = (
        (np.array(spans, dtype=np.int64).reshape(-1, 2).T) if spans is not None else ([-np.inf], [np.inf])
    )
    late = misses(flips, hold)
    start = (flips["flags"] & SEGMENT) != 0
    regular = ~start
    regular[_late(flips)] = False
    due = flips["v"] // hold
    off = regular & (flips["frame"] != due) & np.any((due[:, None] >= lo) & (due[:, None] < hi), axis=1)
    return {
        "flips": int(flips.size),
        "segments": int(np.count_nonzero(start)),
        "missed": int(np.count_nonzero(np.any((late[:, :1] < hi) & (late[:, 1:] >= lo), axis=1))),
        "out_of_cadence": int(np.count_nonzero(off)),
    }


class Playout:
    """Geometry and frame packing shared by the players; ``runner`` reaches the Pi."""

    def __init__(
        self,
        host: str = DEFAULT_HOST,
        *,
        width: int = 1920,
        height: int = 1080,
        ssh_timeout_s: float = DEFAULT_TIMEOUT,
        runner: Runner | None = None,
    ):
        self.width = int(width)
        self.height = int(height)
        self._runner = ssh_runner(host, timeout=ssh_timeout_s) if runner is None else runner

    @property
    def frame_bytes(self) -> int:
        """Size of one displayed frame."""
        return self.width * self.height * 2

    def encode(self, frame: np.ndarray) -> bytes:
        """Pack one frame, checking it matches the display geometry."""
        if frame.shape[:2] != (self.height, self.width):
            raise ValueError(f"frame must be ({self.height}, {self.width}), got {frame.shape[:2]}")
        return to_fb565(frame)


class KmsPlayer(Playout):
    """Starts, watches and stops the Pi's vsync-locked player, addressed by its pidfile.

    Subclasses say what it plays through :meth:`source_args`. A fault the player reports
    is raised as :class:`PlayoutError` when it is stopped or checked, never swallowed.
    """

    def __init__(
        self,
        host: str = DEFAULT_HOST,
        *,
        directory: str = DEFAULT_DIR,
        loop: bool = False,
        preroll_s: float = 2.0,
        ssh_timeout_s: float = UPLOAD_TIMEOUT,
        timeout_s: float = 90.0,
        poll_s: float = 0.25,
        sleep: Callable[[float], None] = time.sleep,
        clock: Callable[[], float] = time.monotonic,
        **kwargs: Any,
    ):
        super().__init__(host, ssh_timeout_s=ssh_timeout_s, **kwargs)
        if poll_s <= 0:
            raise ValueError(f"poll_s must be positive, got {poll_s}")
        self.directory = directory.rstrip("/")
        self.loop = bool(loop)
        self.preroll_s = float(preroll_s)
        self.timeout_s = float(timeout_s)
        self.poll_s = float(poll_s)
        self._sleep = sleep
        self._clock = clock

    def path(self, name: str) -> str:
        """A file the player keeps in its directory on the Pi."""
        return f"{self.directory}/{name}"

    @property
    def pid_path(self) -> str:
        """Pidfile the player's shell records the player's pid in."""
        return self.path("player.pid")

    def source_args(self) -> str:
        """The player's source arguments."""
        raise NotImplementedError

    def span_args(self, spans: Sequence[tuple[int, int]] | None) -> str:
        """The arguments playing only ``spans`` of the source; a player that cannot refuses any."""
        if spans:
            raise ValueError(f"{type(self).__name__} plays no spans")
        return ""

    def command(self, fps: Fraction, spans: Sequence[tuple[int, int]] | None = None) -> str:
        """The player's command line on the Pi, playing only ``spans`` of the source when given."""
        return (
            f"python3 {self.path('kmsplay.py')} {self.source_args()}{self.span_args(spans)} --fps {fps} "
            f"--width {self.width} --height {self.height} --preroll-s {self.preroll_s:g} "
            f"--flips {self.path('flips.bin')} --status {self.path('status.json')}"
            + (" --loop" if self.loop else "")
        )

    def pid(self) -> int | None:
        """Pid in the pidfile, or None when none is recorded."""
        out = str(self._runner(f"cat {self.pid_path} 2>/dev/null || true") or "").strip()
        return int(out) if out.isdigit() else None

    def _alive(self, pid: int) -> bool:
        """Liveness by signal 0, which names one process and cannot match another."""
        return self._runner(f"kill -0 {pid} 2>/dev/null && echo {RUNNING} || echo {STOPPED}") == RUNNING

    def is_running(self) -> bool:
        """Whether the process named by the pidfile is alive."""
        pid = self.pid()
        return pid is not None and self._alive(pid)

    def status(self) -> dict:
        """The player's own status file; empty when it has written none."""
        out = str(self._runner(f"cat {self.path('status.json')} 2>/dev/null || true") or "").strip()
        return json.loads(out) if out else {}

    def log(self) -> str:
        """Tail of the player's output, where a fault explains itself."""
        return str(self._runner(f"tail -n 5 {self.path('player.log')} 2>/dev/null || true") or "")

    def flips(self) -> np.ndarray:
        """Every flip of the last play, as :data:`FLIP` records."""
        out = str(self._runner(f"base64 -w0 {self.path('flips.bin')} 2>/dev/null || true") or "")
        data = base64.b64decode(out.strip())
        return np.frombuffer(data[: len(data) - len(data) % FLIP.itemsize], dtype=FLIP)

    def _fault(self, status: dict) -> PlayoutError:
        """The error for a player that reported a fault."""
        return PlayoutError(f"the Pi player faulted: {status.get('error') or self.log() or 'no diagnostic'}")

    def check(self) -> dict:
        """The status, raising if the player faulted or died without being stopped."""
        status = self.status()
        if status.get("state") == "failed":
            raise self._fault(status)
        if status.get("state") == "playing" and not self.is_running():
            raise PlayoutError(f"the Pi player died while playing: {self.log() or 'no diagnostic'}")
        return status

    def start(
        self,
        *,
        fps: float | str | Fraction = DEFAULT_LOOP_FPS,
        spans: Sequence[tuple[int, int]] | None = None,
    ) -> int:
        """Restart the player at ``fps``, its pid once it is flipping; ``spans`` as for :meth:`command`.

        The pid is recorded by the shell that execs the player, so the pidfile holds the
        player's own pid and survives the ssh session ending.
        """
        exact = rate(fps)
        if exact <= 0:
            raise ValueError(f"fps must be positive, got {fps}")
        self.stop()
        self._runner(f"mkdir -p {self.directory} && cat > {self.path('kmsplay.py')}", player_source())
        self._runner(
            f"rm -f {self.path('status.json')}; setsid sh -c 'echo $$ > {self.pid_path}; "
            f"sudo -n prlimit --rtprio={RTPRIO_LIMIT} --pid $$; "
            f"exec {self.command(exact, spans)}' < /dev/null > {self.path('player.log')} 2>&1 &"
        )
        deadline = self._clock() + self.timeout_s
        while (state := self.status().get("state")) not in SETTLED:
            if self._clock() >= deadline:
                raise PlayoutError(
                    f"player did not start at {exact} fps after {self.timeout_s}s: {self.log()}"
                )
            self._sleep(self.poll_s)
        if state != "playing":
            self.stop()
            raise PlayoutError(f"player exited as it started: {self.log() or 'no diagnostic'}")
        if not self.status().get("realtime"):
            LOG.warning(
                "the Pi player runs without SCHED_FIFO: a stall can make it miss a vblank, which it reports"
            )
        return self.pid()

    def stop(self) -> dict:
        """Stop the player, its last frame left standing; raises if it had faulted."""
        pid = self.pid()
        if pid is not None and self._alive(pid):
            self._runner(f"kill {pid} 2>/dev/null || true")
            deadline = self._clock() + self.timeout_s
            while self._alive(pid):
                if self._clock() >= deadline:
                    raise PlayoutError(f"player pid {pid} did not exit after {self.timeout_s}s: {self.log()}")
                self._sleep(self.poll_s)
        if pid is not None:
            self._runner(f"rm -f {self.pid_path}")
        status = self.status() if pid is not None else {}
        if status.get("state") == "failed":
            raise self._fault(status)
        return status

    @contextmanager
    def playing(
        self,
        *,
        fps: float | str | Fraction = DEFAULT_LOOP_FPS,
        spans: Sequence[tuple[int, int]] | None = None,
    ) -> Iterator["KmsPlayer"]:
        """Play for the block, stopping afterwards; a fault fails the block unless it already failed."""
        self.start(fps=fps, spans=spans)
        try:
            yield self
        except BaseException:
            try:
                self.stop()
            except PlayoutError:
                pass
            raise
        self.stop()


class ClipPlayer(KmsPlayer):
    """Plays a clip decoded on the Pi, so the host pushes nothing per frame."""

    @property
    def clip_path(self) -> str:
        """Where the clip sits in tmpfs."""
        return self.path("clip.mkv")

    def remote_digest(self) -> str:
        """SHA-256 of the clip the Pi holds in tmpfs, or empty when it holds none."""
        out = str(self._runner(f"sha256sum {self.clip_path} 2>/dev/null || true") or "")
        return out.split()[0] if out.strip() else ""

    def upload(self, path: str) -> int:
        """Put the clip in tmpfs unless the Pi's copy hashes the same; returns the bytes put on the wire.

        The copy is hashed on the Pi after sending, and one that differs is a :class:`PlayoutError`.
        """
        want = file_digest(path)
        if self.remote_digest() == want:
            LOG.info("%s: the Pi already holds this clip (sha256 %s), so nothing is sent", path, want[:12])
            return 0
        data = Path(path).read_bytes()
        self._runner(f"mkdir -p {self.directory} && cat > {self.clip_path}", data)
        got = self.remote_digest()
        if got != want:
            raise PlayoutError(f"the Pi's copy of {path} hashes {got or 'to nothing'}, not {want}")
        return len(data)

    def source_args(self) -> str:
        """Decode the uploaded clip."""
        return f"--clip {self.clip_path}"

    def span_args(self, spans: Sequence[tuple[int, int]] | None) -> str:
        """Play the clip's frames ``a..b-1`` of each span in turn, with no seam between them."""
        return f" --spans {','.join(f'{a}:{b}' for a, b in spans)}" if spans else ""


class LoopPlayer(KmsPlayer):
    """Cycles an uploaded loop of frames, uploaded once into tmpfs and cycled by the Pi."""

    def __init__(self, host: str = DEFAULT_HOST, **kwargs: Any):
        kwargs.setdefault("loop", True)
        super().__init__(host, **kwargs)

    @property
    def frames_dir(self) -> str:
        """Where the loop's raw frames sit in tmpfs."""
        return self.path("frames")

    def upload(self, frames: Iterable[np.ndarray]) -> int:
        """Send the loop, compressed, into tmpfs as one raw frame per file; returns bytes sent."""
        blob = b"".join(self.encode(frame) for frame in frames)
        count = len(blob) // self.frame_bytes
        if count < 1:
            raise ValueError("loop must contain at least one frame")
        payload = gzip.compress(blob, GZIP_LEVEL)
        digits = max(3, len(str(count - 1)))
        self._runner(
            f"mkdir -p {self.frames_dir} && rm -f {self.frames_dir}/frame.* && "
            f"gzip -dc | split -b {self.frame_bytes} -d -a {digits} - {self.frames_dir}/frame.",
            payload,
        )
        return len(payload)

    def frame_count(self) -> int:
        """How many loop frames are currently uploaded."""
        out = str(self._runner(f"ls {self.frames_dir}/frame.* 2>/dev/null | wc -l") or "").strip()
        return int(out) if out.isdigit() else 0

    def source_args(self) -> str:
        """Cycle the uploaded frames."""
        return f"--frames {self.frames_dir}"

    def show(self, frame: np.ndarray, *, fps: float | str | Fraction = DEFAULT_LOOP_FPS) -> int:
        """Hold one still frame up until the player is stopped or restarted; returns its pid."""
        self.upload([frame])
        return self.start(fps=fps)

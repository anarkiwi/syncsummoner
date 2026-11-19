"""Stimulus playout: a vsync-locked DRM/KMS player on the Pi feeding the device's input.

The Pi runs :mod:`syncsummoner.device.kmsplay`, which flips one frame per vblank and fails
loudly on any missed one. The host uploads the stimulus once, starts and stops the player
by its pidfile, and reads back its status and its log of every flip. While it plays, the host
follows every flip live (:class:`Shown`), so it knows a frame that was on screen at any moment.
"""

from __future__ import annotations

import base64
import gzip
import hashlib
import bisect
import json
import subprocess
import threading
import time
from contextlib import contextmanager
from fractions import Fraction
from pathlib import Path
from typing import Any, Callable, Iterable, Iterator, Sequence

import numpy as np

from syncsummoner.progress import LOG

from .host import DEFAULT_HOST, DEFAULT_TIMEOUT, Runner, ssh_runner, ssh_stream
from .kmsplay import RECORD, REPLY, SEGMENT, read_exactly

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
#: One distinct frame followed live: the player's stream frame and when the host first heard of it.
SHOWN = np.dtype([("frame", "<i8"), ("t_ns", "<i8"), ("base", "<i8")])
#: One request to the player: when the host sent it, when the reply came, and the last flip done between.
ASKED = np.dtype([("t0_ns", "<i8"), ("t1_ns", "<i8"), ("frame", "<i8"), ("base", "<i8")])
#: The frame :meth:`Shown.latest` reports before any flip has been heard of.
NONE = int(np.iinfo(np.int64).min)


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


def flip_summary(flips: np.ndarray, hold: int) -> dict:
    """Counts a flip log is judged by: flips, segments, and any vblank or frame out of cadence."""
    if not flips.size:
        return {"flips": 0, "segments": 0, "missed": 0, "out_of_cadence": 0}
    start = (flips["flags"] & SEGMENT) != 0
    inside = ~start[1:]
    missed = int(np.count_nonzero(np.diff(flips["seq"])[inside] != np.diff(flips["v"])[inside]))
    due = flips["v"] // hold
    off = int(np.count_nonzero((flips["frame"] != due)[np.r_[False, inside]]))
    return {"flips": int(flips.size), "segments": int(start.sum()), "missed": missed, "out_of_cadence": off}


class Shown:
    """The host's view of the frame on screen, from the player's live flip records on ``stream``.

    A record counts only once it has been read whole, and is stamped with ``clock`` then, so every frame
    reported was on screen no later than the moment it is reported. :meth:`ask` writes one byte to
    ``asks``, which the player answers with the last flip done once it has read it, so the flip after
    that one came after the request left. Each record keeps its segment's base, its display vblank less
    its clip vblank, which the player sets afresh each time the link comes up. ``procs`` are the
    processes feeding ``stream`` and reading ``asks``, terminated on :meth:`close`; with none,
    ``stream`` must end for it to return.
    """

    def __init__(
        self,
        stream: Any,
        *,
        asks: Any = None,
        procs: Sequence[subprocess.Popen] = (),
        clock: Callable[[], int] = time.monotonic_ns,
    ):
        self._stream, self._asks, self._procs, self._clock = stream, asks, tuple(procs), clock
        self._frames: list[int] = []
        self._times: list[int] = []
        self._bases: list[int] = []
        self._sent: list[int] = []
        self._replies: list[tuple[int, int, int]] = []
        self._ended = False
        self._cond = threading.Condition()
        self._watched: set[threading.Event] = set()
        self._thread = threading.Thread(target=self._read, daemon=True)
        self._thread.start()

    def _read(self) -> None:
        """Note each distinct frame as its record completes, until the stream ends."""
        try:
            while (record := read_exactly(self._stream, RECORD.size)) is not None:
                t_ns, fields = self._clock(), RECORD.unpack(record)
                frame, base = fields[2], fields[1] - fields[0]
                with self._cond:
                    if fields[4] & REPLY:
                        self._replies.append((t_ns, frame, base))
                    elif not self._frames or frame > self._frames[-1]:
                        self._frames.append(frame)
                        self._times.append(t_ns)
                        self._bases.append(base)
                        self._cond.notify_all()
        except (OSError, ValueError):
            pass
        finally:
            with self._cond:
                self._ended = True
                self._cond.notify_all()

    def latest(self) -> int:
        """The frame of the last record received, or :data:`NONE` before any."""
        with self._cond:
            return self._frames[-1] if self._frames else NONE

    def _watch(self, stop: threading.Event) -> None:
        """Wake every waiter once ``stop`` is set."""

        def wake() -> None:
            stop.wait()
            with self._cond:
                self._cond.notify_all()

        with self._cond:
            if stop not in self._watched:
                self._watched.add(stop)
                threading.Thread(target=wake, daemon=True).start()

    def wait(self, frame: int, stop: threading.Event | None = None) -> int | None:
        """When the host first heard of a frame at or after ``frame``, blocking until it has.

        None once ``stop`` is set or the stream has ended without one.
        """
        if stop is not None:
            self._watch(stop)
        with self._cond:
            while (at := bisect.bisect_left(self._frames, frame)) == len(self._frames):
                if self._ended or (stop is not None and stop.is_set()):
                    return None
                self._cond.wait()
            return self._times[at]

    def ask(self) -> None:
        """Ask the player for the frame on screen, stamped before it leaves; quiet once the Pi stops."""
        with self._cond:
            if self._asks is None:
                return
            self._sent.append(self._clock())
            try:
                self._asks.write(b"?")
                self._asks.flush()
            except (BrokenPipeError, ValueError):
                self._sent.pop()
                self._asks = None

    def asked(self) -> np.ndarray:
        """Every request answered so far, paired with its reply in order, as :data:`ASKED` records."""
        with self._cond:
            n = min(len(self._sent), len(self._replies))
            out = np.empty(n, dtype=ASKED)
            out["t0_ns"] = self._sent[:n]
            out["t1_ns"], out["frame"], out["base"] = np.array(self._replies[:n], np.int64).reshape(n, 3).T
        return out

    def log(self) -> np.ndarray:
        """Every distinct frame heard of so far, as :data:`SHOWN` records."""
        with self._cond:
            out = np.empty(len(self._frames), dtype=SHOWN)
            out["frame"], out["t_ns"], out["base"] = self._frames, self._times, self._bases
        return out

    def close(self) -> None:
        """Stop following: end the processes owned, and wait for the reader."""
        for proc in self._procs:
            proc.terminate()
        self._thread.join()
        for proc in self._procs:
            proc.wait()
        if self._procs:
            self._stream.close()


class Playout:
    """Geometry and frame packing shared by the players; ``runner`` and ``stream`` reach the Pi."""

    def __init__(
        self,
        host: str = DEFAULT_HOST,
        *,
        width: int = 1920,
        height: int = 1080,
        ssh_timeout_s: float = DEFAULT_TIMEOUT,
        runner: Runner | None = None,
        stream: Callable[[str], subprocess.Popen] | None = None,
    ):
        self.width = int(width)
        self.height = int(height)
        self._runner = ssh_runner(host, timeout=ssh_timeout_s) if runner is None else runner
        self._stream = ssh_stream(host) if stream is None else stream

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
            f"--flips {self.path('flips.bin')} --live {self.path('live.fifo')} --ask {self.path('ask.fifo')} "
            f"--status {self.path('status.json')}" + (" --loop" if self.loop else "")
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
        heard: Callable[[], bool] | None = None,
    ) -> int | None:
        """Restart the player at ``fps``, its pid once it is flipping; ``spans`` as for :meth:`command`.

        The pid is recorded by the shell that execs the player, so the pidfile holds the
        player's own pid and survives the ssh session ending. With ``heard``, true once a flip has
        been heard live (:meth:`following`), it returns None on that, asking the Pi nothing more.
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
        while True:
            if heard is not None and heard():
                return None
            if (state := self.status().get("state")) in SETTLED:
                break
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
        heard: Callable[[], bool] | None = None,
    ) -> Iterator["KmsPlayer"]:
        """Play for the block, stopping afterwards; a fault fails the block unless it already failed."""
        self.start(fps=fps, spans=spans, heard=heard)
        try:
            yield self
        except BaseException:
            try:
                self.stop()
            except PlayoutError:
                pass
            raise
        self.stop()

    @contextmanager
    def following(self, *, clock: Callable[[], int] = time.monotonic_ns) -> Iterator[Shown]:
        """Follow the player's flips live for the block, from before it starts, its pipes made first."""
        self._runner(
            f"mkdir -p {self.directory} && for p in live ask; do f={self.directory}/$p.fifo; "
            '[ -p "$f" ] || { rm -f "$f"; mkfifo "$f"; }; done',
            None,
        )
        proc = self._stream(f"exec cat {self.path('live.fifo')}")
        asker = self._stream(f"exec cat > {self.path('ask.fifo')}")
        shown = Shown(proc.stdout, asks=asker.stdin, procs=(proc, asker), clock=clock)
        try:
            yield shown
        finally:
            shown.close()


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

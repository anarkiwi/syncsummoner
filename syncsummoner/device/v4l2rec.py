"""Capture card recorder keeping every frame's V4L2 sequence number and timestamp.

Frames are piped untouched into ffmpeg; a sidecar log keeps each one's driver sequence, so a lost one
shows as a gap. Run as ``python -m syncsummoner.device.v4l2rec``; ``q`` on stdin or SIGTERM ends it,
and a ``roll`` line splits one gap-free run between files (:func:`watch_stdin`).
"""

from __future__ import annotations

import argparse
import hashlib
import queue
import signal
import subprocess
import sys
import threading
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Iterable

import numpy as np

#: One capture: driver sequence, CLOCK_MONOTONIC time (ns), bytes used, buffer flags, digest of its bytes.
FRAME = np.dtype(
    [("sequence", "<i8"), ("t_ns", "<i8"), ("bytes", "<i8"), ("flags", "<i8"), ("digest", "<u8")]
)
SUFFIX = ".v4l2"
#: ``V4L2_BUF_FLAG_ERROR``: the driver delivered the buffer with corrupted data.
FLAG_ERROR = 0x40
#: Buffers queued to the driver, so a slow write never leaves it without one.
BUFFERS = 16
#: Dequeued frames held in memory while storage catches up: ten seconds at 60 captures a second.
BACKLOG = 600


def sidecar(path: str | Path) -> Path:
    """Where a recording's frame log lives."""
    return Path(f"{path}{SUFFIX}")


def read_log(path: str | Path) -> np.ndarray:
    """A recording's frame log, one :data:`FRAME` record per stored frame."""
    data = sidecar(path).read_bytes()
    return np.frombuffer(data[: len(data) - len(data) % FRAME.itemsize], dtype=FRAME)


def encoder_argv(
    fmt: str, width: int, height: int, fps: float, out_args: list[str], path: str, *, ffmpeg: str
) -> list[str]:
    """ffmpeg storing frames piped to it in the card's own format."""
    source = (
        ["-f", "mjpeg"]
        if fmt == "mjpeg"
        else ["-f", "rawvideo", "-pix_fmt", fmt, "-video_size", f"{width}x{height}"]
    )
    return [
        ffmpeg,
        "-loglevel",
        "error",
        "-y",
        *source,
        "-framerate",
        f"{fps:g}",
        "-i",
        "pipe:0",
        *out_args,
        path,
    ]


@dataclass(eq=False)
class Output:
    """One file captures go to over ``[opens, closes)`` of the capture clock, closed once done with."""

    sink: Any
    log: Any
    close: Callable[[], None] = field(default=lambda: None)
    opens: float = float("-inf")
    closes: float = float("inf")


class Outputs:
    """Where each capture goes: every output whose window holds its time, rolled on as asked.

    :meth:`roll` may be called from another thread while :meth:`at` is consulted per capture.
    """

    def __init__(self, first: Output, opener: Callable[[str], Output] | None = None):
        self.opened, self.opener = [first], opener
        self._lock = threading.Lock()

    def roll(self, opens: int, closes: int, path: str) -> None:
        """Send captures from ``opens`` on to ``path`` and end the last output's at ``closes``."""
        if self.opener is None:
            raise ValueError("this recording cannot roll to another file")
        out = self.opener(path)
        out.opens = float(opens)
        with self._lock:
            self.opened[-1].closes = float(closes)
            self.opened.append(out)

    def at(self, t_ns: int) -> tuple[Output, ...]:
        """The outputs a capture at ``t_ns`` goes to."""
        with self._lock:
            return tuple(o for o in self.opened if o.opens <= t_ns < o.closes)


def record(
    frames: Iterable[Any],
    sink: Any,
    log: Any,
    *,
    stop: Callable[[], bool],
    limit: int | None = None,
    backlog: int = BACKLOG,
    outputs: Outputs | None = None,
) -> int:
    """Write frames to ``sink`` and their records to ``log`` until ``stop`` or ``limit``; returns how many.

    With ``outputs``, each frame goes wherever they say for its time, each closed once frames move past it.
    A writer thread drains up to ``backlog`` dequeued frames, so stalled storage holds them in memory
    rather than leaving the driver without a buffer. A write that fails is raised here.
    """
    pending: queue.Queue = queue.Queue(maxsize=max(1, int(backlog)))
    failed: list[BaseException] = []
    outputs = Outputs(Output(sink, log)) if outputs is None else outputs

    def drain() -> None:
        live: tuple[Output, ...] = ()
        while (item := pending.get()) is not None:
            if failed:
                continue
            try:
                for done in set(live) - set(item[2]):
                    done.close()
                live = item[2]
                for out in live:
                    out.sink.write(item[0])
                    out.log.write(item[1])
            except BaseException as err:  # pylint: disable=broad-exception-caught
                failed.append(err)

    writer = threading.Thread(target=drain, name="v4l2rec-writer", daemon=True)
    writer.start()
    kept = 0
    try:
        for frame in frames:
            if stop() or (limit is not None and kept >= limit) or failed:
                break
            buff, data = frame.buff, bytes(frame.data)
            t_ns = int(buff.timestamp.secs) * 1_000_000_000 + int(buff.timestamp.usecs) * 1000
            digest = int.from_bytes(hashlib.blake2b(data, digest_size=8).digest(), "little")
            row = np.array([(buff.sequence, t_ns, buff.bytesused, buff.flags, digest)], dtype=FRAME)
            pending.put((data, row.tobytes(), outputs.at(t_ns)))
            kept += 1
    finally:
        pending.put(None)
        writer.join()
    if failed:
        raise failed[0]
    return kept


def frame_limit(seconds: float | None, rate: float) -> int | None:
    """Frames in ``seconds`` at the rate the card actually runs at, which need not be the one asked for."""
    return None if seconds is None else int(round(seconds * rate))


def watch_stdin(
    stream: Any, flag: threading.Event, roll: Callable[[int, int, str], None] | None = None
) -> None:
    """Set ``flag`` once ``q`` arrives and hand each ``roll T0 T1 PATH`` line to ``roll``; EOF asks nothing.

    ``stream`` is unbuffered in use: a daemon thread blocked on a buffered reader aborts shutdown.
    """
    line = b""
    while got := stream.read(1):
        got = got.encode() if isinstance(got, str) else got
        if got == b"q" and not line:
            flag.set()
            return
        if got != b"\n":
            line += got
            continue
        words = line.decode().split(maxsplit=3)
        line = b""
        if len(words) == 4 and words[0] == "roll" and roll is not None:
            roll(int(words[1]), int(words[2]), words[3])


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """The recorder's command line; everything after ``--`` is ffmpeg's output arguments."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--device", default="/dev/video0")
    parser.add_argument("--width", type=int, default=1920)
    parser.add_argument("--height", type=int, default=1080)
    parser.add_argument("--fps", type=float, default=30.0)
    parser.add_argument("--input-format", default="mjpeg")
    parser.add_argument("--seconds", type=float, help="stop after this long at the card's own frame rate")
    parser.add_argument("--ffmpeg", default="ffmpeg")
    parser.add_argument("out")
    parser.add_argument("out_args", nargs=argparse.REMAINDER)
    args = parser.parse_args(argv)
    args.out_args = [a for a in args.out_args if a != "--"]
    return args


def main(argv: list[str] | None = None) -> int:  # pragma: no cover - needs a capture card
    """Record until stopped; non-zero when the card or any encoder fails."""
    # pylint: disable=import-outside-toplevel
    from linuxpy.video.device import BufferType, Device, VideoCapture

    args = parse_args(argv)
    done = threading.Event()
    signal.signal(signal.SIGTERM, lambda *_: done.set())
    raw = open(sys.stdin.fileno(), "rb", buffering=0, closefd=False)  # pylint: disable=consider-using-with
    fourcc = "MJPG" if args.input_format == "mjpeg" else "YUYV"
    encoders: list[subprocess.Popen] = []
    with Device(args.device) as cam:
        cam.set_format(BufferType.VIDEO_CAPTURE, args.width, args.height, fourcc)
        cam.set_fps(BufferType.VIDEO_CAPTURE, args.fps)
        rate = float(cam.get_fps(BufferType.VIDEO_CAPTURE))

        def opener(path: str) -> Output:
            argv_out = encoder_argv(
                args.input_format, args.width, args.height, rate, args.out_args, path, ffmpeg=args.ffmpeg
            )
            # An archive may hold the old take as a hard link; a fresh inode leaves it intact.
            if Path(path).is_file():
                Path(path).unlink()
            enc = subprocess.Popen(argv_out, stdin=subprocess.PIPE)  # pylint: disable=consider-using-with
            log = open(sidecar(path), "wb", buffering=0)  # pylint: disable=consider-using-with
            encoders.append(enc)

            def close() -> None:
                enc.stdin.close()
                log.close()

            return Output(enc.stdin, log, close)

        outputs = Outputs(opener(args.out), opener)
        threading.Thread(target=watch_stdin, args=(raw, done, outputs.roll), daemon=True).start()
        try:
            with VideoCapture(cam, size=BUFFERS) as stream:
                kept = record(
                    stream,
                    None,
                    None,
                    stop=done.is_set,
                    limit=frame_limit(args.seconds, rate),
                    outputs=outputs,
                )
        except BrokenPipeError:
            print("an encoder exited while recording", file=sys.stderr)
            return 1
        finally:
            for out in outputs.opened:
                try:
                    out.close()
                except (OSError, ValueError):
                    pass
        code = max((enc.wait() for enc in encoders), default=0)
    print(f"recorded {kept} frames at {rate:g} fps over {len(encoders)} files", file=sys.stderr)
    return code


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())

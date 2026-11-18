"""Capture card recorder keeping every frame's V4L2 sequence number and timestamp.

Frames are piped untouched into ffmpeg, each run of byte-identical captures stored once; a sidecar log
keeps every capture's driver sequence and the stored frame showing it, so a lost one shows as a gap and
every capture reads back as it arrived (:mod:`~syncsummoner.device.stored`). Run as
``python -m syncsummoner.device.v4l2rec``; ``q`` on stdin or SIGTERM ends it, and a ``roll`` line splits
one gap-free run between files (:func:`watch_stdin`).
"""

from __future__ import annotations

import argparse
import hashlib
import queue
import signal
import struct
import subprocess
import sys
import threading
from dataclasses import dataclass, field
from fractions import Fraction
from pathlib import Path
from typing import Any, Callable, Iterable

import numpy as np

#: One capture: driver sequence, CLOCK_MONOTONIC time (ns), bytes used, buffer flags, digest of its bytes.
FRAME = np.dtype(
    [("sequence", "<i8"), ("t_ns", "<i8"), ("bytes", "<i8"), ("flags", "<i8"), ("digest", "<u8")]
)
#: A capture in a deduplicated take: its :data:`FRAME`, the stored frame showing it and its timestamp.
STORED = np.dtype(FRAME.descr + [("stored", "<i8"), ("pts", "<i8")])
SUFFIX = ".v4l2"
#: Opens a sidecar of :data:`STORED` rows; one without it is :data:`FRAME` rows from its first byte.
MAGIC = b"\x89V4L2\r\n"
VERSION = 2
#: Magic, version, the stored frames' fourcc, width and height, and their time base.
HEADER = struct.Struct("<7sB4sHHII")
#: IVF, the framing ffmpeg reads a timestamped stream from: file header, then one header per frame.
IVF = struct.Struct("<4sHH4sHHIIII")
IVF_FRAME = struct.Struct("<IQ")
#: Timestamps are kept in milliseconds, the time base Matroska stores.
TIME_BASE = Fraction(1, 1000)
#: ``V4L2_BUF_FLAG_ERROR``: the driver delivered the buffer with corrupted data.
FLAG_ERROR = 0x40
#: Buffers queued to the driver, so a slow write never leaves it without one.
BUFFERS = 16
#: Dequeued frames held in memory while storage catches up: ten seconds at 60 captures a second.
BACKLOG = 600
#: The card format each stored frame is framed as.
FOURCC = {"mjpeg": b"MJPG", "yuyv422": b"YUY2"}


@dataclass(frozen=True)
class Layout:
    """What a take stores: the frames' fourcc and size, and the time base of their timestamps."""

    fourcc: bytes = FOURCC["mjpeg"]
    width: int = 0
    height: int = 0
    tb: Fraction = TIME_BASE

    def sidecar_header(self) -> bytes:
        """The header opening a deduplicated take's sidecar."""
        return HEADER.pack(
            MAGIC, VERSION, self.fourcc, self.width, self.height, self.tb.numerator, self.tb.denominator
        )

    def ivf_header(self) -> bytes:
        """The header opening an IVF stream of these frames."""
        return IVF.pack(
            b"DKIF",
            0,
            IVF.size,
            self.fourcc,
            self.width,
            self.height,
            self.tb.denominator,
            self.tb.numerator,
            0,
            0,
        )


def ivf_frame(data: bytes, pts: int) -> bytes:
    """The header framing one IVF frame of ``data`` at ``pts``."""
    return IVF_FRAME.pack(len(data), pts)


def sidecar(path: str | Path) -> Path:
    """Where a recording's frame log lives."""
    return Path(f"{path}{SUFFIX}")


def read_sidecar(path: str | Path) -> tuple[np.ndarray, Layout | None]:
    """A recording's rows as stored, :data:`STORED` with its :class:`Layout` or :data:`FRAME` and None.

    A torn last row, from a recorder stopped mid-write, is dropped.
    """
    data = sidecar(path).read_bytes()
    layout, dtype, body = None, FRAME, data
    if data[: len(MAGIC)] == MAGIC:
        _, version, fourcc, width, height, num, den = HEADER.unpack_from(data)
        if version != VERSION:
            raise ValueError(f"{sidecar(path)}: sidecar version {version}, not {VERSION}")
        layout, dtype, body = Layout(fourcc, width, height, Fraction(num, den)), STORED, data[HEADER.size :]
    return np.frombuffer(body[: len(body) - len(body) % dtype.itemsize], dtype=dtype), layout


def read_log(path: str | Path) -> np.ndarray:
    """A recording's frame log, one :data:`FRAME` record per capture whatever the sidecar's version."""
    rows, layout = read_sidecar(path)
    if layout is None:
        return rows
    log = np.empty(rows.size, dtype=FRAME)
    for name in FRAME.names:
        log[name] = rows[name]
    return log


def timestamp(index: Any, rate: Fraction, tb: Fraction = TIME_BASE) -> Any:
    """Capture ``index``'s timestamp at ``rate``, rounded to the nearest tick of ``tb``, halves up."""
    num, den = rate.denominator * tb.denominator, rate.numerator * tb.numerator
    return (2 * index * num + den) // (2 * den)


def card_rate(rate: float) -> Fraction:
    """The card's rate as the rational ffmpeg reads from its ``%g`` spelling."""
    return Fraction(f"{rate:g}").limit_denominator(1001000)


def encoder_argv(out_args: list[str], path: str, *, ffmpeg: str) -> list[str]:
    """ffmpeg storing the timestamped frames piped to it as IVF."""
    return [ffmpeg, "-loglevel", "error", "-y", "-f", "ivf", "-i", "pipe:0", *out_args, path]


@dataclass(eq=False)
class Output:
    """One file captures go to over ``[opens, closes)`` of the capture clock, closed once done with.

    ``held`` is the last frame stored, with its digest; ``stored`` counts frames stored and ``count``
    captures logged, so each capture's row names the stored frame showing it.
    """

    sink: Any
    log: Any
    close: Callable[[], None] = field(default=lambda: None)
    opens: float = float("-inf")
    closes: float = float("inf")
    held: bytes | None = None
    digest: int = -1
    stored: int = 0
    count: int = 0

    def write(self, data: bytes, row: tuple, digest: int, layout: Layout, rate: Fraction) -> None:
        """Log one capture, storing its frame unless it repeats the last one stored byte for byte."""
        if not self.count:
            self.sink.write(layout.ivf_header())
            self.log.write(layout.sidecar_header())
        pts = timestamp(self.count, rate, layout.tb)
        if not (digest == self.digest and data == self.held):
            self.sink.write(ivf_frame(data, pts))
            self.sink.write(data)
            self.held, self.digest, self.stored = data, digest, self.stored + 1
        self.log.write(np.array([(*row, digest, self.stored - 1, pts)], dtype=STORED).tobytes())
        self.count += 1


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
    layout: Layout = Layout(),
    rate: Fraction = Fraction(30),
) -> int:
    """Write frames to ``sink`` and their records to ``log`` until ``stop`` or ``limit``; returns how many.

    Frames are framed as IVF at ``rate`` on ``layout``'s time base, each output storing a run of
    byte-identical captures once (:meth:`Output.write`). With ``outputs``, each frame goes wherever
    they say for its time, each closed once frames move past it.
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
                    out.write(item[0], item[1], item[3], layout, rate)
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
            row = (buff.sequence, t_ns, buff.bytesused, buff.flags)
            pending.put((data, row, outputs.at(t_ns), digest))
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
    parser.add_argument("--input-format", default="mjpeg", choices=sorted(FOURCC))
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
        layout = Layout(FOURCC[args.input_format], args.width, args.height)

        def opener(path: str) -> Output:
            argv_out = encoder_argv(args.out_args, path, ffmpeg=args.ffmpeg)
            # An archive may hold the old take as a hard link; a fresh inode leaves it intact.
            for old in (Path(path), sidecar(path)):
                if old.is_file():
                    old.unlink()
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
                    layout=layout,
                    rate=card_rate(rate),
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

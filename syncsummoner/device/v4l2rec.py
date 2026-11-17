"""Capture card recorder keeping every frame's V4L2 sequence number and timestamp.

Frames are dequeued with linuxpy and piped, untouched, into ffmpeg for storage; a sidecar
log keeps each frame's driver sequence, so a frame the card or driver lost shows as a gap.
Run as ``python -m syncsummoner.device.v4l2rec``; ``q`` on stdin or SIGTERM ends it.
"""

from __future__ import annotations

import argparse
import hashlib
import signal
import subprocess
import sys
import threading
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


def record(
    frames: Iterable[Any],
    sink: Any,
    log: Any,
    *,
    stop: Callable[[], bool],
    limit: int | None = None,
) -> int:
    """Write frames to ``sink`` and their records to ``log`` until ``stop`` or ``limit``; returns how many."""
    kept = 0
    for frame in frames:
        if stop() or (limit is not None and kept >= limit):
            break
        buff = frame.buff
        sink.write(frame.data)
        t_ns = int(buff.timestamp.secs) * 1_000_000_000 + int(buff.timestamp.usecs) * 1000
        digest = int.from_bytes(hashlib.blake2b(frame.data, digest_size=8).digest(), "little")
        log.write(
            np.array([(buff.sequence, t_ns, buff.bytesused, buff.flags, digest)], dtype=FRAME).tobytes()
        )
        kept += 1
    return kept


def frame_limit(seconds: float | None, rate: float) -> int | None:
    """Frames in ``seconds`` at the rate the card actually runs at, which need not be the one asked for."""
    return None if seconds is None else int(round(seconds * rate))


def watch_stdin(stream: Any, flag: threading.Event) -> None:
    """Set ``flag`` once ``q`` arrives; a closed stdin alone asks for nothing.

    ``stream`` is unbuffered in use: a daemon thread blocked on a buffered reader aborts shutdown.
    """
    while got := stream.read(1):
        if got in (b"q", "q"):
            flag.set()
            return


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
    """Record until stopped; non-zero when the card or the encoder fails."""
    # pylint: disable=import-outside-toplevel
    from linuxpy.video.device import BufferType, Device, VideoCapture

    args = parse_args(argv)
    done = threading.Event()
    signal.signal(signal.SIGTERM, lambda *_: done.set())
    raw = open(sys.stdin.fileno(), "rb", buffering=0, closefd=False)  # pylint: disable=consider-using-with
    threading.Thread(target=watch_stdin, args=(raw, done), daemon=True).start()
    fourcc = "MJPG" if args.input_format == "mjpeg" else "YUYV"
    with Device(args.device) as cam:
        cam.set_format(BufferType.VIDEO_CAPTURE, args.width, args.height, fourcc)
        cam.set_fps(BufferType.VIDEO_CAPTURE, args.fps)
        rate = float(cam.get_fps(BufferType.VIDEO_CAPTURE))
        argv_out = encoder_argv(
            args.input_format, args.width, args.height, rate, args.out_args, args.out, ffmpeg=args.ffmpeg
        )
        with (
            subprocess.Popen(argv_out, stdin=subprocess.PIPE) as enc,
            open(sidecar(args.out), "wb", buffering=0) as log,
        ):
            try:
                with VideoCapture(cam, size=BUFFERS) as stream:
                    kept = record(
                        stream, enc.stdin, log, stop=done.is_set, limit=frame_limit(args.seconds, rate)
                    )
            except BrokenPipeError:
                print("the encoder exited while recording", file=sys.stderr)
                return 1
            finally:
                enc.stdin.close()
            code = enc.wait()
    print(f"recorded {kept} frames at {rate:g} fps", file=sys.stderr)
    return code


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())

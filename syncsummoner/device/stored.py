"""Every capture of a take as it arrived, from a file storing each run of identical captures once.

The sidecar of a deduplicated take (:mod:`~syncsummoner.device.v4l2rec`) keeps every capture's row, the
stored frame showing it and its timestamp. :func:`captures` is how any reader opens a take, so each
sees the stream a take storing every capture would give it.
"""

from __future__ import annotations

import hashlib
import math
import os
import subprocess
import tempfile
import threading
from contextlib import ExitStack, closing, contextmanager
from dataclasses import dataclass
from fractions import Fraction
from pathlib import Path
from typing import Iterator

import numpy as np

from .v4l2rec import Layout, ivf_frame, read_sidecar, sidecar

#: Suffix of an ffconcat listing, the concat demuxer's input of several parts.
LISTING_SUFFIX = ".ffconcat"


class StoredError(RuntimeError):
    """A deduplicated take's file and sidecar disagree."""


def digest_of(data: bytes) -> int:
    """The digest the recorder keeps of a capture's bytes."""
    return int.from_bytes(hashlib.blake2b(data, digest_size=8).digest(), "little")


def micros(seconds: float | Fraction) -> int:
    """``seconds`` to the microsecond, as ffmpeg parses a seek."""
    return round(float(seconds) * 1_000_000)


@dataclass(frozen=True)
class Stored:
    """A deduplicated take's sidecar: per capture its row, the stored frame showing it and its timestamp."""

    path: Path
    rows: np.ndarray
    layout: Layout

    @classmethod
    def of(cls, path: str | Path) -> "Stored | None":
        """The take's :class:`Stored`, or None when it stores every capture or has no sidecar."""
        if not sidecar(path).is_file():
            return None
        rows, layout = read_sidecar(path)
        if layout is None:
            return None
        step = np.diff(rows["stored"])
        if rows.size and (rows["stored"][0] != 0 or np.any((step != 0) & (step != 1))):
            raise StoredError(f"{sidecar(path)}: stored frames are not numbered in order")
        return cls(Path(path), rows, layout)

    @property
    def index(self) -> np.ndarray:
        """Per capture, the stored frame showing it."""
        return self.rows["stored"]

    @property
    def firsts(self) -> np.ndarray:
        """Per stored frame, the first capture showing it."""
        return np.flatnonzero(np.r_[True, np.diff(self.index) != 0])

    def seconds(self) -> np.ndarray:
        """Each capture's timestamp in seconds."""
        return self.rows["pts"] * self.layout.tb.numerator / self.layout.tb.denominator

    def at(self, start_s: float | Fraction) -> int:
        """The first capture an input seek to ``start_s`` keeps: timestamped at or after it."""
        tick = math.ceil(Fraction(micros(start_s), 1_000_000) / self.layout.tb)
        return int(np.searchsorted(self.rows["pts"], tick, side="left"))

    def duration(self) -> float:
        """Seconds the captures span, one mean capture interval past the last."""
        t = self.seconds()
        return float(t[-1] + (t[-1] - t[0]) / (t.size - 1)) if t.size > 1 else float(t.sum())

    def frames(self, start: int = 0, *, ffmpeg: str = "ffmpeg") -> Iterator[tuple[int, bytes]]:
        """Every capture from ``start`` on as ``(pts, bytes)``, until the file runs out of stored frames.

        Stored frames are copied out of the file untouched from an exact seek, and each must have the
        size and digest logged for its first capture, or :class:`StoredError` is raised.
        """
        if start >= self.rows.size:
            return
        firsts, index = self.firsts, self.index
        first = int(index[start])
        seek = f"{math.floor(Fraction(int(self.rows['pts'][firsts[first]])) * self.layout.tb * 1_000_000)}us"
        argv = [ffmpeg, "-nostdin", "-loglevel", "error", *(["-copyts", "-ss", seek] if first else [])]
        argv += ["-i", str(self.path), "-map", "0:v:0", "-c", "copy", *(["-ss", seek] if first else [])]
        sizes, digests = self.rows["bytes"][firsts], self.rows["digest"][firsts]
        proc = subprocess.Popen(  # pylint: disable=consider-using-with
            [*argv, "-f", "rawvideo", "pipe:1"],
            stdin=subprocess.DEVNULL,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
        )
        try:
            held, at = b"", -1
            for capture in range(start, self.rows.size):
                if index[capture] != at:
                    at = int(index[capture])
                    held = proc.stdout.read(int(sizes[at]))
                    if len(held) < sizes[at]:
                        return
                    if digest_of(held) != int(digests[at]):
                        raise StoredError(f"{self.path}: stored frame {at} is not the capture logged for it")
                yield int(self.rows["pts"][capture]), held
        finally:
            proc.kill()
            proc.stdout.close()
            proc.wait()


def _feed(fifo: Path, stored: Stored, start: int, ffmpeg: str, failed: list) -> None:
    """Write every capture from ``start`` into ``fifo`` as IVF, stopping quietly when the reader goes."""
    try:
        with open(fifo, "wb", buffering=0) as out, closing(stored.frames(start, ffmpeg=ffmpeg)) as frames:
            out.write(stored.layout.ivf_header())
            for pts, data in frames:
                out.write(ivf_frame(data, pts))
                out.write(data)
    except BrokenPipeError:
        pass
    except BaseException as err:  # pylint: disable=broad-exception-caught
        failed.append(err)


@contextmanager
def _fed(stored: Stored, start: int, ffmpeg: str, fifo: Path) -> Iterator[Path]:
    """``fifo`` carrying ``stored``'s captures from ``start`` on while the block runs.

    A FIFO its reader never opened is opened here, so the writer always ends; its failure is raised.
    """
    os.mkfifo(fifo)
    failed: list[BaseException] = []
    writer = threading.Thread(target=_feed, args=(fifo, stored, start, ffmpeg, failed), daemon=True)
    writer.start()
    try:
        yield fifo
    finally:
        if writer.is_alive():
            os.close(os.open(fifo, os.O_RDONLY | os.O_NONBLOCK))
        writer.join()
    if failed:
        raise failed[0]


def _lines(listing: Path) -> list[list[str]]:
    """An ffconcat listing's directives after its header, each as ``[directive, argument]``."""
    lines = listing.read_text(encoding="utf-8").splitlines()[1:]
    return [line.split(" ", 1) for line in lines if line.strip()]


@contextmanager
def captures(
    path: str | Path,
    *,
    start_s: float | Fraction | None = None,
    from_end_s: float | None = None,
    ffmpeg: str = "ffmpeg",
) -> Iterator[list[str]]:
    """ffmpeg input arguments reading every capture of ``path`` from ``start_s`` on, or near its end.

    A take storing every capture is read directly, seeked as ffmpeg seeks; a deduplicated one is fed
    through a FIFO as IVF, each capture with its own timestamp. An ffconcat listing with deduplicated
    parts is read through a listing of its parts so fed, from its start.
    """
    path = Path(path)
    listing = path.suffix == LISTING_SUFFIX
    lines = _lines(path) if listing else [["file", f"'{path.name}'"]]
    stored = {i: Stored.of(path.parent / a.strip("'")) for i, (k, a) in enumerate(lines) if k == "file"}
    if all(found is None for found in stored.values()):
        seek = [] if start_s is None else ["-ss", f"{micros(start_s) // 10**6}.{micros(start_s) % 10**6:06d}"]
        yield [*seek, *([] if from_end_s is None else ["-sseof", f"{-from_end_s:g}"]), "-i", str(path)]
        return
    with ExitStack() as stack:
        folder = Path(stack.enter_context(tempfile.TemporaryDirectory(prefix="captures-")))
        if not listing:
            one = stored[0]
            if from_end_s is not None:
                start_s = one.duration() - from_end_s
            start = 0 if start_s is None else one.at(start_s)
            yield ["-f", "ivf", "-i", str(stack.enter_context(_fed(one, start, ffmpeg, folder / "take.ivf")))]
            return
        if start_s or from_end_s is not None:
            raise ValueError(f"{path}: a listing of deduplicated parts is read from its start")
        text = "ffconcat version 1.0\n"
        for i, (key, arg) in enumerate(lines):
            if key == "file":
                part = (path.parent / arg.strip("'")).resolve()
                fed = (
                    part
                    if stored[i] is None
                    else stack.enter_context(_fed(stored[i], 0, ffmpeg, folder / f"{i}.ivf"))
                )
                arg = f"'{fed}'"
            text += f"{key} {arg}\n"
        (folder / "listing.ffconcat").write_text(text, encoding="utf-8")
        yield ["-f", "concat", "-safe", "0", "-i", str(folder / "listing.ffconcat")]

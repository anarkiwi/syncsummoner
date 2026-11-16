"""Check archive decoders against each other: bit-exactness, and throughput under concurrency.

FFV1 is lossless, so every backend must hand the refit the same bytes; a digest of
each program's whole decoded stream proves it. Throughput is measured, not assumed,
across concurrent decoders, since a GPU saturates at a stream count of its own.
"""

from __future__ import annotations

import hashlib
import subprocess
import time
from concurrent.futures import ThreadPoolExecutor
from itertools import cycle, islice
from typing import Any, Callable, Iterable, Sequence

from syncsummoner.progress import track

__all__ = ["digest", "rate", "survey"]


def digest(reader: Any, *, width: int | None = None) -> tuple[str, int]:
    """SHA-256 of every frame ``reader`` decodes, in stream order, and how many there were."""
    hashed, frames = hashlib.sha256(), 0
    for frame in reader.stream(width=width):
        hashed.update(frame)
        frames += 1
    return hashed.hexdigest(), frames


def rate(
    archive: Any,
    names: Sequence[str],
    decode: str,
    streams: int,
    *,
    runs: int | None = None,
    frames: int | None = None,
    clock: Callable[[], float] = time.monotonic,
    run: Callable[..., Any] = subprocess.run,
) -> float:
    """Frames per second ``streams`` concurrent decoders sustain, each decoding whole programs.

    ``runs`` programs are decoded in all, ``names`` cycled to fill them, two per
    stream by default so no stream idles at the end while others finish, each cut
    to its first ``frames`` if given. Output is discarded by the decoder itself, so
    this is the decoder's rate and not a reader's.
    """
    readers = [archive.reader(name, decode=decode) for name in islice(cycle(names), runs or 2 * streams)]

    def decode_one(reader: Any) -> int:
        count = min(reader.count, frames or reader.count)
        argv = reader.command(count=count)
        result = run(argv, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE, check=False)
        if result.returncode:
            raise RuntimeError(f"{decode} decode of {reader.video} failed: {result.stderr!r}")
        return count

    start = clock()
    with ThreadPoolExecutor(max_workers=streams) as pool:
        total = sum(track(pool.map(decode_one, readers), desc=f"{decode} x{streams}", total=len(readers)))
    return total / max(clock() - start, 1e-9)


def survey(
    archive: Any, names: Iterable[str], decoders: Sequence[str], streams: Sequence[int], **kwargs: Any
) -> dict[str, dict[int, float]]:
    """:func:`rate` for every decoder at every stream count."""
    chosen = list(names)
    return {decode: {n: rate(archive, chosen, decode, n, **kwargs) for n in streams} for decode in decoders}

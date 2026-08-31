"""What an archived program did to the loop: how far it registers, and how pointwise it is.

Every archive was recorded while the codeframe loop played out, so the stimulus is
rebuilt from the run's seed and this costs no rig time. One native decode feeds two
regimes, and they are not the same width: the pointwise fit samples native pixels
because area-averaging mixes neighbours and co-location is what it asks about, while
registration is area-averaged, the chain having lowpassed the carrier away at native.
"""

from __future__ import annotations

from collections.abc import Sequence
from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace
from itertools import groupby
from typing import Any, Callable, Iterable, NamedTuple

import cv2  # pylint: disable=no-member  ; cv2 is a compiled extension pylint cannot introspect
import numpy as np

from syncsummoner.aesthetics.structure import pointwise_r2
from syncsummoner.probe import codeframes
from syncsummoner.probe.archive import GAP
from syncsummoner.probe.harvest import HarvestConfig, stimulus_loop

__all__ = ["BINS", "Behaviour", "behaviours", "program_behaviour"]

#: Dwells read per program and captures per dwell: 120 native frames, spread over the sweep.
DWELL_SAMPLES = 5
FRAME_SAMPLES = 24
#: Captures pointwise-fitted per dwell, and pixels sampled from each: subsampled, never scaled.
POINTWISE_SAMPLES = 8
PIXEL_SAMPLES = 100_000
#: A steep map needs fine bins: the RGB-path passthrough reads 0.24 at 32 bins and 0.83 at 256.
BINS = 256
SEED = 7
#: Width registration reads at, which is not the pointwise fit's. The chain lowpasses the
#: carrier, so at native it is gone and everything reads unregistered: Kaledos correlates
#: 0.51 at 640 and 0.00 at 1920, and the passthrough control 0.85 against 0.75.
REGISTER_WIDTH = 640


class Behaviour(NamedTuple):
    """The two ``ProgramProfile`` fields a refit can only get from the frames themselves.

    ``registered`` is how strongly the capture correlates with the loop's carrier once
    aligned, so an inverting program still registers; ``style`` is not here, being a
    classification of ``pointwise`` across the library rather than a per-program reading.
    """

    registered: float = 0.0
    pointwise: float = 0.0


def _picks(count: int, samples: int) -> np.ndarray:
    """Evenly spread positions, at most ``samples`` of them."""
    return np.unique(np.linspace(0, count - 1, min(samples, count)).astype(np.intp))


def _dwells(rows: Sequence[Any], *, samples: int = DWELL_SAMPLES, frames: int = FRAME_SAMPLES) -> list:
    """Spread dwells as ``(start, count)`` spans of the frame stream.

    A dwell is a run of consecutive frames held at one setpoint, so its captures are
    contiguous in time, taken under one vector, and decodable in one pass. Runs cut
    short by dropped frames are passed over while any full one remains.
    """
    runs = [
        (int(group[0].frame), len(group))
        for setpoint, group in ((k, list(g)) for k, g in groupby(rows, key=lambda row: row.setpoint))
        if setpoint != GAP
    ]
    usable = [run for run in runs if run[1] >= frames // 2] or runs
    return [(usable[i][0], min(usable[i][1], frames)) for i in _picks(len(usable), samples)]


def _target(shape: tuple) -> tuple[int, int]:
    """Size registration runs at, which is the capture's own where that is smaller."""
    height, width = int(shape[0]), int(shape[1])
    if width <= REGISTER_WIDTH:
        return width, height
    return REGISTER_WIDTH, 2 * round(height * REGISTER_WIDTH / width / 2)


def _reduced(frame: np.ndarray, size: tuple[int, int]) -> np.ndarray:
    """One capture in unit range, area-averaged to registration's own scale."""
    scaled = np.asarray(frame, dtype=np.float32) / np.float32(255.0)
    if (scaled.shape[1], scaled.shape[0]) == size:
        return scaled
    return cv2.resize(scaled, size, interpolation=cv2.INTER_AREA)  # pylint: disable=no-member


def _scaled(loop: codeframes.CodeLoop, width: int) -> codeframes.CodeLoop:
    """The loop at a smaller width, its carrier resampled rather than regenerated.

    Band noise built at another size is another texture, so the reference has to be
    the one the run played, area-averaged exactly like the captures it is read against.
    """
    if width >= loop.width:
        return loop
    height = 2 * round(loop.height * width / loop.width / 2)
    strip = max(1, round(loop.strip_px * height / loop.height))
    size = (width, height - strip)
    texture = cv2.resize(loop.texture, size, interpolation=cv2.INTER_AREA)  # pylint: disable=no-member
    texture = texture - texture.mean()
    texture /= float(np.abs(texture).max()) or 1.0
    return replace(loop, width=width, height=height, strip_px=strip, texture=texture.astype(np.float32))


def _sources(loop: codeframes.CodeLoop, shape: tuple, index: np.ndarray) -> list[np.ndarray]:
    """Every loop position as the same sampled pixels, at the capture's own geometry."""
    out = []
    for k in range(loop.count):
        frame = loop.frame(k)
        if frame.shape[:2] != tuple(shape[:2]):
            frame = cv2.resize(frame, (int(shape[1]), int(shape[0])))  # pylint: disable=no-member
        out.append(np.ascontiguousarray(frame.reshape(-1, 3)[index]))
    return out


def _pointwise(captures: Sequence[np.ndarray], sources: Sequence[np.ndarray], index: np.ndarray) -> float:
    """Best pointwise fit any loop position offers, median over the sampled captures.

    Best over positions rather than the temporally aligned one: a program may delay or
    freeze, and this exists to say whether a value curve could explain the output at
    all, not to date it.
    """
    scores = []
    for i in _picks(len(captures), POINTWISE_SAMPLES):
        pixels = captures[i].reshape(-1, 3)[index].astype(np.float32) / np.float32(255.0)
        scores.append(max(pointwise_r2(source, pixels, bins=BINS) for source in sources))
    return float(np.median(scores)) if scores else 0.0


def _registered(captures: Sequence[np.ndarray], loop: codeframes.CodeLoop, size: tuple) -> float:
    """Median strength the loop's carrier is recovered at, over a few of the dwell's frames.

    The carrier is common to every loop position, so registration needs no index and a
    few frames settle it; the magnitude is signless because an inverting program still
    registers.
    """
    picks = _picks(len(captures), codeframes.REGISTER_SAMPLES)
    fits = [codeframes.register(_reduced(captures[i], size), loop) for i in picks]
    return float(np.median([abs(fit.correlation) for fit in fits]))


def program_behaviour(
    archive: Any,
    program: str,
    *,
    loop: codeframes.CodeLoop | None = None,
    seed: int = SEED,
    log: Callable[[str], None] | None = None,
) -> Behaviour:
    """Read one archived program's own frames into the two fields only they carry.

    A program that cannot be decoded, or that holds no measured dwell, reports the
    profile's own defaults and says why: a refit covers the library, and one
    unreadable program must not end it.
    """
    note = log if log is not None else lambda _message: None
    reference = stimulus_loop(HarvestConfig()) if loop is None else loop
    try:
        reader = archive.reader(program)
        if reader is None:
            raise LookupError("no committed archive")
        spans = _dwells(reader.rows)
        if not spans:
            raise LookupError("no measured dwell")
        rng = np.random.default_rng(seed)
        sources: list[np.ndarray] | None = None
        index, size, reduced = np.empty(0, dtype=np.intp), (0, 0), reference
        rows = []
        for start, count in spans:
            captures = [np.asarray(frame) for frame in reader.stream(start=start, count=count)]
            if not captures:
                continue
            if sources is None:
                shape = captures[0].shape
                pixels = int(shape[0]) * int(shape[1])
                index = rng.choice(pixels, size=min(PIXEL_SAMPLES, pixels), replace=False)
                sources = _sources(reference, shape, index)
                size = _target(shape)
                reduced = _scaled(reference, size[0])
            rows.append((_registered(captures, reduced, size), _pointwise(captures, sources, index)))
        if not rows:
            raise LookupError("no frames decoded")
        return Behaviour(
            registered=float(np.median([row[0] for row in rows])),
            pointwise=float(np.median([row[1] for row in rows])),
        )
    except Exception as exc:
        note(f"{program}: loop behaviour not measured ({type(exc).__name__}: {exc})")
        return Behaviour()


def behaviours(
    archive: Any,
    programs: Iterable[str],
    *,
    loop: codeframes.CodeLoop | None = None,
    jobs: int = 1,
    log: Callable[[str], None] | None = None,
) -> dict[str, Behaviour]:
    """Loop behaviour of every named program, keyed by name and defaulted where unread.

    ``jobs`` reads that many programs at once, as the replay does: a decoder subprocess
    and array work both drop the lock, so threads are enough to overlap them.
    """
    names = list(programs)
    reference = stimulus_loop(HarvestConfig()) if loop is None else loop

    def measure(name: str) -> tuple[str, Behaviour]:
        return name, program_behaviour(archive, name, loop=reference, log=log)

    with ThreadPoolExecutor(max_workers=max(1, int(jobs))) as pool:
        return dict(pool.map(measure, names))

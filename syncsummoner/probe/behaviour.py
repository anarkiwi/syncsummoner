"""What an archived program did to the loop: how far it registers, and how pointwise it is.

Every archive was recorded while the codeframe loop played out, so the stimulus is
rebuilt from the run's seed and this costs no rig time. One native decode feeds two
regimes, and they are not the same width: the pointwise fit samples native pixels
because area-averaging mixes neighbours and co-location is what it asks about, while
registration is area-averaged, the chain having lowpassed the carrier away at native.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from dataclasses import replace
from itertools import groupby
from typing import Any, NamedTuple

import cv2  # pylint: disable=no-member  ; cv2 is a compiled extension pylint cannot introspect
import numpy as np

from syncsummoner.aesthetics.structure import pointwise_r2
from syncsummoner.probe import codeframes
from syncsummoner.probe.archive import GAP
from syncsummoner.probe.harvest import HarvestConfig, stimulus_loop

__all__ = ["BINS", "Behaviour", "Dwell", "behaviour", "dwells", "positions"]

#: Dwells read per program, spread over the sweep, and the run of captures each spans.
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
    """Spread dwells as ``(start, count)`` spans of stream positions.

    A dwell is a run of consecutive frames held at one setpoint, so its captures are
    contiguous in time and taken under one vector. Runs cut short by dropped frames
    are passed over while any full one remains.
    """
    runs = [
        (group[0][0], len(group))
        for setpoint, group in (
            (k, list(g)) for k, g in groupby(enumerate(rows), key=lambda item: item[1].setpoint)
        )
        if setpoint != GAP
    ]
    usable = [run for run in runs if run[1] >= frames // 2] or runs
    return [(usable[i][0], min(usable[i][1], frames)) for i in _picks(len(usable), samples)]


class Dwell(NamedTuple):
    """Stream positions of one spread dwell's captures that each fit reads."""

    pointwise: np.ndarray
    registered: np.ndarray


def dwells(rows: Sequence[Any]) -> list[Dwell]:
    """The spread dwells of a program, as the native stream positions each fit reads."""
    return [
        Dwell(start + _picks(count, POINTWISE_SAMPLES), start + _picks(count, codeframes.REGISTER_SAMPLES))
        for start, count in _dwells(rows)
    ]


def positions(spread: Iterable[Dwell]) -> set[int]:
    """Every native stream position the dwells read, so one decode can keep just those."""
    return {int(p) for dwell in spread for part in dwell for p in part}


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
    """Best pointwise fit any loop position offers, median over the captures given.

    Best over positions rather than the temporally aligned one: a program may delay or
    freeze, and this exists to say whether a value curve could explain the output at
    all, not to date it.
    """
    scores = []
    for capture in captures:
        pixels = capture.reshape(-1, 3)[index].astype(np.float32) / np.float32(255.0)
        scores.append(max(pointwise_r2(source, pixels, bins=BINS) for source in sources))
    return float(np.median(scores)) if scores else 0.0


def _registered(captures: Sequence[np.ndarray], loop: codeframes.CodeLoop, size: tuple) -> float:
    """Median strength the loop's carrier is recovered at, over the captures given.

    The carrier is common to every loop position, so registration needs no index and a
    few frames settle it; the magnitude is signless because an inverting program still
    registers.
    """
    fits = [codeframes.register(_reduced(capture, size), loop) for capture in captures]
    return float(np.median([abs(fit.correlation) for fit in fits]))


def behaviour(
    spread: Sequence[Dwell],
    frames: Mapping[int, np.ndarray],
    *,
    loop: codeframes.CodeLoop | None = None,
    seed: int = SEED,
) -> Behaviour:
    """The two fields only the frames carry, from the native captures at ``positions``.

    ``frames`` maps stream position to the native capture there. Raises
    ``LookupError`` when there is no measured dwell or none of its frames is held,
    which a caller reports as the profile's own defaults.
    """
    if not spread:
        raise LookupError("no measured dwell")
    reference = stimulus_loop(HarvestConfig()) if loop is None else loop
    rng = np.random.default_rng(seed)
    sources: list[np.ndarray] | None = None
    index, size, reduced = np.empty(0, dtype=np.intp), (0, 0), reference
    rows = []
    for dwell in spread:
        pointwise = [np.asarray(frames[int(p)]) for p in dwell.pointwise if int(p) in frames]
        registered = [np.asarray(frames[int(p)]) for p in dwell.registered if int(p) in frames]
        if not pointwise or not registered:
            continue
        if sources is None:
            shape = pointwise[0].shape
            pixels = int(shape[0]) * int(shape[1])
            index = rng.choice(pixels, size=min(PIXEL_SAMPLES, pixels), replace=False)
            sources = _sources(reference, shape, index)
            size = _target(shape)
            reduced = _scaled(reference, size[0])
        rows.append((_registered(registered, reduced, size), _pointwise(pointwise, sources, index)))
    if not rows:
        raise LookupError("no frames decoded")
    return Behaviour(
        registered=float(np.median([row[0] for row in rows])),
        pointwise=float(np.median([row[1] for row in rows])),
    )

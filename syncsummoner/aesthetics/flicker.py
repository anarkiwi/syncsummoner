"""Whether temporal change is a whole-frame level shift or spatial structure.

Cycling a value remap can only brighten and darken the picture, which reads as
a blinking light; cycling a spatial transform moves content, which reads as
motion. The split decides how fast a parameter may usefully be driven.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from syncsummoner.aesthetics.levels import luma_709

#: Below this the frames are not changing enough to classify at all.
QUIET = 1e-5


@dataclass(frozen=True)
class TemporalSplit:
    """How a clip's temporal change divides between level and structure."""

    flicker: float
    structure: float
    energy: float

    @property
    def quiet(self) -> bool:
        """True when nothing is moving enough to classify."""
        return self.energy < QUIET


def temporal_split(frames: np.ndarray, *, luma_only: bool = True) -> TemporalSplit:
    """Split frame-to-frame change into whole-frame level and residual structure.

    The per-frame mean of a difference is the level component; what remains is
    spatial. Their shares sum to one, so either one names the behaviour.
    """
    stack = np.asarray(frames, dtype=np.float32)
    if stack.ndim == 4 and luma_only:
        stack = luma_709(stack)
    if stack.shape[0] < 2:
        return TemporalSplit(0.0, 0.0, 0.0)
    delta = np.diff(stack, axis=0)
    axes = tuple(range(1, delta.ndim))
    energy = float((delta**2).mean())
    if energy < QUIET:
        return TemporalSplit(0.0, 0.0, energy)
    level = delta.mean(axis=axes)
    flicker = float(np.clip((level**2).mean() / energy, 0.0, 1.0))
    return TemporalSplit(flicker, 1.0 - flicker, energy)


def flicker_ratio(frames: np.ndarray) -> float:
    """Share of temporal change carried by whole-frame level; 1.0 is pure flicker."""
    return temporal_split(frames).flicker


def usable_rate_hz(
    split: TemporalSplit, *, flicker_limit: float = 0.5, slow_hz: float = 1.0, fast_hz: float = 15.0
) -> float:
    """Fastest modulation rate this response stays legible at.

    Flicker-dominated change gains nothing above a low rate and becomes a
    blinking light; structured change reads as motion up to the frame-rate
    Nyquist the design already imposes.
    """
    if split.quiet:
        return slow_hz
    if split.flicker >= flicker_limit:
        return slow_hz
    reach = (flicker_limit - split.flicker) / flicker_limit
    return float(slow_hz + (fast_hz - slow_hz) * reach)

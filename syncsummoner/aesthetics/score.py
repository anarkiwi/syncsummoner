"""Clip descriptor and the weighted aggregate score over it.

Objective terms follow design section 4.5: penalize mud, illegal levels and
boredom; reward natural spectral statistics, motion and surprisal. Unbounded
quantities are mapped through ``x / (1 + x)`` so no term needs a tuned scale.
"""

import sys
from dataclasses import dataclass, fields

import numpy as np

from syncsummoner.aesthetics.channels import gabor_energy
from syncsummoner.aesthetics.dynamics import DynamicsResult, analyze_dynamics
from syncsummoner.aesthetics.levels import LevelStats, level_stats, passthrough_distance
from syncsummoner.aesthetics.motion import MotionStats, motion_stats
from syncsummoner.aesthetics.spectrum import spectral_stats
from syncsummoner.aesthetics.surprisal import information_content

NATURAL_SLOPE_BAND = (-1.4, -1.0)
PREFERRED_FRACTAL_BAND = (1.3, 1.5)


@dataclass(frozen=True)
class ScoreWeights:
    """Relative weights of the aggregate score terms; all non-negative."""

    mud: float = 1.0
    spectrum: float = 1.0
    fractal: float = 1.0
    legality: float = 1.0
    motion: float = 0.5
    surprisal: float = 1.0
    boredom: float = 0.5


DEFAULT_WEIGHTS = ScoreWeights()


@dataclass(frozen=True)
class ClipDescriptor:
    """Aggregate perceptual description of a frame stack."""

    analyzer_version: str
    n_frames: int
    fps: float
    channel_energy: np.ndarray
    concentration: float
    spectral_slope: float
    fractal_dimension: float
    levels: LevelStats
    motion: MotionStats
    dynamics: DynamicsResult
    information_content: np.ndarray
    passthrough_distance: float | None


def _analyzer_version() -> str:
    return str(getattr(sys.modules[__package__], "__version__", "unknown"))


def _mean_dataclass(items: list, cls: type):
    return cls(*(float(np.mean([getattr(i, f.name) for i in items])) for f in fields(cls)))


def _as_stack(frames: np.ndarray) -> np.ndarray:
    stack = np.asarray(frames, dtype=np.float32)
    if stack.ndim != 4 or stack.shape[-1] != 3 or stack.shape[0] == 0:
        raise ValueError(f"expected a non-empty (T, H, W, 3) frame stack, got shape {stack.shape}")
    return stack


def _band_distance(value: float, band: tuple[float, float]) -> float:
    return float(max(band[0] - value, 0.0, value - band[1]))


def _saturate(value: float) -> float:
    return float(value / (1.0 + value)) if value > 0.0 else 0.0


def describe_clip(
    frames: np.ndarray, *, fps: float, rng: np.random.Generator, source: np.ndarray | None = None
) -> ClipDescriptor:
    """Describe a frame stack with every per-sample metric in design section 3.5.

    Dynamics run on the recurrence series (distance to the first frame), surprisal
    on the frame-difference series.
    """
    stack = _as_stack(frames)
    channels = [gabor_energy(f) for f in stack]
    spectra = [spectral_stats(f) for f in stack]
    levels = [level_stats(f) for f in stack]
    pairs = [motion_stats(a, b) for a, b in zip(stack[:-1], stack[1:])]
    energy = np.mean([c.energy for c in channels], axis=0).astype(np.float32)
    cells = energy.size
    herfindahl = float(np.sum(np.square(energy, dtype=np.float64)))
    recurrence = np.mean(np.square(stack - stack[0]), axis=(1, 2, 3)).astype(np.float32)
    activity = np.zeros(stack.shape[0], dtype=np.float32)
    if pairs:
        activity[1:] = [m.framediff_energy for m in pairs]
    motion = _mean_dataclass(pairs, MotionStats) if pairs else MotionStats(0.0, 0.0, 0.0)
    distance = None
    if source is not None:
        src = np.asarray(source, dtype=np.float32)
        src = src[None] if src.ndim == 3 else src
        distance = float(np.mean([passthrough_distance(a, b) for a, b in zip(src, stack[: len(src)])]))
    return ClipDescriptor(
        analyzer_version=_analyzer_version(),
        n_frames=int(stack.shape[0]),
        fps=float(fps),
        channel_energy=energy,
        concentration=(herfindahl * cells - 1.0) / (cells - 1.0) if cells > 1 else 1.0,
        spectral_slope=float(np.mean([s.slope for s in spectra])),
        fractal_dimension=float(np.mean([s.fractal_dimension for s in spectra])),
        levels=_mean_dataclass(levels, LevelStats),
        motion=motion,
        dynamics=analyze_dynamics(recurrence, fps=fps),
        information_content=information_content(activity, rng=rng),
        passthrough_distance=distance,
    )


def score_clip(descriptor: ClipDescriptor, weights: ScoreWeights | None = None) -> float:
    """Weighted aggregate of a descriptor in [-1, 1]; higher is better."""
    w = weights or DEFAULT_WEIGHTS
    reward = {
        "fractal": 1.0 - _saturate(_band_distance(descriptor.fractal_dimension, PREFERRED_FRACTAL_BAND)),
        "motion": _saturate(descriptor.motion.flow_magnitude),
        "surprisal": _saturate(float(np.mean(descriptor.information_content))),
    }
    penalty = {
        "mud": float(np.clip(descriptor.concentration, 0.0, 1.0)),
        "spectrum": _saturate(_band_distance(descriptor.spectral_slope, NATURAL_SLOPE_BAND)),
        "legality": float(np.clip(descriptor.levels.clip_frac + descriptor.levels.illegal_frac, 0.0, 1.0)),
        "boredom": 1.0 - _saturate(descriptor.levels.luma_std),
    }
    total = sum(abs(getattr(w, f.name)) for f in fields(ScoreWeights))
    if total == 0.0:
        return 0.0
    signed = sum(getattr(w, k) * v for k, v in reward.items()) - sum(
        getattr(w, k) * v for k, v in penalty.items()
    )
    return float(signed / total)

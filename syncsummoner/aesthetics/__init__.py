"""Perceptual metrics for video and audio.

A library-within-a-project, built to be extracted: it imports nothing from
``syncsummoner.*``, does no I/O outside :mod:`syncsummoner.aesthetics.io`, and
keeps no global RNG state. Boundary types are pinned in ``docs/contracts.md``.
"""

from syncsummoner.aesthetics.channels import ChannelEnergy, gabor_energy, gabor_bank
from syncsummoner.aesthetics.dynamics import (
    DynamicsResult,
    StabilityClass,
    analyze_dynamics,
    winding_number,
)
from syncsummoner.aesthetics.levels import LevelStats, level_stats, passthrough_distance
from syncsummoner.aesthetics.motion import MotionStats, motion_stats
from syncsummoner.aesthetics.safety import (
    FlashRisk,
    dead_output,
    flash_risk,
    mitigate_flashes,
    relative_luminance,
)
from syncsummoner.aesthetics.score import (
    ClipDescriptor,
    ScoreWeights,
    describe_clip,
    score_clip,
)
from syncsummoner.aesthetics.spectrum import SpectralStats, spectral_stats
from syncsummoner.aesthetics.structure import PointwiseFit, pointwise_fit, pointwise_r2
from syncsummoner.aesthetics.surprisal import (
    SurprisalModel,
    entropy_rate,
    information_content,
)
from syncsummoner.aesthetics.sync import SyncResult, av_correlation

__version__ = "0.1.0"

__all__ = [
    "ChannelEnergy",
    "ClipDescriptor",
    "DynamicsResult",
    "FlashRisk",
    "LevelStats",
    "MotionStats",
    "PointwiseFit",
    "ScoreWeights",
    "SpectralStats",
    "StabilityClass",
    "SurprisalModel",
    "SyncResult",
    "analyze_dynamics",
    "av_correlation",
    "dead_output",
    "describe_clip",
    "entropy_rate",
    "flash_risk",
    "gabor_bank",
    "gabor_energy",
    "information_content",
    "level_stats",
    "mitigate_flashes",
    "motion_stats",
    "passthrough_distance",
    "pointwise_fit",
    "pointwise_r2",
    "relative_luminance",
    "score_clip",
    "spectral_stats",
    "winding_number",
    "__version__",
]

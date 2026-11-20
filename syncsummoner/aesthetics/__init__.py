"""Perceptual metrics for video and audio.

A library-within-a-project, built to be extracted: it imports nothing from
``syncsummoner.*``, does no I/O outside :mod:`syncsummoner.aesthetics.io`, and
keeps no global RNG state. Boundary types are pinned in ``docs/contracts.md``.
"""

from syncsummoner.aesthetics.channels import ChannelEnergy, gabor_energy, gabor_bank
from syncsummoner.aesthetics.departure import DELTA_E_SPAN, Effect, delta_e, effect, ssim
from syncsummoner.aesthetics.flicker import (
    TemporalSplit,
    flicker_ratio,
    temporal_split,
    usable_rate_hz,
)
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
    transitions,
)
from syncsummoner.aesthetics.score import (
    ClipDescriptor,
    ScoreWeights,
    describe_clip,
    score_clip,
)
from syncsummoner.aesthetics.spectrum import SpectralStats, spectral_stats
from syncsummoner.aesthetics.structure import PointwiseFit, pointwise_fit, pointwise_r2
from syncsummoner.aesthetics.survival import (
    LEADS,
    Survival,
    critical_z,
    effective_samples,
    follow_z,
    QUANTUM,
    information,
    live_floor,
    quantisation_si,
    quantisation_ti,
    spatial_information,
    survival_stats,
    temporal_information,
)
from syncsummoner.aesthetics.surprisal import (
    SurprisalModel,
    entropy_rate,
    information_content,
)
from syncsummoner.aesthetics.sync import SyncResult, av_correlation

__version__ = "0.2.0"

__all__ = [
    "DELTA_E_SPAN",
    "Effect",
    "delta_e",
    "effect",
    "ssim",
    "transitions",
    "ChannelEnergy",
    "ClipDescriptor",
    "DynamicsResult",
    "QUANTUM",
    "FlashRisk",
    "LevelStats",
    "MotionStats",
    "PointwiseFit",
    "ScoreWeights",
    "SpectralStats",
    "StabilityClass",
    "SurprisalModel",
    "Survival",
    "TemporalSplit",
    "SyncResult",
    "analyze_dynamics",
    "av_correlation",
    "critical_z",
    "LEADS",
    "dead_output",
    "describe_clip",
    "effective_samples",
    "entropy_rate",
    "flicker_ratio",
    "flash_risk",
    "follow_z",
    "gabor_bank",
    "gabor_energy",
    "information",
    "live_floor",
    "information_content",
    "level_stats",
    "mitigate_flashes",
    "motion_stats",
    "passthrough_distance",
    "pointwise_fit",
    "pointwise_r2",
    "quantisation_si",
    "quantisation_ti",
    "relative_luminance",
    "score_clip",
    "spatial_information",
    "spectral_stats",
    "survival_stats",
    "temporal_information",
    "temporal_split",
    "usable_rate_hz",
    "winding_number",
    "__version__",
]

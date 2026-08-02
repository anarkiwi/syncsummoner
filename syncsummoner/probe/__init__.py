"""Probe: stimulus generation, sweep plans, measurement, simulation, profile fitting.

Nothing here imports :mod:`syncsummoner.aesthetics` at module scope; the analyzer is
injected, so a probe run can be re-scored under a different metrics version.
"""

from syncsummoner.probe import archive, codeframes, fit, patterns, plans, runner, sim, store
from syncsummoner.probe.archive import FrameArchive, FrameReader, FrameRow, FrameWriter
from syncsummoner.probe.codeframes import (
    ChannelState,
    CodeLoop,
    LoopReport,
    Verdict,
    analyse,
    build_loop,
)
from syncsummoner.probe.fit import fit_profile, load_profile, save_measurements, save_profile
from syncsummoner.probe.patterns import PATTERNS, crop_strip, generate, read_state_index, with_state_index
from syncsummoner.probe.plans import hysteresis, oat, sobol, tongue_raster
from syncsummoner.probe.runner import run_plan
from syncsummoner.probe.store import KeyKind, ProgramKey, ResultStore, program_key

__all__ = [
    "PATTERNS",
    "ChannelState",
    "CodeLoop",
    "FrameArchive",
    "FrameReader",
    "FrameRow",
    "FrameWriter",
    "KeyKind",
    "LoopReport",
    "ProgramKey",
    "ResultStore",
    "Verdict",
    "analyse",
    "archive",
    "build_loop",
    "codeframes",
    "crop_strip",
    "fit",
    "fit_profile",
    "generate",
    "hysteresis",
    "load_profile",
    "oat",
    "patterns",
    "plans",
    "program_key",
    "read_state_index",
    "run_plan",
    "runner",
    "save_measurements",
    "save_profile",
    "sim",
    "sobol",
    "store",
    "tongue_raster",
    "with_state_index",
]

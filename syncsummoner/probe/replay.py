"""Recompute measurements from an archived run, so a re-fit costs no rig time.

Each program is one in-order decode, at the analysis width for the metrics and at
native size for the dwell captures the behaviour fits read, run whole in a worker
process; the parent only gathers what the workers return. A footage archive's setpoints
are also scored on what they kept of the clip frames their captures show.
"""

from __future__ import annotations

import importlib
import multiprocessing
import os
from contextlib import closing, contextmanager
from concurrent.futures import Executor, ProcessPoolExecutor, as_completed
from dataclasses import replace
from pathlib import Path
from types import ModuleType
from typing import Any, Callable, Iterable, Iterator, NamedTuple

import cv2  # pylint: disable=no-member  ; cv2 is a compiled extension pylint cannot introspect
import numpy as np
from scipy.stats import norm
from threadpoolctl import threadpool_limits

from syncsummoner import aesthetics
from syncsummoner.device.declared import UndeclaredProgramError
from syncsummoner.device.profile import MeasurementRecord, ProgramProfile, Source
from syncsummoner.device.recorder import INSPECT_SIZE
from syncsummoner.probe import behaviour as B
from syncsummoner.probe.archive import GAP, is_footage, stimulus_of
from syncsummoner.probe.fit import fit_profile
from syncsummoner.probe.footage import (
    ClockModel,
    attribute,
    clock_model,
    motion_model,
    source_frame,
    source_series,
    unit_frames,
)
from syncsummoner.probe.harvest import HarvestConfig, stimulus_loop
from syncsummoner.probe.names import archived_specs
from syncsummoner.probe.patterns import read_timecode
from syncsummoner.probe.runner import ANALYSIS_WIDTH, measurement
from syncsummoner.progress import LOG, track

__all__ = [
    "STIMULUS",
    "Replayed",
    "clip_series",
    "group",
    "program_records",
    "replay",
    "replay_program",
    "setpoints",
    "pooled_follow",
    "select_shift",
    "survival_metrics",
]

#: What an archive without a stimulus record played out; the loop is invariant across programs and setpoints.
STIMULUS = "codeframes"
#: Thread pools a worker's native libraries size from the environment as they load.
SINGLE_THREADED = {name: "1" for name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS")}


class Replayed(NamedTuple):
    """One program's refit: its measurements, its fit, its loop behaviour, and notes."""

    records: list[MeasurementRecord]
    profile: ProgramProfile | None
    behaviour: B.Behaviour
    notes: list[str]


def group(pairs: Iterable[tuple[Any, np.ndarray]]) -> Iterator[tuple[int, tuple[int, ...], list[np.ndarray]]]:
    """Group ``(row, frame)`` pairs into ``(setpoint, params, frames)``, in stream order.

    Frames the sweep captured between setpoints carry :data:`GAP` and are
    dropped: they were taken while the parameters were still moving.
    """
    current: list[np.ndarray] = []
    setpoint, params = None, ()
    for row, frame in pairs:
        if row.setpoint == GAP:
            continue
        if row.setpoint != setpoint:
            if current:
                yield setpoint, params, current
            setpoint, params, current = row.setpoint, row.params, []
        current.append(np.asarray(frame, dtype=np.float32) / np.float32(255.0))
    if current:
        yield setpoint, params, current


def setpoints(
    reader: Any, *, width: int = ANALYSIS_WIDTH
) -> Iterator[tuple[int, tuple[int, ...], list[np.ndarray]]]:
    """Group one archive's stream at ``width`` into ``(setpoint, params, frames)``."""
    return group(zip(reader.rows, reader.stream(width=width)))


def _records(groups: Iterable, reader: Any, program: str, *, analyzer: Any, source: Source, **metrics: Any):
    """One measurement record per setpoint group, stamped with the archive's firmware."""
    firmware = str(reader.meta.get("key_material", "unknown"))
    return [
        measurement(
            frames,
            analyzer,
            program=program,
            firmware=firmware,
            source=source,
            params=params,
            state_index=setpoint,
            stimulus=str(stimulus_of(reader.meta).get("kind", STIMULUS)),
            **metrics,
        )
        for setpoint, params, frames in groups
    ]


def program_records(
    archive: Any,
    program: str,
    *,
    analyzer: Any,
    source: Source = Source.HW,
    analysis_width: int = ANALYSIS_WIDTH,
    **metrics: Any,
) -> list[MeasurementRecord]:
    """Measurements for every setpoint of one archived program."""
    reader = archive.reader(program)
    if reader is None:
        return []
    groups = setpoints(reader, width=analysis_width)
    return _records(
        groups, reader, program, analyzer=analyzer, source=source, analysis_width=analysis_width, **metrics
    )


def clip_series(archive: Any, meta: dict[str, Any]) -> tuple[np.ndarray, np.ndarray]:
    """SI and TI of every frame of the clip a footage archive played, stored beside it."""
    stimulus = stimulus_of(meta)
    return source_series(Path(archive.directory) / stimulus["clip"], strip_px=int(stimulus["strip_px"]))


def _survival_frames(
    frames: Iterable[np.ndarray], into: list, keep: np.ndarray | None = None
) -> Iterator[np.ndarray]:
    """Pass frames through, keeping those ``keep`` marks (all by default) as 8-bit luma at the series' size.

    A frame not kept is held as None, so indices still follow the stream.
    """
    for index, frame in enumerate(frames):
        if keep is not None and not (index < keep.size and keep[index]):
            into.append(None)
        else:
            luma = cv2.cvtColor(np.ascontiguousarray(frame), cv2.COLOR_RGB2GRAY)  # pylint: disable=no-member
            small = cv2.resize(luma, INSPECT_SIZE, interpolation=cv2.INTER_AREA)  # pylint: disable=no-member
            into.append(small)
        yield frame


def _clock(
    times: np.ndarray, codes: np.ndarray, frames: list, stimulus: dict, source_ti: np.ndarray
) -> tuple[ClockModel | None, str]:
    """The player's clock from the strip where it reads, else from the motion, and which it was."""
    fps, period = float(stimulus["fps"]), int(stimulus["frames"])
    model = clock_model(times, codes, fps=fps, bits=int(stimulus["bits"]), period=period)
    if model is not None:
        return model, "strip"
    if len(frames) > 1:
        _, per_capture = aesthetics.information(unit_frames(frames))
        model = motion_model(times, per_capture, source_ti, fps=fps, period=period)
    return (model, "motion") if model is not None else (None, "none")


def _measured(frames: list, captures: np.ndarray, index: np.ndarray, series: tuple, period: int) -> tuple:
    """Source and output SI and TI over the clip frames ``captures`` show at unwrapped ``index``.

    SI is read at each capture and TI against the capture of the clip frame before.
    """
    order = np.argsort(index)
    captures, span = captures[order], index[order] - index[order[0]]
    unit = [frames[c].astype(np.float32) / np.float32(255.0) for c in captures]
    out_si, out_ti = np.full(int(span[-1]) + 1, np.nan), np.full(int(span[-1]) + 1, np.nan)
    out_si[span] = [aesthetics.spatial_information(frame) for frame in unit]
    for k in np.flatnonzero(np.diff(span) == 1) + 1:
        out_ti[span[k]] = aesthetics.temporal_information(unit[k - 1], unit[k])
    clip = (index[order[0]] + np.arange(span[-1] + 1)) % period
    return series[0][clip], series[1][clip], out_si, out_ti


def pooled_follow(measured: list[tuple], ambiguity: int) -> np.ndarray:
    """Per shift ``a`` back from the highest candidate, the run's settings' follow evidence pooled into one z.

    Each setting is scored at that shift's own leads; its best lead's z, Bonferroni over those leads, is
    a one-sided p, and the settings' p values combine by Fisher in logs (:func:`matched.fisher_z_log`).
    """
    from syncsummoner.compose.matched import fisher_z_log  # pylint: disable=import-outside-toplevel

    leads = len(aesthetics.LEADS)
    out = np.zeros(int(ambiguity) + 1)
    for a in range(out.size):
        found = [
            aesthetics.survival_stats(*m, leads=[lead + a for lead in aesthetics.LEADS]) for m in measured
        ]
        z = np.array([f.z for f in found if f.frames > 2 and np.isfinite(f.z)])
        log_p = np.minimum(0.0, np.log(leads) + norm.logsf(z))
        out[a] = fisher_z_log(log_p)[0] if log_p.size else 0.0
    return out


def select_shift(pooled: np.ndarray) -> dict:
    """The shift whose pooled follow beats every other's past the bound for a difference, if one does.

    A difference of two unit z scores has twice their variance; the bound is Bonferroni over the shifts.
    """
    from syncsummoner.compose.matched import critical  # pylint: disable=import-outside-toplevel

    best = int(np.argmax(pooled))
    others = np.delete(pooled, best)
    margin = float(np.min(pooled[best] - others) / np.sqrt(2.0)) if others.size else float("inf")
    bound = critical(pooled.size)
    return {
        "pooled_z": [float(z) for z in pooled],
        "shift": best if margin > bound else None,
        "margin_z": margin,
        "critical_z": bound,
    }


def _kept(
    frames: list, captures: np.ndarray, index: np.ndarray, series: tuple, period: int, ambiguity: int = 0
) -> dict:
    """Survival of the clip frames ``captures`` show at unwrapped ``index``, against those same source frames.

    As in a take check, SI is read at each capture and TI against the capture of the clip frame
    before; the follow is left out where too few frames moved to correlate. ``index`` may be up to
    ``ambiguity`` frames late, so the leads tried are each candidate's, one search at the bound for them all;
    where the candidates' own searches disagree on following, the group is attribution-ambiguous and kept
    out. The best lead, a program-delay measurement, is the interval its candidates allow, in source frames.
    """
    if captures.size == 0:
        return {}
    measured = _measured(frames, captures, index, series, period)
    shifts = range(int(ambiguity) + 1)
    union = sorted({lead + a for lead in aesthetics.LEADS for a in shifts})
    bound = aesthetics.critical_z(len(union))
    kept = aesthetics.survival_stats(*measured, leads=union)
    verdicts = {
        aesthetics.survival_stats(*measured, leads=[lead + a for lead in aesthetics.LEADS]).z > bound
        for a in shifts
    }
    if kept.frames > 2 and len(verdicts) > 1:
        return {"attribution_ambiguous": 1.0}
    found = {"si_ratio": kept.si_ratio, "ti_ratio": kept.ti_ratio}
    if kept.frames > 2:
        found.update(
            {
                "motion_follow": kept.follow,
                "motion_lead_lo": kept.lead - int(ambiguity),
                "motion_lead_hi": kept.lead,
            }
        )
    return {name: float(value) for name, value in found.items() if np.isfinite(value)}


def survival_metrics(
    rows: list, frames: list, strips: list, stimulus: dict, series: tuple
) -> tuple[list[dict[str, float]], str]:
    """Per setpoint group, in :func:`group` order, what it kept of the clip, and how the clock was found.

    A locked take names the frame it played of every capture it keeps, lead-in included, and the
    series starts at clip frame 0 (:func:`footage.source_series`); otherwise captures are attributed
    by the fitted clock and ordered unwrapped, so a dwell across the loop's wrap stays consecutive.
    With neither, every group comes back empty.
    """
    rows = rows[: len(frames)]
    setpoint = np.array([row.setpoint for row in rows], dtype=np.int64)
    live = np.flatnonzero(setpoint != GAP)
    groups = np.split(live, np.flatnonzero(np.diff(setpoint[live]) != 0) + 1) if live.size else []
    named = np.array([row.shown for row in rows], dtype=np.int64)
    period = int(stimulus["frames"])
    if live.size and np.all(named[live] >= 0):
        late = np.array([getattr(row, "ambiguity", 0) for row in rows], dtype=np.int64)
        clip = source_frame(named, int(stimulus.get("lead", 0)))
        return [
            _kept(frames, members, clip[members], series, period, int(late[members].max()))
            for members in groups
        ], "take"
    times = np.array([row.captured for row in rows], dtype=np.float64)
    bits, strip_px = int(stimulus["bits"]), int(stimulus["strip_px"])
    codes = np.array(
        [
            -1 if code is None else code
            for strip in strips[: len(frames)]
            for code in [read_timecode(strip.astype(np.float32) / 255.0, bits=bits, strip_px=strip_px)]
        ],
        dtype=np.int64,
    )
    model, how = _clock(times, codes, frames, stimulus, series[1])
    if model is None:
        return [{} for _ in groups], how
    shown = attribute(times, model)
    unwrapped = np.rint((times - model.origin) / model.interval).astype(np.int64)
    return [
        _kept(frames, picked, unwrapped[picked], series, period)
        for members in groups
        for picked in [members[shown[members] >= 0]]
    ], how


def replay_program(
    archive: Any,
    program: str,
    *,
    analyzer: Any,
    loop: Any = None,
    source: Source = Source.HW,
    analysis_width: int = ANALYSIS_WIDTH,
    series: tuple[np.ndarray, np.ndarray] | None = None,
    **metrics: Any,
) -> Replayed:
    """Measure, fit and read the loop behaviour of one program from a single decode.

    A failure is a note rather than the end of a library refit. An ``analyzer`` given as a module
    path is imported here, since a module cannot be sent to a worker. A footage archive is scored
    against ``series``, its clip's :func:`clip_series`, read here when not given.
    """
    if isinstance(analyzer, str):
        analyzer = importlib.import_module(analyzer)
    notes: list[str] = []
    found, records, profile = B.Behaviour(), [], None
    reader = archive.reader(program)
    if reader is None:
        return Replayed(records, profile, found, [f"{program}: no committed archive"])
    footage, stimulus = is_footage(reader.meta), stimulus_of(reader.meta)
    live = np.array([row.setpoint != GAP for row in reader.rows], dtype=bool)
    named = bool(live.any()) and all(row.shown >= 0 for row in reader.rows if row.setpoint != GAP)
    spread = [] if footage else B.dwells(reader.rows)
    natives: dict[int, np.ndarray] = {}
    strips: list[np.ndarray] = []
    small: list[np.ndarray] = []
    try:
        with closing(
            reader.passes(
                width=analysis_width,
                native=B.positions(spread),
                into=natives,
                strip_px=int(stimulus.get("strip_px", 0)) if footage else 0,
                strips=strips if footage and not named else None,
            )
        ) as passes:
            frames = _survival_frames(passes, small, live if named else None) if footage else passes
            records = _records(
                group((row, frame) for frame, row in zip(frames, reader.rows)),
                reader,
                program,
                analyzer=analyzer,
                source=source,
                analysis_width=analysis_width,
                **metrics,
            )
        if footage:
            notes.append(f"{program}: loop behaviour reads the codeframe loop, which footage does not play")
        else:
            found = B.behaviour(spread, natives, loop=loop)
    except Exception as exc:
        notes.append(f"{program}: loop behaviour not measured ({type(exc).__name__}: {exc})")
    if footage and records:
        try:
            kept, how = survival_metrics(
                reader.rows,
                small,
                strips,
                stimulus,
                clip_series(archive, reader.meta) if series is None else series,
            )
            records = [replace(r, metrics={**r.metrics, **extra}) for r, extra in zip(records, kept)]
            notes.append(f"{program}: footage clock from {how}")
        except Exception as exc:
            notes.append(f"{program}: survival not measured ({type(exc).__name__}: {exc})")
    if records:
        try:
            specs = archived_specs(reader.meta, program)
        except UndeclaredProgramError as exc:
            specs = None
            notes.append(f"{program}: {exc}")
        else:
            if specs is None:
                notes.append(f"{program}: no program info archived, so parameters stay unnamed")
        profile = fit_profile(records, specs=specs)
    return Replayed(records, profile, found, notes)


@contextmanager
def _environment(values: dict[str, str]) -> Iterator[None]:
    """Set ``values`` in the environment while the block runs, then put it back."""
    saved = {name: os.environ.get(name) for name in values}
    os.environ.update(values)
    try:
        yield
    finally:
        for name, value in saved.items():
            if value is None:
                os.environ.pop(name, None)
            else:
                os.environ[name] = value


def _single_threaded() -> None:
    """Hold a worker's native libraries to one thread, the pool being the parallelism."""
    cv2.setNumThreads(1)  # pylint: disable=no-member
    threadpool_limits(1)


def replay(
    archive: Any,
    *,
    analyzer: Any,
    programs: Iterable[str] | None = None,
    log: Callable[[str], None] | None = None,
    jobs: int | None = None,
    loop: Any = None,
    executor: Callable[..., Executor] = ProcessPoolExecutor,
    **kwargs: Any,
) -> dict[str, Replayed]:
    """Every named program's refit, keyed by name; all committed ones by default.

    ``jobs`` workers each take a program whole, one per usable core by default.
    A program that measured nothing is noted and left out.
    """
    committed = archive.committed()
    names = sorted(committed) if programs is None else list(programs)
    note = log if log is not None else lambda _message: None
    reference = stimulus_loop(HarvestConfig()) if loop is None else loop
    clips = {
        name: stimulus_of(committed[name])["clip"]
        for name in names
        if name in committed and is_footage(committed[name])
    }
    played: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    for name, clip in clips.items():
        if clip not in played:
            played[clip] = clip_series(archive, committed[name])
    workers = max(1, min(int(jobs or len(os.sched_getaffinity(0))), len(names) or 1))
    # A forked child inherits OpenCV's thread pool mid-state and can hang in it; a fresh one, started
    # under SINGLE_THREADED, also loads its BLAS without a pool per core.
    options = (
        {"initializer": _single_threaded, "mp_context": multiprocessing.get_context("forkserver")}
        if executor is ProcessPoolExecutor
        else {}
    )
    shipped = analyzer.__name__ if isinstance(analyzer, ModuleType) else analyzer
    out = {}
    with _environment(SINGLE_THREADED), executor(max_workers=workers, **options) as pool:
        futures = {
            pool.submit(
                replay_program,
                archive,
                name,
                analyzer=shipped,
                loop=reference,
                series=played.get(clips.get(name)),
                **kwargs,
            ): name
            for name in names
        }
        progress = track(as_completed(futures), desc="refit", total=len(names), unit="program")
        for done, future in enumerate(progress, 1):
            name, result = futures[future], future.result()
            for message in result.notes:
                note(message)
            if result.records:
                out[name] = result
            LOG.info("%s: %d setpoints measured, %d/%d", name, len(result.records), done, len(names))
    return out

"""Recompute measurements from an archived run, so a re-fit costs no rig time.

Each program is one in-order decode, at the analysis width for the metrics and at
native size for the dwell captures the behaviour fits read, run whole in a worker
process; the parent only gathers what the workers return.
"""

from __future__ import annotations

import importlib
import multiprocessing
import os
from contextlib import closing
from concurrent.futures import Executor, ProcessPoolExecutor, as_completed
from types import ModuleType
from typing import Any, Callable, Iterable, Iterator, NamedTuple

import cv2  # pylint: disable=no-member  ; cv2 is a compiled extension pylint cannot introspect
import numpy as np
from threadpoolctl import threadpool_limits

from syncsummoner.device.profile import MeasurementRecord, ProgramProfile, Source
from syncsummoner.probe import behaviour as B
from syncsummoner.probe.archive import GAP
from syncsummoner.probe.fit import fit_profile
from syncsummoner.probe.harvest import HarvestConfig, stimulus_loop
from syncsummoner.probe.names import archived_specs
from syncsummoner.probe.runner import ANALYSIS_WIDTH, measurement
from syncsummoner.progress import LOG, track

__all__ = ["STIMULUS", "Replayed", "group", "program_records", "replay", "replay_program", "setpoints"]

#: What the archive run plays out; the loop is invariant across programs and setpoints.
STIMULUS = "codeframes"


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
            stimulus=STIMULUS,
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


def replay_program(
    archive: Any,
    program: str,
    *,
    analyzer: Any,
    loop: Any = None,
    source: Source = Source.HW,
    analysis_width: int = ANALYSIS_WIDTH,
    **metrics: Any,
) -> Replayed:
    """Measure, fit and read the loop behaviour of one program from a single decode.

    A failure is a note rather than the end of a library refit, and the program
    comes back with whatever was measured before it. An ``analyzer`` given as a
    module path is imported here, since a module cannot be sent to a worker.
    """
    if isinstance(analyzer, str):
        analyzer = importlib.import_module(analyzer)
    notes: list[str] = []
    found, records, profile = B.Behaviour(), [], None
    reader = archive.reader(program)
    if reader is None:
        return Replayed(records, profile, found, [f"{program}: no committed archive"])
    spread = B.dwells(reader.rows)
    natives: dict[int, np.ndarray] = {}
    try:
        with closing(reader.passes(width=analysis_width, native=B.positions(spread), into=natives)) as passes:
            records = _records(
                group((row, frame) for frame, row in zip(passes, reader.rows)),
                reader,
                program,
                analyzer=analyzer,
                source=source,
                analysis_width=analysis_width,
                **metrics,
            )
        found = B.behaviour(spread, natives, loop=loop)
    except Exception as exc:
        notes.append(f"{program}: loop behaviour not measured ({type(exc).__name__}: {exc})")
    if records:
        specs = archived_specs(reader.meta)
        if specs is None:
            notes.append(f"{program}: no program info archived, so parameters stay unnamed")
        profile = fit_profile(records, specs=specs)
    return Replayed(records, profile, found, notes)


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
    names = sorted(archive.committed()) if programs is None else list(programs)
    note = log if log is not None else lambda _message: None
    reference = stimulus_loop(HarvestConfig()) if loop is None else loop
    workers = max(1, min(int(jobs or len(os.sched_getaffinity(0))), len(names) or 1))
    # A forked child inherits OpenCV's thread pool mid-state and can hang in it; a fresh one cannot.
    options = (
        {"initializer": _single_threaded, "mp_context": multiprocessing.get_context("forkserver")}
        if executor is ProcessPoolExecutor
        else {}
    )
    shipped = analyzer.__name__ if isinstance(analyzer, ModuleType) else analyzer
    out = {}
    with executor(max_workers=workers, **options) as pool:
        futures = {
            pool.submit(replay_program, archive, name, analyzer=shipped, loop=reference, **kwargs): name
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

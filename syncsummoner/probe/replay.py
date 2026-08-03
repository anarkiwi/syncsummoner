"""Recompute measurements from an archived run, so a re-fit costs no rig time.

The archive stores the card's own frames with the vector that produced them, and
every metric is derived here rather than on the rig. A reflash is then the only
reason to probe again, and a metric change costs a re-read instead of hours.
"""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from typing import Any, Iterable, Iterator

import numpy as np

from syncsummoner.device.profile import MeasurementRecord, Source
from syncsummoner.probe.archive import GAP
from syncsummoner.probe.runner import ANALYSIS_WIDTH, measurement

__all__ = ["STIMULUS", "program_records", "replay", "setpoints"]

#: What the archive run plays out; the loop is invariant across programs and setpoints.
STIMULUS = "codeframes"


def setpoints(
    reader: Any, *, width: int = ANALYSIS_WIDTH
) -> Iterator[tuple[int, tuple[int, ...], list[np.ndarray]]]:
    """Group one archive into ``(setpoint, params, frames)``, in stream order.

    Frames the sweep captured between setpoints carry :data:`GAP` and are
    dropped: they were taken while the parameters were still moving.
    """
    current: list[np.ndarray] = []
    setpoint, params = None, ()
    for row, frame in zip(reader.rows, reader.stream(width=width)):
        if row.setpoint == GAP:
            continue
        if row.setpoint != setpoint:
            if current:
                yield setpoint, params, current
            setpoint, params, current = row.setpoint, row.params, []
        current.append(np.asarray(frame, dtype=np.float32) / np.float32(255.0))
    if current:
        yield setpoint, params, current


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
            analysis_width=analysis_width,
            **metrics,
        )
        for setpoint, params, frames in setpoints(reader, width=analysis_width)
    ]


def replay(
    archive: Any,
    *,
    analyzer: Any,
    programs: Iterable[str] | None = None,
    log: Any = None,
    jobs: int = 1,
    **kwargs: Any,
) -> dict[str, list[MeasurementRecord]]:
    """Measurements for every committed program, keyed by program name.

    ``jobs`` reads that many programs at once: the cost is a decoder subprocess
    and array work that both drop the lock, so threads are enough to overlap them.
    """
    names = sorted(archive.committed()) if programs is None else list(programs)
    note = log if log is not None else lambda _message: None

    def measure(name: str) -> tuple[str, list[MeasurementRecord]]:
        return name, program_records(archive, name, analyzer=analyzer, **kwargs)

    out = {}
    with ThreadPoolExecutor(max_workers=max(1, int(jobs))) as pool:
        for name, records in pool.map(measure, names):
            if records:
                out[name] = records
            note(f"{name}: {len(records)} setpoints measured")
    return out

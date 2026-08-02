"""Recompute measurements from an archived run, so a re-fit costs no rig time.

The archive stores the card's own frames with the vector that produced them, and
every metric is derived here rather than on the rig. A reflash is then the only
reason to probe again, and a metric change costs a re-read instead of hours.
"""

from __future__ import annotations

import hashlib
from collections import Counter
from typing import Any, Iterable, Iterator

import numpy as np

from syncsummoner.device.capture import bgr_to_rgb, yvyu_to_bgr
from syncsummoner.device.profile import MeasurementRecord, Source
from syncsummoner.probe.runner import ANALYSIS_WIDTH, downscale, measurement

__all__ = ["STIMULUS", "blank_digest", "digest", "program_records", "replay", "setpoints"]

#: What the archive run plays out; the loop is invariant across programs and setpoints.
STIMULUS = "codeframes"


def digest(frame: np.ndarray) -> str:
    """Content digest of one native frame, for recognising a frame the device repeats."""
    return hashlib.blake2b(np.ascontiguousarray(frame).tobytes(), digest_size=8).hexdigest()


def _analysis_rgb(frame: np.ndarray, width: int) -> np.ndarray:
    """One archived frame as the analyzer's RGB, shrunk before anything accumulates."""
    return downscale([bgr_to_rgb(yvyu_to_bgr(frame))], width)[0]


def setpoints(reader: Any) -> Iterator[tuple[int, tuple[int, ...], list[np.ndarray]]]:
    """Group one archive into ``(setpoint, params, frames)``, in stream order."""
    current: list[np.ndarray] = []
    setpoint, params = None, ()
    for row, frame in zip(reader.rows, reader.stream()):
        if row.setpoint != setpoint:
            if current:
                yield setpoint, params, current
            setpoint, params, current = row.setpoint, row.params, []
        current.append(frame)
    if current:
        yield setpoint, params, current


def blank_digest(archive: Any, programs: Iterable[str]) -> str | None:
    """The frame the device emits while blanked, taken from the run rather than assumed.

    Two programs cannot both produce the same frame from different vectors, so a
    digest shared by their opening setpoints is the device's own blanking frame.
    """
    opening: Counter[str] = Counter()
    for name in programs:
        reader = archive.reader(name)
        if reader is None:
            continue
        opening.update(digest(frame) for frame in reader.stream(count=1))
    shared, seen = opening.most_common(1)[0] if opening else (None, 0)
    return shared if seen > 1 else None


def program_records(
    archive: Any,
    program: str,
    *,
    analyzer: Any,
    blank: str | None = None,
    source: Source = Source.HW,
    analysis_width: int = ANALYSIS_WIDTH,
    **metrics: Any,
) -> list[MeasurementRecord]:
    """Measurements for every setpoint of one archived program.

    A setpoint holding nothing but ``blank`` is dropped: the device was still
    blanked from the load, so the vector on those rows produced no picture.
    """
    reader = archive.reader(program)
    if reader is None:
        return []
    firmware = str(reader.meta.get("key_material", "unknown"))
    records = []
    for setpoint, params, frames in setpoints(reader):
        if blank is not None and all(digest(frame) == blank for frame in frames):
            continue
        records.append(
            measurement(
                [_analysis_rgb(frame, analysis_width) for frame in frames],
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
        )
    return records


def replay(
    archive: Any,
    *,
    analyzer: Any,
    programs: Iterable[str] | None = None,
    log: Any = None,
    **kwargs: Any,
) -> dict[str, list[MeasurementRecord]]:
    """Measurements for every committed program, keyed by program name."""
    names = sorted(archive.committed()) if programs is None else list(programs)
    note = log if log is not None else lambda _message: None
    blank = blank_digest(archive, names)
    note(f"blanking frame {blank}" if blank else "no blanking frame shared between programs")
    out = {}
    for name in names:
        records = program_records(archive, name, analyzer=analyzer, blank=blank, **kwargs)
        if records:
            out[name] = records
        note(f"{name}: {len(records)} setpoints measured")
    return out

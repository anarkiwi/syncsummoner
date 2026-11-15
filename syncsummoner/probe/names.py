"""Back-fill each archive with the device's ``program info``, so a refit can name parameters.

An archive recorded before the reply was kept still holds every frame a refit needs,
and only lacks the names and native ranges. Asking the device costs a program load
and one shell command per program, against an hour of capture to re-archive it.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any, Callable, Iterable, Mapping

from syncsummoner.device.profile import ParamSpec
from syncsummoner.device.transport import ProgramInfo
from syncsummoner.probe.archive import PROGRAM_INFO
from syncsummoner.probe.harvest import WEDGED_NOTE, wedged
from syncsummoner.progress import track

__all__ = ["NamesReport", "archived_specs", "backfill", "plain_name", "read_program_info"]


@dataclass
class NamesReport:
    """Which archives were annotated, which already were, and why any were not."""

    annotated: list[str] = field(default_factory=list)
    skipped: list[str] = field(default_factory=list)
    failed: dict[str, str] = field(default_factory=dict)
    wedged: bool = False


def archived_specs(meta: Mapping[str, Any] | None) -> list[ParamSpec] | None:
    """Parameter specs from an archive's stored ``program info``, or None where it has none."""
    payload = (meta or {}).get(PROGRAM_INFO)
    return None if payload is None else ProgramInfo.from_json(payload).params


def plain_name(program: str) -> str:
    """The name the device reports for a program, which never carries its author."""
    return program.rpartition("/")[2]


def read_program_info(
    transport: Any,
    program: str,
    *,
    timeout_s: float = 20.0,
    poll_s: float = 0.5,
    sleep: Callable[[float], None] = time.sleep,
    clock: Callable[[], float] = time.monotonic,
) -> dict[str, Any]:
    """Load ``program`` and return its ``program info`` reply once it names that program.

    A load returns before the device has switched, so a reply naming the previous
    program is stale rather than wrong, and is asked again until the deadline.
    """
    transport.load_program(program)
    deadline = clock() + timeout_s
    while True:
        payload = transport.program_payload()
        if payload.get("name") == plain_name(program):
            return payload
        if clock() >= deadline:
            raise LookupError(f"device still reports {payload.get('name')!r} after loading {program!r}")
        sleep(poll_s)


def backfill(
    archive: Any,
    transport: Any,
    *,
    programs: Iterable[str] | None = None,
    log: Callable[[str], None] | None = None,
    **read: Any,
) -> NamesReport:
    """Annotate every committed archive that lacks ``program info``, resuming where one stopped.

    The metadata is rewritten only once the reply names its program, so an archive
    either carries a reply or is asked again next run. A load that fails on a
    device that will no longer hold any program ends the run, as it does an archive run.
    """
    note = log if log is not None else lambda _message: None
    committed = archive.committed()
    names = sorted(committed) if programs is None else [p for p in programs if p in committed]
    report = NamesReport()
    for name in track(names, desc="program info", unit="program"):
        if PROGRAM_INFO in committed[name]:
            report.skipped.append(name)
            continue
        try:
            archive.annotate(name, **{PROGRAM_INFO: read_program_info(transport, name, **read)})
        except Exception as exc:
            report.failed[name] = f"{type(exc).__name__}: {exc}"
            note(f"{name}: {report.failed[name]}")
            if wedged(transport):
                report.wedged = True
                note(WEDGED_NOTE)
                break
            continue
        report.annotated.append(name)
    return report

"""Back-fill each archive with the device's ``program info``, so a refit can name parameters.

An archive recorded before the reply was kept still holds every frame a refit needs,
and only lacks the names and native ranges. Asking the device costs a program load
and one shell command per program, against an hour of capture to re-archive it.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Iterable, Mapping

from pyvmancer.errors import DeviceGoneError

from syncsummoner.device.declared import declaration
from syncsummoner.device.profile import ParamSpec
from syncsummoner.device.transport import ProgramInfo
from syncsummoner.program.library import Library
from syncsummoner.probe.archive import PROGRAM_INFO
from syncsummoner.probe.harvest import GONE_NOTE
from syncsummoner.progress import track

__all__ = ["NamesReport", "archived_specs", "backfill", "plain_name", "read_program_info"]


@dataclass
class NamesReport:
    """Which archives were annotated, which already were, and why any were not."""

    annotated: list[str] = field(default_factory=list)
    skipped: list[str] = field(default_factory=list)
    failed: dict[str, str] = field(default_factory=dict)
    gone: bool = False


def archived_specs(
    meta: Mapping[str, Any] | None, program: str, library: Library | None = None
) -> list[ParamSpec] | None:
    """Specs from an archive's stored ``program info``, typed by ``program``'s declaration; None without one.

    Raises :class:`~syncsummoner.device.declared.UndeclaredProgramError` when no
    ``.vmprog`` declares ``program``.
    """
    payload = (meta or {}).get(PROGRAM_INFO)
    return None if payload is None else ProgramInfo.from_json(payload, declaration(program, library)).params


def plain_name(program: str) -> str:
    """The name the device reports for a program, which never carries its author."""
    return program.rpartition("/")[2]


def read_program_info(transport: Any, program: str) -> dict[str, Any]:
    """Load ``program`` and return its ``program info`` reply, which must name that program.

    The load returns only once the device runs the program and answers
    ``program info``, so a reply naming another program is a wrong answer, not
    a stale one.
    """
    transport.load_program(program)
    payload = transport.program_payload()
    if payload.get("name") != plain_name(program):
        raise LookupError(f"device reports {payload.get('name')!r} after loading {program!r}")
    return payload


def backfill(
    archive: Any,
    transport: Any,
    *,
    programs: Iterable[str] | None = None,
    log: Callable[[str], None] | None = None,
) -> NamesReport:
    """Annotate every committed archive that lacks ``program info``, resuming where one stopped.

    The metadata is rewritten only once the reply names its program, so an archive
    either carries a reply or is asked again next run. The device leaving USB
    ends the run, as it does an archive run.
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
            archive.annotate(name, **{PROGRAM_INFO: read_program_info(transport, name)})
        except Exception as exc:
            report.failed[name] = f"{type(exc).__name__}: {exc}"
            note(f"{name}: {report.failed[name]}")
            if isinstance(exc, DeviceGoneError):
                report.gone = True
                note(GONE_NOTE)
                break
            continue
        report.annotated.append(name)
    return report

"""Per-program result store, so a long sweep resumes instead of restarting.

Results are keyed on the identity of the program binary, so a firmware or program
update invalidates them rather than being silently reused. Each program commits
independently and atomically, so an interrupted run keeps everything already done.
"""

from __future__ import annotations

import enum
import hashlib
import json
import os
import re
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Iterable, Mapping, Sequence

from syncsummoner.device.profile import MeasurementRecord
from syncsummoner.probe.fit import load_measurements, save_measurements

__all__ = [
    "STORE_SCHEMA_VERSION",
    "KeyKind",
    "ProgramKey",
    "ResultStore",
    "atomic_write",
    "program_key",
    "slug",
    "temp_sibling",
]

STORE_SCHEMA_VERSION = 1
#: Where the manifest's per-program file names are rooted.
PROGRAM_DIR = "sd:/programs"
_UNSAFE = re.compile(r"[^A-Za-z0-9._-]+")


class KeyKind(enum.Enum):
    """How a result was keyed. ``NAME_FIRMWARE`` is the weaker fallback."""

    BINARY = "binary"
    NAME_FIRMWARE = "name_firmware"


@dataclass(frozen=True)
class ProgramKey:
    """What a stored result is keyed on: the material, and how it was obtained."""

    program: str
    material: str
    kind: KeyKind = KeyKind.BINARY

    @property
    def digest(self) -> str:
        """Comparison digest over program, kind and material."""
        blob = "\0".join((self.program, self.kind.value, self.material)).encode("utf-8")
        return hashlib.sha256(blob).hexdigest()[:16]


def _binary_path(name: str) -> str:
    """Absolute device path of a manifest file entry."""
    return name if ":" in name or name.startswith("/") else f"{PROGRAM_DIR}/{name}"


def _firmware(transport: Any) -> str:
    try:
        return str(transport.firmware())
    except Exception:
        return "unknown"


def program_key(
    transport: Any, program: str, *, firmware: str | None = None, manifest: Any = None
) -> ProgramKey:
    """Key ``program`` by its binary digest, falling back to name plus firmware.

    Hashing runs over the serial shell at wire speed and can time out or be
    unavailable for firmware built-ins, which carry no manifest entry.
    """
    try:
        manifest = transport.program_manifest() if manifest is None else manifest
        digest = transport.file_hash(_binary_path(manifest.get(program).file))
        if digest:
            return ProgramKey(program, str(digest), KeyKind.BINARY)
    except Exception:
        pass
    return ProgramKey(
        program, _firmware(transport) if firmware is None else str(firmware), KeyKind.NAME_FIRMWARE
    )


def slug(program: str) -> str:
    """Filesystem-safe stem, disambiguated when characters had to be replaced."""
    safe = _UNSAFE.sub("_", program).strip("._-")
    if safe == program:
        return safe
    return f"{safe or 'program'}-{hashlib.sha256(program.encode('utf-8')).hexdigest()[:8]}"


def temp_sibling(path: Path) -> Path:
    """Uncommitted path beside ``path``, to be published by ``os.replace``."""
    return path.with_name(f".{path.name}.{os.getpid()}.tmp")


def atomic_write(path: Path, write: Callable[[Path], None]) -> None:
    """Write through a temp file in the same directory, then ``os.replace``."""
    tmp = temp_sibling(path)
    try:
        write(tmp)
        os.replace(tmp, path)
    finally:
        tmp.unlink(missing_ok=True)


class ResultStore:
    """One measurement artefact per program under ``directory``.

    The metadata sidecar is written last and is the commit marker: a process
    killed mid-write leaves at most an uncommitted Parquet file, which the next
    run overwrites. An artefact from a newer schema is ignored, not trusted.
    """

    def __init__(self, directory: str | os.PathLike[str], *, clock: Callable[[], float] = time.time):
        self.directory = Path(directory)
        self._clock = clock

    def paths(self, program: str) -> tuple[Path, Path]:
        """Data and metadata paths for one program."""
        stem = slug(program)
        return self.directory / f"{stem}.parquet", self.directory / f"{stem}.json"

    def meta(self, program: str) -> dict[str, Any] | None:
        """Committed metadata for ``program``, or None when nothing usable is stored."""
        data_path, meta_path = self.paths(program)
        try:
            meta = json.loads(meta_path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            return None
        if int(meta.get("schema_version", 0)) > STORE_SCHEMA_VERSION or not data_path.exists():
            return None
        return meta

    def has(self, program: str, key: ProgramKey) -> bool:
        """True when a committed result for ``program`` was keyed on ``key``."""
        meta = self.meta(program)
        return meta is not None and meta.get("key_digest") == key.digest

    def get(self, program: str, key: ProgramKey | None = None) -> list[MeasurementRecord] | None:
        """Committed records for ``program``, or None when absent or keyed differently."""
        if self.meta(program) is None or (key is not None and not self.has(program, key)):
            return None
        return [MeasurementRecord.from_row(row) for row in load_measurements(self.paths(program)[0])]

    def put(self, program: str, key: ProgramKey, records: Sequence[MeasurementRecord]) -> Path:
        """Commit ``records`` for ``program``: data first, then the metadata marker."""
        if not records:
            raise ValueError(f"no records to store for {program!r}")
        self.directory.mkdir(parents=True, exist_ok=True)
        data_path, meta_path = self.paths(program)
        atomic_write(data_path, lambda tmp: save_measurements(records, tmp))
        meta = {
            "schema_version": STORE_SCHEMA_VERSION,
            "program": program,
            "key_kind": key.kind.value,
            "key_material": key.material,
            "key_digest": key.digest,
            "firmware": records[0].firmware,
            "analyzer": records[0].analyzer,
            "records": len(records),
            "data": data_path.name,
            "created": datetime.fromtimestamp(self._clock(), timezone.utc).isoformat(),
        }
        atomic_write(meta_path, lambda tmp: tmp.write_text(json.dumps(meta, indent=2), encoding="utf-8"))
        return data_path

    def pending(
        self,
        programs: Iterable[str],
        keys: Mapping[str, ProgramKey] | Callable[[str], ProgramKey | None],
    ) -> list[str]:
        """Programs still needing work: never committed, or committed under another key.

        ``keys`` resolves a program name to its :class:`ProgramKey`, as a mapping or
        as a callable; a name that resolves to None is always pending.
        """
        lookup = keys.get if isinstance(keys, Mapping) else keys
        return [p for p in programs if (key := lookup(p)) is None or not self.has(p, key)]

    def committed(self) -> dict[str, dict[str, Any]]:
        """Metadata of every committed artefact, keyed by program name."""
        found = {}
        for path in sorted(self.directory.glob("*.json")):
            try:
                program = str(json.loads(path.read_text(encoding="utf-8")).get("program", ""))
            except (OSError, ValueError):
                continue
            meta = self.meta(program) if program else None
            if meta is not None:
                found[program] = meta
        return found

"""Rebuild ``programs/manifest.json`` from the ``.vmprog`` files on a card.

The metadata is read from each package's config block (layout: Videomancer SDK
``docs/vmprog-format.md``), so the manifest can never disagree with the files
it describes. Truncated or corrupt packages are reported rather than listed.
"""

from __future__ import annotations

import hashlib
import json
import struct
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from syncsummoner.progress import LOG, track

MAGIC = b"VMPG"
HEADER_BYTES = 64
TOC_BYTES = 64
TOC_CONFIG = 1
CONFIG_BYTES = 7936
PROGRAM_TYPES = ("processing", "synthesis")


class PackageError(ValueError):
    """A ``.vmprog`` that the device would not be able to load."""


@dataclass
class Report:
    """What a rebuild found: the manifest, and why anything was left out."""

    manifest: dict[str, Any]
    errors: dict[str, str] = field(default_factory=dict)
    shared_names: dict[str, list[str]] = field(default_factory=dict)

    @property
    def ok(self) -> bool:
        """True when every package was listed and no identity is duplicated."""
        return not self.errors


def _text(raw: bytes) -> str:
    return raw.split(b"\0", 1)[0].decode("utf-8")


def _toc(data: bytes) -> list[tuple[int, int, int, bytes]]:
    if len(data) < HEADER_BYTES or data[:4] != MAGIC:
        raise PackageError("not a vmprog")
    size = struct.unpack_from("<I", data, 12)[0]
    toc_offset, toc_bytes, count = struct.unpack_from("<3I", data, 20)
    if size != len(data):
        raise PackageError(f"header says {size} bytes, file has {len(data)}")
    if toc_bytes != TOC_BYTES * count or toc_offset + toc_bytes > len(data):
        raise PackageError("table of contents outside the file")
    entries = []
    for i in range(count):
        kind, _, offset, length = struct.unpack_from("<4I", data, toc_offset + TOC_BYTES * i)
        digest = data[toc_offset + TOC_BYTES * i + 16 : toc_offset + TOC_BYTES * i + 48]
        if offset + length > len(data):
            raise PackageError(f"section {i} runs past the end of the file")
        entries.append((kind, offset, length, digest))
    return entries


def read_config(data: bytes) -> bytes:
    """The verified config block of a package; every section's BLAKE2b-256 must match."""
    config = None
    for kind, offset, length, digest in _toc(data):
        if hashlib.blake2b(data[offset : offset + length], digest_size=32).digest() != digest:
            raise PackageError(f"section type {kind} fails its hash")
        if kind == TOC_CONFIG:
            config = data[offset : offset + length]
    if config is None or len(config) != CONFIG_BYTES:
        raise PackageError("no config section")
    return config


def entry(config: bytes, file: str) -> dict[str, Any]:
    """One manifest row from a config block; ``name`` is the file stem."""
    major, minor, patch = struct.unpack_from("<3H", config, 64)
    kind = config[7916]
    if kind >= len(PROGRAM_TYPES):
        raise PackageError(f"unknown program type {kind}")
    return {
        "name": Path(file).stem,
        "program_id": _text(config[0:64]),
        "program_name": _text(config[86:118]),
        "program_version": f"{major}.{minor}.{patch}",
        "categories": [_text(config[214 + 32 * i : 246 + 32 * i]) for i in range(config[729])],
        "program_type": PROGRAM_TYPES[kind],
        "description": _text(config[470:598]),
        "author": _text(config[118:182]),
        "file": file,
    }


def _groups(rows: list[dict[str, Any]], key) -> dict[Any, list[str]]:
    seen = defaultdict(list)
    for row in rows:
        seen[key(row)].append(row["file"])
    return {k: v for k, v in seen.items() if len(v) > 1}


def rebuild(root: str | Path, now: datetime | None = None) -> Report:
    """Manifest for every ``<author>/<name>.vmprog`` under ``root``.

    A package that fails verification, or whose ``program_id`` or author and
    display name another package already holds, is left out and reported.
    Names shared across authors are legitimate and only reported.
    """
    root = Path(root)
    old = json.loads((root / "manifest.json").read_text()) if (root / "manifest.json").exists() else {}
    rows, errors = [], {}
    for path in track(sorted(root.glob("*/*.vmprog")), desc="vmprog", unit="file"):
        file = path.relative_to(root).as_posix()
        try:
            rows.append(entry(read_config(path.read_bytes()), file))
        except (PackageError, struct.error, UnicodeDecodeError) as err:
            errors[file] = str(err)
    for label, key in (
        ("program_id", lambda r: r["program_id"]),
        ("program", lambda r: (r["author"], r["program_name"].lower())),
    ):
        for clash, files in _groups(rows, key).items():
            for file in files:
                errors[file] = f"duplicate {label} {clash!r}: {', '.join(files)}"
    kept = [r for r in rows if r["file"] not in errors]
    manifest = {k: v for k, v in old.items() if k != "programs"}
    manifest.setdefault("format_version", "1.0")
    manifest.setdefault("product", "videomancer")
    stamp = (now or datetime.now(timezone.utc)).strftime("%Y-%m-%dT%H:%M:%SZ")
    manifest.update(created=stamp, programs=kept)
    shared = _groups(kept, lambda r: r["program_name"].lower())
    for file, why in errors.items():
        LOG.warning("%s: %s", file, why)
    return Report(manifest, errors, shared)


def write(root: str | Path, report: Report) -> Path:
    """Replace ``manifest.json`` atomically."""
    path = Path(root) / "manifest.json"
    tmp = path.with_name(".manifest.json.tmp")
    tmp.write_text(json.dumps(report.manifest, indent=2) + "\n")
    tmp.replace(path)
    return path

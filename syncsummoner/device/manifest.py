"""Rebuild ``programs/manifest.json`` from the ``.vmprog`` files on a card.

The metadata is read from each package's config block (layout: Videomancer SDK
``docs/vmprog-format.md``), so the manifest can never disagree with the files
it describes. Truncated or corrupt packages are reported rather than listed.
"""

from __future__ import annotations

import json
import struct
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from syncsummoner.program.vmprog import (
    CONFIG_BYTES,
    HEADER_BYTES,
    TOC_BYTES,
    PackageError,
    SectionType,
    parse_config,
    read_config,
)
from syncsummoner.progress import LOG, track

__all__ = ["CONFIG_BYTES", "HEADER_BYTES", "TOC_BYTES", "TOC_CONFIG", "PackageError", "read_config", "entry"]

TOC_CONFIG = SectionType.CONFIG


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


def entry(config: bytes, file: str) -> dict[str, Any]:
    """One manifest row from a config block; ``name`` is the file stem."""
    cfg = parse_config(config)
    return {
        "name": Path(file).stem,
        "program_id": cfg.program_id,
        "program_name": cfg.name,
        "program_version": cfg.version_text,
        "categories": list(cfg.categories),
        "program_type": cfg.program_type,
        "description": cfg.description,
        "author": cfg.author,
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

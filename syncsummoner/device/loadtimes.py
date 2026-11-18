"""Every program load timed on this rig, in one record every takes directory shares.

The record sits beside the program cache (``SYNCSUMMONER_LOADS`` overrides it), one JSON line a load,
appended under an exclusive lock. The longest load on record bounds a lead-in's load and the device
stall guard. Run as ``python -m syncsummoner.device.loadtimes ROOT...`` to seed it from takes on disk.
"""

from __future__ import annotations

import argparse
import fcntl
import json
import os
import sys
from pathlib import Path
from typing import Any, Iterable, Iterator

import numpy as np

from syncsummoner.progress import LOG, track

#: The record's file name in the program cache.
RECORD = "loads.jsonl"
#: A takes directory's own load record, from before loads were kept per rig.
DIRECTORY_FILE = "loads.json"


class NoLoadRecordError(RuntimeError):
    """No load has been timed on this rig, so nothing bounds a device wait."""


def record_path(path: str | Path | None = None) -> Path:
    """The rig's load record: ``path``, else ``SYNCSUMMONER_LOADS``, else beside the program cache."""
    if path is not None:
        return Path(path)
    if os.environ.get("SYNCSUMMONER_LOADS"):
        return Path(os.environ["SYNCSUMMONER_LOADS"])
    return Path(os.environ.get("SYNCSUMMONER_PROGRAM_CACHE") or ".program-cache", RECORD).resolve()


def read(path: str | Path | None = None) -> list[dict]:
    """Every load on record, oldest first; none when there is no record."""
    where = record_path(path)
    if not where.is_file():
        return []
    return [json.loads(line) for line in where.read_text(encoding="utf-8").splitlines() if line.strip()]


def append(entries: Iterable[dict], path: str | Path | None = None) -> int:
    """Append loads to the record in one write under an exclusive lock, skipping any already on it.

    A load is the same one when its program and start time are, or with no start time its seconds;
    returns how many were added.
    """
    where = record_path(path)
    where.parent.mkdir(parents=True, exist_ok=True)
    fd = os.open(where, os.O_RDWR | os.O_CREAT | os.O_APPEND, 0o664)
    with os.fdopen(fd, "r+", encoding="utf-8") as handle:
        fcntl.flock(handle, fcntl.LOCK_EX)
        try:
            handle.seek(0)
            seen = {_key(e) for e in map(json.loads, filter(str.strip, handle))}
            fresh = [e for e in dict((_key(e), e) for e in entries).values() if _key(e) not in seen]
            if fresh:
                handle.write("".join(json.dumps(e, sort_keys=True) + "\n" for e in fresh))
                handle.flush()
                os.fsync(handle.fileno())
        finally:
            fcntl.flock(handle, fcntl.LOCK_UN)
    return len(fresh)


def _key(entry: dict) -> tuple:
    start = entry.get("t0_ns")
    return entry.get("program"), "t0", start if start is not None else entry.get("load_s")


def record(
    program: str, t0_ns: int, t1_ns: int, *, firmware: str | None = None, path: str | Path | None = None
) -> float:
    """Keep one timed load of ``program`` on ``firmware``; its seconds."""
    seconds = (int(t1_ns) - int(t0_ns)) / 1e9
    append([{"program": program, "firmware": firmware, "t0_ns": int(t0_ns), "load_s": seconds}], path)
    return seconds


def longest(path: str | Path | None = None) -> float | None:
    """The longest load on record, any program's and firmware's; None when none is."""
    loads = [float(e["load_s"]) for e in read(path)]
    return max(loads) if loads else None


def stall_guard(stall_s: float | None, path: str | Path | None = None) -> float:
    """The device stall guard: ``stall_s`` when given, else the longest load on record.

    With neither, a wait could hang unbounded, so it is a :class:`NoLoadRecordError`.
    """
    if stall_s is not None:
        return float(stall_s)
    found = longest(path)
    if found is None:
        raise NoLoadRecordError(
            f"no program load is on record in {record_path(path)}, so nothing bounds a device wait: pass "
            "--stall, or seed the record with `python -m syncsummoner.device.loadtimes captures/`"
        )
    return found


def _firmware(take: Path) -> str | None:
    """The firmware a take's meta record says it ran on, if it says."""
    meta = take.with_name(take.name.removesuffix(".wire.npz") + ".meta.json")
    try:
        said = json.loads(meta.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    attempts = [a for a in said.values() if isinstance(a, dict)] if isinstance(said, dict) else []
    found = [a.get("before", {}).get("firmware") for a in attempts]
    return next((str(f) for f in found if f), None)


def _timed(loads: Any, firmware: str | None) -> Iterator[dict]:
    for load in loads:
        if "t0_ns" in load and "t1_ns" in load:
            seconds = (int(load["t1_ns"]) - int(load["t0_ns"])) / 1e9
            yield {
                "program": load["program"],
                "firmware": firmware,
                "t0_ns": int(load["t0_ns"]),
                "load_s": seconds,
            }
        elif "load_s" in load:
            yield {
                "program": load["program"],
                "firmware": firmware,
                "t0_ns": None,
                "load_s": float(load["load_s"]),
            }


def found_in(roots: Iterable[str | Path]) -> list[dict]:
    """Every timed load the takes under ``roots`` kept: directories' load files and take records' loads."""
    files = [
        p
        for root in roots
        for pattern in (DIRECTORY_FILE, "*.wire.npz", "*.lock.npz")
        for p in Path(root).rglob(pattern)
    ]
    out: list[dict] = []
    for path in track(files, desc="seeding loads", unit="file"):
        if path.name == DIRECTORY_FILE:
            out += _timed(json.loads(path.read_text(encoding="utf-8")).get("loads", []), None)
            continue
        try:
            with np.load(path) as held:
                said = json.loads(str(held["loads"])) if "loads" in held.files else []
        except (OSError, ValueError) as err:
            LOG.warning("%s: unreadable, no loads taken from it: %s", path, err)
            continue
        out += _timed(said, _firmware(path))
    return out


def seed(roots: Iterable[str | Path], path: str | Path | None = None) -> int:
    """Add every load found under ``roots`` to the rig's record; how many were new."""
    return append(found_in(roots), path)


def main(argv: list[str] | None = None) -> int:
    """Seed the record from the takes under each root, then say the longest load on it."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("roots", nargs="*", help="directories whose takes to take timed loads from")
    parser.add_argument("--record", help="the record to keep them in (default: beside the program cache)")
    args = parser.parse_args(argv)
    added = seed(args.roots, args.record) if args.roots else 0
    print(
        f"{added} loads added; {len(read(args.record))} on record in {record_path(args.record)}, "
        f"longest {longest(args.record)} s"
    )
    return 0


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())

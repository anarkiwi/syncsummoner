"""How long the capture card takes to relock once a program's dry control turns wet, per program, on record.

Each dry session measures its own (:func:`measure`) into ``relocks.jsonl`` beside the load record
(``SYNCSUMMONER_RELOCKS`` overrides it); a program's wet stretch holds its longest (:func:`lead_in`).
``python -m syncsummoner.compose.relocks ROOT...`` seeds the record from takes on disk.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from dataclasses import replace
from pathlib import Path
from typing import Iterable

import numpy as np

from syncsummoner.compose import frameline
from syncsummoner.compose.evidence import START
from syncsummoner.compose.locktake import LeadIn
from syncsummoner.device import loadtimes
from syncsummoner.progress import LOG, track

#: The record's file name, beside the rig's load record.
RECORD = "relocks.jsonl"


def record_path(path: str | Path | None = None) -> Path:
    """The rig's relock record: ``path``, else ``SYNCSUMMONER_RELOCKS``, else beside the load record."""
    if path is not None:
        return Path(path)
    if os.environ.get("SYNCSUMMONER_RELOCKS"):
        return Path(os.environ["SYNCSUMMONER_RELOCKS"])
    return loadtimes.record_path().with_name(RECORD)


def measure(log: np.ndarray, sends: np.ndarray) -> tuple[int, float] | None:
    """The wet write's send time (ns) and the seconds until the card held a picture again; None unwritten.

    The wet write is the first send leaving the start state; the relock is every capture-log break begun
    after it and before the next frame's writes. The card holds a picture once a capture repeats the bytes
    of the one before, each device frame being captured twice and its restart pictures never; 0 unbroken.
    """
    moved = np.flatnonzero(sends["target"] != START)
    if not moved.size:
        return None
    wet, began = int(sends["target"][moved[0]]), int(sends["t0_ns"][moved[0]])
    later = sends["t0_ns"][sends["target"] > wet]
    until = int(later.min()) if later.size else np.iinfo(np.int64).max
    t = log["t_ns"].astype(np.int64)
    seg = frameline.segments(t, log["sequence"])
    breaks = np.flatnonzero(np.diff(seg)) + 1
    mine = breaks[(t[breaks - 1] >= began) & (t[breaks - 1] < until)]
    if not mine.size:
        return began, 0.0
    run = np.flatnonzero(seg == seg[mine[-1]])
    held = run[1:][log["digest"][run[1:]] == log["digest"][run[:-1]]]
    return began, float(t[held[0] if held.size else run[-1]] - began) / 1e9


def record(
    program: str, t0_ns: int, relock_s: float, *, firmware: str | None = None, path: str | Path | None = None
) -> int:
    """Keep one relock of ``program`` after its wet write sent at ``t0_ns``; how many were new."""
    entry = {"program": program, "firmware": firmware, "t0_ns": int(t0_ns), "relock_s": float(relock_s)}
    return loadtimes.append([entry], record_path(path))


def longest(program: str, path: str | Path | None = None) -> float | None:
    """The longest relock on record for ``program``, of any firmware; None when none is."""
    found = [float(e["relock_s"]) for e in loadtimes.read(record_path(path)) if e.get("program") == program]
    return max(found) if found else None


def lead_in(program: str, base: LeadIn = LeadIn(), path: str | Path | None = None) -> LeadIn:
    """``base`` with its wet stretch held at least as long as ``program``'s longest relock on record."""
    found = longest(program, path)
    return base if found is None or found <= base.wet_s else replace(base, wet_s=found)


def found_in(roots: Iterable[str | Path]) -> list[dict]:
    """Every relock the dry sessions under ``roots`` show: each session once, from any take's records."""
    locks = sorted(p for root in roots for p in Path(root).rglob("*.lock.npz"))
    out: dict[tuple, dict] = {}
    for path in track(locks, desc="seeding relocks", unit="take"):
        wire = path.with_name(path.name.removesuffix(".lock.npz") + ".wire.npz")
        try:
            with np.load(path) as held, np.load(wire) as said:
                if "session_log" not in held.files or json.loads(str(said["dry"])) is None:
                    continue
                program, found = str(said["program"]), measure(held["session_log"], said["sends"])
        except (OSError, ValueError, KeyError) as err:
            LOG.warning("%s: unreadable, no relock taken from it: %s", path, err)
            continue
        if found is not None:
            out.setdefault((program, found[0]), {"program": program, "t0_ns": found[0], "relock_s": found[1]})
    return list(out.values())


def main(argv: list[str] | None = None) -> int:
    """Seed the record from the takes under each root, then say each program's longest relock on it."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("roots", nargs="*", help="directories whose takes to measure relocks from")
    parser.add_argument("--record", help="the record to keep them in (default: beside the load record)")
    args = parser.parse_args(argv)
    added = loadtimes.append(found_in(args.roots), record_path(args.record)) if args.roots else 0
    held = loadtimes.read(record_path(args.record))
    print(f"{added} relocks added; {len(held)} on record in {record_path(args.record)}")
    for program in sorted({e["program"] for e in held}):
        print(f"{program}: longest {longest(program, args.record):.3f} s")
    return 0


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())

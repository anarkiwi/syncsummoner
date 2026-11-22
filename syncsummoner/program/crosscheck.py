"""The BDDs, the SAT miter and the ternary simulator set against each other on one program's queries.

The miter is exact under the register rules (docs/program-model.md) and the other two over-approximate
it, so a bit the miter finds live and another method finds dead is a fault of that method.
"""

from __future__ import annotations

import json
import multiprocessing
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from typing import Iterator, Sequence

import numpy as np

from syncsummoner.progress import track

from . import observe
from .declared import field
from .library import Library
from .model import TIMING, ProgramModel
from .model import model as build

METHODS = ("bdd", "sat", "ternary")


def queries(found: ProgramModel, presets: bool = False) -> list[tuple[str, np.ndarray, tuple[int, ...]]]:
    """``(state name, state, freed controls)``: the initial state, and with ``presets`` each preset, with
    each declared control freed alone and together with every knob, as liveness and judging takes query them.
    """
    pids = [d.parameter_id for d in found.params]
    knobs = {p for p in pids if len(field(p).bits) > 1}
    frees = sorted({(p,) for p in pids} | {tuple(sorted(knobs | {p})) for p in pids})
    cfg = found.program.package.config
    states = [("initial", found.initial())] + [(p.name, found.preset(p.name)) for p in cfg.presets if presets]
    return [(name, state, free) for name, state in states for free in frees]


def verdicts(program, presets: bool = False, **kwargs) -> dict:
    """Every :func:`queries` answer by each method for one program (as :func:`~.model.model` takes it):
    the seconds each method took, the queries on which they differ, and the methods at fault.
    """
    first = build(program, method="bdd", **kwargs)
    models = {"bdd": first} | {
        m: ProgramModel(
            first.program,
            first.library,
            config=first.config,
            cache=first.cache.parents[1],
            source=first.source,
            method=m,
        )
        for m in METHODS[1:]
    }
    asked = queries(first, presets)
    out = {"program": first.program.name, "source": first.source, "queries": len(asked), "seconds": {}}
    answers: dict[str, list] = {}
    for method, found in models.items():
        start = time.monotonic()
        try:
            found.prepare([*free, TIMING] for free in {free for _, _, free in asked})
            answers[method] = [
                found.reach(state, free)
                for _, state, free in track(asked, desc=f"{out['program']} {method}", unit="query")
            ]
        except observe.Exhausted as err:
            out[method] = str(err)
        out["seconds"][method] = round(time.monotonic() - start, 1)
    out["disagreements"] = [
        {"state": name, "free": list(free), "param": p} | {m: got[i][p] for m, got in answers.items()}
        for i, (name, _, free) in enumerate(asked)
        for p in free
        if len({tuple(got[i][p]) for got in answers.values()}) > 1
    ]
    exact = answers.get("sat")
    out["faults"] = sorted(
        {m for d in out["disagreements"] for m in answers if exact is not None and set(d["sat"]) - set(d[m])}
    )
    return out


def crosschecks(names: Sequence, jobs: int = 1, presets: bool = False, **kwargs) -> Iterator[dict]:
    """:func:`verdicts` of each program, ``jobs`` at once in their own processes, each as it finishes."""
    jobs = max(1, min(jobs, len(names)))
    if jobs == 1:
        yield from track((verdicts(n, presets, **kwargs) for n in names), total=len(names), desc="crosscheck")
        return
    context = multiprocessing.get_context("spawn")
    with ProcessPoolExecutor(max_workers=jobs, mp_context=context) as pool:
        done = as_completed([pool.submit(verdicts, n, presets, **kwargs) for n in names])
        yield from track((f.result() for f in done), total=len(names), desc="crosscheck")


def add_parser(program_sub) -> None:
    """The ``program crosscheck`` command."""
    found = program_sub.add_parser(
        "crosscheck", help="answer each program's liveness queries by every method and say where they differ"
    )
    found.add_argument("names", nargs="+", help="device names or program ids, or all")
    found.add_argument("--jobs", type=int, default=1, help="programs at once")
    found.add_argument(
        "--presets", action="store_true", help="query every preset too, not just the initial state"
    )
    found.set_defaults(func=_crosscheck_cmd)


def _crosscheck_cmd(args) -> int:
    names = [p.name for p in Library.default().programs] if args.names == ["all"] else args.names
    faults = 0
    for row in crosschecks(names, args.jobs, args.presets):
        print(json.dumps(row), flush=True)
        faults += bool(row["faults"])
    return int(faults > 0)

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


def verdicts(
    program, presets: bool = False, memory: int | None = None, faults: bool = False, **kwargs
) -> dict:
    """Every :func:`queries` answer by each method for one program (as :func:`~.model.model` takes it),
    the BDDs within ``memory`` bytes (default: what the process may use): the seconds each method took,
    the queries on which they differ, and the methods at fault.

    With ``faults``, the miter answers only the queries where a bit the ternary simulator leaves live
    is dead by the BDDs: elsewhere a BDD dead bit is proven dead by the simulator, so no fault of the
    BDDs is missed, though their over-approximations go uncounted and the simulator is not checked.
    """
    first = build(program, method="bdd", **kwargs)
    first.memory = memory
    models = {"bdd": first} | {
        m: ProgramModel(
            first.program,
            first.library,
            config=first.config,
            cache=first.cache.parents[1],
            source=first.source,
            method=m,
        )
        for m in ("ternary", "sat")
    }
    asked = queries(first, presets)
    out = {"program": first.program.name, "source": first.source, "queries": len(asked), "seconds": {}}
    answers: dict[str, list] = {}
    for method, found in models.items():
        todo = list(range(len(asked)))
        if method == "sat" and faults and {"bdd", "ternary"} <= set(answers):
            todo = [
                i
                for i in todo
                if any(set(answers["ternary"][i][p]) - set(answers["bdd"][i][p]) for p in asked[i][2])
            ]
        start = time.monotonic()
        try:
            found.prepare([*asked[i][2], TIMING] for i in todo)
            got = [None] * len(asked)
            for i in track(todo, desc=f"{out['program']} {method}", unit="query"):
                got[i] = found.reach(asked[i][1], asked[i][2])
            answers[method] = got
        except observe.Exhausted as err:
            out[method] = str(err)
        out["seconds"][method] = round(time.monotonic() - start, 1)
    out["asked"] = {m: sum(a is not None for a in got) for m, got in answers.items()}
    out["disagreements"] = []
    for i, (name, _, free) in enumerate(asked):
        said = {m: got[i] for m, got in answers.items() if got[i] is not None}
        for p in free:
            if len({tuple(a[p]) for a in said.values()}) > 1:
                out["disagreements"].append(
                    {"state": name, "free": list(free), "param": p} | {m: a[p] for m, a in said.items()}
                )
    out["faults"] = sorted(
        {
            m
            for d in out["disagreements"]
            if "sat" in d
            for m in METHODS
            if m in d and set(d["sat"]) - set(d[m])
        }
    )
    return out


def crosschecks(
    names: Sequence, jobs: int = 1, presets: bool = False, faults: bool = False, **kwargs
) -> Iterator[dict]:
    """:func:`verdicts` of each program, ``jobs`` at once in their own processes sharing the memory this
    one may use, each as it finishes.
    """
    jobs = max(1, min(jobs, len(names)))
    memory = observe.memory_limit() // jobs
    if jobs == 1:
        yield from track(
            (verdicts(n, presets, memory, faults, **kwargs) for n in names),
            total=len(names),
            desc="crosscheck",
        )
        return
    context = multiprocessing.get_context("spawn")
    with ProcessPoolExecutor(max_workers=jobs, mp_context=context) as pool:
        done = as_completed([pool.submit(verdicts, n, presets, memory, faults, **kwargs) for n in names])
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
    found.add_argument(
        "--faults",
        action="store_true",
        help="ask the miter only where the BDDs drop a bit the simulator keeps",
    )
    found.set_defaults(func=_crosscheck_cmd)


def _crosscheck_cmd(args) -> int:
    names = [p.name for p in Library.default().programs] if args.names == ["all"] else args.names
    faults = 0
    for row in crosschecks(names, args.jobs, args.presets, args.faults):
        print(json.dumps(row), flush=True)
        faults += bool(row["faults"])
    return int(faults > 0)

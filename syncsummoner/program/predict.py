"""The program model's verdict on every step a take sent, from the state it was taken in."""

from __future__ import annotations

import json
import multiprocessing
import resource
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from typing import Iterator, Sequence

import numpy as np

from syncsummoner.progress import track

from . import observe
from .declared import PARAM_COUNT, field, register_value
from .library import Library
from .model import TIMING, ProgramModel
from .model import model as build

#: Per-write verdicts of :func:`classify`, from no step at all to a step the cone says can show.
UNSENT, UNCHANGED, DEAD, NOOP, LIVE = "unsent", "unchanged", "dead", "no-op", "live"


def pre_states(start: np.ndarray, rows: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Each sent write's state before it, and its value before and after, from what was sent.

    Writes landing on one frame move together: each is a step from the frame's new state with its own
    parameter at its value before the frame. Unsent writes get the state they would have stepped from.
    """
    state = np.zeros(PARAM_COUNT, dtype=np.int64)
    state[: np.size(start)] = np.asarray(start, dtype=np.int64)[:PARAM_COUNT]
    sent = rows["sent"] >= 0
    frame = np.where(sent, rows["landed"], np.iinfo(np.int64).max)
    order = np.argsort(frame, kind="stable")
    pre = np.zeros((len(rows), PARAM_COUNT), dtype=np.int64)
    a, b = np.zeros(len(rows), dtype=np.int64), np.zeros(len(rows), dtype=np.int64)
    bounds = np.flatnonzero(np.r_[True, np.diff(frame[order]) != 0, True])
    for lo, hi in zip(bounds[:-1], bounds[1:]):
        group = order[lo:hi]
        after = state.copy()
        for i in group[sent[group]]:
            after[rows["param"][i] - 1] = rows["wire"][i]
        for i in group:
            p = int(rows["param"][i]) - 1
            pre[i], a[i] = after, state[p]
            pre[i, p] = state[p]
            b[i] = after[p] if sent[i] else rows["value"][i]
        state = after
    return pre, a, b


def frame_states(start: np.ndarray, rows: np.ndarray, count: int) -> np.ndarray:
    """Every clip frame's manual state, ``0..count-1``: the start state, each sent write from its landing."""
    state = np.zeros((count, PARAM_COUNT), dtype=np.int64)
    state[:, : np.size(start)] = np.asarray(start, dtype=np.int64)[:PARAM_COUNT]
    sent = rows[rows["sent"] >= 0]
    sent = sent[np.argsort(sent["landed"], kind="stable")]
    frame = np.arange(count)
    for param in np.unique(sent["param"]):
        mine = sent[sent["param"] == param]
        at = np.searchsorted(mine["landed"], frame, side="right") - 1
        state[:, param - 1] = np.where(at >= 0, mine["wire"][np.maximum(at, 0)], state[:, param - 1])
    return state


def regimes(
    model: ProgramModel, start: np.ndarray, rows: np.ndarray, count: int
) -> tuple[np.ndarray, dict[int, int]]:
    """Each clip frame's switch regime as a bit code, and the bit each switch holds in it.

    The switches are the controls a one-bit field drives: the ones a non-knob-gated
    parameter's gating class ties (:func:`gating_class`). A bypass on any other control
    (:meth:`~syncsummoner.program.model.ProgramModel.bypass`) adds one bit, set on the frames it
    holds a proven bypass setting. No switches and no such bypass leave one regime.
    """
    switches = [d for d in model.params if len(field(d.parameter_id).bits) == 1]
    state = frame_states(start, rows, count)
    code = np.zeros(count, dtype=np.int64)
    for bit, decl in enumerate(switches):
        code |= register_value(decl, state[:, decl.parameter_id - 1]).astype(np.int64) << bit
    own = {d.parameter_id: 1 << bit for bit, d in enumerate(switches)}
    dry = model.bypass()
    if dry is not None and dry.parameter_id not in own:
        own[dry.parameter_id] = 1 << len(switches)
        code |= dry.holds(state[:, dry.parameter_id - 1]).astype(np.int64) << len(switches)
    return code, own


def gating_class(model: ProgramModel, state: np.ndarray, param: int, gated: frozenset[int]) -> tuple:
    """The controls a step's liveness query ties, and their register values there.

    A parameter the model found gated by a knob at its reference is queried with every other
    control tied; any other is queried with every knob free, so only the switches tie it.
    """
    free = freed(model, param, gated)
    tied = [d.parameter_id for d in model.params if d.parameter_id not in free]
    held = tuple(int(register_value(model.decl(q), int(state[q - 1]))) for q in tied)
    return param, tuple(tied), held, free


def freed(model: ProgramModel, param: int, gated: frozenset[int]) -> tuple[int, ...]:
    """The controls a step of ``param`` is queried with free (:func:`gating_class`)."""
    knobs = {d.parameter_id for d in model.params if len(field(d.parameter_id).bits) > 1}
    return tuple(sorted((set() if param in gated else knobs) | {param}))


def prepare(model: ProgramModel, timing: int | None = None) -> dict[str, float]:
    """Every analysis :func:`classify` reads, built ahead of any take; the seconds each stage took.

    Raises :class:`~syncsummoner.program.observe.Exhausted` where a model answering by BDDs alone
    outgrows its memory.
    """
    spent = {}
    start = time.monotonic()
    gated = knob_gated(model)
    spent["gating"] = time.monotonic() - start
    sets = {freed(model, d.parameter_id, gated) for d in model.params}
    extra = [] if timing is not None else [TIMING]
    start = time.monotonic()
    model.prepare([*s, *extra] for s in sets)
    for s in track(sorted(sets), desc=f"{model.program.name} gating classes", unit="set"):
        model.conditions([*s, *extra])
    spent["classes"] = time.monotonic() - start
    return spent


def knob_gated(model: ProgramModel, reference: np.ndarray | None = None) -> frozenset[int]:
    """Parameters dead at ``reference`` (default: initial) that some knob revives."""
    knobs = {f"P{d.parameter_id}" for d in model.params if len(field(d.parameter_id).bits) > 1}
    live = model.live(reference)
    return frozenset(p for p, l in live.items() if not l.live and set(l.gated_by) & knobs)


def classify(
    model: ProgramModel, start: np.ndarray, rows: np.ndarray, timing: int | None = None
) -> np.ndarray:
    """Each landed write's step as the model's observability predicts it from the state it was taken in.

    ``dead`` when no bit of the parameter reaches the video, ``no-op`` when the step changes only dead
    bits, ``live`` otherwise. One query per gating class (:func:`gating_class`), their analyses built
    together first.
    """
    pre, a, b = pre_states(start, rows)
    out = np.full(len(rows), UNSENT, dtype=object)
    sent = rows["sent"] >= 0
    out[sent & (a == b)] = UNCHANGED
    todo = np.flatnonzero(sent & (a != b))
    gated = knob_gated(model)
    read = model.resolution()
    keys = [gating_class(model, pre[i], int(rows["param"][i]), gated) for i in todo]
    first = {}
    for i, key in zip(todo, keys):
        first.setdefault(key, i)

    def query(key: tuple) -> tuple[int, ...]:
        param, free = key[0], key[3]
        if model.decl(param) is None or not read.get(param):
            return ()
        return tuple(model.reach(pre[first[key]], free, timing)[param])

    extra = [] if timing is not None else [TIMING]
    model.prepare([*k[3], *extra] for k in first if model.decl(k[0]) is not None and read.get(k[0]))
    live = {key: query(key) for key in track(list(first), desc=f"{model.program.name} classes", unit="class")}
    for i, key in track(list(zip(todo, keys)), desc=f"{model.program.name} steps", unit="step"):
        p, bits = int(rows["param"][i]), live[key]
        if not bits:
            out[i] = DEAD
            continue
        decl = model.decl(p)
        mask = sum(1 << bit for bit in bits) if len(field(p).bits) > 1 else 1
        changed = int(register_value(decl, int(a[i]))) ^ int(register_value(decl, int(b[i])))
        out[i] = LIVE if changed & mask else NOOP
    return out


def survey(program, memory: int | None = None, **kwargs) -> dict:
    """Build the analyses :func:`classify` reads for one program (as :func:`~.model.model` takes it)
    by BDDs alone, within ``memory`` bytes (default: what the process may use): the seconds each stage
    took, or why they outgrew it.
    """
    found = build(program, method="bdd", **kwargs)
    found.memory = memory
    out = {"program": found.program.name, "source": found.source}
    try:
        out |= {"seconds": prepare(found), "outcome": "bdd"}
    except observe.Exhausted as err:
        out |= {"outcome": str(err)}
    peak = max(resource.getrusage(w).ru_maxrss for w in (resource.RUSAGE_SELF, resource.RUSAGE_CHILDREN))
    return out | {"peak_bytes": peak * 1024}


def surveys(names: Sequence, jobs: int = 1, **kwargs) -> Iterator[dict]:
    """:func:`survey` of each program, ``jobs`` at once in their own processes sharing the memory this
    one may use, each as it finishes; ``kwargs`` go to :func:`~.model.model`.
    """
    jobs = max(1, min(jobs, len(names)))
    memory = observe.memory_limit() // jobs
    if jobs == 1:
        yield from track((survey(n, memory, **kwargs) for n in names), total=len(names), desc="survey")
        return
    context = multiprocessing.get_context("spawn")
    with ProcessPoolExecutor(max_workers=jobs, mp_context=context) as pool:
        done = as_completed([pool.submit(survey, n, memory, **kwargs) for n in names])
        yield from track((f.result() for f in done), total=len(names), desc="survey")


def add_parser(program_sub) -> None:
    """The ``program survey`` command."""
    found = program_sub.add_parser(
        "survey", help="build and time the BDD analyses judging a take reads, or say why they outgrow memory"
    )
    found.add_argument("names", nargs="+", help="device names or program ids, or all")
    found.add_argument("--jobs", type=int, default=1, help="programs at once, sharing the memory")
    found.set_defaults(func=_survey_cmd)


def _survey_cmd(args) -> int:
    names = [p.name for p in Library.default().programs] if args.names == ["all"] else args.names
    for row in surveys(names, args.jobs):
        print(json.dumps(row), flush=True)
    return 0

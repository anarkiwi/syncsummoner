"""The program model's verdict on every step a take sent, from the state it was taken in."""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor

import numpy as np

from syncsummoner.progress import track

from .hdl import cpus
from .declared import PARAM_COUNT, field, register_value
from .model import ProgramModel

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
    parameter's gating class ties (:func:`gating_class`). No switches leave one regime.
    """
    switches = [d for d in model.params if len(field(d.parameter_id).bits) == 1]
    state = frame_states(start, rows, count)
    code = np.zeros(count, dtype=np.int64)
    for bit, decl in enumerate(switches):
        code |= register_value(decl, state[:, decl.parameter_id - 1]).astype(np.int64) << bit
    return code, {d.parameter_id: 1 << bit for bit, d in enumerate(switches)}


def gating_class(model: ProgramModel, state: np.ndarray, param: int, gated: frozenset[int]) -> tuple:
    """The controls a step's liveness query ties, and their register values there.

    A parameter the model found gated by a knob at its reference is queried with every other
    control tied; any other is queried with every knob free, so only the switches tie it.
    """
    knobs = {d.parameter_id for d in model.params if len(field(d.parameter_id).bits) > 1}
    tied = [d.parameter_id for d in model.params if d.parameter_id != param]
    if param not in gated:
        tied = [q for q in tied if q not in knobs]
    held = tuple(int(register_value(model.decl(q), int(state[q - 1]))) for q in tied)
    free = sorted((set() if param in gated else knobs) | {param})
    return param, tuple(tied), held, tuple(free)


def knob_gated(model: ProgramModel, reference: np.ndarray | None = None) -> frozenset[int]:
    """Parameters dead at ``reference`` (default: initial) that some knob revives."""
    knobs = {f"P{d.parameter_id}" for d in model.params if len(field(d.parameter_id).bits) > 1}
    live = model.live(reference)
    return frozenset(p for p, l in live.items() if not l.live and set(l.gated_by) & knobs)


def classify(
    model: ProgramModel, start: np.ndarray, rows: np.ndarray, timing: int | None = None
) -> np.ndarray:
    """Each landed write's step as the model's folded cone predicts it from the state it was taken in.

    ``dead`` when no bit of the parameter reaches the video, ``no-op`` when the step changes only dead
    bits, ``live`` otherwise. One fold per gating class (:func:`gating_class`), memoised on disk.
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

    classes = list(first)
    with ThreadPoolExecutor(max_workers=cpus()) as pool:
        live = dict(
            zip(
                classes,
                track(
                    pool.map(query, classes),
                    total=len(classes),
                    desc=f"{model.program.name} classes",
                    unit="class",
                ),
            )
        )
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

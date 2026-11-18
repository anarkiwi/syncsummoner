"""Gestures only on live parameters: each write is checked against the program model along the plan.

A write to a parameter no output depends on in the state the plan holds then is either paired
with the gate move that makes it live there, at the same instant, or refused.
"""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from typing import Any, Callable, Mapping, Sequence

import numpy as np

from syncsummoner.progress import LOG, track
from syncsummoner.program.hdl import cpus
from syncsummoner.compose.vocabulary import Automation

Live = Callable[[np.ndarray, int], bool]
Revive = Callable[[np.ndarray, int], Sequence[Sequence[tuple[int, int]]]]


@dataclass(frozen=True)
class Gated:
    """The automation to play, what gating did to it, and the state it leaves."""

    auto: Automation
    paired: int = 0
    refused: int = 0
    state: np.ndarray | None = None


def _states(auto: Automation, start: np.ndarray) -> np.ndarray:
    """The state each event is written into: everything before it applied."""
    state = np.asarray(start, dtype=np.int64).copy()
    out = np.zeros((len(auto), state.size), dtype=np.int64)
    for i, (p, v) in enumerate(zip(auto.indices.tolist(), auto.values.tolist())):
        out[i] = state
        state[p - 1] = v
    return out


def gate_automation(auto: Automation, start: np.ndarray, live: Live, revive: Revive) -> Gated:
    """Keep each write whose parameter is live where it lands; pair or refuse the rest.

    ``revive(state, p)`` offers gate moves; the first under which ``live`` holds is written just before.
    """
    state = np.asarray(start, dtype=np.int64).copy()
    times, params, values = [], [], []
    paired = refused = 0
    for t, p, v in zip(auto.times.tolist(), auto.indices.tolist(), auto.values.tolist()):
        if state[p - 1] != v and not live(state, p):
            moves = next(
                (m for m in revive(state, p) if live(_moved(state, m), p)),
                None,
            )
            if moves is None:
                refused += 1
                continue
            paired += 1
            for g, w in moves:
                times.append(t)
                params.append(g)
                values.append(w)
                state[g - 1] = w
        times.append(t)
        params.append(p)
        values.append(v)
        state[p - 1] = v
    out = Automation(
        np.asarray(times, np.float64), np.asarray(params, np.int32), np.asarray(values, np.int32)
    )
    return Gated(out, paired, refused, state)


def _moved(state: np.ndarray, moves: Sequence[tuple[int, int]]) -> np.ndarray:
    out = state.copy()
    for g, w in moves:
        out[g - 1] = w
    return out


@dataclass
class ModelGate:
    """Liveness from a program model's folded cone, and gate moves from its covering search."""

    m: Any
    revivals: Mapping[int, list[tuple[tuple[int, int], ...]]] = field(default_factory=dict)
    _seen: dict = field(default_factory=dict, repr=False)

    def key(self, state: np.ndarray, param: int) -> tuple:
        """The registers a parameter's liveness depends on: every word with its own field cleared."""
        from syncsummoner.program.declared import registers  # pylint: disable=import-outside-toplevel

        held = np.asarray(state, dtype=np.int64).copy()
        held[param - 1] = 0
        return param, tuple(sorted(registers(self.m.params, held, None).items()))

    def live(self, state: np.ndarray, param: int) -> bool:
        """Whether any bit of ``param`` reaches the video with the rest at ``state``."""
        key = self.key(state, param)
        if key not in self._seen:
            self._seen[key] = param in self.m.resolution() and bool(self.m.reach(state, [param])[param])
        return self._seen[key]

    def start(self) -> np.ndarray:
        """The state the program loads in, which every plan is gated from."""
        return np.asarray(self.m.initial(), dtype=np.int64)

    def revive(self, _state: np.ndarray, param: int) -> list[tuple[tuple[int, int], ...]]:
        """Gate moves the program model found revive ``param``."""
        return list(self.revivals.get(param, []))

    def prefetch(self, auto: Automation, start: np.ndarray) -> None:
        """Ask every state the ungated plan writes into at once, in parallel."""
        jobs = {}
        for state, p in zip(_states(auto, start), auto.indices.tolist()):
            jobs.setdefault(self.key(state, p), (state, p))
        todo = [job for key, job in jobs.items() if key not in self._seen]
        if todo:
            with ThreadPoolExecutor(max_workers=cpus()) as pool:
                list(
                    track(
                        pool.map(lambda job: self.live(*job), todo),
                        total=len(todo),
                        desc="gating",
                        unit="state",
                    )
                )

    def __call__(self, auto: Automation) -> Automation:
        start = self.start()
        self.prefetch(auto, start)
        gated = gate_automation(auto, start, self.live, self.revive)
        if gated.paired or gated.refused:
            LOG.info(
                "%s: %d writes paired with the gate move that makes them live, %d refused as dead",
                self.m.program.name,
                gated.paired,
                gated.refused,
            )
        return gated.auto


def model_gate(program: str) -> ModelGate | None:
    """The gate for a program from its model and covering search; None when the model cannot be built."""
    from syncsummoner.program import model  # pylint: disable=import-outside-toplevel
    from syncsummoner.program.gates import cover  # pylint: disable=import-outside-toplevel

    try:
        m = model(program)
        found = cover(m)
    except (KeyError, RuntimeError, OSError) as err:
        LOG.warning("%s: plan not gated, no program model: %s", program, err)
        return None
    revivals: dict[int, list[tuple[tuple[int, int], ...]]] = {}
    for found_here in found.revivals:
        for p, rs in found_here.items():
            for r in rs:
                if r.moves not in revivals.setdefault(p, []):
                    revivals[p].append(r.moves)
    return ModelGate(m, revivals)

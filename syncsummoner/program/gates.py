"""Which controls make each parameter live, and the fewest base states that step every one where it is.

Live means a written bit reaches the video and the representative steps are not all proven no-ops;
revivals are found by exact refutation with controls freed, then scans of each remaining control;
the states they reach are covered by a lexicographic 0-1 programme (fewest states, then steps).
"""

from __future__ import annotations

import itertools
from functools import cached_property
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from typing import Callable, Iterable, Sequence

import numpy as np
from scipy.optimize import Bounds, LinearConstraint, milp
from scipy.sparse import coo_matrix

from syncsummoner.progress import track

from .hdl import cpus
from .declared import VALUE_MAX, classes, registers
from .model import ProgramModel, Step

LIVE, NOOP, DEAD = "live", "no-op", "dead"


@dataclass(frozen=True)
class Status:
    """One parameter at one state: its live bits and its representative steps' verdicts."""

    param: int
    live_bits: tuple[int, ...]
    steps: tuple[Step, ...] = ()

    @property
    def state(self) -> str:
        """``live``; ``no-op`` if its bits reach the video but each checked step is proven inert; ``dead``."""
        if not self.live_bits:
            return DEAD
        return LIVE if any(s.matters is not False for s in self.steps) else NOOP

    @property
    def witnessed(self) -> bool:
        """Whether a checked step has a concrete witness of changing the video."""
        return any(s.matters is True for s in self.steps)


@dataclass(frozen=True)
class Revival:
    """Moving ``moves`` from the state it was found at makes ``param`` live.

    For a single control, ``spans`` are the maximal intervals of its manual values over which the
    parameter is live with everything else unchanged; the move sits mid-way in the widest.
    """

    param: int
    kind: str
    moves: tuple[tuple[int, int], ...]
    spans: tuple[tuple[int, int], ...] = ()

    @property
    def gate(self) -> tuple[int, ...]:
        """The controls moved."""
        return tuple(g for g, _ in self.moves)

    def __str__(self) -> str:
        if self.spans:
            return f"P{self.moves[0][0]} in " + ", ".join(
                f"{lo}" if lo == hi else f"{lo}..{hi}" for lo, hi in self.spans
            )
        return " and ".join(f"P{g} = {v}" for g, v in self.moves)


def representatives(m: ProgramModel, param: int, live_bits: Sequence[int]) -> np.ndarray:
    """One manual value per setting the program can tell apart through ``live_bits``, ascending."""
    groups = classes(m.decl(param), tuple(live_bits))
    return np.sort(np.array([g[len(g) // 2] for g in groups], dtype=np.int64))


def probe_steps(values: np.ndarray) -> list[tuple[int, int]]:
    """Lowest to middle and middle to highest setting: two steps that between them span every bit."""
    lo, mid, hi = int(values[0]), int(values[len(values) // 2]), int(values[-1])
    return [(a, b) for a, b in ((lo, mid), (mid, hi)) if a != b]


def status(m: ProgramModel, state: np.ndarray, param: int, timing: int | None = None) -> Status:
    """A parameter's live bits at ``state`` and its representative steps there."""
    bits = tuple(m.reach(state, [param], timing)[param])
    if not bits:
        return Status(param, ())
    values = representatives(m, param, bits)
    return Status(
        param, bits, tuple(m.step_matters(state, param, a, b, timing) for a, b in probe_steps(values))
    )


def intervals(ok: np.ndarray) -> list[tuple[int, int]]:
    """Maximal runs of consecutive indices where ``ok`` holds, as inclusive ``(lo, hi)``."""
    edges = np.diff(np.r_[0, ok.astype(np.int8), 0])
    return [(int(a), int(b) - 1) for a, b in zip(np.flatnonzero(edges == 1), np.flatnonzero(edges == -1))]


def _apply(state: np.ndarray, moves: Iterable[tuple[int, int]]) -> np.ndarray:
    out = np.asarray(state, dtype=np.int64).copy()
    for gate, value in moves:
        out[gate - 1] = value
    return out


@dataclass
class Gating:
    """The model queries a cover is built from, run in parallel and remembered per state."""

    m: ProgramModel
    timing: int | None = None
    workers: int = field(default_factory=cpus)
    _status: dict = field(default_factory=dict, repr=False)

    def key(self, state: np.ndarray) -> tuple:
        """What the program sees of a state: its register words."""
        return tuple(sorted(registers(self.m.params, np.asarray(state), self.timing).items()))

    @cached_property
    def read(self) -> dict[int, tuple[int, ...]]:
        """Each declared parameter's bits the program reads at all; unread parameters are left out."""
        return {p: bits for p, bits in self.m.resolution().items() if bits}

    def map(self, fn: Callable, jobs: Sequence, desc: str) -> list:
        """``fn`` over ``jobs`` on the pool, in order, with progress."""
        if not jobs:
            return []
        with ThreadPoolExecutor(max_workers=self.workers) as pool:
            return list(track(pool.map(fn, jobs), total=len(jobs), desc=desc, unit="query"))

    def statuses(self, state: np.ndarray) -> dict[int, Status]:
        """Every read parameter's status at ``state``."""
        key = self.key(state)
        if key not in self._status:
            params = sorted(self.read)
            found = self.map(lambda p: status(self.m, state, p, self.timing), params, "status")
            self._status[key] = dict(zip(params, found))
        return self._status[key]

    def _alive(self, state: np.ndarray, param: int, kind: str) -> bool:
        if kind == DEAD:
            return bool(self.m.reach(state, [param], self.timing)[param])
        return status(self.m, state, param, self.timing).state == LIVE

    def _could(self, state: np.ndarray, param: int, kind: str, free: Sequence[int]) -> bool:
        """False only when no values of ``free`` can revive ``param``: an exact refutation."""
        if kind == DEAD:
            return bool(self.m.reach(state, [param, *free], self.timing)[param])
        values = representatives(self.m, param, self.read[param])
        return any(
            self.m.step_matters(state, param, a, b, self.timing, free=free).matters is not False
            for a, b in probe_steps(values)
        )

    def revivals(self, state: np.ndarray, st: Status) -> list[Revival]:
        """Every way of making ``st.param`` live by moving the fewest other controls from ``state``."""
        p, kind = st.param, st.state
        others = [q for q in sorted(self.read) if q != p]
        if not self._could(state, p, kind, others):
            return []
        for size in range(1, len(others) + 1):
            sets = list(itertools.combinations(others, size))
            hits = self.map(lambda qs: self._could(state, p, kind, list(qs)), sets, f"P{p} gates of {size}")
            found = [r for qs, hit in zip(sets, hits) if hit for r in self.scan(state, p, kind, qs)]
            if found:
                return found
        return []

    def scan(self, state: np.ndarray, param: int, kind: str, gates: Sequence[int]) -> list[Revival]:
        """Settings of ``gates`` that revive ``param``: intervals for one control, a witness for several."""
        gate, rest = gates[0], list(gates[1:])
        groups = classes(self.m.decl(gate), self.read[gate])
        reps = [int(g[len(g) // 2]) for g in groups]

        def test(v: int) -> bool:
            moved = _apply(state, [(gate, v)])
            return self._could(moved, param, kind, rest) if rest else self._alive(moved, param, kind)

        hits = self.map(test, reps, f"P{param} by P{gate}")
        if not rest:
            ok = np.zeros(VALUE_MAX + 1, dtype=bool)
            for g, hit in zip(groups, hits):
                ok[g] = hit
            spans = intervals(ok)
            if not spans:
                return []
            lo, hi = max(spans, key=lambda s: s[1] - s[0])
            return [Revival(param, kind, ((gate, (lo + hi) // 2),), tuple(spans))]
        for v, hit in zip(reps, hits):
            deeper = self.scan(_apply(state, [(gate, v)]), param, kind, rest) if hit else []
            if deeper:
                return [Revival(param, kind, ((gate, v), *deeper[0].moves))]
        return []


@dataclass
class Cover:
    """The states reached from ``reference`` by revivals, the cover chosen from them, and its schedule."""

    m: ProgramModel
    states: list[np.ndarray]
    status: list[dict[int, Status]]
    revivals: list[dict[int, list[Revival]]]
    chosen: list[int]
    stepped: dict[int, list[int]]
    uncovered: list[tuple]
    read: dict[int, tuple[int, ...]]

    @property
    def reference(self) -> np.ndarray:
        """The state the search started from."""
        return self.states[0]

    def schedule(self, index: int, param: int) -> np.ndarray:
        """Manual values stepping ``param`` through every setting it shows at state ``index``."""
        return representatives(self.m, param, self.status[index][param].live_bits)

    def steps(self) -> int:
        """Values the schedule steps through off its base states, in all."""
        return sum(len(self.schedule(i, p)) for i, ps in self.stepped.items() for p in ps)

    def where(self, param: int) -> list[int]:
        """The chosen base states ``param`` is stepped in."""
        return [i for i in self.chosen if param in self.stepped.get(i, [])]

    def vectors(self) -> list[tuple[np.ndarray, np.ndarray]]:
        """``(from, to)`` full states in sweep order: each base state, then one parameter at a time off it."""
        out, prev = [], self.reference
        for i in self.chosen:
            base = self.states[i]
            out.append((prev, base))
            prev = base
            for p in self.stepped.get(i, []):
                for v in self.schedule(i, p):
                    if int(v) != int(prev[p - 1]):
                        out.append((prev, _apply(prev, [(p, int(v))])))
                        prev = out[-1][1]
                if not np.array_equal(prev, base):
                    out.append((prev, base))
                    prev = base
        return out

    def gating(self) -> dict[int, list[str]]:
        """Per parameter, what makes it live: its state at the reference, then each revival found."""
        out = {}
        for p in sorted(self.read):
            lines = [f"{self.status[0][p].state} at the reference"]
            for i, found in enumerate(self.revivals):
                for r in found.get(p, []):
                    verb = "live" if r.kind == DEAD else "steps matter"
                    line = f"{verb} when {r}" + (f" from state {i}" if i else "")
                    if line not in lines:
                        lines.append(line)
            out[p] = lines
        return out

    def as_dict(self) -> dict:
        """JSON-ready."""
        return {
            "program": self.m.program.name,
            "states": [s.tolist() for s in self.states],
            "status": [{str(p): st.state for p, st in s.items()} for s in self.status],
            "chosen": self.chosen,
            "stepped": {str(i): ps for i, ps in self.stepped.items()},
            "steps": self.steps(),
            "one_at_a_time": {
                "states": 1,
                "steps": sum(len(representatives(self.m, p, bits)) for p, bits in self.read.items()),
                "live": sum(st.state == LIVE for st in self.status[0].values()),
            },
            "gating": {str(p): lines for p, lines in self.gating().items()},
            "uncovered": [list(e) for e in self.uncovered],
        }

    def table(self) -> str:
        """Plain text for the CLI."""
        names = {d.parameter_id: d.name for d in self.m.params}
        lines = [
            f"{self.m.program.name}: {len(self.chosen)}/{len(self.states)} base states, {self.steps()} steps"
        ]
        for p, said in self.gating().items():
            lines.append(f"  P{p:<2} {names.get(p, '')[:16]:16s} " + "; ".join(said))
        for i in self.chosen:
            stepped = ", ".join(f"P{p}" for p in self.stepped.get(i, []))
            lines.append(f"  state {i}: {self.states[i].tolist()} steps {stepped or 'nothing'}")
        lines += [f"  never covered: {e}" for e in self.uncovered]
        return "\n".join(lines)


def elements(read: dict[int, tuple[int, ...]], revivals: list[dict[int, list[Revival]]]) -> list[tuple]:
    """What a cover must reach: every read bit, and each parameter live with each gate reviving it."""
    gates = {
        ("gate", r.param, r.gate, r.spans, () if r.spans else tuple(v for _, v in r.moves))
        for found in revivals
        for rs in found.values()
        for r in rs
    }
    return [("bit", p, b) for p, bits in sorted(read.items()) for b in bits] + sorted(gates)


def covers(element: tuple, state: np.ndarray, st: Status) -> bool:
    """Whether stepping ``st.param`` at ``state`` reaches ``element``."""
    if st.state != LIVE or element[1] != st.param:
        return False
    if element[0] == "bit":
        return element[2] in st.live_bits
    _, _, gates, spans, values = element
    if spans:
        return any(lo <= int(state[gates[0] - 1]) <= hi for lo, hi in spans)
    return all(int(state[g - 1]) == v for g, v in zip(gates, values))


def solve(
    states: Sequence[np.ndarray],
    found: Sequence[dict[int, Status]],
    wanted: Sequence[tuple],
    cost: Callable[[int, int], float],
) -> tuple[list[int], dict[int, list[int]], list[tuple]]:
    """Fewest states, then fewest steps, stepping parameters so every coverable element is reached.

    Binary ``y[s]`` picks a state and ``z[s, p]`` steps ``p`` there with ``z <= y``; each element needs
    one covering ``z``. The state weight exceeds every step cost combined, so the order is lexicographic.
    """
    pairs = [(s, p) for s, sts in enumerate(found) for p, st in sts.items() if st.state == LIVE]
    hit = [[k for k, (s, p) in enumerate(pairs) if covers(e, states[s], found[s][p])] for e in wanted]
    missed = [e for e, ks in zip(wanted, hit) if not ks]
    rows = [ks for ks in hit if ks]
    if not rows:
        return [], {}, missed
    n_y, n_z = len(states), len(pairs)
    costs = np.array([cost(s, p) for s, p in pairs], dtype=np.float64)
    r = [i for i, ks in enumerate(rows) for _ in ks] + [len(rows) + k for k in range(n_z) for _ in (0, 1)]
    col = [n_y + k for ks in rows for k in ks] + [c for k, (s, _) in enumerate(pairs) for c in (n_y + k, s)]
    val = [1.0] * sum(map(len, rows)) + [1.0, -1.0] * n_z
    a = coo_matrix((val, (r, col)), shape=(len(rows) + n_z, n_y + n_z)).tocsr()
    res = milp(
        np.r_[np.full(n_y, 1.0 + costs.sum()), costs],
        constraints=LinearConstraint(
            a,
            np.r_[np.ones(len(rows)), np.full(n_z, -np.inf)],
            np.r_[np.full(len(rows), np.inf), np.zeros(n_z)],
        ),
        integrality=np.ones(n_y + n_z),
        bounds=Bounds(0, 1),
    )
    if not res.success:
        raise RuntimeError(f"set cover found no solution: {res.message}")
    x = np.rint(res.x).astype(bool)
    stepped: dict[int, list[int]] = {}
    for k, (s, p) in enumerate(pairs):
        if x[n_y + k]:
            stepped.setdefault(s, []).append(p)
    return [s for s in range(n_y) if x[s]], stepped, missed


def cover(m: ProgramModel, reference: np.ndarray | None = None, *, gating: Gating | None = None) -> Cover:
    """Gating structure and the covering base states of a program, from ``reference`` (default: initial).

    Revivals are followed from every state reached, each control moved at most once along a path.
    """
    g = gating or Gating(m)
    states = [m.initial() if reference is None else np.asarray(reference, dtype=np.int64)]
    moved, seen = [frozenset()], {g.key(states[0])}
    found_status, found_revivals = [], []
    for i in itertools.count():
        if i == len(states):
            break
        sts = g.statuses(states[i])
        here = {p: g.revivals(states[i], st) for p, st in sts.items() if st.state != LIVE}
        for r in (r for rs in here.values() for r in rs):
            gates = {gate for gate, _ in r.moves}
            nxt = _apply(states[i], r.moves)
            if not gates & moved[i] and g.key(nxt) not in seen:
                seen.add(g.key(nxt))
                states.append(nxt)
                moved.append(moved[i] | gates)
        found_status.append(sts)
        found_revivals.append(here)
    read = g.read
    chosen, stepped, missed = solve(
        states,
        found_status,
        elements(read, found_revivals),
        lambda s, p: len(representatives(m, p, found_status[s][p].live_bits)),
    )
    return Cover(m, states, found_status, found_revivals, chosen, stepped, missed, read)

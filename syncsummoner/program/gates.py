"""Which controls make each parameter live, and the fewest base states that step every one where it is.

Live means a written bit is observable at the video (:mod:`.observe`); revivals are read off each
parameter's observability condition, restricted to the state, as the smallest sets of controls whose
settings over their resolution classes satisfy it (by SAT where there is no condition, by the ternary
simulator where there is no miter either);
the states they reach are covered by a lexicographic 0-1 programme (fewest states, then steps).
"""

from __future__ import annotations

import itertools
import json
from functools import cached_property
from dataclasses import dataclass, field
from typing import Callable, Iterable, Sequence

import numpy as np
from scipy.optimize import Bounds, LinearConstraint, milp
from scipy.sparse import coo_matrix

from syncsummoner.progress import track

from . import declared
from .declared import REGISTER_BITS, TIMING_REGISTER, VALUE_MAX, classes, flat_bit, register_value, registers
from .model import TIMING, ProgramModel

LIVE, DEAD = "live", "dead"


@dataclass(frozen=True)
class Status:
    """One parameter at one state: the bits of it observable at the video."""

    param: int
    live_bits: tuple[int, ...]

    @property
    def state(self) -> str:
        """``live`` or ``dead``."""
        return LIVE if self.live_bits else DEAD


@dataclass(frozen=True)
class Revival:
    """Moving ``moves`` from the state it was found at makes ``param`` live.

    For a single control, ``spans`` are the maximal intervals of its manual values over which the
    parameter is live with everything else unchanged; the move sits mid-way in the widest.
    """

    param: int
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


def tour(values: np.ndarray, base: int) -> np.ndarray:
    """``values`` in the order of a closed walk from ``base`` and back that moves at most two settings a step.

    Every closed walk on a line that reaches both ends moves ``2 * (max - min)`` in all; this one, the
    evens ascending and the odds descending, also keeps each move to the next setting but one.
    The walk starts beside ``base`` and ends beside it; ``base`` itself is left out.
    """
    line = np.union1d(np.asarray(values, dtype=np.int64), [int(base)])
    cycle = np.r_[np.arange(0, line.size, 2), np.arange(line.size - 1 - (line.size % 2 == 1), 0, -2)]
    at = int(np.flatnonzero(line[cycle] == int(base))[0])
    walk = line[np.roll(cycle, -at)[1:]]
    return walk[np.isin(walk, values)]


def intervals(ok: np.ndarray) -> list[tuple[int, int]]:
    """Maximal runs of consecutive indices where ``ok`` holds, as inclusive ``(lo, hi)``."""
    edges = np.diff(np.r_[0, ok.astype(np.int8), 0])
    return [(int(a), int(b) - 1) for a, b in zip(np.flatnonzero(edges == 1), np.flatnonzero(edges == -1))]


def _apply(state: np.ndarray, moves: Iterable[tuple[int, int]]) -> np.ndarray:
    out = np.asarray(state, dtype=np.int64).copy()
    for gate, value in moves:
        out[gate - 1] = value
    return out


def condition(m: ProgramModel, param: int, timing: int | None = None):
    """Where some bit of ``param`` is observable, as a BDD over the other declared controls' bits.

    The parameter itself (and, with no ``timing``, register 8) varies; undeclared bits are 0.
    """
    cond = m.conditions({param, *([TIMING] if timing is None else [])})
    bdd = cond.bdd
    named = {f"c{k}" for d in m.params for k in declared.field(d.parameter_id).flat}
    clock = {f"c{flat_bit(TIMING_REGISTER, b)}": b for b in range(REGISTER_BITS)}
    u = bdd.false
    for k in declared.field(param).flat:
        u |= cond.condition(f"c{k}")
    own = {f"c{k}" for k in declared.field(param).flat}
    fixed = {v: False for v in bdd.vars if v not in named | set(clock)}
    if timing is not None:
        fixed |= {v: bool((timing >> b) & 1) for v, b in clock.items() if v in bdd.vars}
    return bdd, bdd.exist(own | (set(clock) if timing is None else set()), bdd.let(fixed, u))


def setting(m: ProgramModel, param: int, manual: int) -> dict[str, bool]:
    """The condition variables of ``param``'s bits at one manual value."""
    f = declared.field(param)
    value = int(register_value(m.decl(param), int(manual)))
    word = value << f.bits[0] if len(f.bits) == 1 else value
    return {f"c{k}": bool((word >> b) & 1) for k, b in zip(f.flat, f.bits)}


def _anyof(bdd, assignments: Iterable[dict]):
    out = bdd.false
    for a in assignments:
        out |= bdd.cube(a)
    return out


def _exactly(bdd, names: Sequence[str], k: int):
    """True where exactly ``k`` of the variables ``names`` are."""
    row = [bdd.true] + [bdd.false] * k
    for n in reversed(names):
        x = bdd.var(n)
        row = [~x & row[0]] + [(~x & row[j]) | (x & row[j - 1]) for j in range(1, k + 1)]
    return row[k]


def fewest(bdd, u, held: dict[int, dict], domains: dict[int, list[dict]]) -> list[tuple[int, ...]]:
    """Every smallest set of gates whose moves satisfy ``u``, the rest held, in lexicographic order.

    ``u`` is over the gates' variables; each gate holds the assignment ``held[g]`` unless moved to one
    of ``domains[g]``. A selector per gate picks moved or held, the gates' variables are quantified
    away, and the smallest satisfiable selector count is read off by intersecting with a cardinality.
    """
    gates = sorted(domains)
    sel = {g: f"s{g}" for g in gates}
    bdd.declare(*sel.values())
    moved = bdd.true
    for g in gates:
        s = bdd.var(sel[g])
        moved &= (s & _anyof(bdd, domains[g])) | (~s & bdd.cube(held[g]))
    names = {n for g in gates for n in held[g]}
    v = bdd.exist(names, u & moved) if names else u & moved
    for k in range(1, len(gates) + 1):
        w = v & _exactly(bdd, [sel[g] for g in gates], k)
        if w != bdd.false:
            picks = bdd.pick_iter(w, care_vars=set(sel.values()))
            return sorted(tuple(g for g in gates if a[sel[g]]) for a in picks)
    return []


def witness(bdd, u, gates: Sequence[int], domains: dict[int, list[dict]]) -> tuple[list[int], np.ndarray]:
    """The first setting of each gate but the last, in domain order, that some later settings complete,
    and which settings of the last gate then satisfy ``u``.
    """
    picks = []
    for i, g in enumerate(gates[:-1]):
        rest = gates[i + 1 :]
        names = {n for r in rest for n in domains[r][0]}
        allowed = bdd.true
        for r in rest:
            allowed &= _anyof(bdd, domains[r])
        for j, a in enumerate(domains[g]):
            f = bdd.let(a, u)
            if bdd.exist(names, f & allowed) != bdd.false:
                picks.append(j)
                u = f
                break
        else:
            return picks, np.zeros(len(domains[gates[-1]]), dtype=bool)
    return picks, np.array([bdd.let(a, u) != bdd.false for a in domains[gates[-1]]], dtype=bool)


def branches(m: ProgramModel, bdd, u, gates: Sequence[int]) -> list[list[tuple[int, list[tuple[int, int]]]]]:
    """``u`` as disjoint branches, each a list of ``(gate, manual-value intervals)`` over ``gates`` in order.

    A gate's manual values are grouped by the condition left once its bits take their value, and each
    group is stated as maximal intervals; gates outside the remaining support are skipped.
    """
    if u == bdd.false:
        return []
    if u == bdd.true or not gates:
        return [[]]
    gate, rest = gates[0], list(gates[1:])
    names = [f"c{k}" for k in declared.field(gate).flat]
    if not set(names) & bdd.support(u):
        return branches(m, bdd, u, rest)
    values = register_value(m.decl(gate), np.arange(VALUE_MAX + 1))
    by_value: dict[int, object] = {}
    cofactor = []
    for manual, v in enumerate(values.tolist()):
        if v not in by_value:
            by_value[v] = bdd.let(setting(m, gate, manual), u)
        cofactor.append(by_value[v])
    out = []
    for f in dict.fromkeys(cofactor):
        ok = np.array([c == f for c in cofactor])
        for tail in branches(m, bdd, f, rest):
            out.append([(gate, intervals(ok)), *tail])
    return out


def structure(m: ProgramModel, timing: int | None = None) -> dict[int, list[str]]:
    """Per declared parameter, every setting of the others that makes it live, read off its condition."""
    out = {}
    for d in m.params:
        p = d.parameter_id
        others = [q.parameter_id for q in m.params if q.parameter_id != p]
        found = branches(m, *condition(m, p, timing), others)
        out[p] = [
            " and ".join(
                f"P{g} in " + ", ".join(f"{lo}" if lo == hi else f"{lo}..{hi}" for lo, hi in spans)
                for g, spans in branch
            )
            or "always"
            for branch in found
        ] or ["never"]
    return out


@dataclass
class Gating:
    """The liveness queries a cover is built from, remembered per state."""

    m: ProgramModel
    timing: int | None = None
    _status: dict = field(default_factory=dict, repr=False)
    _cond: dict = field(default_factory=dict, repr=False)

    def key(self, state: np.ndarray) -> tuple:
        """What the program sees of a state: its register words."""
        return tuple(sorted(registers(self.m.params, np.asarray(state), self.timing).items()))

    @cached_property
    def read(self) -> dict[int, tuple[int, ...]]:
        """Each declared parameter's bits the program reads at all; unread parameters are left out."""
        return {p: bits for p, bits in self.m.resolution().items() if bits}

    def statuses(self, state: np.ndarray) -> dict[int, Status]:
        """Every read parameter's status at ``state``."""
        key = self.key(state)
        if key not in self._status:
            self.m.prepare({p, *([TIMING] if self.timing is None else [])} for p in self.read)
            self._status[key] = {
                p: Status(p, tuple(self.m.reach(state, [p], self.timing)[p])) for p in sorted(self.read)
            }
        return self._status[key]

    def _could(self, state: np.ndarray, param: int, free: Sequence[int]) -> bool:
        """False only when no held values of ``free`` can revive ``param``: an exact refutation."""
        return self.m.revivable(state, param, free, self.timing)

    @cached_property
    def _classes(self) -> dict[int, tuple[list[np.ndarray], list[int]]]:
        """Each read control's resolution classes and a representative manual value of each."""
        out = {}
        for q, bits in self.read.items():
            groups = classes(self.m.decl(q), bits)
            out[q] = (groups, [int(g[len(g) // 2]) for g in groups])
        return out

    def _condition(self, param: int):
        if param not in self._cond:
            self._cond[param] = condition(self.m, param, self.timing)
        return self._cond[param]

    def revivals(self, state: np.ndarray, st: Status) -> list[Revival]:
        """Every way of making ``st.param`` live by moving the fewest other controls from ``state``.

        Answered from the parameter's observability condition where the program has one, else by SAT,
        else by the ternary simulator.
        """
        p = st.param
        others = [q for q in sorted(self.read) if q != p]
        if self.m.conditions({p, *([TIMING] if self.timing is None else [])}) is None:
            if self.m.method == "sat":
                return self._solved(state, p, others)
            return self._simulated(state, p, others)
        bdd, u = self._condition(p)
        held = {
            d.parameter_id: setting(self.m, d.parameter_id, state[d.parameter_id - 1]) for d in self.m.params
        }
        u = bdd.let({n: x for q, a in held.items() if q != p and q not in others for n, x in a.items()}, u)
        domains = {q: [setting(self.m, q, v) for v in self._classes[q][1]] for q in others}
        found = []
        for qs in track(
            fewest(bdd, u, {q: held[q] for q in others}, domains), desc=f"P{p} gates", unit="set"
        ):
            f = bdd.let({n: x for q in others if q not in qs for n, x in held[q].items()}, u)
            picks, ok = witness(bdd, f, qs, domains)
            groups = self._classes[qs[-1]][0]
            full = np.zeros(VALUE_MAX + 1, dtype=bool)
            for g, hit in zip(groups, ok):
                full[g] = hit
            spans = intervals(full)
            lo, hi = max(spans, key=lambda s: s[1] - s[0])
            moves = (*((g, self._classes[g][1][j]) for g, j in zip(qs, picks)), (qs[-1], (lo + hi) // 2))
            found.append(Revival(p, moves, tuple(spans) if len(qs) == 1 else ()))
        return found

    def _alive(self, states: Sequence[np.ndarray], p: int, free: Sequence[int]) -> list[bool]:
        """Per state, whether ``p`` is live by SAT with the controls in ``free`` left free."""
        ties = [self.m.ties(s, self.timing, [p, *free]) for s in states]
        return self.m.exact.alive(ties, declared.field(p).flat)

    def _solved(self, state: np.ndarray, p: int, others: list[int]) -> list[Revival]:
        """:meth:`revivals` by SAT, in the order and form the conditions give them.

        A set of gates is tried with its controls free, which refutes it exactly; a set that survives is
        completed one gate at a time over its classes, the first that the rest can still complete kept.
        """
        if not self._alive([state], p, others)[0]:
            return []
        for size in range(1, len(others) + 1):
            sets = list(itertools.combinations(others, size))
            found = []
            for qs in track(sets, desc=f"P{p} gates of {size}", unit="set"):
                if self._alive([state], p, qs)[0]:
                    found += self._complete(state, p, qs, ())
            if found:
                return found
        return []

    def _complete(self, state: np.ndarray, p: int, qs: Sequence[int], picks: tuple) -> list[Revival]:
        """The revival through ``qs`` with the first gates held at the class indices ``picks``."""
        moved = _apply(state, ((g, self._classes[g][1][j]) for g, j in zip(qs, picks)))
        gate = qs[len(picks)]
        groups, reps = self._classes[gate]
        if len(picks) < len(qs) - 1:
            rest = qs[len(picks) + 1 :]
            for j, v in enumerate(reps):
                nxt = _apply(moved, [(gate, v)])
                if self._alive([nxt], p, rest)[0]:
                    found = self._complete(state, p, qs, (*picks, j))
                    if found:
                        return found
            return []
        ok = self._alive([_apply(moved, [(gate, v)]) for v in reps], p, ())
        full = np.zeros(VALUE_MAX + 1, dtype=bool)
        for g, hit in zip(groups, ok):
            full[g] = hit
        spans = intervals(full)
        if not spans:
            return []
        lo, hi = max(spans, key=lambda s: s[1] - s[0])
        head = tuple((g, self._classes[g][1][j]) for g, j in zip(qs, picks))
        return [Revival(p, (*head, (gate, (lo + hi) // 2)), tuple(spans) if len(qs) == 1 else ())]

    def _simulated(self, state: np.ndarray, p: int, others: list[int]) -> list[Revival]:
        if not self._could(state, p, others):
            return []
        for size in range(1, len(others) + 1):
            sets = list(itertools.combinations(others, size))
            hits = [self._could(state, p, list(qs)) for qs in track(sets, desc=f"P{p} gates of {size}")]
            found = [r for qs, hit in zip(sets, hits) if hit for r in self.scan(state, p, qs)]
            if found:
                return found
        return []

    def scan(self, state: np.ndarray, param: int, gates: Sequence[int]) -> list[Revival]:
        """Settings of ``gates`` that revive ``param``: intervals for one control, a witness for several."""
        gate, rest = gates[0], list(gates[1:])
        groups = classes(self.m.decl(gate), self.read[gate])
        reps = [int(g[len(g) // 2]) for g in groups]
        moved = [_apply(state, [(gate, v)]) for v in reps]
        hits = [bool(r[param]) for r in self.m.reach_many(moved, [param, *rest], self.timing)]
        if not rest:
            ok = np.zeros(VALUE_MAX + 1, dtype=bool)
            for g, hit in zip(groups, hits):
                ok[g] = hit
            spans = intervals(ok)
            if not spans:
                return []
            lo, hi = max(spans, key=lambda s: s[1] - s[0])
            return [Revival(param, ((gate, (lo + hi) // 2),), tuple(spans))]
        for v, hit in zip(reps, hits):
            deeper = self.scan(_apply(state, [(gate, v)]), param, rest) if hit else []
            if deeper:
                return [Revival(param, ((gate, v), *deeper[0].moves))]
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
    timing: int | None = None

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
        """``(from, to)`` full states in sweep order: each base state, then one parameter at a time off it.

        Each parameter's settings are walked by :func:`tour` from the base value and back.
        """
        out, prev = [], self.reference
        for i in self.chosen:
            base = self.states[i]
            out.append((prev, base))
            prev = base
            for p in self.stepped.get(i, []):
                for v in tour(self.schedule(i, p), int(base[p - 1])):
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
                    line = f"live when {r}" + (f" from state {i}" if i else "")
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
        if self.m.method == "bdd":
            for p, where in structure(self.m, self.timing).items():
                if p in self.read:
                    lines.append(f"  P{p:<2} live where " + "; or ".join(where))
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
    return Cover(m, states, found_status, found_revivals, chosen, stepped, missed, read, g.timing)


def report(
    m: ProgramModel, reference: np.ndarray, timing: int | None = None, *, as_json: bool = False
) -> str:
    """The cover from ``reference`` as the CLI prints it: a table, or JSON."""
    found = cover(m, reference, gating=Gating(m, timing))
    return json.dumps(found.as_dict(), indent=1) if as_json else found.table()

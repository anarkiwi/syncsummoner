"""Exact observability by SAT: two copies of the gate-level netlist as a miter on one incremental solver.

Every free signal (control bit, input bit, storage output) is a pair of variables, one per copy, held
equal unless its enable is free; a query is a set of assumptions over one CNF built once per circuit.
docs/program-model.md gives the register rules this applies exactly.
"""

from __future__ import annotations

from collections import OrderedDict, defaultdict
from typing import Iterable, Mapping, Sequence

import numpy as np
from pysat.solvers import Solver

from syncsummoner.progress import track

from . import hdl, observe
from .observe import GATES, Cell, Circuit, next_state
from .ternary import Ternary

SOLVER = "cadical195"
#: Name prefixes of the video input ports, from source and from the bitstream.
VIDEO = ("data_in", "ADV7611_D")


class Lit:
    """One CNF literal under ``&``, ``|`` and ``~``, built by its :class:`Cnf`."""

    __slots__ = ("cnf", "v")

    def __init__(self, cnf: "Cnf", v: int):
        self.cnf, self.v = cnf, v

    def __and__(self, other: "Lit") -> "Lit":
        return Lit(self.cnf, self.cnf.conj(self.v, other.v))

    def __or__(self, other: "Lit") -> "Lit":
        return Lit(self.cnf, -self.cnf.conj(-self.v, -other.v))

    def __invert__(self) -> "Lit":
        return Lit(self.cnf, -self.v)


class Cnf:
    """Tseitin clauses with constant folding and structural hashing; variable 1 is true."""

    def __init__(self):
        self.top = 1
        self.clauses: list[list[int]] = [[1]]
        self._memo: dict[tuple, int] = {}

    def var(self) -> int:
        """A fresh variable."""
        self.top += 1
        return self.top

    def conj(self, a: int, b: int) -> int:
        """``a & b``."""
        if a == -1 or b == -1 or a == -b:
            return -1
        if a in (1, b):
            return b
        if b == 1:
            return a
        key = ("&", min(a, b), max(a, b))
        if key not in self._memo:
            v = self.var()
            self.clauses += [[-v, a], [-v, b], [v, -a, -b]]
            self._memo[key] = v
        return self._memo[key]

    def xor2(self, a: int, b: int) -> int:
        """``a ^ b``."""
        if abs(a) == 1:
            return -b if a == 1 else b
        if abs(b) == 1:
            return -a if b == 1 else a
        if a == b:
            return -1
        if a == -b:
            return 1
        sign = (a < 0) ^ (b < 0)
        a, b = sorted((abs(a), abs(b)))
        key = ("^", a, b)
        if key not in self._memo:
            v = self.var()
            self.clauses += [[-v, a, b], [-v, -a, -b], [v, -a, b], [v, a, -b]]
            self._memo[key] = v
        return -self._memo[key] if sign else self._memo[key]

    def mux(self, s: int, a: int, b: int) -> int:
        """``s ? a : b``."""
        if s == 1 or a == b:
            return a
        if s == -1:
            return b
        if a == -b:
            return -self.xor2(s, a)
        if abs(a) == 1 or abs(b) == 1 or abs(a) == abs(s) or abs(b) == abs(s):
            return -self.conj(-self.conj(s, a), -self.conj(-s, b))
        key = ("?", s, a, b)
        if key not in self._memo:
            v = self.var()
            self.clauses += [[-s, -a, v], [-s, a, -v], [s, -b, v], [s, b, -v], [-a, -b, v], [a, b, -v]]
            self._memo[key] = v
        return self._memo[key]

    def lit(self, v: int) -> Lit:
        """``v`` as a :class:`Lit`."""
        return Lit(self, v)

    def const(self, value) -> Lit:
        """A constant."""
        return Lit(self, 1 if value else -1)

    def ite(self, s: Lit, a: Lit, b: Lit) -> Lit:
        """``s ? a : b``."""
        return Lit(self, self.mux(s.v, a.v, b.v))

    def xor(self, a: Lit, b: Lit) -> Lit:
        """``a ^ b``."""
        return Lit(self, self.xor2(a.v, b.v))


class Words:
    """Bit-parallel Boolean algebra over ``uint64`` lanes: each bit of each word is one assignment."""

    def __init__(self, width: int):
        self.width = width

    def const(self, value) -> np.ndarray:
        """A constant in every lane; a vector of them gives one row per entry."""
        if np.ndim(value):
            ones = np.where(np.asarray(value, dtype=bool), ~np.uint64(0), np.uint64(0))
            return np.repeat(ones[:, None], self.width, axis=1)
        return np.full(self.width, ~np.uint64(0) if value else np.uint64(0), dtype=np.uint64)

    @staticmethod
    def ite(s: np.ndarray, a: np.ndarray, b: np.ndarray) -> np.ndarray:
        """``s ? a : b``."""
        return (s & a) | (~s & b)

    @staticmethod
    def xor(a: np.ndarray, b: np.ndarray) -> np.ndarray:
        """``a ^ b``."""
        return a ^ b


class Lanes:
    """Random concrete assignments of both copies, simulated together: each lane is a model of the
    :class:`Miter`'s CNF with every black box equal in both copies, so what differs in it is a witness.
    """

    def __init__(self, circuit: Circuit, width: int = 4, seed: int = 0):
        self.ternary = Ternary(circuit)
        self.alg = Words(width)
        self.rng = np.random.default_rng(seed)
        self.rows = len(self.ternary.index) + 3
        self.nets = np.array(sorted(self.ternary.index, key=self.ternary.index.get), dtype=np.int64)
        self.q = np.array([self.row(c.outs["Q"][0]) for c in circuit.storage], dtype=np.int64)
        kinds: dict[tuple, list[int]] = defaultdict(list)
        for i, c in enumerate(circuit.storage):
            kinds[(c.kind, c.code, tuple(sorted(c.ins)))].append(i)
        self.groups = [
            (
                circuit.storage[index[0]],
                np.array(index),
                {
                    p: np.array([self.ternary.at(circuit.storage[i].ins[p][0]) for i in index])
                    for p in circuit.storage[index[0]].ins
                },
            )
            for index in kinds.values()
        ]

    def draw(self) -> np.ndarray:
        """Every row random, the constant rows constant."""
        t = self.ternary
        x = self.rng.integers(0, 2**64, size=(self.rows, self.alg.width), dtype=np.uint64)
        x[t.zero], x[t.one] = self.alg.const(False), self.alg.const(True)
        return x

    def forward(self, x: np.ndarray, alg: Words | None = None) -> None:
        """Every gate output from its inputs, in place."""
        alg = alg or self.alg
        for level in self.ternary.levels:
            for kind, ins, outs in level:
                x[outs] = GATES[kind](alg, *(x[ins[:, k]] for k in range(ins.shape[1])))

    def steps(self, x: np.ndarray, q: np.ndarray, fire: bool, alg: Words | None = None) -> np.ndarray:
        """Every storage bit's next state in every lane, from outputs ``q`` (one row per bit)."""
        alg = alg or self.alg
        out = np.empty_like(q)
        for cell, index, pins in self.groups:
            got = {p: x[rows] for p, rows in pins.items()}
            out[index] = next_state(cell, got.__getitem__, q[index], alg, fire)
        return out

    def row(self, net: int) -> int:
        """Row of a net."""
        return self.ternary.index[net]

    def sensitised(
        self, ties: Sequence[Mapping[int, int]], fixed: Sequence[Mapping[int, bool]], sources: Sequence
    ) -> list[set[int]]:
        """Per lane, the nets its ``sources`` can change by the ternary sensitisation of every pin."""
        t = self.ternary
        rows = np.full((len(ties), max(t.circuit.controls(), default=0) + 1), -1, dtype=np.int8)
        for i, tie in enumerate(ties):
            for k, v in tie.items():
                if k < rows.shape[1]:
                    rows[i, k] = v
        seen = t.reach(rows, fixed, sources)
        return [set(self.nets[np.flatnonzero(seen[: len(self.nets), i])].tolist()) for i in range(len(ties))]


class Miter:
    """Copies ``a`` and ``b`` of one circuit, with each storage bit's next states, on one solver.

    A pair ``(a, b, e)`` stands for one free signal in both copies: ``a == b`` unless ``e``.
    A black box's outputs are equal in both copies wherever all its inputs are.
    """

    def __init__(self, circuit: Circuit, solver: str = SOLVER):
        self.circuit = circuit
        cnf = self.cnf = Cnf()
        order, looped = circuit.order()
        self.boxes = [*circuit.boxes, *looped]
        self.controls = circuit.controls()
        self.inputs = circuit.inputs_named()
        self.storage = list(circuit.storage)
        self.pairs: dict[int, tuple[int, int, int]] = {}
        for net in [*self.controls.values(), *self.inputs, *(c.outs["Q"][0] for c in self.storage)]:
            self.pairs[net] = self._pair()
        self._undef: dict[tuple, int] = {}
        self._eqs: dict[tuple[int, int], int] = {}
        val = self._val = ({n: p[0] for n, p in self.pairs.items()}, {n: p[1] for n, p in self.pairs.items()})
        for box in self.boxes:
            for b in (b for bits in box.outs.values() for b in bits if isinstance(b, int)):
                val[0][b], val[1][b] = cnf.var(), cnf.var()
        for g in order:
            out = next(iter(g.outs.values()))[0]
            for side in val:
                pins = (cnf.lit(self._pin(side, g, p)) for p in sorted(g.ins))
                side[out] = GATES[g.kind](cnf, *pins).v
        self._box(val)
        self.fire: dict[int, int] = {}
        self.keep: dict[int, int] = {}
        self.moves: dict[int, int] = {}
        self._storage(val, order)
        outs = [b for bits in circuit.outputs.values() for b in bits if isinstance(b, int)]
        self.outputs = {o: cnf.xor2(val[0][o], val[1][o]) for o in outs}
        self.gate_nets = [next(iter(g.outs.values()))[0] for g in order]
        self.fanin = {
            next(iter(g.outs.values()))[0]: [b for bits in g.ins.values() for b in bits if isinstance(b, int)]
            for g in order
        } | {
            c.outs["Q"][0]: [b for bits in c.ins.values() for b in bits if isinstance(b, int)]
            for c in self.storage
        }
        self.gate_fanin = {n: self.fanin[n] for n in self.gate_nets}
        self.solver = Solver(name=solver, bootstrap_with=cnf.clauses)
        self.flushed = len(cnf.clauses)
        self.solves = 0
        self.cells = {c.outs["Q"][0]: c for c in self.storage}
        self.readers = self._readers(self.fanin)
        self.gate_readers = self._readers(self.gate_fanin)
        self._after: dict[int, frozenset] = {}
        self.qs = list(self.cells)
        self.start = np.array([circuit.init.get(q, False) for q in self.qs])
        self.lanes = Lanes(circuit)
        self.support = self._supports(order)
        self._links: dict[int, int] = {}

    def _supports(self, order: Sequence[Cell]) -> dict[int, tuple[int, int, bool]]:
        """Per storage output, what its next state with enables fired reads through gates: control bits
        and storage outputs as bit masks, and whether anything else (an input, a box, an undefined bit).
        """
        ctl = {net: k for k, net in self.controls.items()}
        reg = {q: i for i, q in enumerate(self.qs)}
        sup: dict[int, tuple[int, int, bool]] = {}

        def of(b) -> tuple[int, int, bool]:
            if b in ctl:
                return 1 << ctl[b], 0, False
            if b in reg:
                return 0, 1 << reg[b], False
            if b in sup:
                return sup[b]
            return 0, 0, b not in ("0", "1")

        def union(bits) -> tuple[int, int, bool]:
            c = r = 0
            o = False
            for b in bits:
                bc, br, bo = of(b)
                c, r, o = c | bc, r | br, o or bo
            return c, r, o

        for g in order:
            sup[next(iter(g.outs.values()))[0]] = union(b for bits in g.ins.values() for b in bits)
        return {
            c.outs["Q"][0]: union(b for p, bits in c.ins.items() if p not in ("C", "E") for b in bits)
            for c in self.storage
        }

    def links(self, ties: Mapping[int, int], held: Iterable[int], settled: Mapping[int, bool]) -> frozenset:
        """Storage that loads a function of tied and ``held`` control bits alone, through such storage:
        it settles to that function's value at every setting of the held bits, so it is linked to it.
        """
        allowed = sum(1 << k for k in {*ties, *held})
        reg = {q: i for i, q in enumerate(self.qs)}
        known = sum(1 << reg[q] for q in settled)
        todo = [
            q
            for q in self.qs
            if q not in settled and not self.support[q][2] and not self.support[q][0] & ~allowed
        ]
        linked: set[int] = set()
        while True:
            more = {q for q in todo if q not in linked and not self.support[q][1] & ~known}
            if not more:
                return frozenset(linked)
            linked |= more
            known |= sum(1 << reg[q] for q in more)

    def _link(self, q: int) -> int:
        """A literal that, assumed, makes storage output ``q`` its next state with enables fired."""
        if q not in self._links:
            lit, a, f = self.cnf.var(), self.pairs[q][0], self.fire[q]
            self.cnf.clauses += [[-lit, -a, f], [-lit, a, -f]]
            self._links[q] = lit
        return self._links[q]

    def _readers(self, fanin: Mapping[int, Iterable[int]]) -> dict[int, list[int]]:
        """Nets each net feeds within a step: by ``fanin``, and every output of a box it is an input of."""
        out: dict[int, list[int]] = defaultdict(list)
        for net, ins in fanin.items():
            for b in ins:
                out[b].append(net)
        for box in self.boxes:
            outs = [b for bits in box.outs.values() for b in bits if isinstance(b, int)]
            for b in (b for bits in box.ins.values() for b in bits if isinstance(b, int)):
                out[b] += outs
        return out

    def _box(self, val: tuple[dict, dict]) -> None:
        """Each black box's outputs equal in both copies wherever all its inputs are."""
        cnf = self.cnf
        for box in self.boxes:
            diff = [
                cnf.xor2(self._pin(val[0], box, p, i), self._pin(val[1], box, p, i))
                for p, bits in box.ins.items()
                for i in range(len(bits))
            ]
            same = cnf.var()
            cnf.clauses.append([same, *(d for d in diff if d != -1)])
            for o in (b for bits in box.outs.values() for b in bits if isinstance(b, int)):
                cnf.clauses += [[-same, -val[0][o], val[1][o]], [-same, val[0][o], -val[1][o]]]

    def _storage(self, val: tuple[dict, dict], order: Sequence[Cell]) -> None:
        """Each storage bit's next state with enables fired, from its initial value, and in both copies."""
        cnf, init = self.cnf, self.circuit.init
        driver = {next(iter(g.outs.values()))[0]: i for i, g in enumerate(order)}
        for cell in self.storage:
            q = cell.outs["Q"][0]
            self.fire[q] = self._next(cell, val[0], cnf.lit(val[0][q]), True)
            if q in init:
                start = self._from(val[0], order, driver, cell, init[q])
                self.keep[q] = self._next(cell, start, cnf.const(init[q]), False)
            na = self._next(cell, val[0], cnf.lit(val[0][q]), False)
            nb = self._next(cell, val[1], cnf.lit(val[1][q]), False)
            self.moves[q] = cnf.xor2(na, nb)

    def _pair(self) -> tuple[int, int, int]:
        a, b, e = self.cnf.var(), self.cnf.var(), self.cnf.var()
        self.cnf.clauses += [[e, -a, b], [e, a, -b]]
        return a, b, e

    def _pin(self, side: dict, cell: Cell, pin: str, index: int = 0) -> int:
        """A pin's literal in one copy; an undefined bit is one free variable shared by both."""
        b = cell.ins[pin][index]
        if isinstance(b, int):
            return side[b]
        if b in ("0", "1"):
            return 1 if b == "1" else -1
        return self._undef.setdefault((id(cell), pin, index), self.cnf.var())

    def _from(self, side: dict, order: Sequence[Cell], driver: Mapping[int, int], cell: Cell, value: bool):
        """``side`` with ``cell``'s own output at ``value`` through the gates between it and its pins."""
        q = cell.outs["Q"][0]
        cone, stack = set(), [b for bits in cell.ins.values() for b in bits if b in driver]
        while stack:
            i = driver[stack.pop()]
            if i not in cone:
                cone.add(i)
                stack += [b for bits in order[i].ins.values() for b in bits if b in driver]
        out = _Overlay(side, {q: 1 if value else -1})
        for i in sorted(cone):
            g = order[i]
            if any(b in out.own for bits in g.ins.values() for b in bits):
                pins = (self.cnf.lit(self._pin(out, g, p)) for p in sorted(g.ins))
                out.own[next(iter(g.outs.values()))[0]] = GATES[g.kind](self.cnf, *pins).v
        return out

    def _next(self, cell: Cell, side: Mapping, q: Lit, fire: bool) -> int:
        cnf = self.cnf
        return next_state(cell, lambda p: cnf.lit(self._pin(side, cell, p)), q, cnf, fire).v

    def solve(self, assumptions: Sequence[int], budget: int | None = None) -> list[int] | bool | None:
        """A model under ``assumptions``, or None; with a conflict ``budget``, False when it runs out."""
        if len(self.cnf.clauses) > self.flushed:
            self.solver.append_formula(self.cnf.clauses[self.flushed :])
            self.flushed = len(self.cnf.clauses)
        self.solves += 1
        if budget is None:
            found = self.solver.solve(assumptions=list(assumptions))
        else:
            self.solver.conf_budget(budget)
            found = self.solver.solve_limited(assumptions=list(assumptions))
            if found is None:
                return False
        return self.solver.get_model() if found else None

    @property
    def conflicts(self) -> int:
        """Conflicts the solver has spent so far."""
        return self.solver.accum_stats().get("conflicts", 0)

    def _any(self, lits: Iterable[int]) -> int:
        """An activation literal that, assumed, requires one of ``lits``."""
        act = self.cnf.var()
        self.cnf.clauses.append([-act, *lits])
        return act

    def tied(self, ties: Mapping[int, int]) -> list[int]:
        """Assumptions fixing control bits (by flat position) in copy ``a``."""
        return [
            self.pairs[self.controls[k]][0] * (1 if v else -1) for k, v in ties.items() if k in self.controls
        ]

    def _const(self, x: np.ndarray, nets: Mapping[int, bool]) -> None:
        if nets:
            rows = [self.lanes.row(n) for n in nets]
            x[rows] = self.lanes.alg.const(list(nets.values()))

    def _controls(self, ties: Mapping[int, int]) -> dict[int, bool]:
        return {self.controls[k]: bool(v) for k, v in ties.items() if k in self.controls}

    def after(self, bits: Iterable[int]) -> set[int]:
        """Storage outputs whose next states depend, through gates and storage, on the control ``bits``."""
        out: set[int] = set()
        for k in bits:
            if k not in self._after:
                self._after[k] = frozenset(self.cells) & _closure(self.readers, [self.controls[k]])
            out |= self._after[k]
        return out

    def settle(
        self,
        ties: Mapping[int, int],
        known: Mapping[int, bool] | None = None,
        moved: Iterable[int] = (),
    ) -> dict[int, bool]:
        """Storage outputs that settle, and their values, with the untied controls free.

        The least fixpoint of the load rule (next state with enables fired constant over every free
        signal), then the keep rule (the initial value is kept). Random lanes refute candidates; the
        nets they feed that the lanes show constant, and propagation does not, are proven constant in
        order, then the candidates, models refuting more. Given ``known``, settled at ties that differ
        only in the control bits ``moved``, only storage depending on those bits is decided again.
        """
        lanes, init = self.lanes, self.circuit.init
        base = self.tied(ties) + [-p[2] for p in self.pairs.values()]
        open_ = set(self.cells) if known is None else self.after(k for k in moved if k in self.controls)
        settled = {q: v for q, v in (known or {}).items() if q not in open_}
        while True:
            x = lanes.draw()
            self._const(x, self._controls(ties) | settled)
            lanes.forward(x)
            fired = lanes.steps(x, x[lanes.q], True)
            steady = (x == x[:, :1]).all(axis=1) & ((x[:, 0] == 0) | (x[:, 0] == ~np.uint64(0)))
            level = x[:, 0] != 0
            free = [q for q in init if q not in settled]
            self._const(x, {q: init[q] for q in free})
            lanes.forward(x)
            kept = lanes.steps(x, lanes.alg.const(self.start), False) == lanes.alg.const(self.start)
            ones, zeros = (fired == ~np.uint64(0)).all(axis=1), (fired == 0).all(axis=1)
            loads = {
                q: bool(ones[i])
                for i, q in enumerate(self.qs)
                if q in open_ and q not in settled and ones[i] | zeros[i]
            }
            keeps = {
                q
                for i, q in enumerate(self.qs)
                if q in init and q in open_ and q not in settled and kept[i].all()
            }
            ctx = self._under(base + [self.pairs[q][0] * (1 if v else -1) for q, v in settled.items()])
            plain = self._known(ties, settled)
            cone = _closure(self.gate_fanin, [b for q in (*loads, *keeps) for b in self.fanin[q]])
            nets = [
                n for n in self.gate_nets if n in cone and steady[lanes.row(n)] and not plain[lanes.row(n)]
            ]
            items = [(self._val[0][n], bool(level[lanes.row(n)])) for n in nets]
            items += [(self.fire[q], v) for q, v in loads.items()]
            items += [(self.keep[q], init[q]) for q in keeps]
            proven = self._constants(ctx, items)
            self.cnf.clauses.append([-ctx])
            loads = {q: v for q, v in loads.items() if (self.fire[q], v) in proven}
            new = loads | {q: init[q] for q in keeps if q not in loads and (self.keep[q], init[q]) in proven}
            if not new:
                return settled
            settled |= new

    def _known(self, ties: Mapping[int, int], settled: Mapping[int, bool]) -> np.ndarray:
        """Rows the ternary simulator knows, by propagation alone, at ``ties`` with ``settled`` held."""
        t = self.lanes.ternary
        row = np.full((1, max(self.controls, default=0) + 1), -1, dtype=np.int8)
        for k, v in ties.items():
            if k < row.shape[1]:
                row[0, k] = v
        z, o = t.rails(row, [settled])
        return (z | o)[:, 0]

    def _constants(self, ctx: int, items: Sequence[tuple[int, bool]]) -> set[tuple[int, bool]]:
        """The literals of ``items`` proven at their values under ``ctx``, in order, each from those
        before; a model refutes every later item it contradicts.
        """
        proven: set[tuple[int, bool]] = set()
        refuted: set[int] = set()
        for lit, v in track(items, desc="constants", unit="net"):
            if lit in refuted or abs(lit) == 1:
                if abs(lit) == 1 and (lit == 1) == v:
                    proven.add((lit, v))
                continue
            model = self.solve([ctx, -lit if v else lit])
            if model is None:
                proven.add((lit, v))
                self.cnf.clauses.append([-ctx, lit if v else -lit])
                continue
            value = _reader(model)
            refuted |= {m for m, w in items if value(m) != w}
        return proven

    def observable(
        self,
        ties: Mapping[int, int],
        settled: Mapping[int, bool],
        source: Sequence[int],
        video: bool = False,
        reach: set[int] | None = None,
        linked: frozenset = frozenset(),
    ) -> bool:
        """Whether flipping control bits ``source`` (or every input bit, with ``video``) can change an
        output, storage in ``settled`` held, in ``linked`` at its next state, and the rest free; no net
        outside ``reach`` can differ.

        A free storage bit may differ between the copies once its next states can, to a least fixpoint.
        Random lanes find differences; the nets that never differ in them are proven equal in order.
        """
        nets = [self.controls[k] for k in source if k in self.controls]
        flip = [lit for n in nets for lit in (-self.pairs[n][0], self.pairs[n][1])]
        if video:
            nets += [n for n, name in self.inputs.items() if name.startswith(VIDEO)]
        rest = {k: v for k, v in ties.items() if k not in source}
        if reach is None:
            reach = self.lanes.sensitised([rest], [settled], [nets])[0]
        return self._decided(rest, settled, nets, flip, reach, linked)

    def _decided(self, rest, settled, nets, flip, reach, linked) -> bool:
        outs = [o for o in self.outputs if o in reach]
        if not outs:
            return False
        open_ = set(nets)
        free = {q for q in self.cells if q in reach and q not in settled and q not in linked}
        held = self._controls(rest) | dict(settled)
        base = self.tied(rest) + flip + [self.pairs[q][0] * (1 if v else -1) for q, v in settled.items()]
        base += [self._link(q) for q in linked]
        inner = [n for n in self.gate_nets if n in reach]
        border = {b for n in reach for b in self.fanin.get(n, ()) if b not in reach}
        base += [self._eq(b) for b in border if b in self._val[0]]
        proven: set[int] = set()
        while True:
            before = set(open_)
            differ = self._witness(held, nets, open_, free, outs, linked)
            if differ is None:
                return True
            proven -= self._downstream(open_ - before) | differ
            pinned = [-p[2] for n, p in self.pairs.items() if n not in open_]
            todo = [n for n in [*inner, *sorted(free - open_)] if n not in differ and n not in proven]
            same = [self._eq(n) for n in proven if n not in self.cells]
            ctx = self._under([*base, *pinned, *same])
            watch = self._any(self.outputs[o] for o in outs)
            try:
                found = self._sweep(ctx, todo, differ, watch, open_, free, outs, proven)
            finally:
                self.cnf.clauses += [[-watch], [-ctx]]
            if found is not None:
                return found

    def _under(self, lits: Iterable[int]) -> int:
        """An activation literal that, assumed, implies every one of ``lits``."""
        ctx = self.cnf.var()
        self.cnf.clauses += [[-ctx, lit] for lit in lits if lit != 1]
        return ctx

    def _downstream(self, opened: set[int]) -> set[int]:
        """Nets whose proven equality a newly differing storage output can break: the gates and boxes
        it feeds within a step, and the storage whose next state reads them.
        """
        if not opened:
            return set()
        cone = _closure(self.gate_readers, opened)
        return cone | {q for q in self.cells if any(b in cone for b in self.fanin[q])}

    def _sweep(
        self, ctx: int, todo, differ: set, watch: int, open_: set, free: set, outs, proven: set
    ) -> bool | None:
        """Prove the nets in ``todo`` equal in both copies in order under ``ctx``, each from those before,
        racing the output query at a doubling share of the conflicts spent; each proven joins ``proven``.
        True when an output can differ, False when none can, None when a storage bit is found to differ
        (it joins ``open_``).
        """
        start, due, quiet = self.conflicts, 0, False
        for n in track(todo, desc="sweep", unit="net"):
            if n in differ or (quiet and n not in self.cells):
                continue
            spent = self.conflicts - start
            if not quiet and spent >= due:
                found = self.solve([ctx, watch], budget=spent + 1)
                if found:
                    return True
                quiet, due = found is None, 2 * spent + 1
            goal = self.moves[n] if n in self.cells else self.cnf.xor2(self._val[0][n], self._val[1][n])
            model = self.solve([ctx, goal])
            if model is None:
                proven.add(n)
                if n not in self.cells:
                    a, b = self._val[0][n], self._val[1][n]
                    self.cnf.clauses += [[-ctx, -a, b], [-ctx, a, -b]]
                continue
            value = _reader(model)
            if any(value(self.outputs[o]) for o in outs):
                return True
            moved = {q for q in free - open_ if value(self.moves[q])}
            if moved:
                open_ |= moved
                return None
            differ |= {
                m for m in todo if m not in self.cells and value(self._val[0][m]) != value(self._val[1][m])
            }
        return not (quiet or self.solve([ctx, watch]) is None)

    def _eq(self, net: int) -> int:
        """A literal that, assumed, makes ``net`` equal in both copies."""
        a, b = self._val[0][net], self._val[1][net]
        if a == b:
            return 1
        if (a, b) not in self._eqs:
            e = self.cnf.var()
            self.cnf.clauses += [[-e, -a, b], [-e, a, -b]]
            self._eqs[(a, b)] = e
        return self._eqs[(a, b)]

    def _witness(self, held, nets, open_: set, free: set, outs, linked: frozenset) -> set[int] | None:
        """Grow ``open_`` from random lanes to their fixpoint: the nets that differ in some lane, or
        None once an output does. Lanes are drawn again while each draw shows a net differing anew.
        """
        lanes = self.lanes
        out_rows = [lanes.row(o) for o in outs]
        diff = np.zeros(lanes.rows, dtype=bool)
        while True:
            a = lanes.draw()
            self._const(a, held)
            self._settle_links(a, np.array([q in linked for q in self.qs], dtype=bool)[:, None], lanes.alg)
            b, fresh = a.copy(), lanes.draw()
            rows = [lanes.row(q) for q in open_ if q in self.cells or q in self.inputs]
            b[rows] = fresh[rows]
            for n in nets:
                if n not in self.inputs:
                    a[lanes.row(n)], b[lanes.row(n)] = lanes.alg.const(False), lanes.alg.const(True)
            while True:
                lanes.forward(a)
                lanes.forward(b)
                if (a[out_rows] != b[out_rows]).any():
                    return None
                moved = (lanes.steps(a, a[lanes.q], False) != lanes.steps(b, b[lanes.q], False)).any(axis=1)
                more = {q for i, q in enumerate(self.qs) if moved[i] and q in free and q not in open_}
                if not more:
                    break
                open_ |= more
                rows = [lanes.row(q) for q in more]
                b[rows] = fresh[rows]
            seen = (a != b).any(axis=1)
            seen[lanes.q] |= moved
            if not (seen & ~diff).any():
                return {n for n, r in lanes.ternary.index.items() if diff[r]}
            diff |= seen

    def witnesses(
        self,
        ties: Sequence[Mapping[int, int]],
        settled: Sequence[Mapping[int, bool]],
        sources: Sequence[Sequence[int] | None],
        linked: Sequence[frozenset],
    ) -> np.ndarray:
        """Per tie set, its settled storage and its source (control bits flipped, or None for the inputs),
        whether random lanes, one word of them per tie set, make an output differ. A free storage bit
        differs once its next states do.
        """
        n = len(ties)
        found = np.zeros(n, dtype=bool)
        chunk = max(1, observe.memory_limit() // (8 * self.lanes.rows * 8 * hdl.cpus()))
        for start in range(0, n, chunk):
            part = slice(start, min(n, start + chunk))
            found[part] = self._witnesses(ties[part], settled[part], sources[part], linked[part])
        return found

    def _settle_links(self, x: np.ndarray, linked: np.ndarray, alg: Words) -> None:
        """Linked storage outputs (a mask by storage bit and word) at their next states with enables
        fired, to the fixpoint, in place.
        """
        lanes = self.lanes
        if not linked.any():
            return
        while True:
            lanes.forward(x, alg)
            q = np.where(linked, lanes.steps(x, x[lanes.q], True, alg), x[lanes.q])
            if np.array_equal(q, x[lanes.q]):
                return
            x[lanes.q] = q

    def _witnesses(self, ties, settled, sources, linked) -> np.ndarray:
        lanes, n = self.lanes, len(ties)
        alg = Words(n)
        ones = ~np.uint64(0)
        at = {net: i for i, net in enumerate(self.cells)}
        held = np.zeros((len(self.qs), n), dtype=np.int8) - 1
        ctl = sorted(self.controls)
        rows = np.array([lanes.row(self.controls[k]) for k in ctl], dtype=np.int64)
        tied = np.full((len(ctl), n), -1, dtype=np.int8)
        place = {k: i for i, k in enumerate(ctl)}
        flip = np.zeros((len(ctl), n), dtype=bool)
        video = np.zeros(n, dtype=bool)
        for j, (t, s, src) in enumerate(zip(ties, settled, sources)):
            for k, v in t.items():
                if k in place:
                    tied[place[k], j] = v
            for q, v in s.items():
                held[at[q], j] = v
            video[j] = src is None
            for k in src or ():
                if k in place:
                    flip[place[k], j], tied[place[k], j] = True, -1
        vrows = np.array(
            [lanes.row(m) for m, name in self.inputs.items() if name.startswith(VIDEO)], dtype=np.int64
        )

        def draw():
            x = lanes.rng.integers(0, 2**64, size=(lanes.rows, n), dtype=np.uint64)
            x[lanes.ternary.zero], x[lanes.ternary.one] = 0, ones
            x[rows] = np.where(tied == 1, ones, np.where(tied == 0, np.uint64(0), x[rows]))
            x[lanes.q] = np.where(held == 1, ones, np.where(held == 0, np.uint64(0), x[lanes.q]))
            return x

        a, fresh = draw(), draw()
        link = np.zeros((len(self.qs), n), dtype=bool)
        for j, ls in enumerate(linked):
            link[[at[q] for q in ls], j] = True
        self._settle_links(a, link, alg)
        a[rows] = np.where(flip, np.uint64(0), a[rows])
        b = a.copy()
        b[rows] = np.where(flip, ones, b[rows])
        b[vrows] = np.where(video, fresh[vrows], b[vrows])
        outs = [lanes.row(o) for o in self.outputs]
        free = (held < 0) & ~link
        found = np.zeros(n, dtype=bool)
        while True:
            lanes.forward(a, alg)
            lanes.forward(b, alg)
            found |= (a[outs] != b[outs]).any(axis=0)
            more = (
                free
                & ~found
                & (lanes.steps(a, a[lanes.q], False, alg) != lanes.steps(b, b[lanes.q], False, alg))
            )
            if not more.any():
                return found
            free &= ~more
            b[lanes.q] = np.where(more, fresh[lanes.q], b[lanes.q])


def _closure(edges: Mapping[int, Iterable[int]], start: Iterable[int]) -> set[int]:
    seen = set(start)
    stack = list(seen)
    while stack:
        for r in edges.get(stack.pop(), ()):
            if r not in seen:
                seen.add(r)
                stack.append(r)
    return seen


class _Overlay(dict):
    """A mapping that reads ``own`` first, then ``base``."""

    def __init__(self, base: Mapping, own: dict):
        super().__init__()
        self.base, self.own = base, own

    def __getitem__(self, key):
        return self.own[key] if key in self.own else self.base[key]


def _reader(model: Sequence[int]):
    truth = set(model)

    def value(lit: int) -> bool:
        return lit == 1 if abs(lit) == 1 else lit in truth

    return value


class Exact:
    """Liveness queries on one :class:`Miter`, tie sets by flat control bit; untied controls are free.

    A storage bit settles only on tied controls, so a free control that is held is over-approximated:
    each answer is exact for the tie set as given. The ternary simulator screens tie sets in lanes, and
    the sensitised cones of those it leaves live bound the miter's queries.
    """

    def __init__(self, circuit: Circuit, solver: str = SOLVER, keep: int = 1):
        self.miter = Miter(circuit, solver)
        self.keep = keep
        self._recent: OrderedDict[frozenset, dict[int, bool]] = OrderedDict()

    def settled(self, ties: Mapping[int, int]) -> dict[int, bool]:
        """Storage settled at ``ties``; the last ``keep`` tie sets are kept."""
        key = frozenset(ties.items())
        if key in self._recent:
            self._recent.move_to_end(key)
            return self._recent[key]
        known, moved = None, ()
        if self._recent:
            nearest = min(self._recent, key=lambda k: len(k ^ key))
            known, moved = self._recent[nearest], {k for k, _ in nearest ^ key}
        self._recent[key] = self.miter.settle(ties, known, moved)
        while len(self._recent) > self.keep:
            self._recent.popitem(last=False)
        return self._recent[key]

    def _screen(self, ties: Sequence[Mapping[int, int]], names: Sequence[Iterable[str]]) -> list[np.ndarray]:
        """Per group of ``names``, per tie set, whether the ternary simulator leaves any of them live:
        dead there is proven.
        """
        rows = np.full((len(ties), max(self.miter.controls, default=0) + 1), -1, dtype=np.int8)
        for i, t in enumerate(ties):
            for k, v in t.items():
                if k < rows.shape[1]:
                    rows[i, k] = v
        lanes = self.miter.lanes.ternary.live(rows)
        out = []
        for group in names:
            hit = np.zeros(len(ties), dtype=bool)
            for name in group:
                hit |= lanes.get(name, False)
            out.append(hit)
        return out

    def _decide(
        self,
        ties: Sequence[Mapping[int, int]],
        sources: Sequence[int | None],
        first: bool = False,
        held: Sequence[int] = (),
    ) -> list[list[bool]]:
        """Per tie set, per source (a control bit, or None for every input bit), whether it is observable.

        A source the screen leaves live is first sought a witness in random lanes, then bounded by its
        sensitised cone and decided by the miter; with ``first``, a tie set's later sources are decided
        only while the earlier are dead.
        """
        video = [n for n in self.miter.inputs.values() if n.startswith(VIDEO)]
        names = [video if k is None else [f"c{k}"] for k in sources]
        screen = self._screen(ties, names)
        lanes, where = [], []
        for i, t in enumerate(ties):
            for j, k in enumerate(sources):
                if screen[j][i]:
                    lanes.append((t, k))
                    where.append((i, j))
        out = [[False] * len(sources) for _ in ties]
        if not lanes:
            return out
        miter = self.miter
        settled = [self.settled(t) for t, _ in track(lanes, desc="settle", unit="tie set")]
        linked = [miter.links(t, held, s) if held else frozenset() for (t, _), s in zip(lanes, settled)]
        seen = miter.witnesses(
            [t for t, _ in lanes], settled, [None if k is None else [k] for _, k in lanes], linked
        )
        for (i, j), hit in zip(where, seen):
            out[i][j] = bool(hit)
        left = [
            (lane, s, ls, (i, j))
            for lane, s, ls, (i, j) in zip(lanes, settled, linked, where)
            if not (out[i][j] or first and any(out[i]))
        ]
        if not left:
            return out
        lanes, settled, linked, where = (list(x) for x in zip(*left))
        nets = [
            (
                [n for n, name in miter.inputs.items() if name.startswith(VIDEO)]
                if k is None
                else [miter.controls[k]] if k in miter.controls else []
            )
            for _, k in lanes
        ]
        reach = miter.lanes.sensitised([t for t, _ in lanes], settled, nets)
        queries = list(zip(lanes, settled, linked, reach, where))
        for (t, k), s, ls, r, (i, j) in track(queries, desc="miter", unit="source"):
            if first and any(out[i]):
                continue
            out[i][j] = miter.observable(t, s, [] if k is None else [k], video=k is None, reach=r, linked=ls)
        return out

    def screen(self, ties: Sequence[Mapping[int, int]], bits: Sequence[int]) -> np.ndarray:
        """Per tie set, False where the ternary simulator proves every control bit in ``bits`` dead."""
        return self._screen(ties, [[f"c{k}" for k in bits]])[0]

    def live(self, ties: Mapping[int, int], bits: Sequence[int]) -> list[bool]:
        """Whether each control bit in ``bits`` (untied) is observable at ``ties``."""
        return self._decide([ties], list(bits))[0]

    def alive(
        self, ties: Sequence[Mapping[int, int]], bits: Sequence[int], held: Sequence[int] = ()
    ) -> list[bool]:
        """Per tie set, whether any control bit in ``bits`` (untied) is observable there; the untied
        control bits in ``held`` are held at some setting rather than varying.
        """
        return [any(row) for row in self._decide(ties, list(bits), first=True, held=held)]

    def carries(self, ties: Sequence[Mapping[int, int]]) -> list[bool]:
        """Per tie set, whether some input bit is observable there."""
        return [row[0] for row in self._decide(ties, [None])]

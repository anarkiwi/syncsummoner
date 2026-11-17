"""Where each input bit is observable at the outputs, as Boolean conditions over the controls.

Each net is a partition of the control assignments, as BDDs, into the leaf it takes there: 0, 1, a
literal (an input, register or black-box bit, or its complement) or any function. docs/program-model.md
gives the rules; each abstraction in them errs toward observable.
"""

from __future__ import annotations

import heapq
import json
import itertools
import os
import re
import threading
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Iterable, Mapping, Sequence

import numpy as np
from dd import cudd

from syncsummoner.progress import track

CONTROL = re.compile(r"^registers_in_(\d+)$")
_FF = re.compile(r"^\$_(DFFSRE|DFFSR|DFFE|DFF|SDFFCE|SDFFE|SDFF|ALDFFE|ALDFF|DLATCHSR|DLATCH|SR)_([NP01]+)_$")
#: Pin order of each fine-grained storage cell's polarity code, by family and code length.
_PINS = {
    ("DFF", 1): "C",
    ("DFF", 3): "CRv",
    ("DFFE", 2): "CE",
    ("DFFE", 4): "CRvE",
    ("SDFF", 3): "CRv",
    ("SDFFE", 4): "CRvE",
    ("SDFFCE", 4): "CRvE",
    ("DFFSR", 3): "CSR",
    ("DFFSRE", 4): "CSRE",
    ("ALDFF", 2): "CL",
    ("ALDFFE", 3): "CLE",
    ("DLATCH", 1): "E",
    ("DLATCH", 3): "ERv",
    ("DLATCHSR", 3): "ESR",
    ("SR", 2): "SR",
}

#: Yosys's gate library over any algebra ``a`` with ``&``, ``|``, ``~``, ``a.ite`` and ``a.xor``.
GATES: dict[str, Callable] = {
    "$_BUF_": lambda a, A: A,
    "$_NOT_": lambda a, A: ~A,
    "$_AND_": lambda a, A, B: A & B,
    "$_NAND_": lambda a, A, B: ~(A & B),
    "$_OR_": lambda a, A, B: A | B,
    "$_NOR_": lambda a, A, B: ~(A | B),
    "$_XOR_": lambda a, A, B: a.xor(A, B),
    "$_XNOR_": lambda a, A, B: ~a.xor(A, B),
    "$_ANDNOT_": lambda a, A, B: A & ~B,
    "$_ORNOT_": lambda a, A, B: A | ~B,
    "$_MUX_": lambda a, A, B, S: a.ite(S, B, A),
    "$_NMUX_": lambda a, A, B, S: ~a.ite(S, B, A),
    "$_AOI3_": lambda a, A, B, C: ~((A & B) | C),
    "$_OAI3_": lambda a, A, B, C: ~((A | B) & C),
    "$_AOI4_": lambda a, A, B, C, D: ~((A & B) | (C & D)),
    "$_OAI4_": lambda a, A, B, C, D: ~((A | B) & (C | D)),
}


@dataclass
class Cell:
    """One cell: its kind, input and output pin bits, and a storage cell's polarity code."""

    kind: str
    ins: dict[str, list]
    outs: dict[str, list]
    code: tuple[str, ...] = ()


def next_state(cell: Cell, value: Callable[[str], object], q, alg, fire: bool):
    """Next state of one storage bit from its pins' values (``value(pin)``) in algebra ``alg``.

    ``alg.const(b)`` gives a constant; with ``fire`` every enable is taken as active.
    """
    pol = dict(zip(_PINS[(cell.kind, len(cell.code))], cell.code))

    def active(name: str):
        return value(name) if pol[name] == "P" else ~value(name)

    d = value("D") if "D" in cell.ins else q
    if "L" in pol:
        d = alg.ite(active("L"), value("AD"), d)
    reset = alg.const(pol.get("v") == "1")
    if cell.kind == "SDFFCE":
        d = alg.ite(active("R"), reset, d)
    if "E" in pol and not fire:
        d = alg.ite(active("E"), d, q)
    if "S" in pol:
        d = alg.ite(active("S"), alg.const(True), d)
    if "R" in pol and cell.kind != "SDFFCE":
        d = alg.ite(active("R"), reset, d)
    return d


#: Bytes of one CUDD node (``DdNode``: index, reference count, next pointer and two children).
NODE_BYTES = 32


class Exhausted(RuntimeError):
    """A condition outgrew its share of the memory this process may use."""


def memory_limit(
    meminfo: Path = Path("/proc/meminfo"), cgroup: Path = Path("/sys/fs/cgroup/memory.max")
) -> int:
    """Bytes this process may still use: ``MemAvailable``, cut to the cgroup's ``memory.max``."""
    avail = os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_AVPHYS_PAGES")
    try:
        found = dict(line.split(":", 1) for line in meminfo.read_text(encoding="utf-8").splitlines())
        avail = int(found["MemAvailable"].split()[0]) * 1024
    except (OSError, KeyError, ValueError):
        pass
    try:
        cap = cgroup.read_text(encoding="utf-8").strip()
        avail = avail if cap == "max" else min(avail, int(cap))
    except (OSError, ValueError):
        pass
    return avail


@dataclass
class Circuit:
    """A Yosys JSON module split into gates, storage cells and black boxes, with its ports."""

    inputs: dict[str, list]
    outputs: dict[str, list]
    gates: list[Cell]
    storage: list[Cell]
    boxes: list[Cell]
    init: dict[int, bool]

    @classmethod
    def read(cls, design: dict, top: str) -> "Circuit":
        """Classify every cell of ``top`` by its type and port directions; split storage to bits."""
        module = design["modules"][top]
        ports = module["ports"]
        inputs = {n: p["bits"] for n, p in ports.items() if p["direction"] in ("input", "inout")}
        outputs = {n: p["bits"] for n, p in ports.items() if p["direction"] in ("output", "inout")}
        gates, storage, boxes = [], [], []
        for cell in module["cells"].values():
            lib = design["modules"].get(cell["type"], {}).get("ports", {})
            dirs = cell.get("port_directions") or {p: q["direction"] for p, q in lib.items()}
            conns = cell["connections"]
            ins = {p: b for p, b in conns.items() if dirs.get(p, "input") != "output"}
            outs = {p: b for p, b in conns.items() if dirs.get(p) in ("output", "inout")}
            match = _FF.match(cell["type"])
            if not outs:
                continue
            if cell["type"] in GATES:
                gates.append(Cell(cell["type"], ins, outs))
            elif match and (match.group(1), len(match.group(2))) in _PINS:
                code = tuple(match.group(2))
                for k, q in enumerate(outs["Q"]):
                    pins = {p: [b[k] if len(b) > 1 else b[0]] for p, b in ins.items()}
                    storage.append(Cell(match.group(1), pins, {"Q": [q]}, code))
            else:
                boxes.append(Cell(cell["type"], ins, outs))
        init = {}
        for net in module["netnames"].values():
            value = net.get("attributes", {}).get("init")
            if isinstance(value, str) and len(value) == len(net["bits"]):
                for b, c in zip(net["bits"], reversed(value)):
                    if isinstance(b, int) and c in "01":
                        init[b] = c == "1"
        return cls(inputs, outputs, gates, storage, boxes, init)

    def controls(self) -> dict[int, int]:
        """Net of each ``registers_in`` bit, by its flat position."""
        return {int(m.group(1)): bits[0] for n, bits in self.inputs.items() if (m := CONTROL.match(n))}

    def inputs_named(self) -> dict[int, str]:
        """``port[i]`` of every non-control input bit, by net."""
        return {
            b: f"{n}[{i}]"
            for n, bits in self.inputs.items()
            if not CONTROL.match(n)
            for i, b in enumerate(bits)
            if isinstance(b, int)
        }

    def order(self) -> tuple[list[Cell], list[Cell]]:
        """Gates in dependency order, and those on a combinational loop (to be treated as black boxes)."""
        driver = {b: i for i, g in enumerate(self.gates) for bits in g.outs.values() for b in bits}
        deps = [{driver[b] for bits in g.ins.values() for b in bits if b in driver} for g in self.gates]
        users = defaultdict(list)
        for i, ds in enumerate(deps):
            for j in ds:
                users[j].append(i)
        count = [len(d) for d in deps]
        ready = [i for i, n in enumerate(count) if not n]
        order = []
        while ready:
            i = ready.pop()
            order.append(i)
            for j in users[i]:
                count[j] -= 1
                if not count[j]:
                    ready.append(j)
        looped = sorted(set(range(len(self.gates))) - set(order))
        return [self.gates[i] for i in order], [self.gates[i] for i in looped]


def manager(controls: Iterable[int]) -> cudd.BDD:
    """A BDD manager over ``c<flat bit>`` for the given controls, declared in order, reordered by sifting."""
    bdd = cudd.BDD()
    bdd.declare(*(f"c{k}" for k in controls))
    bdd.configure(reordering=True)
    return bdd


@dataclass
class Conditions:
    """Observability of input bits as BDDs over the control variables ``c<flat bit>``."""

    bdd: cudd.BDD
    observed: dict[str, cudd.Function]
    _lock: threading.Lock = field(default_factory=threading.Lock, repr=False)

    def live(self, name: str, ties: Mapping[int, int]) -> bool:
        """Whether ``name`` is observable for some value of every control not in ``ties``."""
        u = self.observed.get(name)
        if u is None:
            return False
        sub = {f"c{k}": bool(v) for k, v in ties.items() if f"c{k}" in self.bdd.vars}
        with self._lock:
            return self.bdd.let(sub, u) != self.bdd.false

    def heard(self, names: Iterable[str], ties: Mapping[int, int], bits: Sequence[int]) -> np.ndarray:
        """Per value of the controls ``bits`` (``bits[i]`` its ``2**i`` place), whether some source in
        ``names`` is observable for some value of every control in neither ``ties`` nor ``bits``.
        """
        keep = {f"c{k}" for k in bits}
        out = np.zeros(1 << len(bits), dtype=bool)
        with self._lock:
            u = self.bdd.false
            for name in names:
                u |= self.condition(name)
            u = self.bdd.let({f"c{k}": bool(v) for k, v in ties.items() if f"c{k}" in self.bdd.vars}, u)
            u = self.bdd.exist(self.bdd.support(u) - keep, u)
            for a in self.bdd.pick_iter(u, care_vars=keep):
                out[sum(int(a[f"c{k}"]) << i for i, k in enumerate(bits))] = True
        return out

    def condition(self, name: str) -> cudd.Function:
        """The observability condition of one input bit (false when it is never observable)."""
        return self.observed.get(name, self.bdd.false)

    def save(self, path: Path) -> Path:
        """Write as JSON atomically: nodes ``[variable, low, high]``, with 0 and 1 the constants."""
        bdd = self.bdd
        index = {bdd.false: 0, bdd.true: 1}
        nodes: list[list] = []
        with self._lock:
            for root in self.observed.values():
                stack = [root]
                while stack:
                    u = stack[-1]
                    if u in index:
                        stack.pop()
                        continue
                    lo, hi = bdd.let({u.var: False}, u), bdd.let({u.var: True}, u)
                    todo = [c for c in (lo, hi) if c not in index]
                    if todo:
                        stack += todo
                        continue
                    nodes.append([u.var, index[lo], index[hi]])
                    index[u] = len(nodes) + 1
                    stack.pop()
            roots = {name: index[u] for name, u in sorted(self.observed.items())}
        tmp = path.with_name(f"{path.stem}.{os.getpid()}.tmp")
        tmp.write_text(json.dumps({"nodes": nodes, "roots": roots}), encoding="utf-8")
        tmp.replace(path)
        return path

    @classmethod
    def load(cls, path: Path, controls: Iterable[int]) -> "Conditions":
        """Read what :meth:`save` wrote."""
        bdd = manager(controls)
        data = json.loads(path.read_text(encoding="utf-8"))
        built = [bdd.false, bdd.true]
        for v, lo, hi in data["nodes"]:
            built.append(bdd.ite(bdd.var(v), built[hi], built[lo]))
        return cls(bdd, {name: built[i] for name, i in data["roots"].items()})


#: Leaves of a symbolic value: the constants, any function, and literals (signed variable ids).
ZERO, ONE, ANY = "0", "1", "*"
#: Literal ids standing for an unknown argument while one pin is probed; above every net id.
PROBE = 1 << 48


def _not(x):
    return {ZERO: ONE, ONE: ZERO, ANY: ANY}.get(x, -x if isinstance(x, int) else x)


def _and(x, y):
    if ZERO in (x, y):
        return ZERO
    if x == ONE:
        return y
    if y in (ONE, x):
        return x
    return ZERO if isinstance(x, int) and isinstance(y, int) and x == -y else ANY


def _or(x, y):
    return _not(_and(_not(x), _not(y)))


def _xor(x, y):
    if x in (ZERO, ONE):
        return y if x == ZERO else _not(y)
    if y in (ZERO, ONE):
        return x if y == ZERO else _not(x)
    if isinstance(x, int) and isinstance(y, int) and abs(x) == abs(y):
        return ZERO if x == y else ONE
    return ANY


def _mux(s, a, b):
    if s in (ZERO, ONE):
        return b if s == ONE else a
    if a == b:
        return a
    return _or(_and(s, b), _and(_not(s), a))


class Sym:
    """A symbolic value: each leaf with the BDD over the controls where the net takes it."""

    __slots__ = ("dom", "parts")

    def __init__(self, dom: "Domain", parts: dict):
        self.dom, self.parts = dom, parts

    def __and__(self, other: "Sym") -> "Sym":
        return self.dom.combine(_and, self, other)

    def __or__(self, other: "Sym") -> "Sym":
        return self.dom.combine(_or, self, other)

    def __invert__(self) -> "Sym":
        return Sym(self.dom, {_not(k): c for k, c in self.parts.items()})

    def __eq__(self, other) -> bool:
        return isinstance(other, Sym) and self.parts == other.parts

    __hash__ = None

    def where(self, *leaves) -> cudd.Function:
        """The controls under which the value is one of ``leaves``."""
        u = self.dom.bdd.false
        for leaf in leaves:
            u |= self.parts.get(leaf, self.dom.bdd.false)
        return u


class Domain:
    """The algebra of :class:`Sym` over one BDD manager."""

    def __init__(self, bdd: cudd.BDD):
        self.bdd = bdd

    @property
    def top(self) -> Sym:
        """Any function everywhere."""
        return Sym(self, {ANY: self.bdd.true})

    def const(self, value) -> Sym:
        """A constant everywhere."""
        return Sym(self, {ONE if value else ZERO: self.bdd.true})

    def leaf(self, leaf) -> Sym:
        """One leaf everywhere."""
        return Sym(self, {leaf: self.bdd.true})

    def control(self, name: str) -> Sym:
        """A control bit: 1 where its variable holds, 0 elsewhere."""
        v = self.bdd.var(name)
        return Sym(self, {ONE: v, ZERO: ~v})

    def combine(self, op: Callable, *args: Sym) -> Sym:
        """``op`` leaf by leaf over every combination of the arguments' parts."""
        out: dict = {}
        false = self.bdd.false
        for combo in itertools.product(*(a.parts.items() for a in args)):
            cond = combo[0][1]
            for _, c in combo[1:]:
                cond &= c
                if cond == false:
                    break
            if cond != false:
                leaf = op(*(k for k, _ in combo))
                out[leaf] = out[leaf] | cond if leaf in out else cond
        return Sym(self, out)

    def sensitised(self, fn: Callable[..., Sym], args: list[Sym], k: int) -> cudd.Function:
        """Where ``fn(*args)`` can differ between ``args[k]`` at 0 and at 1, the other arguments held.

        Each other argument's unknown leaf becomes a literal of its own, so an output that merely passes
        it on is told apart from one the pin changes; an output unknown both ways is taken to differ.
        """
        held = [
            Sym(self, {(PROBE + j if leaf == ANY else leaf): c for leaf, c in a.parts.items()})
            for j, a in enumerate(args)
        ]
        lo = fn(*(self.const(False) if j == k else a for j, a in enumerate(held)))
        hi = fn(*(self.const(True) if j == k else a for j, a in enumerate(held)))
        return self.combine(lambda u, v: ONE if u != v or u == ANY else ZERO, lo, hi).where(ONE)

    def xor(self, x: Sym, y: Sym) -> Sym:
        """``x ^ y``."""
        return self.combine(_xor, x, y)

    def ite(self, s: Sym, a: Sym, b: Sym) -> Sym:
        """``s ? a : b``."""
        return self.combine(_mux, s, b, a)


class Analysis:
    """Symbolic evaluation of one circuit over ``controls``; ``loose`` bits are free, the rest 0.

    Controls in ``varying`` may change while the rest are held: no register settles on their value.
    """

    def __init__(
        self,
        circuit: Circuit,
        controls: Iterable[int],
        loose: Iterable[int] = (),
        limit: int | None = None,
        varying: Iterable[int] = (),
    ):
        self.circuit = circuit
        self.controls = tuple(controls)
        self.varying = {f"c{k}" for k in varying}
        nets = len(circuit.gates) + len(circuit.storage) + 1
        self.nodes = (memory_limit() if limit is None else limit) // (NODE_BYTES * nets)
        self.bdd = manager(self.controls)
        self.dom = Domain(self.bdd)
        self.order, looped = circuit.order()
        self.boxes = [*circuit.boxes, *looped]
        self.boxed = {id(b) for b in self.boxes}
        self.value: dict[int, Sym] = {}
        self.sens: dict[int, dict[int, cudd.Function]] = {}
        self.settled: dict[int, cudd.Function] = {}
        self.registers: set[str] = set()
        loose = set(loose)
        for k, net in circuit.controls().items():
            if k in self.controls:
                self.value[net] = self.dom.control(f"c{k}")
            elif k not in loose:
                self.value[net] = self.dom.const(False)
        held = {b for cell in (*circuit.storage, *self.boxes) for bits in cell.outs.values() for b in bits}
        nets = set(circuit.inputs_named()) | {n for k, n in circuit.controls().items() if k in loose}
        for b in sorted(nets | {b for b in held if isinstance(b, int)}):
            self.value[b] = self.dom.leaf(b)

    def pin(self, b) -> Sym:
        """Value of a connection bit; undefined bits may be anything."""
        if isinstance(b, int):
            return self.value[b]
        return self.dom.const(b == "1") if b in ("0", "1") else self.dom.top

    def check(self, value: Sym) -> Sym:
        """``value``, unless its conditions outgrow one net's share of memory (:class:`Exhausted`)."""
        size = sum(c.dag_size for c in value.parts.values())
        if size > self.nodes:
            raise Exhausted(f"a condition of {size} BDD nodes exceeds the {self.nodes} each net may hold")
        return value

    def gate(self, g: Cell) -> Sym:
        """A gate's output."""
        return GATES[g.kind](self.dom, *(self.pin(g.ins[p][0]) for p in sorted(g.ins)))

    def next_state(self, cell: Cell, fire: bool = False) -> Sym:
        """Next state of one storage bit; ``fire`` takes every enable as active."""
        q = self.dom.leaf(cell.outs["Q"][0])
        return next_state(cell, lambda p: self.pin(cell.ins[p][0]), q, self.dom, fire)

    def settle(self, cell: Cell) -> Sym:
        """A storage bit: its constant load once enables fire, else a kept initial value, else free.

        Either rule must hold for every value of the varying controls and register variables. Where it
        is already constant it stays so; elsewhere its own output is still free, so neither rule assumes
        the value it concludes. A free bit whose next state reads only controls and itself is a register
        variable, so the logic it feeds keeps its correlations; any other free bit is a literal.
        """
        net = cell.outs["Q"][0]
        old = self.value[net]
        held = self.settled.get(net, self.bdd.false)
        rest = ~held
        if rest == self.bdd.false:
            return old
        parts = {
            leaf: c & held
            for leaf, c in old.parts.items()
            if leaf in (ZERO, ONE) and c & held != self.bdd.false
        }
        step = self.next_state(cell)
        found = [(leaf, self.next_state(cell, fire=True).where(leaf)) for leaf in (ZERO, ONE)]
        if net in self.circuit.init:
            leaf = ONE if self.circuit.init[net] else ZERO
            found.append((leaf, step.where(leaf, net)))
        free = self.varying | self.registers
        for leaf, c in found:
            c = (self.bdd.forall(free, c) if free else c) & rest
            if c != self.bdd.false:
                parts[leaf] = parts[leaf] | c if leaf in parts else c
                rest &= ~c
        self.settled[net] = ~rest
        if rest != self.bdd.false:
            if (
                f"r{net}" in self.registers
                or not (step.where(ANY) | self._literals(step, net)) & rest != self.bdd.false
            ):
                if f"r{net}" not in self.registers:
                    self.bdd.declare(f"r{net}")
                    self.registers.add(f"r{net}")
                r = self.bdd.var(f"r{net}")
                for leaf, c in ((ONE, rest & r), (ZERO, rest & ~r)):
                    parts[leaf] = parts[leaf] | c if leaf in parts else c
            else:
                parts[net] = rest
        return Sym(self.dom, parts)

    def evaluate(self) -> None:
        """Every net's value, with storage settled to the least fixpoint of :meth:`settle`."""
        rank = {id(g): i for i, g in enumerate(self.order)}
        readers: dict[int, list[Cell]] = defaultdict(list)
        for g in self.order:
            for bits in g.ins.values():
                for b in bits:
                    if isinstance(b, int):
                        readers[b].append(g)
        feeds: dict[int, list[Cell]] = defaultdict(list)
        for cell in self.circuit.storage:
            for bits in cell.ins.values():
                for b in bits:
                    if isinstance(b, int):
                        feeds[b].append(cell)

        def run(cells: Iterable[Cell]) -> set[int]:
            heap = [(rank[id(g)], id(g), g) for g in cells]
            heapq.heapify(heap)
            done, changed = set(), set()
            while heap:
                _, key, g = heapq.heappop(heap)
                if key in done:
                    continue
                done.add(key)
                out = next(iter(g.outs.values()))[0]
                new = self.check(self.gate(g))
                if self.value.get(out) != new:
                    self.value[out] = new
                    changed.add(out)
                    for r in readers[out]:
                        heapq.heappush(heap, (rank[id(r)], id(r), r))
            return changed

        run(self.order)
        pending = self.circuit.storage
        for _ in track(itertools.count(), desc="settle registers", unit="round"):
            moved = set()
            for cell in pending:
                net = cell.outs["Q"][0]
                new = self.check(self.settle(cell))
                if self.value[net] != new:
                    self.value[net] = new
                    moved.add(net)
            if not moved:
                break
            touched = moved | run({id(r): r for b in moved for r in readers[b]}.values())
            pending = list({id(c): c for b in touched for c in feeds[b]}.values())

    def sensitised(self, cell: Cell) -> dict[int, cudd.Function]:
        """Where each net a gate or storage bit reads can change what it computes, the others held."""
        pins = sorted(cell.ins)
        args = [self.pin(cell.ins[p][0]) for p in pins]
        nets = [cell.ins[p][0] for p in pins]
        if cell.kind in GATES:

            def fn(*xs):
                return GATES[cell.kind](self.dom, *xs)

        else:
            q = cell.outs["Q"][0]
            args, nets = [*args, self.dom.leaf(q)], [*nets, q]

            def fn(*xs):
                return next_state(cell, dict(zip(pins, xs)).__getitem__, xs[-1], self.dom, False)

        out: dict[int, cudd.Function] = {}
        for k, b in enumerate(nets):
            if isinstance(b, int):
                c = self.dom.sensitised(fn, args, k)
                out[b] = out[b] | c if b in out else c
        return out

    def _literals(self, v: Sym, own: int) -> cudd.Function:
        """Where ``v`` is a literal other than ``own`` or its complement."""
        u = self.bdd.false
        for leaf, c in v.parts.items():
            if isinstance(leaf, int) and abs(leaf) != own:
                u |= c
        return u

    def readers(self) -> dict[int, list[Cell]]:
        """Every gate, storage bit and box reading each net."""
        out: dict[int, list[Cell]] = defaultdict(list)
        for cell in (*self.order, *self.circuit.storage, *self.boxes):
            for bits in cell.ins.values():
                for b in bits:
                    if isinstance(b, int):
                        out[b].append(cell)
        return out

    def _cofactors(self, v: Sym, var: str | None) -> tuple[Sym, Sym]:
        if var is None:
            return v, v
        false = self.bdd.false
        return tuple(
            Sym(self.dom, {k: u for k, c in v.parts.items() if (u := self.bdd.let({var: b}, c)) != false})
            for b in (False, True)
        )

    def _differs(self, pair: tuple[Sym, Sym], diff: Mapping, sens: Mapping) -> cudd.Function:
        """Where a value can differ: its two cofactors take different leaves, or one leaf that differs.

        A literal differs where its net does; a leaf of any function differs where an input it is
        sensitised to (``sens``) does.
        """
        false = self.bdd.false
        anyin = false
        for b, c in sens.items():
            if b in diff:
                anyin |= diff[b] & c
        out = false
        for (a, ca), (b, cb) in itertools.product(pair[0].parts.items(), pair[1].parts.items()):
            c = ca & cb
            if c == false:
                continue
            if a != b:
                out |= c
            elif isinstance(a, int):
                out |= c & diff.get(abs(a), false)
            elif a == ANY:
                out |= c & anyin
        return out

    def _pairs(self, v: Sym, var: str | None) -> tuple:
        """Cofactors of ``v`` on ``var``, and on each register variable it depends on."""
        regs = {r: self._cofactors(v, r) for r in self._support(v) & self.registers}
        return self._cofactors(v, var), regs

    def _changes(self, pairs: tuple, diff: Mapping, sens: Mapping, moved: Mapping) -> cudd.Function:
        """Where a value can differ, from ``var`` and from every register variable that does."""
        pair, regs = pairs
        d = self._differs(pair, diff, sens)
        for r, rpair in regs.items():
            if r in moved:
                d |= moved[r] & self._differs(rpair, {}, {})
        return d

    def _support(self, v: Sym) -> set[str]:
        out: set[str] = set()
        for c in v.parts.values():
            out |= self.bdd.support(c)
        return out

    def _change(self, cell: Cell, var: str | None, diff: dict, moved: dict, pairs: dict) -> cudd.Function:
        """Where one cell's output can change, given what its inputs and register variables do."""
        key = id(cell)
        outs = [b for bits in cell.outs.values() for b in bits if isinstance(b, int)]
        if key in self.boxed:
            d = self.bdd.false
            for bits in cell.ins.values():
                for b in bits:
                    d |= diff.get(b, self.bdd.false)
            return d
        if key not in pairs:
            step = self.next_state(cell) if cell.code else None
            pairs[key] = (self._pairs(self.value[outs[0]], var), step and self._pairs(step, var))
        if key not in self.sens:
            self.sens[key] = self.sensitised(cell)
        (here, step), sens = pairs[key], self.sens[key]
        d = self._changes(here, diff, sens, moved)
        if step is not None:
            q = outs[0]
            d |= ~self.settled.get(q, self.bdd.false) & self._changes(step, diff, sens, moved)
            if f"r{q}" in self.registers:
                moved[f"r{q}"] = d | moved.get(f"r{q}", self.bdd.false)
        return d

    def difference(self, nets: Iterable[int], var: str | None, readers: Mapping) -> cudd.Function:
        """Where flipping ``var`` (or changing the ``nets``, when ``var`` is None) can change an output.

        Forward from the nets to a fixpoint through storage and boxes. A free register differs where its
        next state does, and a value differs where a register variable it depends on does.
        """
        rank = {id(g): i for i, g in enumerate(self.order)}
        false = self.bdd.false
        diff: dict[int, cudd.Function] = {n: self.bdd.true for n in nets}
        moved: dict[str, cudd.Function] = {}
        pairs: dict[int, tuple] = {}
        heap = [(rank.get(id(c), -1), id(c), c) for n in diff for c in readers.get(n, ())]
        heapq.heapify(heap)
        queued = {key for _, key, _ in heap}
        while heap:
            _, key, cell = heapq.heappop(heap)
            queued.discard(key)
            d = self._change(cell, var, diff, moved, pairs)
            for o in (b for bits in cell.outs.values() for b in bits if isinstance(b, int)):
                old = diff.get(o, false)
                if (old | d) == old:
                    continue
                diff[o] = self.check(Sym(self.dom, {ONE: old | d})).parts[ONE]
                for r in readers.get(o, ()):
                    if id(r) not in queued:
                        queued.add(id(r))
                        heapq.heappush(heap, (rank.get(id(r), -1), id(r), r))
        out = false
        for bits in self.circuit.outputs.values():
            for b in bits:
                if isinstance(b, int):
                    out |= diff.get(b, false)
        return self.bdd.exist(self.registers, out) if self.registers else out

    def observe(self, sources: Mapping[str, tuple[str | None, list[int]]]) -> Conditions:
        """Each named source's observability condition, carried into a manager over the controls.

        A source is a control variable with its net, or (variable None) a set of input nets.
        """
        self.evaluate()
        readers = self.readers()
        target = manager(self.controls)
        observed = {}
        for name, (var, nets) in track(list(sources.items()), desc="observability", unit="source"):
            u = self.difference(nets, var, readers)
            if u != self.bdd.false:
                observed[name] = self.bdd.copy(u, target)
        return Conditions(target, observed)

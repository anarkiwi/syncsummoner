"""A vectorised ternary (0/1/X) simulator of the gate-level netlist: a sound proof of dead bits per tie set.

Each lane is one tie set; untied controls, other inputs and unsettled storage are X. A bit is dead in a
lane when no path of sensitised gates (output X, and X on that input alone leaving it X) reaches an output.
"""

from __future__ import annotations

import itertools
from collections import defaultdict
from dataclasses import dataclass
from types import SimpleNamespace
from typing import Callable

import numpy as np

from .observe import GATES, Cell, Circuit, next_state


@dataclass
class Tri:
    """Ternary values as two rails: ``z`` where the value is 0, ``o`` where it is 1."""

    z: np.ndarray
    o: np.ndarray

    def __and__(self, other: "Tri") -> "Tri":
        return Tri(self.z | other.z, self.o & other.o)

    def __or__(self, other: "Tri") -> "Tri":
        return Tri(self.z & other.z, self.o | other.o)

    def __invert__(self) -> "Tri":
        return Tri(self.o, self.z)

    @property
    def x(self) -> np.ndarray:
        """Where the value is unknown."""
        return ~(self.z | self.o)


class Rails:
    """The ternary algebra over lanes of one width."""

    def __init__(self, lanes: int):
        self.lanes = lanes

    @staticmethod
    def ite(s: Tri, a: Tri, b: Tri) -> Tri:
        """``s ? a : b``, known where both arms agree even when ``s`` is not."""
        return Tri((s.o & a.z) | (s.z & b.z) | (a.z & b.z), (s.o & a.o) | (s.z & b.o) | (a.o & b.o))

    def xor(self, a: Tri, b: Tri) -> Tri:
        """``a ^ b``."""
        return self.ite(a, ~b, b)

    def const(self, v: bool) -> Tri:
        """A constant in every lane."""
        return Tri(np.full(self.lanes, not v), np.full(self.lanes, bool(v)))

    def unknown(self) -> Tri:
        """X in every lane."""
        return Tri(np.zeros(self.lanes, dtype=bool), np.zeros(self.lanes, dtype=bool))


class Bit(int):
    """0 or 1 under ``&``, ``|`` and ``~``."""

    def __and__(self, other):
        return Bit(int(self) & int(other))

    def __or__(self, other):
        return Bit(int(self) | int(other))

    def __invert__(self):
        return Bit(1 - int(self))


BITS = SimpleNamespace(
    ite=lambda s, a, b: a if s else b,
    xor=lambda a, b: Bit(int(a) ^ int(b)),
    const=lambda v: Bit(bool(v)),
)


def sensitised(fn: Callable[..., Bit], args: list[Tri], k: int) -> np.ndarray:
    """Lanes where ``fn`` may differ between ``args[k]`` at 0 and 1, the others as their rails allow."""
    out = np.zeros_like(args[0].z)
    for bits in itertools.product((0, 1), repeat=len(args) - 1):
        vals = [Bit(b) for b in bits[:k]] + [Bit(0)] + [Bit(b) for b in bits[k:]]
        lo = fn(*vals)
        vals[k] = Bit(1)
        if lo != fn(*vals):
            ok = np.ones_like(out)
            for j, (a, v) in enumerate(zip(args, vals)):
                if j != k:
                    ok &= ~a.z if v else ~a.o
            out |= ok
    return out


class Ternary:
    """Lanes of tie sets simulated together over one circuit."""

    def __init__(self, circuit: Circuit):
        self.circuit = circuit
        order, looped = circuit.order()
        self.boxes = [*circuit.boxes, *looped]
        cells = (*circuit.gates, *circuit.storage, *circuit.boxes)
        nets = {b for c in cells for pins in (c.ins, c.outs) for bits in pins.values() for b in bits}
        nets |= {b for port in (circuit.inputs, circuit.outputs) for bits in port.values() for b in bits}
        self.index = {b: i for i, b in enumerate(sorted(b for b in nets if isinstance(b, int)))}
        self.zero, self.one, self.unknown = len(self.index), len(self.index) + 1, len(self.index) + 2
        self.size = len(self.index) + 3
        self.levels = self._levels(order)

    def at(self, b) -> int:
        """Row of a connection bit; constants and undefined bits share three fixed rows."""
        if isinstance(b, int):
            return self.index[b]
        return {"0": self.zero, "1": self.one}.get(b, self.unknown)

    def _levels(self, order: list[Cell]) -> list[list[tuple[str, np.ndarray, np.ndarray]]]:
        """Gates grouped by depth and type, as input-row and output-row arrays."""
        driver = {b: g for g in order for bits in g.outs.values() for b in bits}
        depth: dict[int, int] = {}
        for g in order:
            depth[id(g)] = 1 + max(
                (depth[id(driver[b])] for bits in g.ins.values() for b in bits if b in driver), default=-1
            )
        groups: dict[tuple[int, str], list[Cell]] = defaultdict(list)
        for g in order:
            groups[(depth[id(g)], g.kind)].append(g)
        out: list[list] = [[] for _ in range(max(depth.values(), default=-1) + 1)]
        for (d, kind), cells in sorted(groups.items()):
            pins = sorted(cells[0].ins)
            ins = np.array([[self.at(c.ins[p][0]) for p in pins] for c in cells], dtype=np.int64)
            outs = np.array([self.at(next(iter(c.outs.values()))[0]) for c in cells], dtype=np.int64)
            out[d].append((kind, ins, outs))
        return out

    def _forward(self, z: np.ndarray, o: np.ndarray, alg: Rails) -> None:
        for level in self.levels:
            for kind, ins, outs in level:
                y = GATES[kind](alg, *(Tri(z[ins[:, k]], o[ins[:, k]]) for k in range(ins.shape[1])))
                z[outs], o[outs] = y.z, y.o

    def _state(self, cell: Cell, rails: tuple, alg: Rails, q: Tri, fire: bool) -> Tri:
        z, o = rails

        def value(p: str) -> Tri:
            row = self.at(cell.ins[p][0])
            return Tri(z[row], o[row])

        return next_state(cell, value, q, alg, fire)

    def _settle(self, z: np.ndarray, o: np.ndarray, alg: Rails) -> bool:
        moved = False
        for cell in self.circuit.storage:
            net = cell.outs["Q"][0]
            row = self.index[net]
            free = ~(z[row] | o[row])
            load = self._state(cell, (z, o), alg, alg.unknown(), True)
            nz, no = z[row] | (load.z & free), o[row] | (load.o & free)
            if net in self.circuit.init:
                i = self.circuit.init[net]
                kept = self._state(cell, (z, o), alg, alg.const(i), False)
                hold = (kept.o if i else kept.z) & ~(nz | no)
                nz, no = nz | (hold & (not i)), no | (hold & i)
            if not (np.array_equal(nz, z[row]) and np.array_equal(no, o[row])):
                z[row], o[row], moved = nz, no, True
        return moved

    def simulate(self, ties: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Rails of every row for ``ties`` (controls by lane, 0/1, or -1 for X), storage settled."""
        alg = Rails(ties.shape[0])
        z = np.zeros((self.size, alg.lanes), dtype=bool)
        o = np.zeros((self.size, alg.lanes), dtype=bool)
        z[self.zero], o[self.one] = True, True
        for k, net in self.circuit.controls().items():
            z[self.index[net]], o[self.index[net]] = ties[:, k] == 0, ties[:, k] == 1
        while True:
            self._forward(z, o, alg)
            if not self._settle(z, o, alg):
                return z, o

    def _edges(self, z: np.ndarray, o: np.ndarray, x: np.ndarray) -> tuple[list, list]:
        """Sensitised gate pins ``(outputs, sources, lanes)`` and storage pins ``(row, source, lanes)``."""
        sens = []
        for level in self.levels:
            for kind, ins, outs in level:
                tris = [Tri(z[ins[:, k]], o[ins[:, k]]) for k in range(ins.shape[1])]
                for k in range(ins.shape[1]):
                    live = sensitised(lambda *v, kind=kind: GATES[kind](BITS, *v), tris, k)
                    sens.append((outs, ins[:, k], live & x[ins[:, k]]))
        held = []
        for cell in self.circuit.storage:
            row = self.index[cell.outs["Q"][0]]
            pins = sorted(cell.ins)
            tris = [Tri(z[self.at(cell.ins[p][0])], o[self.at(cell.ins[p][0])]) for p in pins]

            def fn(*v, cell=cell, pins=pins):
                return next_state(cell, dict(zip(pins, v[1:])).__getitem__, v[0], BITS, False)

            q = Tri(z[row], o[row])
            for k, p in enumerate(pins):
                b = self.at(cell.ins[p][0])
                live = x[b] if p == "C" else sensitised(fn, [q, *tris], k + 1) & x[b]
                held.append((row, b, live))
        return sens, held

    def observed(self, ties: np.ndarray) -> np.ndarray:
        """Rows with a sensitised path to an output, by lane."""
        z, o = self.simulate(ties)
        x = ~(z | o)
        seen = np.zeros_like(x)
        for bits in self.circuit.outputs.values():
            for b in bits:
                if isinstance(b, int):
                    seen[self.at(b)] = x[self.at(b)]
        sens, held = self._edges(z, o, x)
        while True:
            before = int(seen.sum())
            for outs, src, live in reversed(sens):
                np.logical_or.at(seen, src, seen[outs] & live)
            for row, b, live in held:
                seen[b] |= seen[row] & live
            for box in self.boxes:
                hit = np.zeros(ties.shape[0], dtype=bool)
                for bits in box.outs.values():
                    for b in bits:
                        hit |= seen[self.at(b)]
                for bits in box.ins.values():
                    for b in bits:
                        seen[self.at(b)] |= hit & x[self.at(b)]
            seen[[self.zero, self.one, self.unknown]] = False
            if int(seen.sum()) == before:
                return seen

    def live(self, ties: np.ndarray) -> dict[str, np.ndarray]:
        """Each control bit ``c<k>`` and named input bit, by lane: whether it may reach an output."""
        seen = self.observed(ties)
        out = {f"c{k}": seen[self.index[net]] for k, net in self.circuit.controls().items()}
        out |= {name: seen[self.index[b]] for b, name in self.circuit.inputs_named().items()}
        return out

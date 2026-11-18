"""Random concrete assignments of a gate-level netlist simulated bit-parallel: each bit of a word is one."""

from __future__ import annotations

from collections import defaultdict
from typing import Mapping, Sequence

import numpy as np

from .observe import GATES, Circuit, next_state
from .ternary import Ternary


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

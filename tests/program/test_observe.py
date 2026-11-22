"""Observability conditions and the ternary fallback on hand-made gate-level netlists with known gating."""

# pylint: disable=missing-function-docstring

import numpy as np
import pytest

from syncsummoner.program import observe
from syncsummoner.program.declared import field
from syncsummoner.program.model import CONTROL_BITS, _lanes
from syncsummoner.program.observe import ONE, ZERO
from syncsummoner.program.ternary import Ternary

S, B, G, M, E, K = (field(p).flat[0] for p in (7, 8, 9, 10, 11, 12))
DIRS = {"Y": "output", "Q": "output", "RDATA": "output"}


def cell(kind, **pins):
    return {
        "type": kind,
        "port_directions": {p: DIRS.get(p, "input") for p in pins},
        "connections": {p: bits if isinstance(bits, list) else [bits] for p, bits in pins.items()},
    }


def design(cells, controls, data=(), outputs=(), init=None, lib=None):
    ports = {f"registers_in_{k}": {"direction": "input", "bits": [net]} for k, net in controls.items()}
    ports["data_in"] = {"direction": "input", "bits": list(data)}
    ports["data_out"] = {"direction": "output", "bits": list(outputs)}
    netnames = {f"n{b}": {"bits": [b], "attributes": {"init": v}} for b, v in (init or {}).items()}
    modules = {"top": {"ports": ports, "cells": dict(enumerate(cells)), "netnames": netnames}}
    return {"modules": {**modules, **(lib or {})}}


def analysis(d):
    return observe.Analysis(observe.Circuit.read(d, "top"), CONTROL_BITS)


def sources(d):
    c = observe.Circuit.read(d, "top")
    found = {f"c{k}": (f"c{k}", [net]) for k, net in c.controls().items()}
    return found | {name: (None, [b]) for b, name in c.inputs_named().items()}


def conditions(d, varying=()):
    return observe.Analysis(observe.Circuit.read(d, "top"), CONTROL_BITS, varying=varying).observe(sources(d))


#: ``data_out = s ? data_in ^ b : data_in``: the select shows only where ``b`` flips the other arm.
ALIAS = design(
    [cell("$_XOR_", A=10, B=3, Y=11), cell("$_MUX_", A=10, B=11, S=2, Y=12)],
    {S: 2, B: 3},
    data=[10],
    outputs=[12],
)


def test_a_select_between_equal_arms_is_dead():
    cond = conditions(ALIAS)
    assert not cond.live(f"c{S}", {B: 0}) and cond.live(f"c{S}", {B: 1})
    assert not cond.live(f"c{B}", {S: 0}) and cond.live(f"c{B}", {S: 1})
    assert cond.live("data_in[0]", {S: 0, B: 0})
    assert cond.condition("nothing") == cond.bdd.false


#: A latched control enables a data register masked by another control; a toggle and a counter run beside.
SEQUENTIAL = design(
    [
        cell("$_DFF_P_", C=1, D=4, Q=20),
        cell("$_DFFE_PP_", C=1, D=10, E=20, Q=21),
        cell("$_AND_", A=21, B=5, Y=22),
        cell("$_NOT_", A=23, Y=24),
        cell("$_DFFE_PP_", C=1, D=24, E=6, Q=23),
        cell("$_XOR_", A=25, B=7, Y=26),
        cell("$_DFF_P_", C=1, D=26, Q=25),
    ],
    {G: 4, M: 5, E: 6, K: 7},
    data=[10, 1],
    outputs=[22, 23, 25],
    init={21: "0", 23: "0", 25: "0"},
)


def test_registers_settle_and_carry_observability_across_cycles():
    cond = conditions(SEQUENTIAL)
    assert cond.live(f"c{M}", {G: 1}) and not cond.live(f"c{M}", {G: 0})
    assert cond.live(f"c{G}", {M: 1}) and not cond.live(f"c{G}", {M: 0})
    assert cond.live("data_in[0]", {G: 1, M: 1}) and not cond.live("data_in[0]", {G: 0, M: 1})
    assert not cond.live("data_in[0]", {G: 1, M: 0})
    assert cond.live(f"c{E}", {G: 0, M: 0, K: 0}) and cond.live(f"c{K}", {G: 0, M: 0, E: 0})


def test_a_disabled_register_keeps_its_initial_value():
    a = analysis(SEQUENTIAL)
    a.evaluate()
    g, e, k = (a.bdd.var(f"c{x}") for x in (G, E, K))
    assert a.value[20].where(ONE) == g and a.value[21].where(ZERO) == ~g
    assert a.value[23].where(ONE) == e & a.bdd.var("r23"), "a toggle is a register variable where enabled"
    assert a.settled[25] == ~k and a.settled[23] == ~e


def test_reset_set_and_load_cells_settle_to_their_loads():
    cells = [
        cell("$_SDFF_PP1_", C=1, R=2, D=3, Q=30),
        cell("$_SDFFCE_PP0P_", C=1, R=2, E=3, D=10, Q=31),
        cell("$_DFFSR_PPP_", C=1, S=2, R=3, D=10, Q=32),
        cell("$_ALDFF_PP_", C=1, L=2, AD=3, D=10, Q=33),
        cell("$_DLATCH_N_", E=2, D=3, Q=34),
        cell("$_SR_PP_", S=2, R=3, Q=35),
        cell("$_DFF_PN0_", C=1, R=2, D="x", Q=36),
        cell("$_AOI3_", A=30, B=31, C=32, Y=40),
        cell("$_OAI4_", A=33, B=34, C=35, D=36, Y=41),
        cell("$_NMUX_", A=40, B=41, S=2, Y=42),
    ]
    a = analysis(design(cells, {S: 2, B: 3}, data=[10, 1], outputs=[42]))
    a.evaluate()
    s, b = a.bdd.var(f"c{S}"), a.bdd.var(f"c{B}")
    assert a.value[30].where(ONE) == s | b
    assert a.value[31].where(ZERO) == s and a.value[31].where(31) == ~s
    assert a.value[32].where(ZERO) == b and a.value[32].where(ONE) == s & ~b
    assert a.value[33].where(ONE) == s & b and a.value[33].where(33) == ~s
    assert a.value[34].where(ONE) == b
    assert a.value[35].where(ONE) == (s & ~b) | (~s & ~b & a.bdd.var("r35"))
    assert a.value[36].where(ZERO) == ~s
    assert a.observe({f"c{B}": (f"c{B}", [3])}).live(f"c{B}", {S: 0})


def test_black_boxes_and_loops_pass_everything():
    lib = {"SB_RAM40_4K": {"ports": {"WDATA": {"direction": "input"}, "RDATA": {"direction": "output"}}}}
    box = {"type": "SB_RAM40_4K", "connections": {"WDATA": [2], "RDATA": [50]}}
    loop = [cell("$_AND_", A=3, B=52, Y=51), cell("$_OR_", A=51, B=10, Y=52)]
    cells = [box, *loop, cell("$_ORNOT_", A=50, B=52, Y=53)]
    found = conditions(design(cells, {S: 2, B: 3}, [10], [53], lib=lib))
    assert found.live(f"c{S}", {}) and found.live(f"c{B}", {})


def test_conditions_survive_a_round_trip(tmp_path):
    cond = conditions(SEQUENTIAL)
    back = observe.Conditions.load(cond.save(tmp_path / "c.json"), CONTROL_BITS)
    assert set(back.observed) == set(cond.observed)
    every = [{G: g, M: m, E: e, K: k} for g in (0, 1) for m in (0, 1) for e in (0, 1) for k in (0, 1)]
    for name in cond.observed:
        assert [back.live(name, t) for t in every] == [cond.live(name, t) for t in every]


def test_bdds_beyond_the_memory_limit_are_refused(tmp_path):
    with pytest.raises(observe.Exhausted):
        observe.Analysis(observe.Circuit.read(ALIAS, "top"), CONTROL_BITS, limit=0).observe(sources(ALIAS))
    assert observe.resident() > 0
    (tmp_path / "meminfo").write_text("MemTotal: 100 kB\nMemAvailable: 64 kB\n")
    (tmp_path / "max").write_text("32768\n")
    assert observe.memory_limit(tmp_path / "meminfo", tmp_path / "max") == 32768
    (tmp_path / "max").write_text("max\n")
    assert observe.memory_limit(tmp_path / "meminfo", tmp_path / "max") == 65536
    assert observe.memory_limit(tmp_path / "absent", tmp_path / "absent") > 0


def lanes(*ties):
    return _lanes(list(ties))


def test_the_ternary_fallback_is_sound_and_loses_only_correlation():
    sim = Ternary(observe.Circuit.read(ALIAS, "top"))
    found = sim.live(lanes({S: 1, B: 0}, {S: 0, B: 0}, {B: 1}, {S: 0}))
    assert found[f"c{S}"].tolist() == [False, False, True, False]
    assert found[f"c{B}"].tolist() == [False, False, False, False]
    assert found["data_in[0]"].tolist() == [True, True, True, True]
    assert sim.live(lanes({B: 0}))[f"c{S}"].tolist() == [True], "X ^ 0 is X to it: the select looks live"


def test_the_ternary_fallback_settles_registers():
    sim = Ternary(observe.Circuit.read(SEQUENTIAL, "top"))
    found = sim.live(lanes({M: 1, E: 0, K: 0}, {G: 0, M: 1, E: 0, K: 0}, {G: 1, E: 0, K: 0}))
    assert found[f"c{G}"].tolist() == [True, False, False]
    assert found[f"c{M}"].tolist() == [False, False, True]
    assert np.array_equal(found["data_in[0]"], [True, False, True])


#: ``data_out = s ? (d & ~s) | (d & s)...``: a select whose two arms reconverge to the same bit.
RECONVERGE = design(
    [
        cell("$_XOR_", A=10, B=3, Y=11),
        cell("$_AND_", A=11, B=2, Y=12),
        cell("$_ANDNOT_", A=10, B=2, Y=13),
        cell("$_OR_", A=12, B=13, Y=14),
    ],
    {S: 2, B: 3},
    data=[10],
    outputs=[14],
)


def test_a_select_split_over_gates_is_dead_where_its_arms_agree():
    cond = conditions(RECONVERGE)
    assert not cond.live(f"c{S}", {B: 0}) and cond.live(f"c{S}", {B: 1})


#: A page select routes a pot into one of two registers that both drive the picture.
PAGED = design(
    [
        cell("$_DFFE_PP_", C=1, D=4, E=2, Q=20),
        cell("$_NOT_", A=2, Y=21),
        cell("$_DFFE_PP_", C=1, D=4, E=21, Q=22),
        cell("$_XOR_", A=20, B=22, Y=23),
    ],
    {S: 2, G: 4},
    data=[1],
    outputs=[23],
)


def test_a_page_select_matters_only_while_the_paged_control_varies():
    assert not conditions(PAGED).live(f"c{S}", {G: 0}) and not conditions(PAGED).live(f"c{S}", {G: 1})
    assert conditions(PAGED, varying=[G]).live(f"c{S}", {})


def test_the_memory_an_analysis_takes_on_bounds_it_not_one_conditions_size(monkeypatch):
    circuit = observe.Circuit.read(ALIAS, "top")
    held = {"bytes": 0}
    monkeypatch.setattr(observe, "resident", lambda: held["bytes"])
    early = observe.Analysis(circuit, CONTROL_BITS, limit=2)
    held["bytes"] = 1
    found = early.observe(sources(ALIAS))
    assert found.observed, "any condition fits while the process has taken on less than the limit"
    held["bytes"] = 1
    late = observe.Analysis(circuit, CONTROL_BITS, limit=2)
    held["bytes"] = 3
    with pytest.raises(observe.Exhausted, match="2 bytes"):
        late.observe(sources(ALIAS))

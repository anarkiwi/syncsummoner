"""The SAT miter against the BDD conditions: equal on every tie set of hand-made netlists and the fixtures."""

# pylint: disable=missing-function-docstring,protected-access

import itertools
import json

import numpy as np
import pytest

from syncsummoner.program import gates, observe, sat
from syncsummoner.program.declared import field
from syncsummoner.program.gates import Gating, Status, cover
from syncsummoner.program.model import CONTROL_BITS, ProgramModel
from tests.program.test_observe import ALIAS, PAGED, RECONVERGE, SEQUENTIAL, cell, design

S, B, G, M, E, K = (field(p).flat[0] for p in (7, 8, 9, 10, 11, 12))

#: Reset, set, load and latch cells, a box, a loop and an undefined pin, under two controls.
CELLS = design(
    [
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
        {"type": "SB_RAM40_4K", "connections": {"WDATA": [3], "RDATA": [50]}},
        cell("$_AND_", A=42, B=52, Y=51),
        cell("$_OR_", A=51, B=50, Y=52),
        cell("$_XNOR_", A=52, B=10, Y=53),
    ],
    {S: 2, B: 3},
    data=[10, 1],
    outputs=[53],
    lib={"SB_RAM40_4K": {"ports": {"WDATA": {"direction": "input"}, "RDATA": {"direction": "output"}}}},
)


def every(bits):
    return [dict(zip(bits, v)) for v in itertools.product((0, 1), repeat=len(bits))]


@pytest.mark.parametrize(
    "d", [ALIAS, SEQUENTIAL, RECONVERGE, PAGED, CELLS], ids=["alias", "seq", "rec", "paged", "cells"]
)
def test_the_miter_equals_the_conditions_on_every_tie_set(d):
    circuit = observe.Circuit.read(d, "top")
    exact = sat.Exact(circuit)
    controls = sorted(circuit.controls())
    video = [n for n in circuit.inputs_named().values() if n.startswith("data_in")]
    for k in controls:
        sources = {f"c{j}": (f"c{j}", [circuit.controls()[j]]) for j in controls}
        cond = observe.Analysis(circuit, CONTROL_BITS, varying=[k]).observe(sources)
        for ties in every([j for j in controls if j != k]):
            assert exact.live(ties, [k]) == [cond.live(f"c{k}", ties)], (k, ties)
            assert exact.alive([ties], [k]) == [cond.live(f"c{k}", ties)]
    cond = observe.Analysis(circuit, CONTROL_BITS).observe({"video": (None, list(circuit.inputs_named()))})
    for ties in every(controls):
        expected = cond.live("video", ties) and bool(video)
        assert exact.carries([ties]) == [expected], ties


def test_a_select_whose_arms_agree_is_dead_where_the_ternary_simulator_calls_it_live():
    exact = sat.Exact(observe.Circuit.read(ALIAS, "top"))
    assert exact.live({B: 0}, [S]) == [False] and exact.live({B: 1}, [S]) == [True]
    assert exact._screen([{B: 0}], [[f"c{S}"]])[0].tolist() == [True]


def test_a_paged_control_is_live_only_while_its_page_is_free():
    exact = sat.Exact(observe.Circuit.read(PAGED, "top"))
    assert exact.live({G: 0}, [S]) == [False] and exact.live({G: 1}, [S]) == [False]
    assert exact.live({}, [S]) == [True], "a free control is never settled on"


#: Two pipelines of one control whose agreement masks another control.
TWINS = design(
    [
        cell("$_DFF_P_", C=1, D=4, Q=20),
        cell("$_DFF_P_", C=1, D=4, Q=21),
        cell("$_XOR_", A=20, B=21, Y=22),
        cell("$_AND_", A=22, B=2, Y=23),
        cell("$_XOR_", A=23, B=10, Y=24),
    ],
    {S: 2, G: 4},
    data=[10, 1],
    outputs=[24],
)


def test_a_held_control_links_its_pipelines_where_a_varying_one_frees_them():
    exact = sat.Exact(observe.Circuit.read(TWINS, "top"))
    assert exact.alive([{}], [S], held=[G]) == [False], "both pipelines hold the same setting"
    assert exact.alive([{}], [S]) == [True], "a varying control settles neither"
    assert exact.alive([{G: 1}], [S]) == [False] and exact.miter.links({}, [G], {}) == {20, 21}


def test_settling_needs_every_free_signal_and_takes_a_load_before_a_kept_value():
    miter = sat.Miter(observe.Circuit.read(SEQUENTIAL, "top"))
    assert miter.settle({G: 1, M: 1, E: 0, K: 0}) == {20: True, 23: False, 25: False}
    assert miter.settle({G: 0, M: 1, E: 1, K: 1}) == {20: False, 21: False}


def test_the_tiny_design_is_the_same_by_sat(tiny, tmp_path):
    exact = ProgramModel(tiny.program, tiny.library, cache=tmp_path, method="sat")
    for state in (
        tiny.initial(),
        tiny.preset("Gain"),
        np.r_[tiny.initial()[:6], 0, 1023, tiny.initial()[8:]],
    ):
        assert exact.live(state) == tiny.live(state)
    assert exact.method == "sat" and tiny.method == "bdd"
    assert not exact.live(gates=False)[8].live, "both arms of the dry select are data_in"
    assert exact.resolution() == tiny.resolution()
    assert exact.carries_input(tiny.initial())
    equal = tiny.initial()
    equal[1] = 520
    assert exact.reach(equal, [7])[7] == [] and tiny.reach(equal, [7])[7] == [0], "gain and bias are equal"
    found, bdd = cover(exact), cover(tiny)
    assert [s.tolist() for s in found.states[:4]] == [s.tolist() for s in bdd.states]
    assert found.gating()[1][:3] == bdd.gating()[1]


def test_the_mixer_is_covered_alike(mixer, tmp_path):
    exact = ProgramModel(mixer.program, mixer.library, cache=tmp_path, method="sat")
    assert json.dumps(cover(exact).as_dict()) == json.dumps(cover(mixer).as_dict())


def test_a_two_gate_revival_is_found_as_the_conditions_find_it(tiny, tmp_path):
    exact = ProgramModel(tiny.program, tiny.library, cache=tmp_path, method="sat")
    both = tiny.initial()
    both[[6, 7]] = 0, 1023
    pair = Gating(exact).revivals(both, Status(1, ()))
    assert pair == Gating(tiny).revivals(both, Status(1, ()))
    assert [r.gate for r in pair] == [(7, 8)] and not pair[0].spans
    assert exact.revivable(both, 1, [7, 8]) and not exact.revivable(both, 1, [7])


def test_the_fader_bypass_is_proven_by_sat(mixer, tmp_path):
    exact = ProgramModel(mixer.program, mixer.library, cache=tmp_path, method="sat")
    assert exact.passes(12).tolist() == mixer.passes(12).tolist() == [0]
    assert not exact.passes(7).size and not exact.passes(1).size
    assert exact.bypass() == mixer.bypass()


def test_a_bypass_switch_is_proven_by_sat(tiny, tmp_path):
    exact = ProgramModel(tiny.program, tiny.library, cache=tmp_path, method="sat")
    assert exact.bypass() == tiny.bypass()
    assert exact.passes(8).tolist() == list(range(512, 1024))


def paged(tiny, tmp_path, method):
    """Tiny's declarations over a paged pot: Gain On is the pot, Bypass its page select."""
    m = ProgramModel(tiny.program, tiny.library, cache=tmp_path, method=method)
    cells = [
        cell("$_DFFE_PP_", C=1, D=3, E=2, Q=20),
        cell("$_NOT_", A=2, Y=21),
        cell("$_DFFE_PP_", C=1, D=3, E=21, Q=22),
        cell("$_XOR_", A=20, B=22, Y=23),
    ]
    d = design(cells, {B: 2, S: 3}, data=[1], outputs=[23])
    m._circuit = observe.Circuit.read({"modules": {"program_top": d["modules"]["top"]}}, "program_top")
    m.cache.mkdir(parents=True)
    return m


def test_a_page_select_live_only_while_its_pot_moves_has_no_revival(tiny, tmp_path):
    by_sat, by_bdd = paged(tiny, tmp_path / "s", "sat"), paged(tiny, tmp_path / "b", None)
    state = tiny.initial()
    live = by_sat.live(state)
    assert live == by_bdd.live(state)
    assert not live[8].live and live[8].gated_by == ("P7",), "freeing the pot revives it"
    assert not by_sat.revivable(state, 8, [7])
    assert Gating(by_sat).revivals(state, Status(8, ())) == []
    assert by_sat.reach(state, [7, 8])[8] == [1], "freed together, the page reaches the picture"


def test_without_a_miter_the_ternary_simulator_answers(tiny, tmp_path, monkeypatch):
    def fail(*_args, **_kwargs):
        raise MemoryError("no room")

    monkeypatch.setattr(sat, "Exact", fail)
    coarse = ProgramModel(tiny.program, tiny.library, cache=tmp_path)
    coarse.memory = 0
    assert coarse.live(gates=False)[8].live and coarse.method == "ternary"
    with pytest.raises(RuntimeError, match="ternary"):
        _ = coarse.exact
    assert gates.Gating(coarse).revivals(tiny.initial(), Status(1, ())) is not None


def test_a_bypass_proven_once_is_read_from_the_cache(tiny, tmp_path, monkeypatch):
    first = ProgramModel(tiny.program, tiny.library, cache=tmp_path, method="sat")
    found = first.bypass()
    again = ProgramModel(tiny.program, tiny.library, cache=tmp_path, method="sat")
    monkeypatch.setattr(again, "_bypass", lambda *a: pytest.fail("proven again"))
    assert again.bypass() == found == tiny.bypass()
    held = tiny.initial()
    held[0] = 1023
    monkeypatch.setattr(again, "_bypass", lambda *a: None)
    assert again.bypass(held) is None, "another state is another proof"

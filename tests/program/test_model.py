"""Liveness and step verdicts on a tiny design whose gating is known by construction."""

# pylint: disable=missing-function-docstring

import json

import numpy as np

import pytest

from syncsummoner.compose.evidence import LANDED
from syncsummoner.program import parse_state
from syncsummoner.program.model import ProgramModel, model
from syncsummoner.program.report import rows, table


def test_resolution_is_the_bits_the_vhdl_reads(tiny):
    assert tiny.resolution() == {1: (6, 7, 8, 9), 2: (0, 1, 2, 3, 9), 3: (), 7: (0,), 8: (1,)}


def test_initial_state_gating(tiny):
    live = tiny.live()
    assert not live[1].live and live[1].gated_by == ("P7",)
    assert live[2].live_bits == (0, 1, 2, 3, 9)
    assert not live[3].live and live[3].gated_by == ()
    assert live[7].live
    assert not live[8].live and live[8].gated_by == ("P2", "P7")


def test_a_preset_moves_the_gate(tiny):
    live = tiny.live(tiny.preset("Gain"))
    assert live[1].live_bits == (6, 7, 8, 9)
    assert not live[2].live and live[2].gated_by == ("P7",)


def test_steps(tiny):
    gain = parse_state(tiny, "Gain")
    unread = tiny.step_matters(gain, 1, 512, 512 + 32)
    assert unread.matters is False and "bits no output depends on" in unread.how
    moved = tiny.step_matters(gain, 1, 0, 512)
    assert moved.matters is True and "differ at cycle" in moved.how
    assert tiny.step_matters(None, 1, 0, 512).matters is False
    assert tiny.step_matters(None, 2, 100, 101).matters is False
    assert tiny.step_matters(None, 2, 600, 601).matters is True
    assert tiny.step_matters(None, 9, 0, 1023).how == "the program declares no such parameter"


def test_schedule_offers_one_value_per_visible_setting(tiny):
    plan = tiny.schedule()
    assert set(plan) == {2, 7}
    assert len(plan[2]) == 32 and np.count_nonzero(plan[2] >= 512) == 16
    assert len(tiny.schedule(tiny.preset("Gain"))[1]) == 16


def test_state_parsing_and_the_table(tiny):
    state = parse_state(tiny, ",".join(["0"] * 12), {7: 1023})
    assert state[6] == 1023 and list(parse_state(tiny, None)[:2]) == [512, 100]
    text = table(tiny, tiny.initial())
    assert "gated by P7" in text and "no single control revives it" in text
    row = rows(tiny, tiny.initial())[0]
    assert row["reads"] == "r0[9:6]" and row["live_bits"] == "-" and row["levels"] == 0
    assert rows(tiny, tiny.initial())[1]["reads"] == "r1[9,3:0]"


def test_the_bypass_is_the_switch_that_silences_every_other_control(tiny):
    found = tiny.bypass()
    assert (found.parameter_id, found.value) == (8, 1023)
    assert found.proven == tuple(range(512, 1024)) and found.holds([0, 511, 512, 1023]).tolist() == [
        False,
        False,
        True,
        True,
    ]
    assert tiny.carries_input(tiny.initial())


def test_a_setting_passes_only_where_every_other_control_is_dead_for_all_their_values(tiny):
    assert tiny.passes(8).tolist() == list(range(512, 1024)), "the switch's own proof agrees"
    for p in (1, 2, 3, 7):
        assert not tiny.passes(p).size, f"P{p} leaves the bypass switch live"
    assert not tiny.passes(9).size, "an undeclared control proves nothing"


def test_with_no_switch_a_crossfader_at_the_end_that_returns_its_input_is_the_bypass(mixer):
    assert mixer.passes(12).tolist() == [0], "only an exact pass-through counts: 1023 is the effect"
    assert mixer.passes(12, timing=0).tolist() == [0]
    assert not mixer.passes(1).size and not mixer.passes(7).size
    found = mixer.bypass()
    assert (found.parameter_id, found.value, found.proven) == (12, 0, (0,))
    held = mixer.initial()
    held[11] = 0
    assert not any(l.live for p, l in mixer.live(held, gates=False).items() if p != 12)
    assert mixer.carries_input(held)


def test_the_ternary_simulator_proves_a_crossfader_bypass_as_well(mixer, tmp_path):
    coarse = ProgramModel(mixer.program, mixer.library, cache=tmp_path, method="ternary")
    assert coarse.passes(12).tolist() == [0] and coarse.method == "ternary"
    assert not coarse.passes(7).size


def test_a_source_that_reads_other_bits_than_the_package_yields_to_the_bitstream(tiny, monkeypatch, tmp_path):
    def latched(_self):
        return {**tiny.resolution(), 1: (9,)}

    monkeypatch.setattr(ProgramModel, "latched", latched)
    picked = model(tiny.program, tiny.library, cache=tmp_path)
    assert picked.source == "bitstream" and "P1 [9,8,7,6] where the bitstream latches [9]" in picked.note
    assert model(tiny.program, tiny.library, cache=tmp_path, source="rtl").source == "rtl"
    monkeypatch.undo()
    with pytest.raises(ValueError, match="bitstream"):
        tiny.latched()


def test_a_folded_state_is_reused_by_the_next_step(tiny, monkeypatch):
    from syncsummoner.program import hdl  # pylint: disable=import-outside-toplevel

    real, calls = hdl.fold, []

    def counting(*args, **kwargs):
        calls.append(1)
        return real(*args, **kwargs)

    monkeypatch.setattr(hdl, "fold", counting)
    monkeypatch.setattr(tiny, "_recent", type(tiny._recent)())  # pylint: disable=protected-access
    state = tiny.initial()
    first = tiny._folded(state, None)  # pylint: disable=protected-access
    assert tiny._folded(state.copy(), None) is first and len(calls) == 1  # pylint: disable=protected-access


def test_bdds_beyond_memory_fall_back_to_sat(tiny, tmp_path):
    spent = ProgramModel(tiny.program, tiny.library, cache=tmp_path / "spent")
    spent.memory = 0
    assert spent.live(gates=False) == tiny.live(gates=False)
    assert spent.method == "sat" and (spent.cache / "observe.exhausted").exists()


def test_the_ternary_simulator_over_approximates(tiny, tmp_path):
    coarse = ProgramModel(tiny.program, tiny.library, cache=tmp_path, method="ternary")
    found, exact = coarse.live(gates=False), tiny.live(gates=False)
    assert coarse.method == "ternary" and tiny.method == "bdd"
    for p, liveness in exact.items():
        assert set(liveness.live_bits) <= set(found[p].live_bits)
    assert found[8].live and not exact[8].live, "the ternary simulator cannot see that both arms are data_in"
    assert coarse.carries_input(coarse.initial())


def test_conditions_are_cached_per_free_set(tiny):
    first = tiny.conditions({1, "timing"})
    assert tiny.conditions({1, "timing"}) is first
    assert any(p.name.startswith("observe-1-timing") for p in tiny.cache.iterdir())
    assert tiny.reach(tiny.preset("Gain"), [1], timing=0)[1] == [6, 7, 8, 9]


@pytest.mark.parametrize("fixture", ["tiny", "mixer"])
def test_the_cli_states_the_bypass_setting(fixture, request, monkeypatch, capsys):
    from syncsummoner import cli  # pylint: disable=import-outside-toplevel
    from syncsummoner.program import Library  # pylint: disable=import-outside-toplevel

    built = request.getfixturevalue(fixture)
    monkeypatch.setattr(Library, "default", classmethod(lambda cls: built.library))
    monkeypatch.setattr("syncsummoner.program.model", lambda *a, **k: built)
    assert cli.main(["program", "bypass", "x", "--json"]) == 0
    said = json.loads(capsys.readouterr().out)
    assert (said["parameter_id"], said["value"]) == {"tiny": (8, 1023), "mixer": (12, 0)}[fixture]
    assert cli.main(["program", "bypass", "x", "--state", "Gain" if fixture == "tiny" else "initial"]) == 0
    assert " at " in capsys.readouterr().out


def test_the_cli_says_when_no_setting_passes(tiny, monkeypatch, capsys):
    from syncsummoner import cli  # pylint: disable=import-outside-toplevel
    from syncsummoner.program import Library  # pylint: disable=import-outside-toplevel

    monkeypatch.setattr(Library, "default", classmethod(lambda cls: tiny.library))
    monkeypatch.setattr("syncsummoner.program.model", lambda *a, **k: tiny)
    monkeypatch.setattr(ProgramModel, "bypass", lambda self, *a: None)
    assert cli.main(["program", "bypass", "x", "--json"]) == 0
    assert json.loads(capsys.readouterr().out) is None
    assert cli.main(["program", "bypass", "x"]) == 0
    assert "no setting is proven" in capsys.readouterr().out


ARCHIVED_BYPASS = (
    "com.homegrownvhs.pigment",
    "com.lzxindustries.yuv_amplifier",
    "com.lzxindustries.pinwheel",
    "com.peteappleby.yuv_bit_rotator",
)


@pytest.mark.parametrize("program_id", ARCHIVED_BYPASS)
def test_archived_bypass_programs_keep_their_bypass_switch(program_id):
    """The four programs archived with Bypass on as their dry state keep it: switch 11 turned on."""
    from syncsummoner.compose.locktake import dry_state  # pylint: disable=import-outside-toplevel
    from syncsummoner.program import Library, predict  # pylint: disable=import-outside-toplevel

    library = Library.default()
    try:
        built = model(program_id, library)
    except KeyError:
        pytest.skip(f"{program_id} is not in this library")
    found = built.bypass()
    assert (found.parameter_id, found.value, found.proven) == (11, 1023, tuple(range(512, 1024)))
    assert dry_state(None, lambda info: found) == {11: 1023}
    _, own = predict.regimes(built, built.initial(), np.zeros(0, dtype=LANDED), 2)
    assert 11 in own and set(own) == {d.parameter_id for d in built.params if 7 <= d.parameter_id <= 11}

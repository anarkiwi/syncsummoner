"""Liveness and step verdicts on a tiny design whose gating is known by construction."""

# pylint: disable=missing-function-docstring

import numpy as np

import pytest

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
    assert tiny.bypass() == 8
    assert tiny.carries_input(tiny.initial())


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

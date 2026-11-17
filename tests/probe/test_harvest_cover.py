"""Harvest sweeps from the program model's cover: full declared states, each step's origin recorded."""

# pylint: disable=missing-function-docstring

import dataclasses

from syncsummoner.device.transport import ProgramInfo
from syncsummoner.probe import harvest as H
from tests.device.conftest import COLORBARS_INFO, FakeClock
from tests.probe.test_harvest import (
    CONFIG,
    INFO,
    KEY,
    Archive,
    FakeCover,
    Port,
    Rec,
    Sess,
    fake_plan,
    no_ffmpeg,
)

__all__ = ["no_ffmpeg"]


def test_the_sweep_is_the_covers_full_declared_states():
    vectors = H.sweep_vectors(INFO, FakeCover())
    assert len(vectors) == 2 and all(set(v) == {p.index for p in INFO.used} for v in vectors)
    assert vectors[1][12] == 0 and vectors[0][12] == 512


def test_sweep_of_an_undeclared_program_is_one_empty_vector():
    assert H.sweep_vectors(ProgramInfo.from_json(COLORBARS_INFO), FakeCover()) == [{}]
    assert H.sweep_vectors(INFO, None) == [{}]


def test_sweep_of_a_program_reporting_no_parameters_at_all_is_one_empty_vector():
    info = ProgramInfo.from_json({"name": "Null", "parameters": []})
    assert H.sweep_vectors(info, FakeCover()) == [{}]


def test_focus_steps_are_taken_from_the_base_state_the_parameter_is_live_in():
    focus = (("Alpha", 12, 2), ("Alpha", 2, 0), ("Alpha", 3, 3), ("Alpha", 1, 2), ("Beta", 2, 2))
    config = dataclasses.replace(CONFIG, focus=focus)
    got = H.focus_vectors("Alpha", INFO, config, FakeCover())
    assert [v[12] for v in got[:3]] == [0, 0, 1023] and all(v[2] == 512 for v in got[:3])
    assert [v[2] for v in got[3:]] == [512, 0, 341, 682, 1023] and all(v[12] == 512 for v in got[3:])
    many = dataclasses.replace(CONFIG, focus=(("Alpha", 2, 6),))
    assert [v[2] for v in H.focus_vectors("Alpha", INFO, many, FakeCover())[1:]] == [
        0,
        341,
        682,
        1023,
        0,
        341,
    ]
    assert not H.focus_vectors("Alpha", INFO, config, None)


def test_the_model_cover_is_none_for_an_undeclared_program():
    assert H.model_cover(ProgramInfo.from_json(COLORBARS_INFO)) is None


def test_every_setpoint_records_the_full_state_it_was_stepped_from(tmp_path):
    clock = FakeClock()
    archive = Archive(tmp_path)
    port = Port()
    H.harvest_program(
        Sess(port),
        Rec(clock),
        archive,
        port,
        "Alpha",
        KEY,
        config=CONFIG,
        plan=fake_plan,
        sleep=clock.sleep,
        clock=clock,
    )
    schedule = archive.schedules["Alpha"]
    assert [e["setpoint"] for e in schedule] == [0, 1]
    assert schedule[1]["from"] == schedule[0]["to"] and schedule[1]["to"][11] == 0

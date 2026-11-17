"""Plan predictions: each planned step split into reachable and unreachable, per parameter."""

# pylint: disable=missing-function-docstring

import numpy as np

from syncsummoner.program.predict import Prediction, predict, steps

PLAN = np.dtype([("target", "<i8"), ("param", "<i8"), ("value", "<i8")])


def test_steps_follow_frame_order_and_skip_rewrites():
    plan = np.array([(5, 2, 9), (1, 1, 4), (3, 1, 4), (7, 1, 0)], dtype=PLAN)
    out = steps(np.zeros(12, dtype=int), plan)
    assert [(p, a, b) for _, p, a, b in out] == [(1, 0, 4), (2, 0, 9), (1, 4, 0)]
    assert out[1][0][0] == 4 and out[2][0][1] == 9


def test_verdicts_and_checks():
    assert Prediction(3, 0, 0).check(True) == "no step to predict"
    assert Prediction(3, 0, 2).check(False) == "predicted dead, and did not move"
    assert Prediction(3, 1, 2).check(True) == "predicted live, and moved"


def test_the_tiny_design_predicts_its_gate(tiny):
    plan = np.array([(0, 1, 256), (2, 1, 264), (4, 7, 1023), (6, 1, 0), (8, 2, 700)], dtype=PLAN)
    got = predict(tiny, tiny.initial(), plan)
    assert got[1] == Prediction(1, live=1, dead=2)
    assert got[7].verdict == "live" and got[2] == Prediction(2, live=0, dead=1)


def test_a_take_of_an_unknown_program_gets_no_prediction(monkeypatch, tmp_path):
    from syncsummoner.compose.locktake import predicted  # pylint: disable=import-outside-toplevel

    monkeypatch.setenv("SYNCSUMMONER_CARD", str(tmp_path))
    monkeypatch.setenv("SYNCSUMMONER_FIRMWARE", str(tmp_path))
    plan = np.zeros(0, dtype=PLAN)
    assert not predicted("nobody/Nothing", {"start": np.zeros(12, dtype=int), "plan": plan})

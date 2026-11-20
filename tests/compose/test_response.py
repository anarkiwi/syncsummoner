"""Settling after a step, from scores built with a known transient."""

# pylint: disable=missing-function-docstring

from types import SimpleNamespace

import numpy as np

from syncsummoner.compose import evidence as ev
from syncsummoner.compose import response as R


def landed_rows(frames, params, values):
    rows = np.zeros(len(frames), dtype=ev.LANDED)
    rows["target"] = rows["landed"] = frames
    rows["param"], rows["wire"], rows["value"] = params, values, values
    rows["sent"] = np.arange(len(frames))
    return rows


def scores_with(transient, *, count=6000, spacing=40, seed=0):
    rng = np.random.default_rng(seed)
    scores = rng.standard_normal(count)
    frames = np.arange(spacing, count - spacing, spacing)
    for f in frames:
        scores[f : f + transient.size] += transient
    params = np.where(np.arange(frames.size) % 2, 1, 2)
    return scores, frames, params


def test_a_step_that_settles_after_two_lags_holds_for_two_more():
    scores, frames, params = scores_with(np.array([8.0, 6.0, 4.0]))
    found = R.response("p", [R.differences(scores, frames, params)])
    assert found.settle == 2 and found.bounded and found.hold == 2 + R.MEASURED
    assert found.params == (1, 2) and found.moving(1) == 2
    assert "bounded" in found.table() and found.as_dict()["hold"] == 4


def test_a_regime_change_that_persists_is_settled_not_moving():
    scores, frames, params = scores_with(np.array([8.0]))
    for k, f in enumerate(frames):
        scores[f + 1 : f + 40] += 3.0 * (k % 2)
    assert R.response("p", [R.differences(scores, frames, params)]).settle == 0


def test_no_steps_and_too_few_steps_are_not_bounded():
    empty = R.response("p", [])
    assert empty.settle == 0 and not empty.bounded
    scores, frames, params = scores_with(np.array([8.0]), count=200)
    assert not R.response("p", [R.differences(scores, frames[:3], params[:3])]).bounded


def test_writes_landing_together_have_no_sole_parameter():
    diff, sole = R.differences(np.zeros(100), np.array([10, 10, 50]), np.array([1, 2, 3]))
    assert sole.tolist() == [0, 3] and diff.shape == (2, 25)
    assert np.isnan(diff[0, 20]) and diff[1, 24] == 0


def test_only_value_changing_writes_are_steps():
    rows = landed_rows([5, 9, 12, 12], [1, 1, 2, 3], [0, 7, 3, 3])
    rows["sent"][3] = -1
    frames, params = R.steps_of(rows, np.zeros(12, dtype=np.int64))
    assert frames.tolist() == [9, 12] and params.tolist() == [1, 2]


def test_take_scores_matches_against_step_free_frames():
    count, rng = 400, np.random.default_rng(1)
    small = rng.integers(0, 255, (count, 3, 4, 4)).astype(np.uint8)
    seen = SimpleNamespace(
        found=SimpleNamespace(picked=np.arange(count), lead=0, source_offset=0),
        features=SimpleNamespace(small=small),
        source=SimpleNamespace(small=small),
        proof=SimpleNamespace(landed=SimpleNamespace(rows=landed_rows([50, 100], [1, 2], [5, 6]))),
    )
    scores, frames, params = R.take_scores(seen, np.zeros(12, dtype=np.int64))
    assert scores.shape == (count,) and np.isnan(scores[0]) and np.isfinite(scores[1:]).all()
    assert frames.tolist() == [50, 100] and params.tolist() == [1, 2]


def test_saved_response_round_trips(tmp_path):
    scores, frames, params = scores_with(np.array([8.0]))
    found = R.response("p", [R.differences(scores, frames, params)], ["t.mkv"])
    found.save(tmp_path / "h.json")
    assert '"hold": 2' in (tmp_path / "h.json").read_text()
    assert "source" not in found.as_dict()
    named = R.response("p", [R.differences(scores, frames, params)], ["t.mkv"], "c.mkv")
    assert named.as_dict()["source"] == "c.mkv"

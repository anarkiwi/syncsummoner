"""Region sweeps: follow-test holds from the footage, Sobol settings per regime, and the takes they make."""

# pylint: disable=missing-function-docstring

import json
from pathlib import Path
import math
from types import SimpleNamespace

import numpy as np
import pytest

from syncsummoner import cli
from syncsummoner.aesthetics.survival import SIGMA, critical_z, effective_samples, follow_z
from syncsummoner.compose import locktake as LT
from syncsummoner.compose import timecode
from syncsummoner.compose.session import session_frames
from syncsummoner.compose.render import RenderConfig
from syncsummoner.probe import decoded
from syncsummoner.probe import locked as L
from syncsummoner.probe import regions as R
from syncsummoner.probe import sweep_cmds
from syncsummoner.program.model import Bypass


def ar1(n, a, seed):
    rng = np.random.default_rng(seed)
    x = np.zeros(n)
    noise = rng.standard_normal(n)
    for i in range(1, n):
        x[i] = a * x[i - 1] + noise[i]
    return x + 10.0


def brute_hold(ti, effect, start):
    for h in range(4, ti.size - start + 1):
        w = ti[start : start + h]
        if follow_z(effect, effective_samples(w, w)) >= critical_z() + SIGMA:
            return h
    return 0


def test_a_bars_effect_is_the_least_follow_its_dead_bar_test_resolves():
    ti = ar1(300, 0.5, 1)
    effect = R.effect_sizes(ti, np.array([100, 102]))
    n = effective_samples(ti[:100], ti[:100])
    assert effect[50] == pytest.approx(math.tanh(critical_z() / math.sqrt(n - 3)))
    assert follow_z(effect[50], n) == pytest.approx(critical_z())
    assert np.isnan(effect[100:102]).all() and np.isfinite(effect[102:]).all()


@pytest.mark.parametrize("a", [0.0, 0.6, 0.9])
def test_the_hold_is_the_first_window_whose_follow_test_reaches_the_power(a):
    ti = ar1(2000, a, 3)
    for start, effect in ((0, 0.4), (500, 0.25), (1200, 0.6)):
        assert R.hold_at(ti, effect, start) == brute_hold(ti, effect, start)


def test_motion_that_lingers_needs_a_longer_hold_and_short_footage_none():
    assert R.hold_at(ar1(6000, 0.9, 4), 0.4, 0) > R.hold_at(ar1(6000, 0.0, 4), 0.4, 0) > 0
    assert R.hold_at(ar1(20, 0.0, 4), 0.3, 0) == 0
    assert R.hold_at(ar1(20, 0.0, 4), float("nan"), 0) == 0
    assert R.holds(ar1(400, 0.3, 5), np.array([200]), [0, 250]).tolist() == [
        R.hold_at(ar1(400, 0.3, 5), float(R.effect_sizes(ar1(400, 0.3, 5), np.array([200]))[t]), t)
        for t in (0, 250)
    ]


def test_the_guard_covers_every_slot_written_a_frame_apart_and_the_settle():
    assert R.guard_frames(1) == 1 + R.PARAM_COUNT + 1
    assert R.guard_frames(4) - R.guard_frames(1) == 3


def test_settings_fill_the_clip_take_by_take_within_one_sessions_budget():
    ti = ar1(3000, 0.5, 6)
    edges = np.arange(0, 3000, 90)
    placed = R.slots(ti, edges, guard=4, budget_s=400.0, overhead_s=80.0, fps=30.0)
    takes = {s.take for s in placed}
    assert len(takes) > 1
    for take in takes:
        mine = [s for s in placed if s.take == take]
        assert mine[0].start == 0 and all(b.start == a.start + a.frames for a, b in zip(mine, mine[1:]))
        assert mine[-1].start + mine[-1].frames <= ti.size
    effect = R.effect_sizes(ti, edges)
    assert all(s.hold == R.hold_at(ti, float(effect[s.start + 4]), s.start + 4) for s in placed)
    used = sum(s.frames for s in placed) / 30.0 + 80.0
    assert used <= 400.0, "the takes play back to back in one session, its lead-in and lead-out paid once"
    assert R.slots(ti, edges, guard=4, budget_s=480.0, overhead_s=160.0, fps=30.0) == placed
    assert len(R.slots(ti, edges, guard=4, budget_s=800.0, overhead_s=80.0, fps=30.0)) > len(placed)
    with pytest.raises(L.SweepError):
        R.slots(np.full(30, 10.0), edges, guard=4, budget_s=400.0, overhead_s=80.0, fps=30.0)


def test_regimes_get_power_of_two_shares_by_their_dimensions():
    got = R.allot(100, [1, 3, 4])
    assert got == [8, 32, 32] and sum(got) <= 100
    assert R.allot(3, [1, 12]) == [0, 2] and R.allot(0, [2]) == [0] and R.allot(5, [0]) == [0]


def test_a_fill_gives_exactly_its_total_in_powers_of_two_where_they_reach_it():
    assert R.fill(64, [3, 5, 5], [1024] * 3) == [16, 32, 16]
    assert R.fill(64, [12], [4096]) == [64]
    got = R.fill(64, [1, 1, 11], [2, 2, 4096])
    assert got == [2, 2, 60] and sum(got) == 64
    assert R.fill(64, [0, 2], [1, 16]) == [1, 16]
    assert R.fill(5, [1, 2], [64, 64]) == [2, 3]
    with pytest.raises(R.SweepError):
        R.fill(1, [1, 1], [2, 2])


def test_a_design_of_no_power_of_two_is_the_prefix_of_the_next():
    base = np.zeros(12, dtype=np.int64)
    values = {1: np.arange(0, 1024, 64), 2: np.array([0, 1023])}
    whole = R.design(base, [1, 2], values, 64, seed=3)
    part = R.design(base, [1, 2], values, 48, seed=3)
    assert (part == whole[:48]).all()
    assert np.unique(part[:32, 1], return_counts=True)[1].tolist() == [16, 16]


def test_a_design_spreads_jointly_over_read_settings_only():
    base = np.full(12, 7, dtype=np.int64)
    values = {1: np.array([0, 341, 682, 1023]), 2: np.arange(0, 1024, 8), 7: np.array([0, 1023])}
    got = R.design(base, [1, 2, 7, 9], values, 64, seed=5)
    assert got.shape == (64, 12) and (got[:, 2:6] == 7).all() and (got[:, 8] == 7).all()
    assert np.isin(got[:, 1], values[2]).all()
    assert sorted(np.unique(got[:, 0], return_counts=True)[1].tolist()) == [16] * 4
    assert np.unique(got[:, 6], return_counts=True)[1].tolist() == [32, 32]
    assert abs(np.corrcoef(got[:, 0], got[:, 1])[0, 1]) < 0.3
    assert R.design(base, [9], values, 4, seed=5).tolist() == [base.tolist()] * 4


def test_each_take_writes_its_first_setting_whole_and_then_only_what_moves():
    states = np.zeros((4, 12), dtype=np.int64)
    states[1, 0], states[2, 7], states[3, 0] = 100, 300, 900
    placed = [R.Slot(0, 0, 2, 10), R.Slot(0, 12, 2, 20), R.Slot(1, 0, 2, 10), R.Slot(1, 12, 2, 15)]
    chunks = R.chunks_of(states, placed, settle=1, toggles=[8])
    assert [c.first for c in chunks] == [0, 2] and [c.frames for c in chunks] == [34, 29]
    assert chunks[0].plan[12:].tolist() == [(12, 1, 100)] and chunks[0].owner[12:].tolist() == [1]
    assert chunks[1].plan[7].tolist() == (0, 8, 0)
    assert chunks[1].plan[12:].tolist() == [(12, 1, 900)] and chunks[1].hold == 15
    with pytest.raises(L.SweepError):
        R.chunks_of(states[:3], placed, settle=1, toggles=[8])


def _regions_argv(tmp_path, monkeypatch):
    """A two-regime program's `sweep regions` arguments, its clip, cover and model stubbed."""
    hold = tmp_path / "hold.json"
    hold.write_text(json.dumps({"program": "p", "settle": 1, "hold": 3}))
    cover = tmp_path / "cover.json"
    states = np.zeros((2, 12), dtype=np.int64)
    states[1, 0] = 512
    cover.write_text(
        json.dumps({"states": states.tolist(), "chosen": [0, 1], "stepped": {"0": [1, 2], "1": [2]}})
    )
    lead = 1200
    luma = np.zeros((lead + 3000 + lead, 4, 4), dtype=np.float32)
    rng = np.random.default_rng(7)
    luma[lead : lead + 3000] = np.cumsum(rng.normal(0, 1, (3000, 4, 4)), axis=0)
    monkeypatch.setattr(decoded, "features", lambda *a, **k: SimpleNamespace(luma=luma))
    monkeypatch.setattr(timecode, "clip_layout", lambda clip: timecode.ClipLayout(lead, 3000, lead))
    monkeypatch.setattr(sweep_cmds, "downbeat_frames", lambda audio, fps: np.arange(0, 3000, 84))
    values = {
        1: np.arange(0, 1024, 64),
        2: np.array([0, 512]),
        6: np.array([0, 1023]),
        11: np.array([0, 1023]),
    }
    model = SimpleNamespace(
        resolution=lambda: {p: (0,) for p in values}, bypass=lambda: Bypass(11, 1023, tuple(range(512, 1024)))
    )
    monkeypatch.setattr("syncsummoner.program.model", lambda name: model)
    monkeypatch.setattr("syncsummoner.program.gates.representatives", lambda m, p, bits: values[p])
    cover.write_text(
        json.dumps({"states": states.tolist(), "chosen": [0, 1], "stepped": {"0": [1, 2, 11], "1": [2]}})
    )
    argv = ["probe", "sweep", "regions", "p", "--cover", str(cover), "--hold", str(hold), "--source", "c.mkv"]
    argv += ["--audio", "a.wav", "--takes", str(tmp_path), "--budget-s", "900", "--joint", "11:6"]
    return argv


def test_regions_plans_a_sweep_the_capture_and_collect_steps_read(tmp_path, monkeypatch, capsys):
    argv = _regions_argv(tmp_path, monkeypatch)
    assert cli.main(argv) == 0
    chunks, said = L.read_plan(tmp_path / "sweep-p.json")
    regions = said["regions"]
    assert regions["dims"] == {"0": [1, 2], "1": [2]}
    assert regions["joint"] == 6 and regions["guard_frames"] == R.guard_frames(1)
    assert (
        sum(len(c.states) for c in chunks) == regions["settings"] == sum(regions["per_regime"].values()) + 6
    )
    assert all(c.frames <= 3000 for c in chunks) and regions["rig_s"] <= 900
    ends = session_frames(LT.LeadIn(), 30.0, dry=True)
    session_s = sum(ends) / 30.0 + RenderConfig.for_format("1080p30").tail_s
    assert regions["rig_s"] == pytest.approx(sum(c.frames for c in chunks) / 30.0 + session_s)
    assert "settings" in capsys.readouterr().out
    assert all(math.log2(n).is_integer() for n in regions["per_regime"].values() if n)
    assert said["response"]["settle"] == 1 and said["response"].get("measured", True)
    unheld = [a for i, a in enumerate(argv) if a != "--hold" and argv[i - 1] != "--hold"]
    assert cli.main([a for a in unheld if a not in ("--budget-s", "900")]) == 0
    chunks, said = L.read_plan(tmp_path / "sweep-p.json")
    regions = said["regions"]
    assert said["response"] == {"program": "p", "settle": 0, "measured": False}
    assert regions["guard_frames"] == R.guard_frames(0) and regions["budget_s"] is None
    assert regions["per_regime"] == {"0": 4, "1": 2} and regions["settings"] == 4 + 2 + 6
    assert sum(len(c.states) for c in chunks) == 12
    clip = tmp_path / "c.mkv"
    Path(f"{clip}.stimulus.json").write_text(json.dumps({"stimulus": {"shot_frames": 45}}), encoding="utf-8")
    synthetic = [str(clip) if a == "c.mkv" else a for a in unheld if a not in ("--budget-s", "900")]
    assert cli.main(synthetic) == 0
    chunks, said = L.read_plan(tmp_path / "sweep-p.json")
    assert said["regions"]["shot_frames"] == 45 and said["regions"]["hold_frames"][
        "max"
    ] == 45 - R.guard_frames(0)
    assert sum(len(c.states) for c in chunks) == 12 and sum(c.frames for c in chunks) == 12 * 45


def test_regions_plans_exactly_the_settings_asked_joint_moves_among_them(tmp_path, monkeypatch):
    clip = tmp_path / "c.mkv"
    Path(f"{clip}.stimulus.json").write_text(json.dumps({"stimulus": {"shot_frames": 45}}), encoding="utf-8")
    argv = [str(clip) if a == "c.mkv" else a for a in _regions_argv(tmp_path, monkeypatch)]
    assert cli.main([a for a in argv if a not in ("--budget-s", "900")] + ["--settings", "9"]) == 0
    chunks, said = L.read_plan(tmp_path / "sweep-p.json")
    assert said["regions"]["per_regime"] == {"0": 2, "1": 1} and said["regions"]["settings"] == 9
    assert said["regions"]["joint"] == 6 and sum(len(c.states) for c in chunks) == 9
    with pytest.raises(SystemExit):
        cli.main(argv + ["--settings", "9"])


def test_shot_slots_put_each_setting_on_whole_shots_its_guard_covering_the_cut():
    placed = R.shot_slots(45, 200, guard=13, limit=7)
    assert [(s.take, s.start, s.guard, s.hold) for s in placed[:5]] == [
        (0, 0, 13, 32),
        (0, 45, 13, 32),
        (0, 90, 13, 32),
        (0, 135, 13, 32),
        (1, 0, 13, 32),
    ]
    assert len(placed) == 7 and all(s.start % 45 == 0 for s in placed)
    longer = R.shot_slots(15, 200, guard=13, limit=2)
    assert [(s.start, s.hold) for s in longer] == [
        (0, 17),
        (30, 17),
    ], "a guard past a shot takes the next too"
    with pytest.raises(R.SweepError, match="needs 90 frames"):
        R.shot_slots(45, 60, guard=40, limit=1)


def test_second_order_resolves_every_effect_and_pair_at_a_power_of_two():
    assert [R.second_order(d) for d in (0, 1, 2, 3, 8, 12)] == [1, 2, 4, 8, 64, 128]


def test_slots_stop_at_a_limit_and_need_a_bound():
    ti = ar1(3000, 0.5, 6)
    edges = np.arange(0, 3000, 90)
    full = R.slots(ti, edges, guard=4, budget_s=800.0, overhead_s=80.0, fps=30.0)
    capped = R.slots(ti, edges, guard=4, overhead_s=80.0, fps=30.0, limit=5)
    assert len(full) > 5 and capped == full[:5]
    assert R.slots(ti, edges, guard=4, budget_s=800.0, overhead_s=80.0, fps=30.0, limit=len(full) + 9) == full
    with pytest.raises(ValueError, match="budget or a limit"):
        R.slots(ti, edges, guard=4, overhead_s=80.0, fps=30.0)


def test_clip_ti_compares_each_frame_with_the_one_played_before():
    luma = np.zeros((6, 2, 2), dtype=np.float32)
    luma[2:4] = [[[0, 2], [0, 2]], [[2, 0], [2, 0]]]
    ti = sweep_cmds.clip_ti(luma, lead=2, tail=2)
    assert ti.tolist() == [1.0, 2.0]


def test_region_dims_keep_the_read_stepped_parameters_and_never_the_bypass():
    record = {"chosen": [0, 2], "stepped": {"0": [1, 2, 11], "2": [3, 9]}}
    values = {1: np.arange(4), 2: np.arange(4), 3: np.arange(4), 11: np.array([0, 1023])}
    switch = Bypass(11, 1023, tuple(range(512, 1024)))
    assert R.region_dims(record, R.without_bypass(values, switch)) == [[1, 2], [3]]
    assert R.region_dims(record, R.without_bypass(values, None)) == [[1, 2, 11], [3]]


def test_a_fader_bypass_keeps_its_fader_sampled_away_from_the_bypass_setting():
    record = {"chosen": [0], "stepped": {"0": [1, 12]}}
    values = {1: np.arange(4), 12: np.array([0, 300, 700, 1023])}
    kept = R.without_bypass(values, Bypass(12, 0, (0,)))
    assert kept[12].tolist() == [300, 700, 1023] and kept[1] is values[1]
    assert R.region_dims(record, kept) == [[1, 12]]
    assert R.without_bypass(values, Bypass(5, 0, (0,)))[12] is values[12]


def test_a_design_asks_for_no_more_settings_than_its_dimensions_have():
    values = {7: np.array([0, 1023]), 1: np.arange(0, 1024, 64), 2: np.arange(3)}
    assert R.distinct([7], values) == 2 and R.distinct([7, 1], values) == 32
    assert R.distinct([2], values) == 2 and R.distinct([], values) == 1

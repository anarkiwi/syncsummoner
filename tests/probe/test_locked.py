"""Sweeps as locked takes: planning, matching what was sent, and the rows each setpoint settled over."""

# pylint: disable=missing-function-docstring

from types import SimpleNamespace

import numpy as np
import pytest

from syncsummoner.compose import evidence as ev
from syncsummoner.probe import locked as L
from syncsummoner.probe.archive import GAP, MJPEG_PIX_FMT, FrameArchive, FrameRow
from syncsummoner.probe.fit import load_measurements
from syncsummoner.probe.store import KeyKind, ProgramKey


def sweep_states(n=9):
    base = np.zeros(12, dtype=np.int64)
    out = [base]
    for k in range(1, n):
        state = base.copy()
        if k % 4:
            state[0] = 10 * k
        out.append(state)
    return np.array(out)


def landed_from(chunk, *, late=(), unsent=(), ambiguous=()):
    plan = chunk.plan[chunk.plan["target"] > 0]
    rows = np.zeros(plan.size, dtype=ev.LANDED)
    for name in ("target", "param", "value"):
        rows[name] = plan[name]
    rows["wire"], rows["sent"], rows["landed"] = plan["value"], np.arange(plan.size), plan["target"]
    for i, by in late:
        rows["landed"][i] += by
    rows["first"], rows["last"] = rows["landed"], rows["landed"]
    rows["sent"][list(unsent)] = -1
    rows["last"][list(ambiguous)] += 1
    return rows


def test_a_sweep_is_cut_into_takes_one_hold_a_setpoint():
    states = sweep_states(9)
    chunks = L.plan_sweep(states, settle=1, hold=3, frames=10)
    assert [c.first for c in chunks] == [0, 3, 6]
    assert all(c.frames <= 10 for c in chunks) and sum(len(c.states) for c in chunks) == 9
    first = chunks[0]
    assert first.plan["target"][:12].tolist() == [0] * 12
    assert first.plan["target"][12:].tolist() == [3, 6] and first.owner[12:].tolist() == [1, 2]
    assert L.Chunk.from_dict(first.as_dict()).plan.tolist() == first.plan.tolist()
    with pytest.raises(L.SweepError):
        L.plan_sweep(states, settle=0, hold=11, frames=10)


def test_levels_are_the_covers_states_in_order():
    a, b = np.zeros(12, dtype=np.int64), np.ones(12, dtype=np.int64)
    assert L.levels(SimpleNamespace(vectors=lambda: [(a, b), (b, a)])).tolist() == [b.tolist(), a.tolist()]


def test_writes_moved_by_the_take_still_find_their_setpoint():
    chunk = L.plan_sweep(sweep_states(4), settle=0, hold=2, frames=20)[0]
    rows = landed_from(chunk)
    rows["target"][1] -= 1
    assert L.owners(chunk, rows).tolist() == [1, 2, 3][: rows.size]
    rows["value"][0] += 1
    with pytest.raises(L.SweepError):
        L.owners(chunk, rows)


def test_each_setpoint_keeps_the_frames_after_it_settled_until_the_next_lands():
    chunk = L.plan_sweep(sweep_states(4), settle=1, hold=3, frames=20)[0]
    count = chunk.frames
    picked = np.arange(count) * 2
    shown = np.repeat(np.arange(count), 2)
    landed = landed_from(chunk, late=[(1, 1)])
    lead_out = landed[-1:].copy()
    lead_out["target"], lead_out["landed"], lead_out["param"], lead_out["value"] = count, count, 11, 1023
    rows = L.settled_rows(
        chunk,
        np.concatenate([landed, lead_out]),
        picked=picked,
        shown=shown,
        captured=np.arange(shown.size) / 60.0,
        lead=5,
        program="p",
        bad_frames=[10, 99],
    )
    by_frame = [r.setpoint for r in rows[::2]]
    assert by_frame == [GAP, 0, 0, GAP, 1, 1, 1, GAP, 2, GAP, GAP, 3]
    assert all(r.setpoint == GAP for r in rows[1::2])
    assert rows[8].params[0] == 10 and rows[0].shown == 5 and rows[1].shown == 5


def test_an_unsent_write_and_a_write_on_a_tick_cost_their_setpoint_frames():
    chunk = L.plan_sweep(sweep_states(4), settle=0, hold=3, frames=20)[0]
    count = chunk.frames
    rows = L.settled_rows(
        chunk,
        landed_from(chunk, unsent=[0], ambiguous=[1]),
        picked=np.arange(count),
        shown=np.arange(count),
        captured=np.zeros(count),
        lead=0,
        program="p",
    )
    assert [r.setpoint for r in rows] == [0, 0, 0, 0, 0, 0, GAP, 2, 2, 3, 3, 3]
    assert rows[4].params[0] == 0


def test_plans_and_rows_round_trip(tmp_path):
    chunks = L.plan_sweep(sweep_states(5), settle=0, hold=2, frames=6)
    path = L.write_plan(tmp_path / "sweep.json", chunks, program="p")
    back, said = L.read_plan(path)
    assert said["program"] == "p" and [c.first for c in back] == [c.first for c in chunks]
    rows = L.settled_rows(
        back[0],
        landed_from(back[0]),
        picked=np.arange(back[0].frames),
        shown=np.arange(back[0].frames),
        captured=np.zeros(back[0].frames),
        lead=0,
        program="p",
    )
    L.save_rows(tmp_path / "t.mkv", rows)
    assert len(load_measurements(L.rows_path(tmp_path / "t.mkv"))) == back[0].frames


def test_takes_join_into_one_archive_of_every_capture(tmp_path):
    chunks = L.plan_sweep(sweep_states(5), settle=0, hold=2, frames=6)
    takes = []
    for k, chunk in enumerate(chunks):
        take = tmp_path / f"take-{k}.mkv"
        take.write_bytes(b"v")
        rows = L.settled_rows(
            chunk,
            landed_from(chunk),
            picked=np.arange(chunk.frames),
            shown=np.arange(chunk.frames),
            captured=np.arange(chunk.frames) + 100.0 * k,
            lead=0,
            program="p",
        )
        L.save_rows(take, rows)
        takes.append(take)

    def run(_argv, **_kw):
        return SimpleNamespace(returncode=0, stdout=b"0.2\n", stderr=b"")

    archive = FrameArchive(tmp_path / "archive", run=run)
    meta = L.commit_sweep(
        archive,
        "p",
        ProgramKey("p", "1.0.2-rc.4", KeyKind.NAME_FIRMWARE),
        takes,
        chunks,
        width=4,
        height=2,
        fps=30.0,
        program_info={"name": "p"},
        stimulus={"kind": "footage", "clip": "c.mkv", "sha256": "x"},
    )
    assert meta["parts"] == [f"p.part{k:02d}.mkv" for k in range(len(takes))]
    assert meta["pix_fmt"] == MJPEG_PIX_FMT and meta["codec"] == "mjpeg"
    assert archive.meta("p") is not None
    rows = archive.rows("p")
    assert [r.frame for r in rows] == list(range(len(rows))) and len(rows) == sum(c.frames for c in chunks)
    assert {r.setpoint for r in rows} - {GAP} == set(range(5)) and len(meta["schedule"]) == 5


def test_disagreeing_frames_come_from_the_readback():
    applied = SimpleNamespace(disagree=np.array([(0, 1, 7, 0, 1), (1, 2, 7, 0, 1)], dtype=ev.DISAGREE))
    assert L.disagreeing(applied).tolist() == [7]


def test_toggles_go_out_as_off_or_on_and_repeats_drop():
    states = np.zeros((4, 12), dtype=np.int64)
    states[1, 6], states[2, 6], states[3, 0] = 768, 1000, 5
    out = L.carried(states, [7])
    assert out[:, 6].tolist() == [0, 1023, 0] and out[:, 0].tolist() == [0, 0, 5]
    states[1, 6], states[2, 6] = 511, 512
    assert L.carried(states, [7])[:, 6].tolist() == [0, 1023, 0]


def test_a_recorded_cover_walks_each_stepped_parameter_off_its_base_and_back():
    record = {"states": [[0] * 12, [5] + [0] * 11], "chosen": [1], "stepped": {"1": [2]}}
    out = L.recorded(record, {2: np.array([0, 3, 6])})
    assert out[:, 0].tolist() == [5, 5, 5, 5] and out[:, 1].tolist() == [0, 6, 3, 0]
    assert L.recorded({"states": [[1] * 12], "chosen": [0], "stepped": {}}, {}).shape == (1, 12)


def test_a_selector_is_held_at_each_setting_with_a_mover_move_under_each():
    base = np.zeros((1, 12), dtype=np.int64)
    base[0, 5] = 300
    out = L.joint(base, 11, 6, {11: np.array([0, 1023]), 6: np.array([0, 300, 1023])})
    assert out[:, 10].tolist() == [0, 0, 0, 1023, 1023, 1023, 0]
    assert out[:, 5].tolist() == [300, 1023, 300, 300, 1023, 300, 300]


def judged(rows, **counts):
    """A :class:`evidence.Landed` stand-in over ``rows``, its fault counts zero unless given."""
    base = {"missing": 0, "unechoed": 0, "duplicated": 0, "out_of_order": 0, "late": 0}
    return SimpleNamespace(rows=rows, **{**base, **counts})


def test_a_write_late_on_its_frame_but_landed_before_its_window_is_no_fault():
    chunk = L.plan_sweep(sweep_states(4), settle=1, hold=3, frames=20)[0]
    got = L.verdict(chunk, judged(landed_from(chunk, late=[(1, 1)]), late=1), np.array([3]), chunk.frames)
    assert got.ok and got.late == 1 and got.misapplied == 0
    assert got.as_dict() == {"ok": True, "untaken": 0, "unopened": 0, "misapplied": 0, "late": 1}


def test_a_write_landing_past_its_settings_window_fails_the_take():
    chunk = L.plan_sweep(sweep_states(4), settle=1, hold=3, frames=20)[0]
    rows = landed_from(chunk)
    rows["last"][0] += 2
    got = L.verdict(chunk, judged(rows), np.zeros(0), chunk.frames)
    assert not got.ok and got.unopened == 1


def test_a_readback_disagreeing_inside_an_open_window_fails_the_take():
    chunk = L.plan_sweep(sweep_states(4), settle=1, hold=3, frames=20)[0]
    rows = landed_from(chunk)
    assert L.verdict(chunk, judged(rows), np.array([3, 6, 9]), chunk.frames).ok
    got = L.verdict(chunk, judged(rows), np.array([5]), chunk.frames)
    assert not got.ok and got.misapplied == 1


def test_a_write_never_taken_fails_the_take():
    chunk = L.plan_sweep(sweep_states(4), settle=0, hold=3, frames=20)[0]
    got = L.verdict(chunk, judged(landed_from(chunk), unechoed=1), np.zeros(0), chunk.frames)
    assert not got.ok and got.untaken == 1


def test_rows_carry_how_many_frames_earlier_the_clock_also_admits():
    chunk = L.plan_sweep(sweep_states(4), settle=0, hold=3, frames=20)[0]
    count = chunk.frames
    late = np.r_[np.zeros(count // 2, dtype=np.int64), np.ones(count - count // 2, dtype=np.int64)]
    rows = L.settled_rows(
        chunk,
        landed_from(chunk),
        picked=np.arange(count),
        shown=np.arange(count),
        captured=np.zeros(count),
        lead=0,
        program="p",
        ambiguity=late,
    )
    assert [r.ambiguity for r in rows] == late.tolist()
    back = FrameRow.from_row(rows[-1].as_row())
    assert back.ambiguity == 1
    old = {k: v for k, v in rows[-1].as_row().items() if k != "ambiguity"}
    assert FrameRow.from_row(old).ambiguity == 0


def clocked_rows(ambiguity=1):
    """Two runs of three captures: the first strip-named, the second clock-numbered with ``ambiguity``."""
    rows = [
        FrameRow(
            frame=c,
            program="p",
            params=(0,) * 12,
            setpoint=c // 2,
            shown=10 + c,
            ambiguity=ambiguity * (c >= 3),
        )
        for c in range(6)
    ]
    return rows, np.repeat([0, 2], 3)


@pytest.mark.parametrize("pooled,numbering", [((1.0, 9.0), 7), ((5.0, 5.2), None)], ids=["chosen", "union"])
def test_a_clock_numbered_run_takes_the_numbering_its_settings_pool_to(monkeypatch, pooled, numbering):
    from syncsummoner.probe import replay  # pylint: disable=import-outside-toplevel

    monkeypatch.setattr(replay, "pooled_follow", lambda measured, span: np.array(pooled))
    monkeypatch.setattr(replay, "_measured", lambda *a: ())
    rows, seg = clocked_rows()
    clock = [{"run": 2, "candidates": [7, 8]}]
    got = L.number_runs(rows, clock, seg, np.zeros((6, 2, 2)), (np.zeros(20), np.zeros(20)))
    assert clock[0]["selection"]["numbering"] == numbering
    if numbering is None:
        assert [r.shown for r in got] == [r.shown for r in rows] and [r.ambiguity for r in got][3:] == [1] * 3
    else:
        assert [r.shown for r in got] == [10, 11, 12, 12, 13, 14] and not any(r.ambiguity for r in got)


def test_a_uniquely_numbered_run_records_its_one_numbering():
    rows, seg = clocked_rows(ambiguity=0)
    clock = [{"run": 2, "candidates": [7]}]
    assert L.number_runs(rows, clock, seg, np.zeros((6, 2, 2)), ()) == rows
    assert clock[0]["selection"] == {"shift": 0, "numbering": 7}


def test_pooled_numbering_follows_the_clip_only_when_its_series_counts_the_lead_in():
    """Rows name played frames, lead-in counted; read past the lead-in the series picks the true shift."""
    from syncsummoner import aesthetics  # pylint: disable=import-outside-toplevel

    rng = np.random.default_rng(3)
    lead, count = 30, 400
    field = np.cumsum(rng.standard_normal((16, 400)), axis=1)
    pan = np.cumsum(rng.integers(0, 4, size=lead + count))
    clip = np.stack([np.roll(field, -int(p), axis=1)[:, :16] for p in pan])
    clip = np.rint(255 * (clip - clip.min()) / np.ptp(clip)).astype(np.uint8)
    rows, seg = [], []
    for k in range(4):
        for n in range(lead + 100 * k, lead + 100 * k + 90):
            rows.append(
                FrameRow(frame=len(rows), program="p", params=(0,) * 12, setpoint=k, shown=n, ambiguity=1)
            )
            seg.append(2)
    seg = np.array(seg)
    luma = clip[[r.shown for r in rows]]
    series = aesthetics.information(clip[lead:].astype(np.float32) / 255.0)
    clock = [{"run": 2, "candidates": [7, 8]}]
    L.number_runs(list(rows), clock, seg, luma, series, lead=lead)
    assert clock[0]["selection"]["shift"] == 0
    clock = [{"run": 2, "candidates": [7, 8]}]
    L.number_runs(list(rows), clock, seg, luma, series)
    assert clock[0]["selection"]["shift"] != 0


def test_a_takes_channel_order_is_kept_in_its_attempt_record_and_never_faults_it(monkeypatch, caplog):
    from syncsummoner.compose import chroma as cc  # pylint: disable=import-outside-toplevel

    said = {"order": "chroma exchanged", "ok": False, "judged": True}
    monkeypatch.setattr(cc, "judge", lambda take, clip, config: dict(said))
    assert L.chroma_record("take.mkv", "clip.mkv", None) == {"chroma": said}
    assert "read with swapuv" in caplog.text
    said.clear()
    said["error"] = "ValueError: no strip"
    assert L.chroma_record("take.mkv", "clip.mkv", None) == {"chroma": said}
    assert "no channel order from the lead-in" in caplog.text

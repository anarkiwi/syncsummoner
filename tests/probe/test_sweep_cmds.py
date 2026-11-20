"""The probe hold and probe sweep commands, with the takes, the model and the decodes faked."""

# pylint: disable=missing-function-docstring

import json
from types import SimpleNamespace

import numpy as np
import pytest

from syncsummoner.compose.stimulus import Coverage
from syncsummoner import cli
from syncsummoner.compose import locktake as LT
from syncsummoner.compose import response as response_mod
from syncsummoner.compose import session as SS
from syncsummoner.compose import timecode
from syncsummoner.compose.evidence import DISAGREE, LANDED
from syncsummoner.probe import archive as A
from syncsummoner.probe import decoded
from syncsummoner.probe import locked as L
from syncsummoner.probe import sweep_cmds
from syncsummoner.probe.archive import FrameArchive
from syncsummoner.program import gates


def fake_features(count):
    return SimpleNamespace(small=np.zeros((count, 3, 2, 2), dtype=np.uint8))


def test_hold_writes_each_programs_response(tmp_path, monkeypatch, capsys):
    rng = np.random.default_rng(0)
    scores = rng.standard_normal(4000)
    frames = np.arange(40, 3960, 40)
    scores[frames] += 9.0
    found = response_mod.response(
        "czartroniks/Quilt", [response_mod.differences(scores, frames, frames % 2 + 1)]
    )
    monkeypatch.setattr(response_mod, "of_takes", lambda takes, **_kw: {"czartroniks/Quilt": found})
    assert cli.main(["probe", "hold", "t.mkv", "--source", "c.mkv", "--out", str(tmp_path)]) == 0
    assert json.loads((tmp_path / "hold-czartroniks_Quilt-803bc8ea.json").read_text())["hold"] == 2
    assert "settles 0 frames" in capsys.readouterr().out


def test_of_takes_groups_takes_by_program(monkeypatch):
    rows = np.zeros(1, dtype=response_mod.ev.LANDED)
    rows["landed"], rows["param"], rows["wire"], rows["sent"] = 10, 1, 5, 0
    seen = SimpleNamespace(
        found=SimpleNamespace(picked=np.arange(50), lead=0, source_offset=0),
        features=fake_features(50),
        source=fake_features(50),
        proof=SimpleNamespace(landed=SimpleNamespace(rows=rows)),
        start=np.zeros(12, dtype=np.int64),
        program="p",
    )
    monkeypatch.setattr(response_mod, "judge_offline", lambda take, **_kw: seen)
    out = response_mod.of_takes(["a.mkv", "b.mkv"], clip="c.mkv", config=None)
    assert list(out) == ["p"] and out["p"].takes == ("a.mkv", "b.mkv")


def plan_states():
    states = np.zeros((6, 12), dtype=np.int64)
    states[1:, 0] = [10, 20, 30, 40, 0]
    states[3, 6] = 300
    return states


def test_sweep_plan_cuts_the_cover_into_takes(tmp_path, monkeypatch, capsys):
    hold = tmp_path / "hold.json"
    hold.write_text(json.dumps({"program": "p", "settle": 1, "hold": 3}))
    monkeypatch.setattr(timecode, "clip_layout", lambda clip: timecode.ClipLayout(1200, 12, 1200))
    monkeypatch.setattr("syncsummoner.program.model", lambda name: SimpleNamespace(name=name))
    monkeypatch.setattr(gates, "cover", lambda m, gating=None: "cover")
    monkeypatch.setattr(L, "levels", lambda found: plan_states())
    argv = ["probe", "sweep", "plan", "p", "--hold", str(hold), "--source", "c.mkv", "--takes", str(tmp_path)]
    assert cli.main(argv + ["--structural"]) == 0
    chunks, said = L.read_plan(tmp_path / "sweep-p.json")
    assert said["source"] == "c.mkv" and sum(len(c.states) for c in chunks) == 6
    assert all(c.frames <= 12 for c in chunks) and "6 setpoints" in capsys.readouterr().out
    assert cli.main(argv[:3] + ["q"] + argv[4:]) == 1


def sweep_dir(tmp_path):
    chunks = L.plan_sweep(plan_states(), settle=0, hold=2, frames=6)
    plan = L.write_plan(tmp_path / "sweep-p.json", chunks, program="p", source=str(tmp_path / "c.mkv"))
    (tmp_path / "c.mkv").write_bytes(b"clip")
    return plan, chunks


def test_sweep_collect_and_commit_build_the_archive(tmp_path, monkeypatch, capsys):
    plan, chunks = sweep_dir(tmp_path)

    def collect(take, chunk, **_kw):
        rows = np.zeros(int((chunk.plan["target"] > 0).sum()), dtype=LANDED)
        body = chunk.plan[chunk.plan["target"] > 0]
        for name in ("target", "param", "value"):
            rows[name] = body[name]
        rows["wire"], rows["sent"], rows["landed"] = body["value"], 0, body["target"]
        rows["first"], rows["last"] = body["target"], body["target"]
        n = chunk.frames
        L.save_rows(
            take,
            L.settled_rows(
                chunk,
                rows,
                picked=np.arange(n),
                shown=np.arange(n),
                captured=np.zeros(n),
                lead=0,
                program="p",
            ),
        )
        take.with_name(take.name + ".meta.json").write_text(
            json.dumps(
                {
                    "attempt-1": {"fault": "x"},
                    "attempt-2": {"before": {"firmware": "rc4"}, "program_info": {"n": 1}},
                }
            )
        )
        take.write_bytes(b"take")
        return {"evidence_ok": True}

    monkeypatch.setattr(L, "collect", collect)
    for k in range(len(chunks)):
        assert cli.main(["probe", "sweep", "collect", str(plan), "--chunk", str(k)]) == 0
    monkeypatch.setattr(decoded, "features", lambda *a, **k: fake_features(100))
    monkeypatch.setattr(timecode, "clip_layout", lambda clip: timecode.ClipLayout(30, 40, 30))

    monkeypatch.setattr(A, "container_duration", lambda *a, **k: "0.2")
    assert cli.main(["probe", "sweep", "commit", str(plan), "--archive", str(tmp_path / "a")]) == 0
    meta = FrameArchive(tmp_path / "a").meta("p")
    assert meta["key_material"] == "rc4" and meta["stimulus"]["frames"] == 100
    assert meta["stimulus"]["lead"] == 30, "the archive names the clip's own lead-in"
    assert meta["video"] == "p.ffconcat" and len(meta["parts"]) == len(chunks)
    assert (tmp_path / "a" / "c.mkv").is_symlink() and '"measured"' in capsys.readouterr().out


def test_collect_judges_and_keeps_rows(tmp_path, monkeypatch):
    from syncsummoner.compose import judge  # pylint: disable=import-outside-toplevel
    from syncsummoner.device import v4l2rec, wire  # pylint: disable=import-outside-toplevel

    chunk = L.plan_sweep(plan_states(), settle=0, hold=2, frames=20)[0]
    body = chunk.plan[chunk.plan["target"] > 0]
    rows = np.zeros(body.size, dtype=LANDED)
    for name in ("target", "param", "value"):
        rows[name] = body[name]
    rows["wire"], rows["sent"], rows["landed"] = body["value"], 0, body["target"]
    rows["first"], rows["last"] = body["target"], body["target"]
    n = chunk.frames
    proof = SimpleNamespace(
        ok=True,
        landed=SimpleNamespace(
            rows=rows, summary=lambda: "landed", missing=0, unechoed=0, duplicated=0, out_of_order=0, late=2
        ),
        applied=SimpleNamespace(disagree=np.zeros(0, dtype=DISAGREE), summary=lambda: "applied"),
        received=None,
    )
    seen = SimpleNamespace(
        proof=proof, found=SimpleNamespace(picked=np.arange(n), shown=np.arange(n), lead=3, source_offset=3)
    )
    monkeypatch.setattr(judge, "judge_take", lambda *a, **k: seen)
    monkeypatch.setattr(v4l2rec, "read_log", lambda take: {"t_ns": np.arange(n, dtype=np.int64)})
    monkeypatch.setattr(wire, "load_wire", lambda take: {"program": np.array("p")})
    said = L.collect(tmp_path / "t.mkv", chunk, clip="c.mkv", config=None)
    between = 2
    assert (
        said["setpoints_kept"] == len(chunk.states) and said["rows_measured"] == n - between
    ), "a setpoint moving two slots is written a frame apart; the frame between is no setpoint's"
    assert said["evidence_ok"] and said["verdict"]["late"] == 2
    assert said["settle"] == {"frames": 0, "source": "plan"}
    kept = json.loads((tmp_path / "t.mkv.evidence.json").read_text())
    assert kept["settle"] == said["settle"]
    assert L.rows_path(tmp_path / "t.mkv").is_file()


def test_a_measured_hold_overrides_a_plans_settle_and_says_so(tmp_path):
    chunk = L.plan_sweep(plan_states(), settle=1, hold=2, frames=20)[0]
    hold = tmp_path / "hold-p.json"
    hold.write_text(json.dumps({"program": "p", "settle": 27, "hold": 29}))
    moved, said = sweep_cmds.settled_by(chunk, "p", str(hold), "plan.json")
    assert moved.settle == 27 and said == {"frames": 27, "source": str(hold), "measured": True}
    kept, said = sweep_cmds.settled_by(chunk, "p", None, "plan.json")
    assert kept is chunk and said == {"frames": 1, "source": "plan.json", "measured": True}
    _, said = sweep_cmds.settled_by(chunk, "p", None, "plan.json", measured=False)
    assert said["measured"] is False
    _, said = sweep_cmds.settled_by(chunk, "p", str(hold), "plan.json", measured=False)
    assert said["measured"] is True, "a hold record is a measured settle whatever the plan had"
    with pytest.raises(ValueError, match="not q"):
        sweep_cmds.settled_by(chunk, "q", str(hold), "plan.json")


def test_judge_offline_writes_nothing(tmp_path, monkeypatch):
    from syncsummoner.compose import lock  # pylint: disable=import-outside-toplevel
    from syncsummoner.device import wire  # pylint: disable=import-outside-toplevel

    monkeypatch.setattr(decoded, "features", lambda *a, **k: fake_features(4))
    monkeypatch.setattr(wire, "load_wire", lambda take: {"program": np.array("p"), "start": np.zeros(12)})
    monkeypatch.setattr(lock, "attribute_take", lambda *a, **k: "found")
    monkeypatch.setattr(LT, "take_evidence", lambda *a, **k: "proof")
    seen = response_mod.judge_offline(
        tmp_path / "t.mkv",
        clip="c.mkv",
        config=SimpleNamespace(strip_px=8, bits=16, width=4, height=2, fps=30.0),
    )
    assert seen.program == "p" and seen.proof == "proof" and not list(tmp_path.iterdir())


def test_sweep_plan_takes_a_recorded_cover_and_joint_moves(tmp_path, monkeypatch):
    hold = tmp_path / "hold.json"
    hold.write_text(json.dumps({"program": "p", "settle": 0, "hold": 2}))
    cover_file = tmp_path / "cover.json"
    cover_file.write_text(json.dumps({"states": [[0] * 12], "chosen": [0], "stepped": {"0": [1]}}))
    monkeypatch.setattr(timecode, "clip_layout", lambda clip: timecode.ClipLayout(1200, 40, 1200))
    fake = SimpleNamespace(resolution=lambda: {1: (0, 1), 11: (0,), 6: (0, 1)})
    monkeypatch.setattr("syncsummoner.program.model", lambda name: fake)
    monkeypatch.setattr(gates, "representatives", lambda m, p, bits: np.array([0, 512]))
    argv = ["probe", "sweep", "plan", "p", "--hold", str(hold), "--source", "c.mkv", "--takes", str(tmp_path)]
    assert cli.main(argv + ["--cover", str(cover_file), "--joint", "11:6"]) == 0
    chunks, _ = L.read_plan(tmp_path / "sweep-p.json")
    states = np.concatenate([c.states for c in chunks])
    assert set(states[:, 0].tolist()) == {0, 512} and set(states[:, 10].tolist()) == {0, 1023}


def test_timecode_sizes_the_clip_by_the_stall_guard(monkeypatch, capsys):
    made = {}

    def write(source, out, *, config, lead, tail, **_kw):
        made.update(source=source, out=out, fps=config.fps, lead=lead, tail=tail)
        return 90

    monkeypatch.setattr(timecode, "write_timecoded", write)
    argv = ["--stall", "40", "probe", "timecode", "f.mkv", "--out", "c.mkv"]
    assert cli.main(argv) == 0
    assert made == {
        "source": "f.mkv",
        "out": "c.mkv",
        "fps": 30.0,
        "lead": SS.lead_in_frames(LT.LeadIn(), 30.0, dry=False, load_s=40.0),
        "tail": SS.lead_out_frames(LT.LeadIn(), 30.0, dry=False, load_s=40.0),
    }
    assert "for a 40.0 s load" in capsys.readouterr().out


def test_sweep_capture_passes_the_stall_guard_and_keeps_each_attempt(tmp_path, monkeypatch):
    plan, chunks = sweep_dir(tmp_path)
    seen = {}

    def capture(config, picked, outs, *, program, clip):
        seen.update(stall=config.stall_s, frames=[c.frames for c in picked], program=program, clip=clip)
        return [{"fault": "LeadInTimeError: late"} for _ in outs]

    monkeypatch.setattr(L, "capture", capture)
    argv = ["--stall", "40", "probe", "sweep", "capture", str(plan), "--chunk", "0", "--attempt", "2"]
    assert cli.main(argv) == 2
    assert seen == {
        "stall": 40.0,
        "frames": [chunks[0].frames],
        "program": "p",
        "clip": str(tmp_path / "c.mkv"),
    }
    meta = json.loads((tmp_path / "take-p-00.mkv.meta.json").read_text())
    assert meta == {"attempt-2": {"fault": "LeadInTimeError: late"}}


def test_commit_refuses_rows_that_waited_no_measured_settle(tmp_path, monkeypatch):
    chunks = L.plan_sweep(plan_states(), settle=0, hold=2, frames=6)
    unmeasured = {"program": "p", "settle": 0, "measured": False}
    plan = L.write_plan(tmp_path / "sweep-p.json", chunks, program="p", source="c.mkv", response=unmeasured)
    seen = []

    def collect(take, _chunk, *, settle, **_kw):
        seen.append(settle)
        LT.evidence_path(take).write_text(json.dumps({"settle": settle}), encoding="utf-8")
        return {"evidence_ok": True}

    monkeypatch.setattr(L, "collect", collect)
    for k in range(len(chunks)):
        assert cli.main(["probe", "sweep", "collect", str(plan), "--chunk", str(k)]) == 0
    assert [s["measured"] for s in seen] == [False] * len(chunks)
    take = tmp_path / "take-p-00.mkv"
    assert not sweep_cmds.settle_measured(take) and sweep_cmds.settle_measured(tmp_path / "none.mkv")
    assert cli.main(["probe", "sweep", "commit", str(plan), "--archive", str(tmp_path / "a")]) == 1
    assert FrameArchive(tmp_path / "a").meta("p") is None
    hold = tmp_path / "hold-p.json"
    hold.write_text(json.dumps({"program": "p", "settle": 2, "hold": 4}))
    assert cli.main(["probe", "sweep", "collect", str(plan), "--chunk", "0", "--hold", str(hold)]) == 0
    assert seen[-1] == {"frames": 2, "source": str(hold), "measured": True} and sweep_cmds.settle_measured(
        take
    )


def test_smoke_toggles_one_parameter_captures_it_and_lands_it_from_the_wire(tmp_path, monkeypatch, capsys):
    plan, _ = sweep_dir(tmp_path)
    captured = {}

    def capture(args):
        captured["chunks"], captured["said"] = L.read_plan(args.plan)
        captured["chunk"] = args.chunk
        return 0

    rows = np.zeros(2, dtype=LANDED)
    rows["target"], rows["param"], rows["value"], rows["sent"] = [4, 8], 3, [100, 900], [0, 1]
    rows["first"], rows["last"] = [4, 8], [4, 8]
    landed = LT.ev.Landed(rows=rows, missing=0, duplicated=0, out_of_order=0, late=0, early=0)
    monkeypatch.setattr(sweep_cmds, "_sweep_cmd", capture)
    monkeypatch.setattr(
        LT, "landing", lambda wire, period: (LT.ev.Offset(50, 50), np.array([[54, 54], [58, 58]]), landed)
    )
    monkeypatch.setattr("syncsummoner.device.wire.load_wire", lambda take: {})
    out = tmp_path / "smoke"
    argv = ["probe", "sweep", "smoke", str(plan), "--param", "3", "--values", "100", "900", "--states", "4"]
    assert cli.main(argv + ["--hold", "4", "--out", str(out)]) == 0
    (chunk,) = captured["chunks"]
    assert chunk.states[:, 2].tolist() == [100, 900, 100, 900] and captured["chunk"] == 0
    assert captured["said"]["program"] == "p" and chunk.frames == 16
    said = capsys.readouterr().out
    assert "by 50..50" in said and "device [58, 58], clip 8..8" in said


def test_timecode_makes_a_synthetic_stimulus_clip_and_keeps_its_spec(tmp_path, monkeypatch):
    made = {}

    def write(source, _out, *, lead, tail, seconds=None, **_kw):
        made.update(source=source, seconds=seconds, lead=lead, tail=tail)
        return 90

    monkeypatch.setattr(timecode, "write_timecoded", write)
    out = tmp_path / "c.mkv"
    assert (
        cli.main(
            ["--stall", "40", "probe", "timecode", "--synthetic", "7", "--seconds", "3", "--out", str(out)]
        )
        == 0
    )
    assert made["source"].seed == 7 and made["seconds"] == 3.0
    assert json.loads((tmp_path / "c.mkv.stimulus.json").read_text())["stimulus"]["seed"] == 7
    argv = ["--stall", "40", "probe", "timecode", "--synthetic", "7", "--seconds", "3", "--out", str(out)]
    assert cli.main([*argv, "--shot-frames", "45", "--one-texture"]) == 0
    assert made["source"].shot_frames == 45 and made["source"].coverage == Coverage().one_texture()
    assert cli.main([*argv, "--colour", "bw"]) == 0
    assert made["source"].coverage == Coverage().monochrome()
    assert cli.main([*argv, "--colour", "full"]) == 0
    assert made["source"].coverage == Coverage().full_colour()
    with pytest.raises(SystemExit, match="not both"):
        cli.main(["probe", "timecode", "f.mkv", "--synthetic", "7", "--out", str(out)])


def test_a_setpoint_moving_several_slots_writes_one_a_frame_and_holds_after_the_last():
    states = np.zeros((3, 12), dtype=np.int64)
    states[1, [0, 3, 5]] = [10, 20, 30]
    states[2, 0] = 40
    chunks = L.plan_sweep(states, settle=0, hold=5, frames=40)
    assert len(chunks) == 1
    chunk = chunks[0]
    body = chunk.plan[chunk.plan["target"] > 0]
    assert body[["target", "param"]].tolist() == [(5, 1), (6, 4), (7, 6), (12, 1), (13, 4), (14, 6)]
    assert chunk.frames == 19 and chunk.owner[12:].tolist() == [1, 1, 1, 2, 2, 2]


def test_restagger_spreads_each_setpoint_and_keeps_the_original(tmp_path):
    states = np.zeros((3, 12), dtype=np.int64)
    states[1, [0, 3]] = [10, 20]
    states[2, [0, 3]] = [30, 40]
    rows = [(0, p + 1, 0) for p in range(12)] + [(5, 1, 10), (5, 4, 20), (10, 1, 30), (10, 4, 40)]
    chunk = L.Chunk(0, states, np.array(rows, dtype=L.PLAN), np.r_[[0] * 12, 1, 1, 2, 2], 20, 0, 5)
    plan = L.write_plan(tmp_path / "sweep-p.json", [chunk], program="p", source="c.mkv")
    assert cli.main(["probe", "sweep", "restagger", str(plan)]) == 0
    (got,), said = L.read_plan(plan)
    assert got.plan[got.plan["target"] > 0][["target", "param"]].tolist() == [
        (5, 1),
        (6, 4),
        (10, 1),
        (11, 4),
    ]
    assert said["program"] == "p" and (tmp_path / "sweep-p.json.unstaggered").is_file()
    assert got.plan[got.plan["target"] == 0].size == 12, "the start state stays at frame 0"

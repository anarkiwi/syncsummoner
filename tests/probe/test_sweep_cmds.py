"""The probe hold and probe sweep commands, with the takes, the model and the decodes faked."""

# pylint: disable=missing-function-docstring

import json
from types import SimpleNamespace

import numpy as np

from syncsummoner import cli
from syncsummoner.compose import response as response_mod
from syncsummoner.compose.evidence import DISAGREE, LANDED
from syncsummoner.probe import decoded
from syncsummoner.probe import locked as L
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
        found=SimpleNamespace(picked=np.arange(50), lead=0),
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
    monkeypatch.setattr(decoded, "features", lambda *a, **k: fake_features(1200 + 12 + 1200))
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
        rows["wire"], rows["sent"], rows["landed"], rows["clear_ns"] = body["value"], 0, body["target"], 1.0
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
        return {"evidence_ok": True}

    monkeypatch.setattr(L, "collect", collect)
    for k in range(len(chunks)):
        assert cli.main(["probe", "sweep", "collect", str(plan), "--chunk", str(k)]) == 0
    monkeypatch.setattr(decoded, "features", lambda *a, **k: fake_features(100))

    def concat(_videos, out, **_kw):
        out.write_bytes(b"joined")

    monkeypatch.setattr(L, "concat", concat)
    assert cli.main(["probe", "sweep", "commit", str(plan), "--archive", str(tmp_path / "a")]) == 0
    meta = FrameArchive(tmp_path / "a").meta("p")
    assert meta["key_material"] == "rc4" and meta["stimulus"]["frames"] == 100
    assert (tmp_path / "a" / "c.mkv").is_symlink() and '"measured"' in capsys.readouterr().out


def test_collect_judges_and_keeps_rows(tmp_path, monkeypatch):
    from syncsummoner.compose import judge  # pylint: disable=import-outside-toplevel
    from syncsummoner.device import v4l2rec, wire  # pylint: disable=import-outside-toplevel

    chunk = L.plan_sweep(plan_states(), settle=0, hold=2, frames=20)[0]
    body = chunk.plan[chunk.plan["target"] > 0]
    rows = np.zeros(body.size, dtype=LANDED)
    for name in ("target", "param", "value"):
        rows[name] = body[name]
    rows["wire"], rows["sent"], rows["landed"], rows["clear_ns"] = body["value"], 0, body["target"], 1.0
    n = chunk.frames
    proof = SimpleNamespace(
        ok=True,
        landed=SimpleNamespace(rows=rows, summary=lambda: "landed"),
        applied=SimpleNamespace(disagree=np.zeros(0, dtype=DISAGREE), summary=lambda: "applied"),
        received=None,
    )
    seen = SimpleNamespace(
        proof=proof, found=SimpleNamespace(picked=np.arange(n), shown=np.arange(n), lead=3)
    )
    monkeypatch.setattr(judge, "judge_take", lambda *a, **k: seen)
    monkeypatch.setattr(v4l2rec, "read_log", lambda take: {"t_ns": np.arange(n, dtype=np.int64)})
    monkeypatch.setattr(wire, "load_wire", lambda take: {"program": np.array("p")})
    said = L.collect(tmp_path / "t.mkv", chunk, clip="c.mkv", config=None)
    assert said["setpoints_kept"] == len(chunk.states) and said["rows_measured"] == n
    assert L.rows_path(tmp_path / "t.mkv").is_file()


def test_judge_offline_writes_nothing(tmp_path, monkeypatch):
    from syncsummoner.compose import lock, locktake  # pylint: disable=import-outside-toplevel
    from syncsummoner.device import wire  # pylint: disable=import-outside-toplevel

    monkeypatch.setattr(decoded, "features", lambda *a, **k: fake_features(4))
    monkeypatch.setattr(wire, "load_wire", lambda take: {"program": np.array("p"), "start": np.zeros(12)})
    monkeypatch.setattr(lock, "attribute_take", lambda *a, **k: "found")
    monkeypatch.setattr(locktake, "take_evidence", lambda *a, **k: "proof")
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
    monkeypatch.setattr(decoded, "features", lambda *a, **k: fake_features(2 * 1200 + 40))
    fake = SimpleNamespace(resolution=lambda: {1: (0, 1), 11: (0,), 6: (0, 1)})
    monkeypatch.setattr("syncsummoner.program.model", lambda name: fake)
    monkeypatch.setattr(gates, "representatives", lambda m, p, bits: np.array([0, 512]))
    argv = ["probe", "sweep", "plan", "p", "--hold", str(hold), "--source", "c.mkv", "--takes", str(tmp_path)]
    assert cli.main(argv + ["--cover", str(cover_file), "--joint", "11:6"]) == 0
    chunks, _ = L.read_plan(tmp_path / "sweep-p.json")
    states = np.concatenate([c.states for c in chunks])
    assert set(states[:, 0].tolist()) == {0, 512} and set(states[:, 10].tolist()) == {0, 1023}

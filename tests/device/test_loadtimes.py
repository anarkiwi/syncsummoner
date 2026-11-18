"""The rig's one load record: shared by every takes directory, seeded from takes on disk, required."""

# pylint: disable=missing-function-docstring

import json
import threading

import numpy as np
import pytest

from syncsummoner.device import loadtimes as LD


def test_the_record_sits_beside_the_program_cache_unless_named(monkeypatch, tmp_path):
    monkeypatch.delenv("SYNCSUMMONER_LOADS")
    monkeypatch.setenv("SYNCSUMMONER_PROGRAM_CACHE", str(tmp_path / "cache"))
    assert LD.record_path() == tmp_path / "cache" / LD.RECORD
    assert LD.record_path(tmp_path / "x.jsonl") == tmp_path / "x.jsonl"


def test_loads_are_kept_by_program_and_firmware_and_the_longest_bounds_them():
    assert LD.read() == [] and LD.longest() is None
    assert LD.record("Passthru", 0, 2_500_000_000, firmware="1.0.2-rc.4") == 2.5
    LD.record("czartroniks/Quilt", 10, 4_000_000_010, firmware="1.0.2-rc.4")
    LD.record("Passthru", 20, 1_000_000_020, firmware="1.0.1")
    assert LD.longest() == 4.0
    assert [(e["program"], e["firmware"]) for e in LD.read()] == [
        ("Passthru", "1.0.2-rc.4"),
        ("czartroniks/Quilt", "1.0.2-rc.4"),
        ("Passthru", "1.0.1"),
    ]


def test_writers_at_once_lose_no_load_and_keep_every_line_whole():
    threads = [threading.Thread(target=LD.record, args=(f"P{k}", k, k + 10**9)) for k in range(16)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()
    assert sorted(e["t0_ns"] for e in LD.read()) == list(range(16))


def test_no_load_on_record_refuses_an_unguarded_wait_unless_a_stall_is_given():
    with pytest.raises(LD.NoLoadRecordError, match="--stall"):
        LD.stall_guard(None)
    assert LD.stall_guard(7.0) == 7.0
    LD.record("Passthru", 0, 3_000_000_000)
    assert LD.stall_guard(None) == 3.0 and LD.stall_guard(1.0) == 1.0


def wire(path, loads):
    np.savez(path, loads=np.array(json.dumps(loads)), sends=np.zeros(0))


def test_seeding_takes_every_load_on_disk_once(tmp_path):
    takes = tmp_path / "captures" / "takes8"
    takes.mkdir(parents=True)
    wire(
        takes / "take-q.mkv.wire.npz",
        [
            {"program": "Quilt", "t0_ns": 5, "t1_ns": 13_000_000_005},
            {"program": "Passthru", "t0_ns": 9, "t1_ns": 9},
        ],
    )
    (takes / "take-q.mkv.meta.json").write_text(json.dumps({"attempt-1": {"before": {"firmware": "1.0.2"}}}))
    wire(takes / "take-d.mkv.wire.npz", [])
    np.savez(takes / "take-d.mkv.lock.npz", flips=np.zeros(3))
    (takes / "broken.mkv.wire.npz").write_bytes(b"not an archive")
    (tmp_path / "captures" / "sweep").mkdir()
    (tmp_path / "captures" / "sweep" / LD.DIRECTORY_FILE).write_text(
        json.dumps({"loads": [{"program": "Teletext", "load_s": 21.5}]})
    )
    assert LD.seed([tmp_path / "captures"]) == 3
    assert LD.longest() == 21.5
    got = {(e["program"], e["firmware"], e["load_s"]) for e in LD.read()}
    assert got == {("Quilt", "1.0.2", 13.0), ("Passthru", "1.0.2", 0.0), ("Teletext", None, 21.5)}
    assert LD.seed([tmp_path / "captures"]) == 0, "seeding again adds nothing already on record"


def test_the_seeder_reports_what_it_kept(tmp_path, capsys):
    (tmp_path / LD.DIRECTORY_FILE).write_text(json.dumps({"loads": [{"program": "P", "load_s": 2.0}]}))
    assert LD.main([str(tmp_path)]) == 0
    said = capsys.readouterr().out
    assert "1 loads added; 1 on record" in said and said.rstrip().endswith("longest 2.0 s")

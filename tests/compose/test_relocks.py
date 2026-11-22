"""The card's relock after a dry control turns wet: measured from a session's capture log, kept, waited."""

import json

import numpy as np
import pytest

from syncsummoner.compose import relocks as R
from syncsummoner.compose.evidence import START
from syncsummoner.compose.locktake import LeadIn
from syncsummoner.device import loadtimes
from syncsummoner.device.v4l2rec import FRAME
from syncsummoner.device.wire import SEND

PERIOD = 16_666_667
WET_NS = 10 * PERIOD


def capture_log(gaps=(), restart=4, start_ns=0):
    """Captures every ``PERIOD``, each device frame twice; after each gap ``(at, length)`` the card's
    ``restart`` pictures, each unlike the one before, then frames twice again; the sequence never breaks."""
    t = start_ns + np.arange(200, dtype=np.int64) * PERIOD
    digest = (np.arange(200) // 2).astype(np.uint64)
    for at, length in gaps:
        t[at:] += length
        digest[at : at + restart] = 10_000 + at + np.arange(restart, dtype=np.uint64)
    log = np.zeros(t.size, dtype=FRAME)
    log["t_ns"], log["sequence"], log["digest"] = t, np.arange(t.size), digest
    return log


def sends(*rows):
    """Sends ``(t0_ns, target)``, in order."""
    out = np.zeros(len(rows), dtype=SEND)
    out["t0_ns"], out["target"] = np.array(rows, dtype=np.int64).T
    return out


WRITES = sends((0, START), (WET_NS, -60), (WET_NS, -60), (120 * PERIOD, 45))


def test_the_relock_runs_from_the_wet_write_to_the_card_holding_a_frame_after_its_last_break():
    """Two breaks after the wet write: the relock ends where a capture first repeats after the second."""
    log = capture_log(gaps=[(30, 20 * PERIOD), (70, 45 * PERIOD)])
    began, relock = R.measure(log, WRITES)
    held = 70 + 4 + 1
    assert began == WET_NS and relock == pytest.approx((log["t_ns"][held] - WET_NS) / 1e9)
    assert (
        log["digest"][held] == log["digest"][held - 1] and log["digest"][held - 1] != log["digest"][held - 2]
    )


def test_breaks_before_the_wet_write_or_after_the_next_frames_writes_are_not_its_relock():
    """A load's break before the wet write and a break a later setting causes leave the relock unbroken."""
    log = capture_log(gaps=[(5, 30 * PERIOD), (150, 40 * PERIOD)])
    assert R.measure(log, WRITES) == (WET_NS, 0.0)
    assert R.measure(capture_log(), sends((0, START))) is None


def test_a_run_never_holding_a_frame_relocks_at_its_last_capture():
    """Restart pictures to the end of the log: the relock reaches its last capture."""
    log = capture_log(gaps=[(190, 20 * PERIOD)], restart=10)
    assert R.measure(log, sends((WET_NS, -60)))[1] == pytest.approx((log["t_ns"][-1] - WET_NS) / 1e9)


def test_a_programs_wet_stretch_is_its_longest_relock_on_record_and_never_less_than_the_lead_ins():
    """Only the program's own relocks lengthen its wet stretch, and none shorten it."""
    base = LeadIn()
    assert R.lead_in("czartroniks/Inferno") == base and R.longest("czartroniks/Inferno") is None
    R.record("czartroniks/Inferno", 1, base.wet_s / 2)
    assert R.lead_in("czartroniks/Inferno") == base
    R.record("czartroniks/Inferno", 2, base.wet_s + 0.5)
    R.record("czartroniks/Inferno", 3, base.wet_s + 0.25)
    R.record("czartroniks/Pyre", 4, base.wet_s + 1.0)
    assert R.lead_in("czartroniks/Inferno").wet_s == base.wet_s + 0.5
    assert R.lead_in("czartroniks/Inferno", LeadIn(wet_s=9.0)).wet_s == 9.0
    assert R.lead_in("czartroniks/Fray") == base
    assert R.record_path().parent == loadtimes.record_path().parent


def test_the_record_follows_its_override(monkeypatch, tmp_path):
    """``SYNCSUMMONER_RELOCKS`` names the record."""
    monkeypatch.setenv("SYNCSUMMONER_RELOCKS", str(tmp_path / "r.jsonl"))
    R.record("P", 1, 3.0)
    assert R.record_path() == tmp_path / "r.jsonl" and R.longest("P") == 3.0


def write_take(path, *, program, dry, log, writes, session=True):
    """A take's lock and wire records as a session leaves them."""
    lock = {"session_log": log} if session else {"flips": np.zeros(0)}
    np.savez(f"{path}.lock.npz", **lock)
    np.savez(f"{path}.wire.npz", program=np.array(program), dry=np.array(json.dumps(dry)), sends=writes)


def test_seeding_measures_each_dry_session_once(tmp_path, capsys):
    """Sessions on disk seed the record: one entry a session, none from a take with no dry control or log."""
    log = capture_log(gaps=[(30, 20 * PERIOD), (70, 45 * PERIOD)])
    for k in range(2):
        write_take(tmp_path / f"take-a-0{k}.mkv", program="A", dry={"11": 1023}, log=log, writes=WRITES)
    write_take(tmp_path / "take-b-00.mkv", program="B", dry=None, log=log, writes=WRITES)
    write_take(tmp_path / "take-c-00.mkv", program="C", dry={"1": 0}, log=log, writes=WRITES, session=False)
    (tmp_path / "take-d-00.mkv.lock.npz").write_bytes(b"not a record")
    assert R.main([str(tmp_path)]) == 0
    held = loadtimes.read(R.record_path())
    assert [e["program"] for e in held] == ["A"] and held[0]["relock_s"] == R.measure(log, WRITES)[1]
    assert "1 relocks added" in capsys.readouterr().out
    assert R.main([str(tmp_path)]) == 0 and "0 relocks added" in capsys.readouterr().out

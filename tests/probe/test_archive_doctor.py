"""The offline archive doctor, over a tiny synthetic archive."""

# pylint: disable=missing-function-docstring

import json

import numpy as np
import pytest

from syncsummoner import cli
from syncsummoner.device import v4l2rec as vr
from syncsummoner.device.playout import FLIP
from syncsummoner.probe import archive_doctor as ad

PERIOD = 16_683_350
FRAMES = 600


def write_plan(path, clip, *, rig_s=1000.0, shot_frames=45):
    regions = {"rig_s": rig_s, "guard_frames": 13, "hold_frames": {"min": 1, "median": 2, "max": 3}}
    if shot_frames is not None:
        regions["shot_frames"] = shot_frames
    record = {"program": path.stem, "source": str(clip), "response": {}, "regions": regions, "chunks": []}
    path.write_text(json.dumps(record), encoding="utf-8")
    return path


def write_take(take, *, period=PERIOD, flip=2 * PERIOD, segments=1, size=1000, attempt=None, rows=True):
    """A take of ``FRAMES`` captures ``period`` ns apart, its Pi flips ``flip`` ns apart."""
    take.write_bytes(b"\0" * size)
    stored = np.zeros(FRAMES, dtype=vr.STORED)
    stored["sequence"] = np.arange(FRAMES) + 7
    stored["t_ns"] = 10**12 + np.arange(FRAMES) * period
    vr.sidecar(take).write_bytes(vr.Layout().sidecar_header() + stored.tobytes())
    flips = np.zeros(FRAMES // 2, dtype=FLIP)
    flips["v"] = flips["seq"] = flips["frame"] = np.arange(flips.size)
    flips["t_ns"] = 10**12 + np.arange(flips.size) * flip
    flips["flags"][np.linspace(0, flips.size - 1, segments, dtype=int)] = 1
    np.savez(f"{take}.lock.npz", flips=flips, hold=np.int64(1))
    meta = {"attempt-1": attempt or {"program": "P"}}
    (take.parent / f"{take.name}.meta.json").write_text(json.dumps(meta), encoding="utf-8")
    if rows:
        (take.parent / f"{take.name}.rows.parquet").write_bytes(b"")
    return take


@pytest.fixture(name="archive")
def fixture_archive(tmp_path):
    for sub in ("takes", "holds", "logs", "clip"):
        (tmp_path / sub).mkdir()
    clip = tmp_path / "clip" / "synthetic.mkv"
    clip.write_bytes(b"clip")
    (tmp_path / "clip" / "synthetic.mkv.stimulus.json").write_text(
        json.dumps({"stimulus": {"seed": 1, "shot_frames": 45}}), encoding="utf-8"
    )
    plans = [write_plan(tmp_path / "takes" / f"sweep-{n}.json", clip) for n in ("A", "B")]
    (tmp_path / "plans.txt").write_text("".join(f"{p}\n" for p in plans), encoding="utf-8")
    write_take(tmp_path / "takes" / "take-A-00.mkv")
    return tmp_path


def by_check(findings):
    return {f.check: f for f in findings}


def test_a_sound_archive_passes_every_check(archive):
    found = by_check(ad.diagnose(archive))
    assert list(found) == [
        "plans.readable",
        "plans.one_clip",
        "plans.shot_aligned",
        "holds.clip",
        "disk.budget",
        "rig.time",
        "takes.faults",
        "takes.rate_lock",
        "takes.collected",
        "takes.chroma",
    ]
    assert all(f.ok and not f.fix for f in found.values()), found
    assert found["takes.chroma"].detail == "0 of 1 takes judged, 0 exchanged (0 dry, read put right)"
    assert found["rig.time"].detail == "0.6 rig h left over 2 of 2 plans"
    assert found["takes.rate_lock"].detail == (
        "1 takes measured, 0 multi-segment skipped; ppm +0.0 to +0.0, the tightest bound 1669.4"
    )


def test_unreadable_and_missing_plans_are_named(archive, tmp_path):
    good = archive / "takes" / "sweep-A.json"
    (archive / "takes" / "sweep-B.json").write_text("{", encoding="utf-8")
    plans, errors = ad.read_plans([good, archive / "takes" / "sweep-B.json", tmp_path / "nope.json"])
    got = ad.plans_readable(plans, errors)
    assert not got.ok and got.detail.startswith("1 of 3 plans read")
    assert "sweep-B.json" in got.fix and "nope.json" in got.fix
    assert not ad.plans_readable({}, {}).ok
    listing = by_check(ad.diagnose(archive, tmp_path / "absent.txt"))["plans.readable"]
    assert not listing.ok and "no plan listing" in listing.detail


def test_a_plan_on_another_or_a_missing_clip_fails(archive):
    plans, _ = ad.read_plans((archive / "plans.txt").read_text(encoding="utf-8").split())
    assert ad.one_clip(plans).ok
    other = archive / "takes" / "sweep-B.json"
    write_plan(other, archive / "clip" / "timecoded.mkv")
    plans, _ = ad.read_plans([archive / "takes" / "sweep-A.json", other, archive / "takes" / "sweep-A.json"])
    got = ad.one_clip({**plans, "x": plans[str(archive / "takes" / "sweep-A.json")]})
    assert not got.ok and got.fix.endswith(other.name) and got.detail.startswith("2 of 3 plans play")
    gone = ad.one_clip({str(other): plans[str(other)]})
    assert not gone.ok and gone.detail.endswith(", missing") and "restore" in gone.fix


def test_plans_off_the_stimulus_shots_are_named_and_footage_passes(archive):
    clip = archive / "clip" / "synthetic.mkv"
    off = write_plan(archive / "takes" / "sweep-B.json", clip, shot_frames=30)
    bare = write_plan(archive / "takes" / "sweep-C.json", clip, shot_frames=None)
    plans, _ = ad.read_plans([archive / "takes" / "sweep-A.json", off, bare])
    got = ad.shot_aligned(plans)
    assert not got.ok and "probe sweep regions" in got.fix
    assert off.name in got.fix and bare.name in got.fix and "sweep-A.json" not in got.fix
    footage = archive / "clip" / "footage.mkv"
    plans, _ = ad.read_plans([write_plan(archive / "takes" / "sweep-D.json", footage, shot_frames=None)])
    assert ad.shot_aligned(plans) == ad.Finding("plans.shot_aligned", True, "footage clip", "")


def test_hold_records_are_checked_against_the_clip_or_reported_unverifiable(archive):
    clip = str(archive / "clip" / "synthetic.mkv")
    assert ad.holds_clip([], clip).ok
    right, wrong, bare = (archive / "holds" / f"hold-{n}.json" for n in "ABC")
    right.write_text(json.dumps({"program": "A", "source": clip}), encoding="utf-8")
    wrong.write_text(json.dumps({"program": "B", "clip": str(archive / "clip" / "timecoded.mkv")}), "utf-8")
    bare.write_text(json.dumps({"program": "C", "settle": 3, "takes": []}), encoding="utf-8")
    assert ad.holds_clip([right], clip).ok
    got = ad.holds_clip([right, wrong, bare], clip)
    assert not got.ok and "1 name no clip, so cannot be verified" in got.detail
    assert wrong.name in got.fix and bare.name in got.fix and right.name not in got.fix


def test_the_disk_budget_projects_the_remaining_plans_at_the_measured_rate(archive):
    take = archive / "takes" / "take-A-00.mkv"
    seconds = (FRAMES - 1) * PERIOD / 1e9
    bare = archive / "takes" / "take-A-01.mkv"
    bare.write_bytes(b"\0" * 5000)
    assert ad.write_rate([take, bare]) == pytest.approx(1000 / seconds)
    assert ad.write_rate([]) is None
    (archive / "logs" / "done-A").touch()
    plans, _ = ad.read_plans((archive / "plans.txt").read_text(encoding="utf-8").split())
    left = ad.remaining(archive, plans)
    assert list(left) == [str(archive / "takes" / "sweep-B.json")] and ad.rig_seconds(left) == 1000.0
    fits = ad.disk_budget(3600.0, 1e6, free=5e9, floor=1e9)
    assert fits.ok and fits.detail.endswith("floor reached at rig hour 1.1")
    over = ad.disk_budget(3600.0, 1e6, free=4e9, floor=1e9)
    assert not over.ok and "3.6 GB" in over.detail and over.fix.startswith("free 0.6 GB")
    assert ad.disk_budget(3600.0, 1e6, free=1e9, floor=2e9).detail.endswith("rig hour 0.0")
    unmeasured = ad.disk_budget(3600.0, None, free=5e9, floor=0.0)
    assert not unmeasured.ok and unmeasured.fix == "pass --rate-bytes-s"


def test_the_given_rate_is_used_only_when_no_take_measures_one(archive):
    big = 1e30
    assert by_check(ad.diagnose(archive, rate=big))["disk.budget"].ok
    for path in (archive / "takes").glob("take-*"):
        path.unlink()
    assert not by_check(ad.diagnose(archive, rate=big))["disk.budget"].ok
    assert by_check(ad.diagnose(archive, rate=1.0))["disk.budget"].ok


def test_rig_time_is_detail_only():
    got = ad.rig_time(7200.0, 3, 5, True)
    assert got.ok and got.detail == "2.0 rig h left over 3 of 5 plans, paused"


def test_faults_from_the_log_and_take_attempts_are_counted_by_message(archive):
    assert ad.take_faults([]).ok
    log = archive / "logs" / "faults.tsv"
    long = "CCMapError: " + "x" * 200
    log.write_text(f"2026-01-01 00:00:00\tA\t0\t{long}\n\n2026-01-01 00:00:01\tA\t1\t{long}\n", "utf-8")
    write_take(archive / "takes" / "take-B-00.mkv", attempt={"fault": "RebootError: boot"})
    got = by_check(ad.diagnose(archive))["takes.faults"]
    assert not got.ok
    assert got.detail == f"3 faults, 2 distinct: 2x {long[: ad.MESSAGE]}; 1x RebootError: boot"
    assert ad.logged_faults(archive / "logs" / "absent.tsv") == []


def test_a_drifting_take_fails_the_rate_lock_and_a_multi_segment_one_is_skipped(archive):
    takes = archive / "takes"
    bound = 1e6 / (FRAMES - 1)
    write_take(takes / "take-A-01.mkv", period=PERIOD, flip=round(2 * PERIOD * (1 - 2 * bound / 1e6)))
    write_take(takes / "take-A-02.mkv", segments=2, flip=PERIOD)
    write_take(takes / "take-A-03.mkv", period=PERIOD, flip=round(2 * PERIOD * (1 - bound / 2e6)))
    locks = ad.takes_rate_lock(ad.takes_of(archive))
    assert locks[str(takes / "take-A-02.mkv")] is None
    assert locks[str(takes / "take-A-00.mkv")] == pytest.approx((0.0, bound), abs=1e-6)
    got = ad.rate_lock(locks)
    assert (
        not got.ok
        and got.detail.endswith("drifting: take-A-01.mkv")
        and "1 multi-segment skipped" in got.detail
    )
    assert ad.rate_lock({k: v for k, v in locks.items() if "01" not in k}).ok
    assert (
        ad.rate_lock({}).ok
        and ad.rate_lock({"t": None}).detail == "0 takes measured, 1 multi-segment skipped"
    )


def test_a_relock_break_leaves_the_capture_period_alone():
    rows = np.zeros(200, dtype=vr.STORED)
    rows["sequence"] = np.arange(200)
    rows["t_ns"] = np.arange(200) * PERIOD + np.where(np.arange(200) >= 80, 45 * PERIOD, 0)
    assert ad.capture_period(rows) == pytest.approx(PERIOD)
    assert ad.capture_period(rows[:80]) == pytest.approx(PERIOD)


def test_flip_period_counts_only_consecutive_flips():
    flips = np.zeros(5, dtype=FLIP)
    flips["v"] = [0, 1, 3, 4, 5]
    flips["seq"] = [0, 1, 2, 3, 4]
    flips["t_ns"] = [0, 10, 40, 50, 60]
    assert ad.flip_period(flips) == 10.0
    assert np.isnan(ad.flip_period(flips[1:3]))
    assert ad.lock_ppm(np.zeros(1, dtype=vr.STORED), flips) is None


def test_captured_takes_without_rows_are_listed(archive):
    takes = archive / "takes"
    write_take(takes / "take-A-01.mkv", rows=False)
    write_take(takes / "take-A-02.mkv", rows=False, attempt={"fault": "Stall"})
    got = ad.collected(ad.takes_of(archive))
    assert not got.ok and got.detail == "1 of 2 captured takes collected; not: take-A-01.mkv"
    assert "probe sweep collect" in got.fix


def test_the_cli_exits_by_the_checks_and_prints_the_report(archive, capsys):
    assert cli.main(["probe", "doctor", str(archive), "--json"]) == 0
    said = json.loads(capsys.readouterr().out)
    assert said["ok"] and len(said["findings"]) == 10
    write_plan(archive / "takes" / "sweep-B.json", archive / "clip" / "synthetic.mkv", shot_frames=30)
    listing = archive / "only.txt"
    listing.write_text(f"{archive / 'takes' / 'sweep-B.json'}\n", encoding="utf-8")
    argv = ["probe", "doctor", str(archive), "--plans", str(listing), "--floor-bytes", "0"]
    assert cli.main(argv + ["--rate-bytes-s", "1"]) == 1
    said = json.loads(capsys.readouterr().out)
    failed = [f["check"] for f in said["findings"] if not f["ok"]]
    assert failed == ["plans.shot_aligned"]


def test_long_lists_name_the_first_files_and_count_the_rest():
    paths = [f"/a/sweep-{k}.json" for k in range(ad.NAMED + 3)]
    assert ad.named(paths).endswith("sweep-7.json and 3 more") and "/a/" not in ad.named(paths)
    assert ad.named(paths[:2]) == "sweep-0.json, sweep-1.json"


def test_effects_are_reported_per_program_and_a_profile_without_them_fails(archive, capsys):
    from syncsummoner.device.profile import ProgramProfile, Source  # pylint: disable=import-outside-toplevel
    from syncsummoner.probe.fit import save_profile  # pylint: disable=import-outside-toplevel

    shelf = archive / "profiles-each"
    shelf.mkdir()
    summary = {"settings": 4, "safe": 3, "blank": 1, "static": 1, "flashing": 0, "red_flashing": 0}
    measured = ProgramProfile("A", "fw", "a", Source.HW)
    measured.effects = {"bw": {"summary": {**summary, "intensity": [0.1, 0.8], "safe_intensity": [0.1, 0.5]}}}
    save_profile(measured, shelf / "A.yaml")
    found = by_check(ad.diagnose(archive))["effects.measured"]
    assert found.ok and "A bw: 3/4 safe, intensity 0.100-0.500, blank 1 static 1" in found.detail
    save_profile(ProgramProfile("B", "fw", "a", Source.HW), shelf / "B.yaml")
    found = by_check(ad.diagnose(archive, profiles=shelf))["effects.measured"]
    assert not found.ok and "1 of 2" in found.detail and "probe effects" in found.fix
    assert cli.main(["probe", "doctor", str(archive), "--json"]) == 1
    said = json.loads(capsys.readouterr().out)
    assert said["effects"] == {"A": {"bw": measured.effects["bw"]["summary"]}, "B": {}}
    assert ad.effects_measured({"C": {"bw": {**summary, "safe": 0, "safe_intensity": None}}}).detail.endswith(
        "0/4 safe, intensity none, blank 1 static 1 flashing 0 red 0"
    )


def test_only_passthru_coming_back_exchanged_fails_and_a_kept_verdict_is_read_first(archive):
    dry = {"order": "chroma exchanged", "ok": False, "judged": True, "dry": True}
    chain = {"order": "chroma exchanged", "ok": False, "judged": True, "dry": False}
    unsure = {"order": "chroma exchanged", "ok": False, "judged": False, "dry": False}
    write_take(archive / "takes" / "take-B-00.mkv", attempt={"program": "P", "chroma": dry})
    write_take(archive / "takes" / "take-C-00.mkv", attempt={"program": "P", "chroma": unsure})
    found = ad.chroma(ad.takes_chroma(ad.takes_of(archive), None))
    assert found.ok and found.detail == "1 of 3 takes judged, 1 exchanged (1 dry, read put right)"
    kept = archive / "takes" / "take-A-00.mkv.chroma.json"
    kept.write_text(json.dumps(chain), encoding="utf-8")
    found = ad.chroma(ad.takes_chroma(ad.takes_of(archive), None))
    assert not found.ok and "on Passthru: take-A-00.mkv (chroma exchanged)" in found.detail
    assert "device doctor --loop" in found.fix


def test_a_take_with_no_record_is_measured_on_the_clip(archive, monkeypatch):
    from syncsummoner.compose import chroma as cc  # pylint: disable=import-outside-toplevel
    from syncsummoner.probe.chroma import ChromaOrder  # pylint: disable=import-outside-toplevel

    asked = {}

    def measure(take, clip, *, spec, fps, width, lead):
        asked.update(take=take.name, clip=clip, fps=fps, width=width, lead=lead, bits=spec.bits)
        return ChromaOrder(
            "as played", {o: [1.0, 1.0, 1.0] for o in ("as played", "chroma exchanged")}, 30, 9.0
        )

    monkeypatch.setattr(cc, "lead_in_order", measure)
    take = archive / "takes" / "take-A-00.mkv"
    clip = str(archive / "clip" / "synthetic.mkv")
    monkeypatch.setattr("syncsummoner.device.wire.load_wire", lambda path: {"dry": np.array("null")})
    found = ad.take_chroma(take, clip, 120)
    assert found["order"] == "as played" and found["frames"] == 30 and found["dry"] is False
    assert asked == {"take": take.name, "clip": clip, "fps": 30, "width": 1920, "lead": 120, "bits": 16}
    assert ad.take_chroma(take, clip, None) is None

    def unreadable(*_args, **_kwargs):
        raise ValueError("no strip")

    monkeypatch.setattr(cc, "lead_in_order", unreadable)
    assert ad.take_chroma(take, clip, 120) == {"error": "ValueError: no strip"}


def test_the_latest_attempt_is_the_one_read(archive):
    take = archive / "takes" / "take-A-00.mkv"
    meta = {f"attempt-{n}": {"program": "P", "chroma": {"order": str(n), "ok": True}} for n in (2, 10, 9)}
    (take.parent / f"{take.name}.meta.json").write_text(json.dumps(meta), encoding="utf-8")
    assert ad.last_attempt(take)["chroma"]["order"] == "10"
    assert ad.last_attempt(archive / "takes" / "missing.mkv") == {}

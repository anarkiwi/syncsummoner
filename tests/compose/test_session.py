"""Sessions: one load, takes back to back on one frame line, and the lead-in and lead-out they derive."""

import json

import numpy as np
import pytest

from syncsummoner.compose import lock as L
from syncsummoner.compose import locktake as LT
from syncsummoner.compose import session as SS
from syncsummoner.compose.timecode import ClipLayout, LayoutError
from syncsummoner.device import loadtimes, v4l2rec
from syncsummoner.probe import decoded
from tests.compose import test_locktake
from tests.compose.test_locktake import (  # pylint: disable=unused-import
    FPS,
    LEAD_IN,
    PLAN_ROWS,
    config_for,
    fixture_simulate,
    info_with,
    rigged,
    session_for,
)

COUNTS = [30, 24, 30]


def test_a_layout_plays_the_lead_in_each_take_from_clip_frame_0_and_the_lead_out():
    """The Pi plays the clip file's last lead-in frames, every take from frame 0, then the lead-out."""
    one = SS.Layout(lead=10, counts=(30,), tail=4, source_lead=40)
    assert one.spans() == [(30, 74)] and one.offsets.tolist() == [0] and one.count == 30
    three = SS.Layout(lead=10, counts=(30, 24, 30), tail=4, source_lead=40)
    assert three.spans() == [(30, 70), (40, 64), (40, 74)]
    assert three.offsets.tolist() == [0, 30, 54] and three.count == 84
    assert three.owner(np.array([LT.ev.START, -15, 0, 29, 30, 53, 54, 84, 87])).tolist() == [
        0,
        0,
        0,
        0,
        1,
        1,
        2,
        2,
        2,
    ]


@pytest.mark.parametrize(
    "clip, said",
    [
        (ClipLayout(8, 40, 10), "lead-in needs 10 frames and the clip's holds 8"),
        (ClipLayout(10, 20, 10), "a take plays 30 clip frames and the clip has 20"),
        (ClipLayout(10, 30, 3), "lead-out needs 4 frames after clip frame 30"),
    ],
)
def test_a_clip_too_short_for_its_session_is_refused_with_what_it_lacks(clip, said):
    """Every frame a session plays must be in the clip; what is missing is said, nothing is trimmed."""
    with pytest.raises(LayoutError, match=said):
        SS.Layout(lead=10, counts=(30,), tail=4, source_lead=clip.lead).check(clip)


def test_a_lead_in_holds_its_contents_and_a_load_only_without_a_dry_control():
    """A dry lead-in holds no load; one without grows with the longest load timed and needs one timed."""
    dry = SS.lead_in_frames(LEAD_IN, FPS, dry=True, load_s=None)
    assert dry == SS.lead_in_frames(LEAD_IN, FPS, dry=True, load_s=30.0)
    before = SS.frames_of(LEAD_IN.settle_s + LEAD_IN.bound_s, FPS)
    assert dry == before + SS.frames_of(LEAD_IN.wet_s, FPS) and before >= L.ANCHOR_FRAMES
    with pytest.raises(SS.NoLoadTimeError):
        SS.lead_in_frames(LEAD_IN, FPS, dry=False, load_s=loadtimes.longest())
    short, long_ = (SS.lead_in_frames(LEAD_IN, FPS, dry=False, load_s=s) for s in (1.0, 4.0))
    assert long_ - short == SS.frames_of(3.0, FPS), "one load, decoded by no one before the first write"
    assert (
        SS.lead_out_frames(LEAD_IN, FPS, dry=False, load_s=1.0)
        == SS.frames_of(1.0, FPS) + 1 + L.ANCHOR_FRAMES
    )
    assert (
        SS.lead_out_frames(LEAD_IN, FPS, dry=True, load_s=None)
        == SS.frames_of(LEAD_IN.wet_s, FPS) + L.ANCHOR_FRAMES
    )


def test_a_session_keeps_its_loads_where_a_session_in_another_directory_sizes_by_them(
    monkeypatch, simulate, tmp_path
):
    """Loads timed in one takes directory size the next session's lead-in in any other."""
    for name in "ab":
        (tmp_path / name).mkdir()
    run_session(monkeypatch, simulate, tmp_path / "a", dry=None, counts=[30])
    held = loadtimes.read()
    assert {e["program"] for e in held} == {"Passthru", "Teletext"} and not list(
        (tmp_path / "a").glob("loads*")
    )
    assert SS.session_frames(LEAD_IN, FPS, dry=False)[0] == SS.lead_in_frames(
        LEAD_IN, FPS, dry=False, load_s=max(e["load_s"] for e in held)
    )


def test_a_clip_holds_a_lead_in_no_longer_than_its_own_and_a_lead_out_for_passthru_alone():
    """The lead-in holds the longest load, up to the clip's lead-in; the lead-out holds Passthru's reload."""
    loadtimes.record("Passthru", 0, 1_000_000_000)
    loadtimes.record("czartroniks/Slow", 0, 100_000_000_000)
    least = SS.lead_in_frames(LEAD_IN, FPS, dry=False, load_s=0.0)
    tail = SS.lead_out_frames(LEAD_IN, FPS, dry=False, load_s=1.0)
    roomy = SS.lead_in_frames(LEAD_IN, FPS, dry=False, load_s=200.0)
    assert SS.clip_frames(LEAD_IN, FPS, ClipLayout(roomy, 30, 0), dry=False) == (
        SS.lead_in_frames(LEAD_IN, FPS, dry=False, load_s=100.0),
        tail,
    )
    assert SS.clip_frames(LEAD_IN, FPS, ClipLayout(least + 5, 30, 0), dry=False) == (least + 5, tail)
    assert SS.clip_frames(LEAD_IN, FPS, ClipLayout(least - 1, 30, 0), dry=False) == (least, tail)
    dry = SS.session_frames(LEAD_IN, FPS, dry=True)
    assert SS.clip_frames(LEAD_IN, FPS, ClipLayout(1, 30, 0), dry=True) == dry


def test_a_slow_load_on_record_leaves_a_session_on_its_clip(monkeypatch, simulate, tmp_path):
    """A load longer than the clip's lead-in holds, timed once, does not refuse the sessions after it."""
    monkeypatch.setattr(
        test_locktake, "ends_for", lambda dry, _d=None: SS.session_frames(LEAD_IN, FPS, dry=dry, load_s=0.0)
    )
    loadtimes.record("czartroniks/Slow", 0, 100_000_000_000)
    _, _, _, records, found, proofs, _ = run_session(monkeypatch, simulate, tmp_path, dry=None, counts=[30])
    assert len(records) == 1 and found[0].complete and proofs[0].ok, proofs[0].lines()


class Silent:
    """A Pi heard showing frames up to ``latest``, then nothing more."""

    def __init__(self, latest):
        self.latest_frame, self.waits = latest, []

    def latest(self):
        """The last frame heard."""
        return self.latest_frame

    def wait(self, frame, stop=None, timeout_s=None):  # pylint: disable=unused-argument
        """A frame heard by now, else nothing however long the wait."""
        self.waits.append(timeout_s)
        return 1234 if frame <= self.latest_frame else None


def test_a_frame_the_pi_never_reports_faults_the_session_once_its_frames_and_the_stall_guard_pass():
    """Waiting for the Pi is bounded by the frames left at its rate and the stall guard, then a fault."""
    shown = Silent(10)
    assert SS._heard(shown, 10, 1e9 / 30, 2.0) == 1234  # pylint: disable=protected-access
    with pytest.raises(SS.PlayoutError, match="not heard showing frame 40 within 3.0 s of frame 10"):
        SS._heard(shown, 40, 1e9 / 30, 2.0)  # pylint: disable=protected-access
    with pytest.raises(SS.PlayoutError, match="within 2.5 s of frame -1"):
        SS._heard(Silent(-1), 15, 1e9 / 30, 2.0)  # pylint: disable=protected-access
    assert shown.waits == [2.0, 3.0]


def run_session(monkeypatch, simulate, tmp_path, *, dry, counts=tuple(COUNTS), plans=None):
    """A simulated session of ``counts`` takes, then each take judged from the files it left."""
    info = info_with(dry)
    sim, clip = session_for(simulate, tmp_path, dry=info.dry is not None, counts=counts)
    sim.dry = info.dry or sim.dry
    rig, recorder, load, loads = rigged(sim, info)
    takes = [
        SS.Take(tmp_path / f"take-{k}.mkv", PLAN_ROWS if plans is None else plans[k], c)
        for k, c in enumerate(counts)
    ]
    records = SS.locked_session(
        rig,
        recorder,
        takes,
        program="Teletext",
        config=config_for(),
        working=lambda _info: {1: 0, 3: 0.5, 4: 100},
        load=load,
        clip=clip,
        lead_in=LEAD_IN,
        bypass=lambda name: info.dry,
    )
    pictures = sim.source(clip.lead)
    monkeypatch.setattr(v4l2rec, "read_log", lambda take: sim.frames[sim.captures(take)])

    def features(path, spec, **_):
        del spec
        if str(path) == "clip.mkv":
            return decoded.Features({"small": pictures}, "")
        at = sim.captures(path)
        return decoded.Features({"small": sim.images(at=at), "codes": sim.codes[at]}, "")

    monkeypatch.setattr(LT.decoded, "features", features)
    found = [
        L.attribute_take(t.out, config=config_for(), codes=sim.codes[sim.captures(t.out)]) for t in takes
    ]
    proofs = [
        LT.take_evidence(t.out, program="Teletext", clip="clip.mkv", config=config_for()) for t in takes
    ]
    return sim, rig, takes, records, found, proofs, loads


def assert_each_take_shows_its_own_frames(sim, takes, found):
    """Every take's clip frame ``j`` is the session frame its offset puts it at, from its own captures."""
    offsets = np.r_[0, np.cumsum([t.count for t in takes])[:-1]]
    for take, offset, got in zip(takes, offsets, found):
        at = sim.captures(take.out)
        assert got.complete and got.picked.size == take.count
        assert np.array_equal(at[got.picked] // 2, sim.lead + offset + np.arange(take.count))
        assert np.array_equal(got.shown[got.picked], np.arange(take.count))
        assert got.anchored and not got.faults and not got.strays


def test_a_session_loads_once_and_attributes_every_take_on_its_one_frame_line(
    monkeypatch, simulate, tmp_path
):
    """Three takes back to back: one load, one play, rolls between files, each take exact and proven."""
    sim, rig, takes, records, found, proofs, loads = run_session(monkeypatch, simulate, tmp_path, dry=1)
    assert [name for name, _ in loads] == ["Teletext"]
    ((fps, spans),) = rig.playout.played
    tail = int(np.load(L.lock_path(takes[0].out))["tail"])
    assert fps == FPS and spans == [
        (0, sim.lead + 30),
        (sim.lead, sim.lead + 24),
        (sim.lead, sim.lead + 30 + tail),
    ]
    assert_each_take_shows_its_own_frames(sim, takes, found)
    assert len({f.line.period for f in found}) == 1 and len({f.line.origin for f in found}) == 1
    starts = [int(sim.captures(t.out)[0]) for t in takes]
    assert starts[0] == 0 and all(np.diff(starts) > 0)
    for k, (take, record, proof) in enumerate(zip(takes, records, proofs)):
        lock = np.load(L.lock_path(take.out))
        said = json.loads(str(lock["session"]))
        assert said["index"] == k and said["offsets"] == [0, 30, 54] and said["counts"] == COUNTS
        assert (lock["session_codes"] >= 0).sum() > 0 and lock["session_log"].size == sim.size
        assert proof.ok, (k, proof.lines())
        assert proof.landed.ok and proof.applied.ok and proof.received.ok and proof.loaded["ok"]
        assert not (record.sends["target"] == LT.ev.CALIBRATION).any()
    assert len({json.loads(str(np.load(L.lock_path(t.out))["session"]))["id"] for t in takes}) == 1
    first, later = (np.load(f"{t.out}.wire.npz") for t in takes[:2])
    assert np.array_equal(first["ticks"], later["ticks"])
    assert (
        later["asked"]["frame"] == first["asked"]["frame"] - 30
    ).all(), "each take counts from its frame 0"
    assert records[1].plan["target"].min() == 0 and records[-1].plan["target"].max() == 30


def test_a_bypass_program_plays_no_passthru_at_all(monkeypatch, simulate, tmp_path):
    """A program whose bypass carries the strips is loaded once, before playout, and Passthru never."""
    _, rig, takes, _, found, proofs, loads = run_session(monkeypatch, simulate, tmp_path, dry=1, counts=[30])
    assert [name for name, _ in loads] == ["Teletext"]
    assert "Passthru" not in json.dumps(json.loads(str(np.load(f"{takes[0].out}.wire.npz")["loads"])))
    assert ("midi", "monitor", "on") in rig.transport.device.shell.said
    assert found[0].complete and proofs[0].ok, proofs[0].lines()


def test_a_program_with_no_dry_control_has_passthru_only_at_the_sessions_ends(
    monkeypatch, simulate, tmp_path
):
    """Passthru carries the lead-in and the lead-out, the program loaded once between them for every take."""
    sim, rig, takes, _, found, proofs, loads = run_session(monkeypatch, simulate, tmp_path, dry=None)
    assert [name for name, _ in loads] == ["Passthru", "Teletext", "Passthru"]
    timed = json.loads(str(np.load(f"{takes[0].out}.wire.npz")["loads"]))
    assert [x["program"] for x in timed] == ["Teletext", "Passthru"]
    assert timed[0]["t1_ns"] < sim.frames["t_ns"][2 * sim.lead]
    assert timed[1]["t0_ns"] > sim.frames["t_ns"][2 * (sim.lead + sum(COUNTS)) - 1]
    assert_each_take_shows_its_own_frames(sim, takes, found)
    assert all(p.landed.ok and p.applied.ok for p in proofs), [p.lines() for p in proofs]
    shell = rig.transport.device.shell
    assert all(quiet for _, quiet in shell.loads), "no load shares the shell with the poller or the sender"
    (lead_in, _), (lead_out, _) = shell.loads[-2:]
    assert ("midi", "monitor", "on") not in shell.said[:lead_in], "the lead-in's load precedes the monitor"
    assert shell.said[lead_out - 1] == ("midi", "monitor", "off")


def test_a_lead_in_whose_strips_name_too_few_frames_fails_the_session(monkeypatch, simulate, tmp_path):
    """Two legible frames name one start: nothing fixes the line, said once the take is attributed."""
    sim, clip = session_for(simulate, tmp_path, dry=True, counts=[30])
    sim.codes[:] = -1
    sim.codes[2 * (sim.lead - 3) : 2 * (sim.lead - 1)] = (np.repeat(np.arange(-3, -1), 2)) & 0xFFFF
    rig, recorder, load, _ = rigged(sim, info_with(1))
    SS.locked_take(
        rig,
        recorder,
        tmp_path / "t.mkv",
        program="Teletext",
        seconds=1.0,
        config=config_for(),
        plan=PLAN_ROWS,
        working=lambda info: {1: 0},
        load=load,
        clip=clip,
        lead_in=LEAD_IN,
        bypass=lambda name: info_with(1).dry,
    )
    assert rig.transport.device.shell.said[-1] == ("midi", "monitor", "off")
    monkeypatch.setattr(v4l2rec, "read_log", lambda take: sim.frames[sim.captures(take)])
    with pytest.raises(L.AnchorError, match=f"{L.ANCHOR_FRAMES} consecutive legible strip-coded frames"):
        L.attribute_take(
            tmp_path / "t.mkv", config=config_for(), codes=sim.codes[sim.captures(tmp_path / "t.mkv")]
        )


def test_a_take_judged_against_another_sessions_log_is_refused(monkeypatch, simulate, tmp_path):
    """A take whose captures the record's session does not hold is not attributed by it."""
    sim, _, takes, _, _, _, _ = run_session(monkeypatch, simulate, tmp_path, dry=1, counts=[30, 30])
    shifted = sim.frames[sim.captures(takes[1].out)].copy()
    shifted["sequence"] += sim.size
    monkeypatch.setattr(v4l2rec, "read_log", lambda take: shifted)
    with pytest.raises(L.LockError, match="are not the session's"):
        L.attribute_take(takes[1].out, config=config_for(), codes=np.full(shifted.size, -1))

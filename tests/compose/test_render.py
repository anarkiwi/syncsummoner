"""Render: timecode strip, design 4.4 scheduling, alignment, passes and compositing."""

# pylint: disable=missing-function-docstring,protected-access

import contextlib
import dataclasses
import json
import sys
import time
import types

import numpy as np
import pytest
from pyvmancer import CcMapError

from syncsummoner.compose import lock as L
from syncsummoner.compose import locktake
from syncsummoner.compose import session as compose_session
from syncsummoner.compose import render as R
from syncsummoner.compose.score import GestureInstance, Layer, Score, Section
from syncsummoner.device import loadtimes
from syncsummoner.compose.timecode import ClipLayout
from syncsummoner.compose.vocabulary import Automation
from syncsummoner.device.profile import Profiles
from syncsummoner.probe.patterns import read_timecode
from tests.compose import PROBED, rig_transport
from tests.compose.locked import CLIP, LOCKED, Clock, Proof, locked_pass, render_locked

CONFIG = R.RenderConfig(width=32, height=24, fps=10.0, strip_px=4, bits=16)
#: The session above with a stall guard given, so a rig opens with no load on record.
STALLED = dataclasses.replace(CONFIG, stall_s=4.0)


@pytest.fixture(autouse=True)
def _unjudged(monkeypatch):
    """Takes here are stand-ins with no lock record or picture; the check has tests of its own."""
    monkeypatch.setattr(R, "checked_plan", lambda score, plan, takes, clip, *, config: (list(plan), {}))


@pytest.mark.parametrize("index", [0, 1, 7, 1234, 65535])
def test_timecode_round_trip(index):
    frame = np.zeros((24, 32, 3), np.float32)
    burned = R.burn_timecode(frame, index, bits=16, strip_px=4)
    assert read_timecode(burned, bits=16, strip_px=4) == index
    assert R.crop_strip(burned, strip_px=4).shape == (20, 32, 3)
    assert frame.max() == 0.0


def test_timecode_absent_or_washed_out_reads_none():
    flat = np.full((24, 32, 3), 0.5, np.float32)
    assert read_timecode(flat, bits=16, strip_px=4) is None
    burned = R.burn_timecode(np.zeros((24, 32, 3), np.float32), 5, bits=16, strip_px=4)
    assert read_timecode(burned * 0.1, bits=16, strip_px=4) is None


def test_open_rig_builds_from_the_device_layer(monkeypatch):
    built = {}

    def module(name, **attrs):
        mod = types.ModuleType(name)
        for key, value in attrs.items():
            setattr(mod, key, value)
        monkeypatch.setitem(sys.modules, name, mod)
        monkeypatch.setattr("syncsummoner.device." + name.rsplit(".", 1)[1], mod, raising=False)
        return mod

    def record(key):
        def make(*args, **kwargs):
            built[key] = (args, kwargs)
            return built[key]

        return make

    stalls = []
    module(
        "syncsummoner.device.transport",
        Transport=types.SimpleNamespace(open=lambda stall: stalls.append(stall) or "transport"),
    )
    module("syncsummoner.device.session", Session=record("session"))

    def capture(*args, **kwargs):
        built["capture"] = (args, kwargs)
        return types.SimpleNamespace(open=lambda: "open capture")

    module("syncsummoner.device.capture", Capture=capture)
    module("syncsummoner.device.playout", ClipPlayer=record("playout"))
    module("syncsummoner.device.link", Link=record("link"))
    with pytest.raises(loadtimes.NoLoadRecordError, match="--stall"):
        R.open_rig(CONFIG)
    assert not stalls, "with no load on record and no --stall, no device is opened"
    loadtimes.record("Teletext", 0, 3_000_000_000, firmware="1.0.2")
    loadtimes.record("Passthru", 10, 2_000_000_010, firmware="1.0.2")
    rig = R.open_rig(CONFIG)
    assert rig.session == (("transport",), {})
    assert built["capture"][1]["width"] == 32 and built["playout"][1]["height"] == 24
    assert rig.capture == "open capture", "the rig hands back an opened capture"
    assert rig.link == ((), {}), "a real rig always gets link control, defaulted by the device layer"
    R.open_rig(dataclasses.replace(CONFIG, source_host="pi@rig"))
    assert built["link"][0] == ("pi@rig",) and built["playout"][0] == ("pi@rig",)
    R.open_rig(dataclasses.replace(CONFIG, stall_s=5.0))
    assert stalls == [3.0, 3.0, 5.0], "a wait seeing nothing past the longest load timed fails"


def test_enforce_safety_passes_safe_output_through():
    frames = np.full((30, 32, 32, 3), 0.5, np.float32)
    assert R.enforce_safety(frames, fps=30.0) is frames


def test_enforce_safety_raises_when_mitigation_declined():
    frames = np.zeros((90, 32, 32, 3), np.float32)
    frames[::2] = 1.0
    with pytest.raises(R.UnsafeOutputError, match="flashes/s"):
        R.enforce_safety(frames, fps=30.0, mitigate=False)


def test_open_rig_hands_back_an_open_capture(monkeypatch):
    """A pass grabs frame by frame and never enters the capture as a context."""
    opened = []

    class FakeCapture:
        """Capture stand-in recording open and close."""

        def __init__(self, **kwargs):
            self.kwargs = kwargs

        def open(self):
            """Open."""
            opened.append("open")
            return self

        def close(self):
            """Close."""
            opened.append("close")

    import syncsummoner.device.capture as capture_mod
    import syncsummoner.device.link as link_mod
    import syncsummoner.device.playout as playout_mod
    import syncsummoner.device.session as session_mod
    import syncsummoner.device.transport as transport_mod

    monkeypatch.setattr(capture_mod, "Capture", FakeCapture)
    monkeypatch.setattr(link_mod, "Link", lambda *a, **k: None)
    monkeypatch.setattr(playout_mod, "ClipPlayer", lambda *a, **k: None)
    monkeypatch.setattr(session_mod, "Session", lambda *a, **k: None)
    monkeypatch.setattr(transport_mod.Transport, "open", staticmethod(lambda *a, **k: None))
    rig = R.open_rig(STALLED)
    assert opened == ["open"] and isinstance(rig.capture, FakeCapture)


def test_the_rig_session_is_a_named_format():
    """Playout writes the Pi framebuffer, which is 1920x1080: a 720p session showed nothing."""
    rig = R.RenderConfig.for_format("1080p30")
    assert (rig.width, rig.height, rig.fps) == (1920, 1080, 30.0)


def test_reading_a_timecode_only_touches_the_strip():
    """Only the strip rows are converted, never the whole frame."""
    frame = np.random.random((1080, 1920, 3)).astype(np.float32)
    stamped = R.burn_timecode(frame, 4242, bits=16, strip_px=8)
    assert read_timecode(stamped, bits=16, strip_px=8) == 4242
    start = time.perf_counter()
    for _ in range(20):
        read_timecode(stamped, bits=16, strip_px=8)
    assert (time.perf_counter() - start) / 20 < 0.005, "a strip read is not a whole-frame conversion"


def test_cuts_follow_the_score_sections_in_rotation():
    """Sections are where the music already changes, so they are where a cut belongs."""
    score = Score(
        seed=1,
        bpm=120.0,
        duration=40.0,
        fps=30.0,
        sections=[Section(0.0, 10.0, "A"), Section(10.0, 25.0, "B"), Section(25.0, 40.0, "C")],
    )
    plan = R.cut_plan(score, ["X", "Y"])
    assert [(c.start, c.end, c.program) for c in plan] == [
        (0.0, 10.0, "X"),
        (10.0, 25.0, "Y"),
        (25.0, 40.0, "X"),
    ]
    assert plan[1].duration == 15.0


def test_a_score_with_no_sections_cuts_once():
    score = Score(seed=1, bpm=120.0, duration=12.0, fps=30.0)
    assert [(c.start, c.end, c.program) for c in R.cut_plan(score, ["Solo"])] == [(0.0, 12.0, "Solo")]
    with pytest.raises(ValueError, match="at least one program"):
        R.cut_plan(score, [])


def test_render_cuts_runs_one_pass_per_program_and_refuses_a_blank_one(monkeypatch, tmp_path):
    """A blank pass must not be spliced into a demo as though it were footage."""
    calls = []

    def fake_render(score, source, out, **kwargs):
        del source, kwargs
        calls.append((score.layers[0].program, out))
        blank = score.layers[0].program == "Bad"
        return R.TakeReport(frames=10, picture=0 if blank else 10, frozen=blank, luma=0.4)

    monkeypatch.setattr(R, "splice", lambda *a, **k: None)
    score = Score(
        seed=1,
        bpm=120.0,
        duration=20.0,
        fps=30.0,
        sections=[Section(0.0, 10.0, "A"), Section(10.0, 20.0, "B")],
        layers=[Layer(index=0, program="ignored", gestures=[])],
    )
    plan = R.render_cuts(
        score,
        "clip.mkv",
        tmp_path / "out.mkv",
        profiles={},
        programs=["Good", "Other"],
        takes=tmp_path,
        prepared=True,
        pass_render=fake_render,
    )
    assert [c.program for c in plan] == ["Good", "Other"] and len(calls) == 2
    with pytest.raises(R.BlankTakeError):
        R.render_cuts(
            score,
            "clip.mkv",
            tmp_path / "out.mkv",
            profiles={},
            programs=["Bad"],
            takes=tmp_path,
            prepared=True,
            pass_render=fake_render,
        )


def test_a_recorded_rig_leaves_the_card_for_the_recorder(monkeypatch):
    """A capture held here stops ffmpeg opening the card, and the pass dies at once."""
    import syncsummoner.device.capture as capture_mod
    import syncsummoner.device.link as link_mod
    import syncsummoner.device.playout as playout_mod
    import syncsummoner.device.session as session_mod
    import syncsummoner.device.transport as transport_mod

    opened = []
    monkeypatch.setattr(
        capture_mod, "Capture", lambda **kw: types.SimpleNamespace(open=lambda: opened.append(True))
    )
    monkeypatch.setattr(link_mod, "Link", lambda *a, **k: None)
    monkeypatch.setattr(playout_mod, "ClipPlayer", lambda *a, **k: None)
    monkeypatch.setattr(session_mod, "Session", lambda *a, **k: None)
    monkeypatch.setattr(transport_mod.Transport, "open", staticmethod(lambda *a, **k: None))
    assert R.open_rig(STALLED, capture=False).capture is None and not opened
    R.open_rig(STALLED, capture=True)
    assert opened == [True], "the old path still gets one when it asks"


@pytest.mark.parametrize("corrupted", [False, True])
def test_a_locked_take_is_proven_and_judged_once_its_cc_map_holds(monkeypatch, tmp_path, corrupted):
    """The program loads, the take records frame-locked, the CC map is checked, then the take is proven."""

    def verify():
        rig.log.append(("verify",))
        if corrupted:
            raise CcMapError("device CC map differs from the CCs sent (P12 CC0)")

    rig = locked_pass(monkeypatch, transport=rig_transport(verify_cc_map=verify), load_s={"Teletext": 0.1})
    out = tmp_path / "take.mkv"
    with pytest.raises(CcMapError) if corrupted else contextlib.nullcontext():
        assert render_locked(rig, out) == PROBED
    steps = ["upload", "load", "record", "play", "drive", "stop", "recorded", "flips", "verify"]
    assert rig.steps == steps + ([] if corrupted else ["evidence", "inspect"])
    assert [step[1:] for step in rig.log if step[0] == "load"] == [("Teletext", "link")]
    record = np.load(L.lock_path(out))
    assert (int(record["hold"]), int(record["lead"]), int(record["count"])) == (2, 8, 6)
    assert np.array_equal(record["flips"], rig.take.flips)
    if not corrupted:
        assert rig.take.shows(np.load(L.picked_path(out)))
        assert rig.log[-2] == ("evidence", "Teletext", "tc.mkv", "take.mkv"), "proven against the clip played"
        assert rig.log[-1] == ("inspect", int(np.flatnonzero(rig.take.shown() == 0)[0]))
        assert json.loads(locktake.evidence_path(out).read_text(encoding="utf-8")) == {"ok": True}


def test_an_unproven_take_is_reported_and_still_judged(monkeypatch, tmp_path, caplog):
    rig = locked_pass(monkeypatch, proof=[Proof(ok=False)])
    out = tmp_path / "take.mkv"
    assert render_locked(rig, out) == PROBED
    assert rig.steps[-2:] == ["evidence", "inspect"]
    assert json.loads(locktake.evidence_path(out).read_text(encoding="utf-8")) == {"ok": False}
    assert "Teletext: the take's parameters are not all proven" in caplog.text


def failed_attribution(monkeypatch, sequence):
    """A take whose attribution fails, its sidecar numbering captures as ``sequence``."""
    from syncsummoner.device import v4l2rec  # pylint: disable=import-outside-toplevel

    def attribute(*_, **__):
        raise L.LockError("clip frames 3..5 have no capture")

    monkeypatch.setattr("syncsummoner.compose.judge.attribute_take", attribute)
    monkeypatch.setattr(v4l2rec, "read_log", lambda take: np.array(sequence, dtype=[("sequence", "<i8")]))


def test_a_failed_attribution_with_driver_drops_is_a_take_fault(monkeypatch, tmp_path):
    rig = locked_pass(monkeypatch)
    failed_attribution(monkeypatch, [0, 1, 2, 5, 6, 9])
    with pytest.raises(R.TakeFaultError, match="Teletext: the capture driver lost 4 frames: clip frames 3"):
        render_locked(rig, tmp_path / "take.mkv")
    assert "evidence" not in rig.steps


def test_a_failed_attribution_with_no_driver_drops_stays_a_lock_error(monkeypatch, tmp_path):
    rig = locked_pass(monkeypatch)
    failed_attribution(monkeypatch, [0, 1, 2, 3])
    with pytest.raises(L.LockError) as raised:
        render_locked(rig, tmp_path / "take.mkv")
    assert not isinstance(raised.value, R.TakeFaultError)


def test_a_take_plans_only_the_writes_inside_its_clip(monkeypatch, tmp_path):
    """Writes are targeted on clip frames; one due past the clip's last frame is never sent."""
    rig = locked_pass(monkeypatch)
    planned = []
    take_fn = compose_session.locked_take

    def capture(*args, plan, **kw):
        planned.append(plan.copy())
        return take_fn(*args, plan=plan, **kw)

    monkeypatch.setattr(compose_session, "locked_take", capture)
    auto = Automation.of([0.0, 0.05, 0.3], 2, [100, 200, 300])
    monkeypatch.setattr(R, "plan_automation", lambda *a, **k: {0: auto})
    render_locked(rig, tmp_path / "take.mkv")
    assert len(planned) == 1
    plan = planned[0]
    assert plan["target"].tolist() == [0, 2] and plan["value"].tolist() == [100, 200]


def test_a_passthru_take_loads_nothing_inside_its_lead_in(monkeypatch, tmp_path):
    rig = locked_pass(monkeypatch, clock=Clock(5.0))
    out = tmp_path / "take.mkv"
    loaded = R.play_take(rig.rig, rig, out, program=None, seconds=0.2, config=LOCKED, clip=CLIP)
    assert rig.steps == ["load", "record", "play", "stop", "recorded", "flips"] and loaded == 5.0
    assert L.lock_path(out).is_file()


def test_a_lockcheck_take_loads_its_program_inside_the_lead_in(monkeypatch, tmp_path):
    rig = locked_pass(monkeypatch, load_s={"Teletext": 0.1})
    rig.rig.transport.prepare = lambda name: rig.log.append(("author", name))
    out = tmp_path / "take.mkv"
    loaded = R.play_take(rig.rig, rig, out, program="Teletext", seconds=0.2, config=LOCKED, clip=CLIP)
    assert [step[1] for step in rig.log if step[0] == "load"] == ["Passthru", "Teletext", "Passthru"]
    assert rig.steps[:6] == ["load", "author", "record", "play", "load", "params"]
    assert loaded == pytest.approx(0.1), "the author is switched before the lead-in, outside its timing"
    assert rig.clock.now == pytest.approx((8 + 6 + 8) / 30 + LOCKED.tail_s)
    record = np.load(L.lock_path(out))
    assert (int(record["lead"]), int(record["count"]), int(record["tail"])) == (8, 6, 8)


def test_passthru_loading_past_the_lead_out_is_a_lead_in_error(monkeypatch, tmp_path):
    rig = locked_pass(monkeypatch, load_s={"Passthru": 0.3})
    with pytest.raises(R.LeadInError, match="Teletext: Passthru loaded past the lead-out"):
        R.play_take(
            rig.rig, rig, tmp_path / "take.mkv", program="Teletext", seconds=0.2, config=LOCKED, clip=CLIP
        )
    assert not L.lock_path(tmp_path / "take.mkv").exists()


def test_a_load_that_overruns_the_lead_in_is_a_lead_in_error(monkeypatch, tmp_path):
    rig = locked_pass(monkeypatch, load_s={"Teletext": 0.3})
    out = tmp_path / "take.mkv"
    with pytest.raises(
        R.LeadInError,
        match=r"Teletext: loading took 0.3s, past the 0.3s lead-in; timecode the clip with a longer one",
    ):
        R.play_take(rig.rig, rig, out, program="Teletext", seconds=0.2, config=LOCKED, clip=CLIP)
    assert not L.lock_path(out).is_file()


@pytest.fixture(name="clipped")
def fixture_clipped(monkeypatch, tmp_path):
    """``ensure_clip`` over a scratch clip whose layout is ``have``, needing ``need``; writes logged."""
    state = {"need": (10, 4), "have": ClipLayout(8, 6, 8), "written": []}
    monkeypatch.setattr(locktake, "dry_state", lambda program, bypass: {1: 1023})
    monkeypatch.setattr(compose_session, "session_frames", lambda lead_in, fps, **kw: state["need"])

    def written(*_, **kw):
        state["written"].append((kw["lead"], kw["tail"]))
        state["have"] = ClipLayout(kw["lead"], 6, kw["tail"])

    monkeypatch.setattr(R, "write_timecoded", written)
    monkeypatch.setattr(R.timecode, "clip_layout", lambda path: state["have"])
    (tmp_path / "tc.mkv").write_bytes(b"x")
    return state


def test_a_clip_too_short_for_a_session_is_timecoded_again_never_shorter(clipped, tmp_path):
    got = R.ensure_clip("src.mkv", tmp_path / "tc.mkv", "Teletext", config=LOCKED, seconds=0.2)
    assert clipped["written"] == [(10, 8)] and got == ClipLayout(10, 6, 8)
    clipped["need"] = (9, 8)
    assert R.ensure_clip("src.mkv", tmp_path / "tc.mkv", "Teletext", config=LOCKED, seconds=0.2) == got
    assert clipped["written"] == [(10, 8)], "a clip long enough is reused as it is"


def test_a_prepared_clip_too_short_for_a_session_is_refused(clipped, tmp_path):
    with pytest.raises(R.timecode.LayoutError, match="10-frame lead-in"):
        R.ensure_clip(
            "src.mkv",
            tmp_path / "tc.mkv",
            "Teletext",
            config=LOCKED,
            seconds=0.2,
            prepared=True,
        )
    assert not clipped["written"]


def test_a_layer_with_no_profile_is_refused_before_the_rig_is_touched(monkeypatch, tmp_path):
    rig = locked_pass(monkeypatch)
    score = Score(
        duration=0.2, fps=30.0, sections=[Section(0.0, 0.2, "A")], layers=[Layer("Teletext", 0, [])]
    )
    with pytest.raises(R.MissingProfileError, match=r"no profile for Teletext in /profiles/here"):
        R.render_played(
            score,
            "clip.mkv",
            tmp_path / "take.mkv",
            profiles=Profiles({"Other": object()}, directory="/profiles/here"),
            rig=rig.rig,
            config=LOCKED,
            scratch="tc.mkv",
            prepared=True,
            recorder=rig,
        )
    assert not rig.log


def test_each_cut_program_is_driven_by_its_own_evolved_layer(monkeypatch, tmp_path):
    """The planner evolves a layer per program; a cut that reused the first would throw that away."""
    seen = {}

    def fake_render(score, source, out, **kwargs):
        del source, out, kwargs
        layer = score.layers[0]
        seen[layer.program] = [g.gesture for g in layer.gestures]
        return R.TakeReport(frames=10, picture=10, frozen=False, luma=0.4)

    monkeypatch.setattr(R, "splice", lambda *a, **k: None)
    score = Score(
        duration=20.0,
        sections=[Section(0.0, 7.0, "A"), Section(7.0, 14.0, "B"), Section(14.0, 20.0, "C")],
        layers=[
            Layer(index=0, program="Good", gestures=[GestureInstance("ramp", 1.0, 1.0)]),
            Layer(index=1, program="Other", gestures=[GestureInstance("hold", 2.0, 1.0)]),
        ],
    )
    R.render_cuts(
        score,
        "clip.mkv",
        tmp_path / "out.mkv",
        profiles={},
        programs=["Good", "Other", "Unplanned"],
        takes=tmp_path,
        prepared=True,
        pass_render=fake_render,
    )
    assert seen == {"Good": ["ramp"], "Other": ["hold"], "Unplanned": ["ramp"]}


def test_each_cut_pass_sizes_the_timecoded_source_for_its_own_session(monkeypatch, tmp_path):
    """The cut builds no clip itself: each pass lengthens it only if its program's session needs more."""
    built, passes = [], []
    monkeypatch.setattr(R, "splice", lambda *a, **k: None)
    monkeypatch.setattr(R, "write_timecoded", lambda source, out, **kw: built.append(out) or 1)

    def fake_render(score, source, out, **kwargs):
        del score, source, out
        passes.append(kwargs["prepared"])
        return R.TakeReport(frames=10, picture=10, frozen=False, luma=0.4)

    score = Score(
        duration=20.0,
        sections=[Section(0.0, 10.0, "A"), Section(10.0, 20.0, "B")],
        layers=[Layer("Good", 0, [])],
    )
    R.render_cuts(
        score,
        "clip.mkv",
        tmp_path / "out.mkv",
        profiles={},
        programs=["Good", "Other"],
        takes=tmp_path,
        scratch=tmp_path / "tc.mkv",
        pass_render=fake_render,
    )
    assert not built and passes == [False, False]


def test_cut_render_makes_room_for_its_takes(monkeypatch, tmp_path):
    """A pass that fails on a missing directory has already spent the load on it."""
    monkeypatch.setattr(R, "splice", lambda *a, **k: None)
    takes = tmp_path / "takes"
    R.render_cuts(
        Score(duration=20.0, sections=[Section(0.0, 20.0, "A")], layers=[Layer("Good", 0, [])]),
        "clip.mkv",
        tmp_path / "out.mkv",
        profiles={},
        programs=["Good"],
        takes=takes,
        prepared=True,
        pass_render=lambda *a, **k: R.TakeReport(frames=10, picture=10, frozen=False, luma=0.4),
    )
    assert takes.is_dir()


def test_programs_beyond_the_sections_are_reported_not_dropped(monkeypatch, tmp_path, caplog):
    """Naming eight programs against three sections renders three; silence would read as eight."""
    monkeypatch.setattr(R, "splice", lambda *a, **k: None)
    with caplog.at_level("WARNING", logger="syncsummoner"):
        plan = R.render_cuts(
            Score(
                duration=20.0,
                sections=[Section(0.0, 10.0, "A"), Section(10.0, 20.0, "B")],
                layers=[Layer("Derez", 0, [])],
            ),
            "clip.mkv",
            tmp_path / "out.mkv",
            profiles={},
            programs=["Derez", "Lorenz", "Scramble", "Sfumato"],
            takes=tmp_path,
            prepared=True,
            pass_render=lambda *a, **k: R.TakeReport(frames=10, picture=10, frozen=False, luma=0.4),
        )
    assert len(plan) == 2
    assert "Scramble, Sfumato" in caplog.text


def test_a_long_wait_reports_progress_while_it_waits():
    """A five minute pass would otherwise print nothing for five minutes."""
    clock, said = Clock(), []
    R._wait_until(5.0, clock=clock, sleep=clock.sleep, tick_s=1.0, report=said.append)
    assert clock.slept == pytest.approx([1.0] * 5) and said == pytest.approx([1.0, 2.0, 3.0, 4.0, 5.0])


def test_ticks_never_outrun_the_instant_they_wait_for():
    clock = Clock()
    R._wait_until(2.5, clock=clock, sleep=clock.sleep, tick_s=1.0)
    assert clock.slept == pytest.approx([1.0, 1.0, 0.5]), "the last tick is short, landing on the instant"


def test_a_clock_that_never_advances_ends_the_wait():
    slept = []
    R._wait_until(1.0, clock=lambda: 0.0, sleep=slept.append, tick_s=0.25)
    assert slept == [0.25]


def test_unknown_programs_reports_what_the_device_will_not_load():
    assert R.unknown_programs(["a", "b"], None) == []
    assert R.unknown_programs(["a", "b"], ["a", "b"]) == []
    assert R.unknown_programs(["a", "b"], ["a"]) == ["b"]
    assert R.unknown_programs(["b", "b"], ["a"]) == ["b"]


def test_a_program_the_device_dropped_costs_no_rig_time():
    """Firmware moves under an archive; the pass must not be the thing that finds out."""
    score = Score(duration=4.0, fps=30.0, sections=[Section(0.0, 4.0, "A")])
    started = []

    def never(*args, **_kwargs):
        started.append(args)

    with pytest.raises(R.UnknownProgramError, match="Flock"):
        R.render_cuts(
            score,
            "source.mkv",
            "out.mp4",
            profiles={"Zollner": None, "Flock": None},
            programs=["Zollner", "Flock"],
            available=["Zollner"],
            prepared=True,
            pass_render=never,
        )
    assert not started


def offering(*listings):
    """A lister answering ``listings`` in turn, recording each call."""
    calls = []

    def lister():
        calls.append(len(calls))
        return listings[min(len(calls) - 1, len(listings) - 1)]

    return lister, calls


def cut_render(monkeypatch, tmp_path, available):
    """Render cuts for Quilt and Pinwheel against ``available``, with passes that succeed at once."""
    monkeypatch.setattr(R, "splice", lambda *a, **k: None)
    score = Score(duration=4.0, fps=30.0, sections=[Section(0.0, 2.0, "A"), Section(2.0, 4.0, "B")])
    return R.render_cuts(
        score,
        "source.mkv",
        tmp_path / "out.mp4",
        profiles={},
        programs=["czartroniks/Quilt", "Pinwheel"],
        available=available,
        takes=tmp_path,
        prepared=True,
        pass_render=lambda *a, **k: R.TakeReport(frames=10, picture=10, frozen=False, luma=0.4),
    )


def test_a_short_listing_is_listed_again_before_failing(monkeypatch, tmp_path, caplog):
    """A listing missing a program is retried once, and both results are logged."""
    full = R.Offered(["Pinwheel", "czartroniks/Quilt"], ["lzx", "czartroniks"])
    lister, calls = offering(R.Offered(["Quilt", "Fray"]), full)
    with caplog.at_level("INFO"):
        plan = cut_render(monkeypatch, tmp_path, lister)
    assert [c.program for c in plan] == ["czartroniks/Quilt", "Pinwheel"] and calls == [0, 1]
    assert (
        "listing again" in caplog.text
        and "second listing: 2 programs from authors lzx, czartroniks" in caplog.text
    )


def test_the_does_not_offer_error_says_what_the_device_listed(monkeypatch, tmp_path):
    """The error carries each listing's count, its authors, a sample, and where a missing name was found."""
    lister, calls = offering(R.Offered(["Quilt", "Fray"]), R.Offered(["Quilt", "Fray", "Bitcullis"], None))
    with pytest.raises(R.UnknownProgramError) as caught:
        cut_render(monkeypatch, tmp_path, lister)
    said = str(caught.value)
    assert calls == [0, 1] and said.startswith(
        "the device does not offer czartroniks/Quilt, Pinwheel; it listed "
    )
    assert "2 programs from authors unreported, e.g. Quilt, Fray; listed czartroniks/Quilt as Quilt" in said
    assert " then 3 programs from authors unreported, e.g. Quilt, Fray, Bitcullis" in said


def test_describe_offer_samples_a_long_listing():
    names = [f"P{i}" for i in range(R.LISTING_SAMPLE + 3)]
    said = R.describe_offer(R.Offered(names, ["lzx"]), ["Q"])
    assert said == f"{len(names)} programs from authors lzx, e.g. {', '.join(names[: R.LISTING_SAMPLE])}"
    assert R.describe_offer([]) == "0 programs from authors unreported, e.g. none"


def test_device_programs_reports_the_authors_it_enumerated(monkeypatch):
    from syncsummoner.device import transport as transport_mod

    port = types.SimpleNamespace(
        programs=lambda: ["Pinwheel", "czartroniks/Quilt"], authors=lambda: ["lzx", "czartroniks"]
    )
    port.close = lambda: setattr(port, "closed", True)
    monkeypatch.setattr(transport_mod.Transport, "open", staticmethod(lambda: port))
    offered = R.device_programs()
    assert list(offered) == ["Pinwheel", "czartroniks/Quilt"] and offered.authors == ("lzx", "czartroniks")
    assert port.closed

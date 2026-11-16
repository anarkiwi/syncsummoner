"""Render: timecode strip, design 4.4 scheduling, alignment, passes and compositing."""

# pylint: disable=missing-function-docstring,protected-access

import contextlib
import io
import dataclasses
import sys
import time
import types

import numpy as np
import pytest
from pyvmancer import CcMapError

from syncsummoner.compose import lock as L
from syncsummoner.compose import render as R
from syncsummoner.compose.score import GestureInstance, Layer, Score, Section
from syncsummoner.compose.vocabulary import Automation
from syncsummoner.probe.patterns import read_timecode
from tests.compose import PROBED, rig_transport
from tests.compose.locked import LOCKED, Clock, locked_pass, render_locked

CONFIG = R.RenderConfig(width=32, height=24, fps=10.0, strip_px=4, bits=16)


class FakeSession:
    """Records program loads and CC batches; a real Session would rate-limit and cache."""

    def __init__(self):
        self.programs = []
        self.calls = []
        self.links = []

    def load_program(self, name, *, link=None):
        self.programs.append(name)
        self.links.append(link)

    def set_params(self, values):
        self.calls.append(dict(values))


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


def test_schedule_biases_visuals_early():
    auto = Automation.of([1.0], 2, 500)
    shifted = R.schedule(auto, latency_s=0.08, early_bias_s=R.EARLY_BIAS_S)
    assert shifted.times[0] == pytest.approx(1.0 - 0.08 - 0.015)
    assert 0.010 <= R.EARLY_BIAS_S <= 0.020
    assert R.schedule(auto).times[0] < auto.times[0]


def test_write_latency_defaults_to_a_frame_and_takes_an_override():
    assert R.write_latency(R.RenderConfig.for_format("1080p30"), None) == pytest.approx(1 / 30)
    assert R.write_latency(R.RenderConfig.for_format("720p60"), None) == pytest.approx(1 / 60)
    assert R.write_latency(dataclasses.replace(CONFIG, latency_s=0.02), None) == 0.02
    late = R.schedule(Automation.of([1.0], 2, 500), latency_s=R.write_latency(CONFIG, None), early_bias_s=0.0)
    assert late.times[0] == pytest.approx(1.0 - 1 / CONFIG.fps)


@pytest.mark.parametrize("measured,want", [(0.031, 0.031), (None, 0.02)])
def test_measured_latency_falls_back_to_the_configured_figure(monkeypatch, measured, want):
    from syncsummoner.device import doctor

    seen = []
    monkeypatch.setattr(doctor, "write_latency", lambda transport: seen.append(transport) or measured)
    config = dataclasses.replace(CONFIG, latency_s=0.02, measure_latency=True)
    assert R.write_latency(config, "transport") == want and seen == ["transport"]


def test_schedule_pulls_simultaneous_writes_forward_into_their_own_slots():
    auto = Automation.concat([Automation.of([0.5, 1.0], i, [100, 200]) for i in (1, 2, 3)])
    paced = R.schedule(auto, latency_s=0.0, early_bias_s=0.0, cc_budget_hz=100.0)
    assert paced.times == pytest.approx([0.48, 0.49, 0.5, 0.98, 0.99, 1.0])
    assert paced.indices.tolist() == [1, 2, 3, 1, 2, 3], "order within an instant is kept"
    start = R.schedule(Automation.of([0.0, 0.0, 0.0], 1, [1, 2, 3]), early_bias_s=0.0, cc_budget_hz=100.0)
    assert start.times == pytest.approx([0.0, 0.01, 0.02]), "nothing is due before the pass starts"


class StampedTransport:
    """Records when each CC write leaves, on a clock only sleeps advance."""

    def __init__(self, clock, msgs_per_param):
        self.clock, self.msgs_per_param, self.sent = clock, msgs_per_param, []

    def set_param(self, index, value):
        self.sent.append((self.clock(), index, value))


@pytest.mark.parametrize("msgs_per_param", [1, 2])
def test_a_schedule_at_the_planners_budget_is_delivered_on_time(msgs_per_param):
    from syncsummoner.compose import planner
    from syncsummoner.device.session import Session

    from . import make_profile

    budget, burst = 64.0, 4
    cues = np.repeat(np.arange(1.0, 3.0, burst / budget), burst)
    unsettled = np.array([4, 5, 6, 8], dtype=np.int32)
    auto = Automation(cues, np.tile(unsettled, cues.size // burst), np.arange(cues.size, dtype=np.int32) + 1)
    thinned = planner.thin(auto, make_profile(), fps=30.0, cc_budget_hz=budget)
    assert len(thinned) == len(auto) and thinned.max_rate() == budget
    paced = R.schedule(thinned, latency_s=0.033, early_bias_s=R.EARLY_BIAS_S, cc_budget_hz=budget)
    clock = Clock()
    transport = StampedTransport(clock, msgs_per_param)
    session = Session(transport, cc_budget_hz=budget, clock=clock, sleep=clock.sleep)
    rig = R.Rig(session=session, capture=None, playout=None)
    assert R.drive(rig, paced, duration=3.0, clock=clock, sleep=clock.sleep) == len(auto)
    sent = np.array([t for t, _, _ in transport.sent])
    assert sent == pytest.approx(paced.times, abs=1e-9), "the session never holds a write back"
    order = np.argsort(auto.times, kind="stable")
    lead = auto.times[order] - sent
    assert (
        lead.min() >= 0.033 + R.EARLY_BIAS_S - 1e-9 and lead.max() <= 0.033 + R.EARLY_BIAS_S + burst / budget
    )


def test_drive_wakes_only_when_a_parameter_is_due():
    """The host does nothing per frame: simultaneous writes are one batch, not one call each."""
    auto = Automation.concat([Automation.of([0.0, 0.4], 2, [500, 900]), Automation.of([0.0], 5, 300)])
    rig = R.Rig(session=FakeSession(), capture=None, playout=None)
    clock = Clock()
    written = R.drive(rig, auto, duration=1.0, clock=clock, sleep=clock.sleep)
    assert written == 3
    assert rig.session.calls == [
        {2: 500 / R.PARAM_MAX, 5: 300 / R.PARAM_MAX},
        {2: 900 / R.PARAM_MAX},
    ]
    assert clock.slept == pytest.approx([0.4, 0.6]), "it sleeps to the next event, then out the take"


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

    module("syncsummoner.device.transport", Transport=types.SimpleNamespace(open=lambda stall: "transport"))
    module("syncsummoner.device.session", Session=record("session"))

    def capture(*args, **kwargs):
        built["capture"] = (args, kwargs)
        return types.SimpleNamespace(open=lambda: "open capture")

    module("syncsummoner.device.capture", Capture=capture)
    module("syncsummoner.device.playout", ClipPlayer=record("playout"))
    module("syncsummoner.device.link", Link=record("link"))
    rig = R.open_rig(CONFIG)
    assert rig.session == (("transport",), {"cc_budget_hz": CONFIG.cc_budget_hz})
    assert built["capture"][1]["width"] == 32 and built["playout"][1]["height"] == 24
    assert rig.capture == "open capture", "the rig hands back an opened capture"
    assert rig.link == ((), {}), "a real rig always gets link control, defaulted by the device layer"
    R.open_rig(dataclasses.replace(CONFIG, source_host="pi@rig"))
    assert built["link"][0] == ("pi@rig",) and built["playout"][0] == ("pi@rig",)


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
    rig = R.open_rig(CONFIG)
    assert opened == ["open"] and isinstance(rig.capture, FakeCapture)


def test_the_rig_session_is_a_named_format():
    """Playout writes the Pi framebuffer, which is 1920x1080: a 720p session showed nothing."""
    rig = R.RenderConfig.for_format("1080p30")
    assert (rig.width, rig.height, rig.fps) == (1920, 1080, 30.0)


@pytest.mark.parametrize("lead", [0, 3])
def test_write_timecoded_stamps_every_frame_after_a_wrapped_lead_in(monkeypatch, tmp_path, lead):
    clip = [np.full((4, 4, 3), 0.1 * (i + 1), np.float32) for i in range(6)]
    monkeypatch.setattr(
        "syncsummoner.compose.features.read_frames",
        lambda p, max_frames=None: iter([(CONFIG.fps, f) for f in clip]),
    )
    written = []

    class Pipe:
        """Encoder stdin standing in for ffmpeg, keeping the frames it is handed."""

        def write(self, data):
            """Write."""
            written.append(np.frombuffer(bytes(data), np.uint8))

        def close(self):
            """Close."""

    monkeypatch.setattr(
        R.subprocess,
        "Popen",
        lambda *a, **k: types.SimpleNamespace(
            stdin=Pipe(), stderr=io.BytesIO(b""), wait=lambda: 0, returncode=0
        ),
    )
    total = R.write_timecoded("clip.mkv", tmp_path / "tc.mkv", config=CONFIG, lead=lead)
    assert total == 6 and len(written) == 6 + lead
    shaped = [f.reshape(CONFIG.height, CONFIG.width, 3).astype(np.float32) / 255.0 for f in written]
    codes = [read_timecode(f, bits=CONFIG.bits, strip_px=CONFIG.strip_px) for f in shaped]
    assert codes == [65536 - lead + i for i in range(lead)] + list(range(6))
    body = [f[: -CONFIG.strip_px].mean() for f in shaped]
    assert body[: lead + 1] == [body[lead]] * (lead + 1) and body[lead] < body[lead + 1]


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
    assert R.open_rig(CONFIG, capture=False).capture is None and not opened
    R.open_rig(CONFIG, capture=True)
    assert opened == [True], "the old path still gets one when it asks"


@pytest.mark.parametrize("corrupted", [False, True])
def test_a_locked_take_loads_its_program_in_the_lead_in_and_keeps_its_lock(monkeypatch, tmp_path, corrupted):
    """Passthru first, then recording and player; the program loads inside, the lock saved once stopped."""

    def verify():
        rig.log.append(("verify",))
        if corrupted:
            raise CcMapError("device CC map differs from the CCs sent (P12 CC0)")

    rig = locked_pass(monkeypatch, transport=rig_transport(verify_cc_map=verify), load_s={"Teletext": 0.1})
    out = tmp_path / "take.mkv"
    with pytest.raises(CcMapError) if corrupted else contextlib.nullcontext():
        assert render_locked(rig, out) == PROBED
    steps = [
        "upload",
        "load",
        "record",
        "play",
        "load",
        "params",
        "drive",
        "stop",
        "recorded",
        "flips",
        "verify",
    ]
    assert rig.steps == steps + ([] if corrupted else ["inspect"])
    assert [step[1:] for step in rig.log if step[0] == "load"] == [("Passthru", "link"), ("Teletext", "link")]
    began = next(step[2] for step in rig.log if step[0] == "play")
    assert rig.log[6] == ("drive", pytest.approx(began + 8 / 30), 0.2), "the pass is timed from clip frame 0"
    assert rig.clock.now == pytest.approx(began + 8 / 30 + 0.2 + LOCKED.tail_s)
    record = np.load(L.lock_path(out))
    assert (int(record["hold"]), int(record["lead"]), int(record["count"])) == (2, 8, 6)
    assert np.array_equal(record["flips"], rig.take.flips)
    if not corrupted:
        picked = rig.take.picked()
        assert np.load(L.picked_path(out)).tolist() == picked.tolist()
        assert rig.log[-1] == ("inspect", pytest.approx(picked[0] / 60))


def test_a_passthru_take_loads_nothing_inside_its_lead_in(monkeypatch, tmp_path):
    rig = locked_pass(monkeypatch, clock=Clock(5.0))
    out = tmp_path / "take.mkv"
    loaded = R.play_take(rig.rig, rig, out, program=None, seconds=0.2, config=LOCKED)
    assert rig.steps == ["load", "record", "play", "stop", "recorded", "flips"] and loaded == 5.0
    assert L.lock_path(out).is_file()


@pytest.mark.parametrize(
    "load_s,skew_s,match,locked",
    [
        (0.3, 0.0, r"Teletext: loading took 0.3s, past the 0.3s lead-in; raise lead_in_s", False),
        (0.2, -0.1, r"Teletext: clip frame 0 reached the card 0.03s before the load ended", True),
    ],
    ids=["load-overran", "clip-before-load"],
)
def test_a_load_not_done_before_clip_frame_0_is_a_lead_in_error(
    monkeypatch, tmp_path, load_s, skew_s, match, locked
):
    rig = locked_pass(monkeypatch, load_s={"Teletext": load_s}, skew_s=skew_s)
    out = tmp_path / "take.mkv"
    with pytest.raises(R.LeadInError, match=match):
        render_locked(rig, out)
    assert ("drive" in rig.steps) is locked
    assert L.lock_path(out).is_file() is locked and "inspect" not in rig.steps


def test_lead_frames_rounds_the_lead_in_to_whole_frames():
    assert [R.lead_frames(dataclasses.replace(LOCKED, lead_in_s=s)) for s in (0.99, 1.01, 0.0)] == [30, 30, 0]


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


def test_the_timecoded_source_is_built_once_for_every_cut_pass(monkeypatch, tmp_path):
    """Burning it per pass would cost minutes a program for a clip that never changes."""
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
    assert built == [tmp_path / "tc.mkv"] and passes == [True, True]


def test_an_excerpt_takes_its_footage_from_where_it_starts(monkeypatch):
    """A windowed score plays the footage at its own start, not from frame zero."""
    clip = [np.full((CONFIG.height, CONFIG.width, 3), i / 20, np.float32) for i in range(20)]
    monkeypatch.setattr(
        "syncsummoner.compose.features.read_frames",
        lambda p, max_frames=None: iter([(CONFIG.fps, f) for f in clip]),
    )
    _, stream = R.source_stream("clip.mkv", CONFIG, seconds=0.5, start=1.0)
    assert [round(float(f.mean()), 3) for f in stream] == [round(float(f.mean()), 3) for f in clip[10:15]]


def test_a_dead_encoder_is_reported_as_what_it_said(monkeypatch, tmp_path):
    """Writing to a dead encoder raises a broken pipe, which says nothing about why."""
    monkeypatch.setattr(
        "syncsummoner.compose.features.read_frames",
        lambda p, max_frames=None: iter([(CONFIG.fps, np.zeros((4, 4, 3), np.float32))] * 3),
    )

    class Pipe:
        """Encoder stdin that has already gone away."""

        def write(self, data):
            """Write."""
            del data
            raise BrokenPipeError

        def close(self):
            """Close."""

    monkeypatch.setattr(
        R.subprocess,
        "Popen",
        lambda *a, **k: types.SimpleNamespace(
            stdin=Pipe(),
            stderr=io.BytesIO(b"Error opening output timecoded.mkv: Permission denied\n"),
            wait=lambda: 1,
            returncode=1,
        ),
    )
    with pytest.raises(RuntimeError, match="Permission denied"):
        R.write_timecoded("clip.mkv", tmp_path / "tc.mkv", config=CONFIG)


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
    """A five minute pass with one gesture would otherwise print nothing for five minutes."""
    auto = Automation.of([0.0], 2, 500)
    rig = R.Rig(session=FakeSession(), capture=None, playout=None)
    clock = Clock()
    assert R.drive(rig, auto, duration=5.0, clock=clock, sleep=clock.sleep, tick_s=1.0) == 1
    assert clock.slept == pytest.approx([1.0] * 5), "it wakes each tick, not once at the end"


def test_ticks_never_outrun_the_event_they_wait_for():
    """Subdividing a wait must not move when a parameter lands."""
    auto = Automation.of([0.0, 2.5], 2, [100, 900])
    rig = R.Rig(session=FakeSession(), capture=None, playout=None)
    clock = Clock()
    R.drive(rig, auto, duration=2.5, clock=clock, sleep=clock.sleep, tick_s=1.0)
    assert clock.slept == pytest.approx([1.0, 1.0, 0.5]), "the last tick is short, landing on the event"


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

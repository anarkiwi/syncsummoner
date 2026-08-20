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

from syncsummoner.compose import render as R
from syncsummoner.compose.score import GestureInstance, Layer, Score, Section
from syncsummoner.compose.vocabulary import Automation

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


class FakeLink:
    """Stands in for the source HDMI link; a real one blanks the Pi framebuffer."""


class FakeRig(R.Rig):
    """Loopback rig: the capture returns whatever was last played out, optionally dropping frames."""

    def __init__(self, *, drop_every=0, no_signal=False, link=None):
        session = FakeSession()
        super().__init__(session=session, capture=self, playout=self, link=link)
        self.last = None
        self.n = 0
        self.drop_every = drop_every
        self.no_signal = no_signal

    def show(self, frame):
        self.last = frame

    def read(self):
        self.n += 1
        if self.no_signal:
            return None
        if self.drop_every and self.n % self.drop_every == 0:
            return None
        return self.last


def source_stack(n=12, h=24, w=32):
    rng = np.random.default_rng(0)
    return rng.uniform(0.1, 0.9, (n, h, w, 3)).astype(np.float32)


def simple_score(programs=("glitch",)):
    return Score(
        seed=3,
        duration=1.2,
        fps=CONFIG.fps,
        sections=[Section(0.0, 1.2, "A")],
        layers=[
            Layer(p, i, [GestureInstance("ramp", 1.0, 0.6, "motion_rate", 0.8, (), 1)])
            for i, p in enumerate(programs)
        ],
    )


@pytest.mark.parametrize("index", [0, 1, 7, 1234, 65535])
def test_timecode_round_trip(index):
    frame = np.zeros((24, 32, 3), np.float32)
    burned = R.burn_timecode(frame, index, bits=16, strip_px=4)
    assert R.read_timecode(burned, bits=16, strip_px=4) == index
    assert R.crop_strip(burned, strip_px=4).shape == (20, 32, 3)
    assert frame.max() == 0.0


def test_timecode_absent_or_washed_out_reads_none():
    flat = np.full((24, 32, 3), 0.5, np.float32)
    assert R.read_timecode(flat, bits=16, strip_px=4) is None
    burned = R.burn_timecode(np.zeros((24, 32, 3), np.float32), 5, bits=16, strip_px=4)
    assert R.read_timecode(burned * 0.1, bits=16, strip_px=4) is None


def test_schedule_biases_visuals_early():
    auto = Automation.of([1.0], 2, 500)
    shifted = R.schedule(auto, latency_s=0.08, early_bias_s=R.EARLY_BIAS_S)
    assert shifted.times[0] == pytest.approx(1.0 - 0.08 - 0.015)
    assert 0.010 <= R.EARLY_BIAS_S <= 0.020
    assert R.schedule(auto).times[0] < auto.times[0]


class Clock:
    """Monotonic clock advanced only by the sleeps handed to it."""

    def __init__(self):
        self.now = 0.0
        self.slept = []

    def __call__(self):
        return self.now

    def sleep(self, seconds):
        """Sleep."""
        self.slept.append(seconds)
        self.now += seconds


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

    module("syncsummoner.device.transport", Transport=types.SimpleNamespace(open=lambda: "transport"))
    module("syncsummoner.device.session", Session=record("session"))

    def capture(*args, **kwargs):
        built["capture"] = (args, kwargs)
        return types.SimpleNamespace(open=lambda: "open capture")

    module("syncsummoner.device.capture", Capture=capture)
    module("syncsummoner.device.playout", Playout=record("playout"))
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
    monkeypatch.setattr(playout_mod, "Playout", lambda *a, **k: None)
    monkeypatch.setattr(session_mod, "Session", lambda *a, **k: None)
    monkeypatch.setattr(transport_mod.Transport, "open", staticmethod(lambda *a, **k: None))
    rig = R.open_rig(CONFIG)
    assert opened == ["open"] and isinstance(rig.capture, FakeCapture)


def test_the_rig_session_is_a_named_format():
    """Playout writes the Pi framebuffer, which is 1920x1080: a 720p session showed nothing."""
    rig = R.RenderConfig.for_format("1080p30")
    assert (rig.width, rig.height, rig.fps) == (1920, 1080, 30.0)


def test_write_timecoded_stamps_every_frame(monkeypatch, tmp_path):
    clip = [np.full((4, 4, 3), 0.5, np.float32) for _ in range(6)]
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
    total = R.write_timecoded("clip.mkv", tmp_path / "tc.mkv", config=CONFIG)
    assert total == len(written) == 6
    shaped = [f.reshape(CONFIG.height, CONFIG.width, 3).astype(np.float32) / 255.0 for f in written]
    codes = [R.read_timecode(f, bits=CONFIG.bits, strip_px=CONFIG.strip_px) for f in shaped]
    assert codes == list(range(6)), "each frame carries its own index"


def test_reading_a_timecode_only_touches_the_strip():
    """Converting the whole frame to float64 to read eight rows cost 12ms a frame."""
    frame = np.random.random((1080, 1920, 3)).astype(np.float32)
    stamped = R.burn_timecode(frame, 4242, bits=16, strip_px=8)
    assert R.read_timecode(stamped, bits=16, strip_px=8) == 4242
    start = time.perf_counter()
    for _ in range(20):
        R.read_timecode(stamped, bits=16, strip_px=8)
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


def test_assembling_takes_each_span_from_its_own_pass(monkeypatch, tmp_path):
    seen = {}

    def run(argv, check=False, capture_output=False):
        del check, capture_output
        seen["argv"] = argv
        return types.SimpleNamespace(returncode=0, stderr=b"")

    monkeypatch.setattr(R.subprocess, "run", run)
    cuts = [R.Cut(0.0, 5.0, "X"), R.Cut(5.0, 9.0, "Y"), R.Cut(9.0, 12.0, "X")]
    R.assemble(cuts, {"X": tmp_path / "x.mkv", "Y": tmp_path / "y.mkv"}, tmp_path / "out.mkv")
    argv = seen["argv"]
    assert argv.count("-i") == 3, "one input per cut, from that cut's take"
    assert argv[argv.index("-ss") + 1] == "0.000" and "concat=n=3" in " ".join(argv)
    assert str(tmp_path / "y.mkv") in argv, "the middle span comes from the other program"


def test_render_cuts_runs_one_pass_per_program_and_refuses_a_blank_one(monkeypatch, tmp_path):
    """A blank pass must not be spliced into a demo as though it were footage."""
    calls = []

    def fake_render(score, source, out, **kwargs):
        del source, kwargs
        calls.append((score.layers[0].program, out))
        blank = score.layers[0].program == "Bad"
        return R.TakeReport(frames=10, distinct=1 if blank else 9, blank=10 if blank else 0, luma=0.4)

    monkeypatch.setattr(R, "assemble", lambda *a, **k: None)
    monkeypatch.setattr(R, "picture_start", lambda path, **kw: 0.0)
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


def test_a_pass_waits_for_the_picture_on_the_configs_budget(monkeypatch, tmp_path):
    """The load blacks the output out, so the take may not start until it is back."""
    order = []

    class Session:
        """Session recording the load only."""

        def load_program(self, program, link=None):
            """Load program."""
            del link
            order.append(("load", program))

        def working_point(self, info):
            """Working point."""
            del info
            return {}

        def set_params(self, values):
            """Set params."""
            del values

    class Player:
        """Playout stand-in."""

        def upload(self, path):
            """Upload."""

        def playing(self, fps):
            """Playing."""
            del fps
            return contextlib.nullcontext()

    class Rec:
        """Recorder stand-in recording when the take itself started."""

        def recording(self, path, seconds=None):
            """Recording."""
            del path, seconds
            order.append(("record",))
            return contextlib.nullcontext()

    def fake_settle(recorder, **kwargs):
        del recorder
        order.append(("settle", kwargs))
        return R.TakeReport(4, 4, 0, 0.4)

    monkeypatch.setattr(R, "settle", fake_settle)
    monkeypatch.setattr(R, "drive", lambda *a, **k: 0)
    monkeypatch.setattr(R, "inspect_take", lambda *a, **k: R.TakeReport(4, 4, 0, 0.4))
    monkeypatch.setattr(R, "plan_automation", lambda *a, **k: {})
    config = dataclasses.replace(CONFIG, settle_s=7.5, probe_path=str(tmp_path / "probe.mkv"))
    score = Score(
        seed=1, duration=1.0, fps=CONFIG.fps, layers=[Layer(index=0, program="Teletext", gestures=[])]
    )
    R.render_played(
        score,
        "clip.mkv",
        tmp_path / "out.mkv",
        profiles={},
        rig=R.Rig(
            session=Session(),
            capture=None,
            playout=Player(),
            transport=types.SimpleNamespace(program_info=lambda: None),
        ),
        config=config,
        scratch=tmp_path / "tc.mkv",
        prepared=True,
        recorder=Rec(),
    )
    assert [step[0] for step in order] == ["load", "settle", "record"]
    assert order[1][1] == {
        "program": "Teletext",
        "timeout_s": 7.5,
        "probe_path": config.probe_path,
    }


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
    monkeypatch.setattr(playout_mod, "Playout", lambda *a, **k: None)
    monkeypatch.setattr(playout_mod, "ClipPlayer", lambda *a, **k: None)
    monkeypatch.setattr(session_mod, "Session", lambda *a, **k: None)
    monkeypatch.setattr(transport_mod.Transport, "open", staticmethod(lambda *a, **k: None))
    assert R.open_rig(CONFIG, capture=False).capture is None and not opened
    R.open_rig(CONFIG, capture=True)
    assert opened == [True], "the old path still gets one when it asks"


def test_a_pass_puts_the_program_at_its_working_point(monkeypatch, tmp_path):
    """Parking drives every parameter to zero, and a program at zero renders black."""
    calls = []

    class Session:
        """Session recording load and parameter writes in order."""

        def load_program(self, program, link=None):
            """Load program."""
            del link
            calls.append(("load", program))

        def working_point(self, info):
            """Working point."""
            del info
            return {1: 0.5, 12: 1.0}

        def set_params(self, values):
            """Set params."""
            calls.append(("params", dict(values)))

    class Player:
        """Playout stand-in."""

        def upload(self, path):
            """Upload."""

        def playing(self, fps):
            """Playing."""
            del fps
            return contextlib.nullcontext()

    rig = R.Rig(
        session=Session(),
        capture=None,
        playout=Player(),
        transport=types.SimpleNamespace(program_info=lambda: None),
    )
    monkeypatch.setattr(R, "settle", lambda *a, **k: None)
    monkeypatch.setattr(R, "drive", lambda *a, **k: 0)
    monkeypatch.setattr(R, "inspect_take", lambda *a, **k: R.TakeReport(4, 4, 0, 0.4))
    monkeypatch.setattr(R, "plan_automation", lambda *a, **k: {})

    class Rec:
        """Recorder stand-in."""

        def recording(self, path, seconds=None):
            """Recording."""
            del path, seconds
            return contextlib.nullcontext()

    score = Score(
        seed=1,
        bpm=120.0,
        duration=1.0,
        fps=CONFIG.fps,
        layers=[Layer(index=0, program="Teletext", gestures=[])],
    )
    R.render_played(
        score,
        "clip.mkv",
        tmp_path / "out.mkv",
        profiles={},
        rig=rig,
        config=CONFIG,
        scratch=tmp_path / "tc.mkv",
        prepared=True,
        recorder=Rec(),
    )
    assert calls[0] == ("load", "Teletext")
    assert calls[1][0] == "params" and calls[1][1] == {1: 0.5, 12: 1.0}, "the load is followed by a picture"


def test_each_cut_program_is_driven_by_its_own_evolved_layer(monkeypatch, tmp_path):
    """The planner evolves a layer per program; a cut that reused the first would throw that away."""
    seen = {}

    def fake_render(score, source, out, **kwargs):
        del source, out, kwargs
        layer = score.layers[0]
        seen[layer.program] = [g.gesture for g in layer.gestures]
        return R.TakeReport(frames=10, distinct=9, blank=0, luma=0.4)

    monkeypatch.setattr(R, "assemble", lambda *a, **k: None)
    monkeypatch.setattr(R, "picture_start", lambda path, **kw: 0.0)
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
    monkeypatch.setattr(R, "assemble", lambda *a, **k: None)
    monkeypatch.setattr(R, "picture_start", lambda path, **kw: 0.0)
    monkeypatch.setattr(R, "write_timecoded", lambda source, out, **kw: built.append(out) or 1)

    def fake_render(score, source, out, **kwargs):
        del score, source, out
        passes.append(kwargs["prepared"])
        return R.TakeReport(frames=10, distinct=9, blank=0, luma=0.4)

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
    """A windowed score played from frame zero would show the wrong 30 seconds."""
    clip = [np.full((CONFIG.height, CONFIG.width, 3), i / 20, np.float32) for i in range(20)]
    monkeypatch.setattr(
        "syncsummoner.compose.features.read_frames",
        lambda p, max_frames=None: iter([(CONFIG.fps, f) for f in clip]),
    )
    _, stream = R.source_stream("clip.mkv", CONFIG, seconds=0.5, start=1.0)
    assert [round(float(f.mean()), 3) for f in stream] == [round(float(f.mean()), 3) for f in clip[10:15]]


def test_the_take_gets_the_clip_from_its_first_frame(monkeypatch, tmp_path):
    """A load costs 12 to 19s, so a clip started before it has already run out by the take."""
    order = []

    class Player:
        """ClipPlayer stand-in recording when playback starts and stops."""

        def upload(self, path):
            """Upload."""
            order.append(f"upload:{path}")
            return 1

        @contextlib.contextmanager
        def playing(self, *, fps):
            """Play for the duration of the block."""
            del fps
            order.append("play")
            yield self
            order.append("stop")

    class Session:
        """Session recording the load only."""

        def load_program(self, program, link=None):
            """Load program."""
            del link
            order.append(f"load:{program}")

        def set_params(self, values):
            """Set params."""

        def working_point(self, info):
            """Working point."""
            del info
            return {}

    class Rec:
        """Recorder stand-in."""

        def recording(self, path, seconds=None):
            """Recording."""
            del path, seconds
            return contextlib.nullcontext()

    monkeypatch.setattr(R, "settle", lambda *a, **k: order.append("settle"))
    monkeypatch.setattr(R, "inspect_take", lambda *a, **k: R.TakeReport(300, 280, 0, 0.4))
    monkeypatch.setattr(R, "drive", lambda *a, **k: order.append("drive") or 0)
    monkeypatch.setattr(R, "plan_automation", lambda *a, **k: {})
    rig = R.Rig(
        session=Session(),
        capture=None,
        playout=Player(),
        transport=types.SimpleNamespace(program_info=lambda: None),
    )
    score = Score(duration=10.0, sections=[Section(0.0, 10.0, "A")], layers=[Layer("Passthru", 0, [])])
    R.render_played(
        score,
        "clip.mkv",
        tmp_path / "take.mkv",
        profiles={},
        rig=rig,
        config=CONFIG,
        prepared=True,
        recorder=Rec(),
    )
    assert order == [
        "upload:timecoded.mkv",
        "play",
        "load:Passthru",
        "settle",
        "stop",
        "play",
        "drive",
        "stop",
    ]


def test_picture_start_finds_where_the_played_clip_begins(monkeypatch, tmp_path):
    """A take opens on the previous play's last frame, which carries its own timecode."""
    lead, config = 5, dataclasses.replace(CONFIG, lead_s=1.0)
    stale = R.burn_timecode(
        np.full((CONFIG.strip_px, CONFIG.width, 3), 0.5, np.float32),
        299,
        bits=CONFIG.bits,
        strip_px=CONFIG.strip_px,
    )
    strips = [stale for _ in range(lead)]
    strips += [
        R.burn_timecode(
            np.full((config.strip_px, config.width, 3), 0.5, np.float32),
            i,
            bits=config.bits,
            strip_px=config.strip_px,
        )
        for i in range(4)
    ]
    raw = np.clip(np.stack(strips), 0, 1).astype(np.float32)
    monkeypatch.setattr(
        R.subprocess,
        "run",
        lambda argv, **kw: types.SimpleNamespace(stdout=(raw * 255).astype(np.uint8).tobytes()),
    )
    assert R.picture_start(tmp_path / "take.mkv", config=config) == pytest.approx(lead / config.fps)


def test_picture_start_survives_a_misread_strip(monkeypatch, tmp_path):
    """One strip decoding to a wrong low value would otherwise move the whole clip."""
    config = dataclasses.replace(CONFIG, lead_s=1.0)
    plate = np.full((config.strip_px, config.width, 3), 0.5, np.float32)
    codes = [299, 299, 1, 0, 1, 2, 3, 4, 5, 6]
    raw = np.stack([R.burn_timecode(plate, c, bits=config.bits, strip_px=config.strip_px) for c in codes])
    monkeypatch.setattr(
        R.subprocess,
        "run",
        lambda argv, **kw: types.SimpleNamespace(
            stdout=(np.clip(raw, 0, 1) * 255).astype(np.uint8).tobytes()
        ),
    )
    assert R.picture_start(tmp_path / "take.mkv", config=config) == pytest.approx(3 / config.fps)


def test_picture_start_of_a_take_with_no_strip_is_zero(monkeypatch, tmp_path):
    monkeypatch.setattr(R.subprocess, "run", lambda argv, **kw: types.SimpleNamespace(stdout=b""))
    assert R.picture_start(tmp_path / "take.mkv", config=CONFIG) == 0.0


def test_a_cut_is_taken_past_each_takes_own_lead_in(monkeypatch, tmp_path):
    seen = []
    monkeypatch.setattr(
        R.subprocess,
        "run",
        lambda argv, **kw: seen.append(argv) or types.SimpleNamespace(returncode=0, stderr=b""),
    )
    cuts = [R.Cut(0.0, 5.0, "Derez"), R.Cut(5.0, 9.0, "Lorenz")]
    R.assemble(
        cuts,
        {"Derez": tmp_path / "d.mkv", "Lorenz": tmp_path / "l.mkv"},
        tmp_path / "out.mkv",
        starts={"Derez": 2.0},
    )
    argv = seen[0]
    assert argv[argv.index("-ss") + 1] == "2.000" and argv[argv.index("-to") + 1] == "7.000"
    assert argv[argv.index(str(tmp_path / "l.mkv")) - 3 : argv.index(str(tmp_path / "l.mkv"))] == [
        "-to",
        "9.000",
        "-i",
    ]


def test_picture_start_reads_the_take_at_session_rate(monkeypatch, tmp_path):
    """The card returns its own frame rate; an index only means a time at the session's."""
    seen = {}

    def fake_run(argv, **kw):
        del kw
        seen["argv"] = argv
        return types.SimpleNamespace(stdout=b"")

    monkeypatch.setattr(R.subprocess, "run", fake_run)
    R.picture_start(tmp_path / "take.mkv", config=CONFIG)
    assert seen["argv"][seen["argv"].index("-vf") + 1].startswith(f"fps={CONFIG.fps},crop=")


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

"""Render: timecode strip, design 4.4 scheduling, alignment, passes and compositing."""

# pylint: disable=missing-function-docstring,protected-access

import dataclasses
import sys
import types

import numpy as np
import pytest

from syncsummoner.compose import render as R
from syncsummoner.compose.score import GestureInstance, Layer, Score, Section
from syncsummoner.compose.vocabulary import Automation

from . import make_profile

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


def test_align_holds_the_last_good_frame_over_gaps():
    shape = (2, 2, 3)
    frames = {0: np.zeros(shape, np.float32), 2: np.ones(shape, np.float32)}
    out = R.align(frames, 4, shape)
    assert out[1].max() == 0.0 and out[3].min() == 1.0
    assert R.align({1: np.ones(shape, np.float32)}, 3, shape)[0].min() == 1.0
    assert R.align({}, 2, shape).shape == (2,) + shape


def test_composite_modes():
    a = np.zeros((1, 2, 2, 3), np.float32)
    b = np.ones((1, 2, 2, 3), np.float32)
    assert R.composite([a, b], mode="mean").mean() == pytest.approx(0.5)
    assert R.composite([a, b], mode="max").mean() == pytest.approx(1.0)
    assert R.composite([a, a], mode="screen").mean() == pytest.approx(0.0)


def test_play_pass_loads_once_and_returns_cropped_aligned_frames():
    rig = FakeRig()
    frames = source_stack()
    auto = Automation.of(np.array([0.05, 0.35, 0.65]), 2, np.array([100, 500, 900]))
    out = R.play_pass(rig, frames, auto, program="glitch", config=CONFIG)
    assert rig.session.programs == ["glitch"]
    assert out.shape == (12, 24 - CONFIG.strip_px, 32, 3)
    assert np.allclose(out, frames[:, CONFIG.strip_px :])
    assert [list(c) for c in rig.session.calls] == [[2], [2], [2]]
    assert all(0.0 <= v <= 1.0 for call in rig.session.calls for v in call.values())


def test_play_pass_drops_the_source_link_across_the_program_change():
    link = FakeLink()
    rig = FakeRig(link=link)
    R.play_pass(rig, source_stack(n=2), Automation.empty(), program="glitch", config=CONFIG)
    assert rig.session.links == [link], "every load must hold the source link down"


def test_play_pass_survives_dropped_and_dead_captures():
    frames = source_stack(n=8)
    out = R.play_pass(FakeRig(drop_every=2), frames, Automation.empty(), program="p", config=CONFIG)
    assert np.allclose(out[0], frames[0, CONFIG.strip_px :])
    assert np.allclose(out[1], out[0])
    dead = R.play_pass(FakeRig(no_signal=True), frames, Automation.empty(), program="p", config=CONFIG)
    assert dead.shape[0] == 8 and dead.max() == 0.0


def test_render_writes_one_pass_through_a_sink():
    written = {}
    R.render(
        simple_score(),
        source_stack(),
        "out.mov",
        profiles={"glitch": make_profile()},
        rig=FakeRig(),
        config=CONFIG,
        sink=lambda path, frames, fps: written.update(path=path, frames=frames, fps=fps),
    )
    assert written["path"] == "out.mov" and written["fps"] == CONFIG.fps
    assert written["frames"].shape == (12, 20, 32, 3)


def test_render_refeeds_and_composites_two_passes():
    rig = FakeRig()
    written = {}
    R.render(
        simple_score(("glitch", "blur")),
        source_stack(),
        "out.mov",
        passes=2,
        profiles={"glitch": make_profile("glitch"), "blur": make_profile("blur")},
        rig=rig,
        config=CONFIG,
        sink=lambda path, frames, fps: written.update(frames=frames),
        mode="mean",
    )
    assert rig.session.programs == ["glitch", "blur"]
    assert written["frames"].shape == (12, 24 - 2 * CONFIG.strip_px, 32, 3)


def test_render_requires_profiles_and_layers():
    with pytest.raises(ValueError):
        R.render(simple_score(), source_stack(), "o.mov", rig=FakeRig(), config=CONFIG)
    with pytest.raises(ValueError):
        R.render(
            Score(fps=CONFIG.fps),
            source_stack(),
            "o.mov",
            profiles={},
            rig=FakeRig(),
            config=CONFIG,
            sink=lambda *a: None,
        )
    with pytest.raises(ValueError):
        R.audition(simple_score(), source_stack(), rig=FakeRig(), config=CONFIG)


def test_audition_truncates_and_downscales():
    out = R.audition(
        simple_score(),
        source_stack(n=40),
        seconds=0.5,
        scale=0.5,
        profiles={"glitch": make_profile()},
        rig=FakeRig(),
        config=CONFIG,
    )
    assert out.shape == (5, 12 - CONFIG.strip_px, 16, 3)


def test_audition_composites_multiple_passes():
    out = R.audition(
        simple_score(("glitch", "blur")),
        source_stack(n=10),
        seconds=0.6,
        scale=1.0,
        passes=2,
        profiles={"glitch": make_profile("glitch"), "blur": make_profile("blur")},
        rig=FakeRig(),
        config=CONFIG,
        mode="max",
    )
    assert out.shape[0] == 6


def test_config_formats_and_source_loading(monkeypatch):
    ntsc = R.RenderConfig.for_format("ntsc")
    assert (ntsc.width, ntsc.height) == (720, 480)
    assert R.RenderConfig.for_format("720p60", latency_s=0.1).fps == 60.0
    monkeypatch.setattr(
        "syncsummoner.compose.features.read_frames",
        lambda p, max_frames=None: iter([(CONFIG.fps, np.zeros((4, 4, 3), np.float32))] * 2),
    )
    assert R._source_frames("clip.mp4", CONFIG).shape == (2, CONFIG.height, CONFIG.width, 3)


def test_write_video_uses_the_bgr_boundary(monkeypatch, tmp_path):
    import cv2

    seen = []

    class FakeWriter:
        """Minimal cv2.VideoWriter stand-in."""

        def __init__(self, *args):
            seen.append(args)

        def write(self, frame):
            seen.append(frame)

        def release(self):
            seen.append("released")

    monkeypatch.setattr(cv2, "VideoWriter", FakeWriter)
    monkeypatch.setattr(cv2, "VideoWriter_fourcc", lambda *a: 0, raising=False)
    frames = np.zeros((2, 4, 6, 3), np.float32)
    frames[..., 0] = 1.0
    R.write_video(tmp_path / "x.mov", frames, 30.0)
    assert seen[0][2] == 30.0 and seen[0][3] == (6, 4)
    assert seen[1][0, 0].tolist() == [0, 0, 255]
    assert seen[-1] == "released"


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


def test_enforce_safety_repairs_an_unsafe_pass():
    frames = np.zeros((90, 32, 32, 3), np.float32)
    frames[::2] = 1.0
    from syncsummoner import aesthetics

    assert aesthetics.flash_risk(R.enforce_safety(frames, fps=30.0), fps=30.0).safe


def test_enforce_safety_passes_safe_output_through():
    frames = np.full((30, 32, 32, 3), 0.5, np.float32)
    assert R.enforce_safety(frames, fps=30.0) is frames


def test_enforce_safety_raises_when_mitigation_declined():
    frames = np.zeros((90, 32, 32, 3), np.float32)
    frames[::2] = 1.0
    with pytest.raises(R.UnsafeOutputError, match="flashes/s"):
        R.enforce_safety(frames, fps=30.0, mitigate=False)


def test_source_frames_are_resampled_to_the_session_rate(monkeypatch):
    """A pass shows one source frame per session frame, so a 25fps clip would run fast."""
    clip = [np.full((4, 4, 3), i, np.float32) for i in range(5)]
    monkeypatch.setattr(
        "syncsummoner.compose.features.read_frames",
        lambda p, max_frames=None: iter([(25.0, f) for f in clip]),
    )
    config = R.RenderConfig(fps=50.0)
    got = R._source_frames("clip.mkv", config)
    assert got.shape[0] == 10, "half the rate means every frame is held twice"
    middle = (got.shape[1] // 2, got.shape[2] // 2)
    assert [round(float(f[middle][0])) for f in got] == [0, 0, 1, 1, 2, 2, 3, 3, 4, 4]


def test_source_frames_decode_only_the_span_the_pass_uses(monkeypatch):
    """A three minute clip is 21GB decoded whole; an excerpt must not pay for it."""
    decoded = []

    def frames(path, max_frames=None):
        del path, max_frames
        for i in range(10_000):
            decoded.append(i)
            yield 25.0, np.zeros((4, 4, 3), np.float32)

    monkeypatch.setattr("syncsummoner.compose.features.read_frames", frames)
    got = R._source_frames("clip.mkv", R.RenderConfig(fps=25.0), seconds=2.0)
    assert got.shape[0] == 50 and len(decoded) == 50, "it stopped at the budget"


def test_source_frames_of_an_empty_clip(monkeypatch):
    monkeypatch.setattr("syncsummoner.compose.features.read_frames", lambda p, max_frames=None: iter([]))
    assert R._source_frames("clip.mkv", CONFIG).shape[0] == 0


def test_source_frames_are_fitted_into_the_session_raster(monkeypatch):
    """Playout takes the session geometry and nothing else; a 694x576 clip failed at the pipe."""
    clip = [np.full((576, 694, 3), 0.5, np.float32)]
    monkeypatch.setattr(
        "syncsummoner.compose.features.read_frames", lambda p, max_frames=None: iter([(10.0, clip[0])])
    )
    got = R._source_frames("clip.mkv", CONFIG)
    assert got.shape == (1, CONFIG.height, CONFIG.width, 3)
    filled = got[0].any(axis=2)
    assert filled.all(axis=0).any(), "a full column of picture, so the height is filled"
    assert not filled[:, 0].any() and not filled[:, -1].any(), "pillarboxed, not stretched"


def test_a_source_already_at_the_session_raster_is_untouched(monkeypatch):
    frames = [np.full((CONFIG.height, CONFIG.width, 3), 0.25, np.float32)]
    monkeypatch.setattr(
        "syncsummoner.compose.features.read_frames",
        lambda p, max_frames=None: iter([(CONFIG.fps, frames[0])]),
    )
    got = R._source_frames("clip.mkv", CONFIG)
    assert got.shape == (1, CONFIG.height, CONFIG.width, 3) and float(got.min()) == 0.25


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


def test_render_closes_the_capture_it_opened(monkeypatch):
    closed = []

    class Cap:
        """Capture that records being closed."""

        def close(self):
            """Close."""
            closed.append(True)

    rig = R.Rig(session=None, capture=Cap(), playout=None)
    monkeypatch.setattr(R, "open_rig", lambda config: rig)
    monkeypatch.setattr(R, "_source_frames", lambda *a, **k: np.zeros((1, 4, 4, 3), np.float32))
    monkeypatch.setattr(R, "_passes", lambda *a, **k: [])
    score = Score(seed=1, bpm=120.0, duration=1.0, fps=CONFIG.fps)
    with pytest.raises(ValueError, match="no layers"):
        R.render(score, "clip.mkv", "out.mkv", profiles={}, config=CONFIG)
    assert closed == [True], "a rig it opened is a rig it closes, even when the render fails"


def test_the_rig_session_is_a_named_format():
    """Playout writes the Pi framebuffer, which is 1920x1080: a 720p session showed nothing."""
    rig = R.RenderConfig.for_format("1080p30")
    assert (rig.width, rig.height, rig.fps) == (1920, 1080, 30.0)


def test_a_frame_whose_strip_will_not_decode_is_still_kept():
    """Kaledos decoded 4 of 40 strips; discarding the rest rendered the take as black."""
    frames = {i: np.full((4, 4, 3), i / 10.0, np.float32) for i in range(10)}
    stamps = {4: 2, 6: 4, 8: 6}  # a lag of two, measured from the three that decoded
    placed = R._placed(frames, stamps, 10)
    assert placed[2] is frames[4] and placed[6] is frames[8], "a stamp wins where it exists"
    assert placed[0] is frames[2] and placed[7] is frames[9], "the rest land by arrival less the lag"
    assert len(placed) == 8, "the two that predate the lag fall off the front"


def test_placing_with_no_decoded_stamp_at_all_keeps_arrival_order():
    frames = {i: np.full((2, 2, 3), i, np.float32) for i in range(4)}
    placed = R._placed(frames, {}, 4)
    assert [int(placed[i][0, 0, 0]) for i in sorted(placed)] == [0, 1, 2, 3]


def test_a_pass_that_decodes_nothing_still_returns_the_pictures(monkeypatch):
    """The captured stream is the performance; naming its frames is what alignment adds."""
    monkeypatch.setattr(R, "read_timecode", lambda *a, **k: None)
    shown, grabbed = [], []

    class Playout:
        """Playout stand-in."""

        def show(self, frame):
            """Show."""
            shown.append(frame)

    class Capture:
        """Capture handing back a distinguishable frame each grab."""

        def read(self):
            """Read."""
            got = np.full((8, 8, 3), 0.1 * (len(grabbed) + 1), np.float32)
            grabbed.append(got)
            return got

    class Session:
        """Session stand-in."""

        def load_program(self, program, link=None):
            """Load program."""

        def set_params(self, values):
            """Set params."""

    rig = R.Rig(session=Session(), capture=Capture(), playout=Playout())
    source = np.zeros((5, 8, 8, 3), np.float32)
    out = R.play_pass(rig, source, Automation.empty(), program="Kaledos", config=CONFIG)
    assert out.shape[0] == 5 and float(out.max()) > 0, "the take survives an unreadable strip"


class Emitted:
    """Collects what a sink wrote, standing in for the encoder."""

    def __init__(self):
        self.frames = []

    def write(self, frame):
        """Write."""
        self.frames.append(np.array(frame, copy=True))


def test_a_sink_emits_in_order_and_holds_over_a_gap():
    out = Emitted()
    sink = R.FrameSink(out.write, fps=30.0, window=4)
    for index in (0, 1, 3):  # 2 never arrives
        sink.add(index, np.full((4, 4, 3), (index + 1) / 10.0, np.float32))
    sink.close(5)
    assert len(out.frames) == 5, "the take is as long as it was asked for"
    values = [round(float(f[0, 0, 0]), 1) for f in out.frames]
    assert values[:2] == [0.1, 0.2] and values[2] == 0.2, "the gap holds the last good frame"


def test_a_sink_never_holds_the_whole_take():
    out = Emitted()
    sink = R.FrameSink(out.write, fps=30.0, window=8)
    for index in range(64):
        sink.add(index, np.full((4, 4, 3), 0.5, np.float32))
        assert len(sink.pending) <= 9 and len(sink.buffer) <= 8, "bounded no matter how long the take"
    sink.close(64)
    assert len(out.frames) == 64


def test_render_stream_writes_a_take_it_never_holds(monkeypatch, tmp_path):
    """180s at 1080p30 is 134GB as a stack; one pass has no need of it."""
    clip = [np.full((4, 4, 3), i / 20.0, np.float32) for i in range(12)]
    monkeypatch.setattr(
        "syncsummoner.compose.features.read_frames",
        lambda p, max_frames=None: iter([(CONFIG.fps, f) for f in clip]),
    )
    grabbed = []

    class Capture:
        """Capture handing back the frame it was shown."""

        def __init__(self):
            self.shown = None

        def read(self):
            """Read."""
            grabbed.append(self.shown)
            return self.shown

        def close(self):
            """Close."""

    capture = Capture()

    class Playout:
        """Playout that feeds the capture what it shows."""

        def show(self, frame):
            """Show."""
            capture.shown = frame

    class Session:
        """Session stand-in."""

        def load_program(self, program, link=None):
            """Load program."""

        def set_params(self, values):
            """Set params."""

    out = Emitted()
    rig = R.Rig(session=Session(), capture=capture, playout=Playout())
    score = Score(
        seed=1,
        bpm=120.0,
        duration=12 / CONFIG.fps,
        fps=CONFIG.fps,
        layers=[Layer(index=0, program="glitch", gestures=[])],
    )
    R.render_stream(
        score,
        "clip.mkv",
        tmp_path / "out.mkv",
        profiles={"glitch": make_profile("glitch")},
        rig=rig,
        config=CONFIG,
        sink=R.FrameSink(out.write, fps=CONFIG.fps, window=4),
    )
    assert len(out.frames) == 12 and grabbed, "every frame shown was captured and written"


def test_capture_pass_follows_the_playback_it_is_watching():
    """The source plays itself, so position comes from the frame the card hands over."""
    applied = []

    class Session:
        """Session recording the parameter writes a pass makes."""

        def load_program(self, program, link=None):
            """Load program."""

        def set_params(self, values):
            """Set params."""
            applied.append(dict(values))

    stamps = iter([0, 1, 2, 3, 4, 5])

    class Capture:
        """Capture handing back frames already carrying a timecode."""

        def read(self):
            """Read."""
            index = next(stamps, None)
            if index is None:
                return None
            return R.burn_timecode(
                np.full((16, 16, 3), 0.5, np.float32), index, bits=CONFIG.bits, strip_px=CONFIG.strip_px
            )

    out = Emitted()
    auto = Automation(times=np.array([2 / CONFIG.fps]), indices=np.array([3]), values=np.array([512.0]))
    seen = R.capture_pass(
        R.Rig(session=Session(), capture=Capture(), playout=None),
        auto,
        program="glitch",
        config=CONFIG,
        sink=R.FrameSink(out.write, fps=CONFIG.fps, window=3),
        total=6,
    )
    assert seen == 6 and len(out.frames) == 6
    assert applied and 3 in applied[0], "the automation fired against the decoded position"


def test_write_timecoded_stamps_every_frame(monkeypatch, tmp_path):
    clip = [np.full((4, 4, 3), 0.5, np.float32) for _ in range(6)]
    monkeypatch.setattr(
        "syncsummoner.compose.features.read_frames",
        lambda p, max_frames=None: iter([(CONFIG.fps, f) for f in clip]),
    )
    written = []
    monkeypatch.setattr(R.VideoSink, "write", lambda self, frame: written.append(frame))
    monkeypatch.setattr(R.VideoSink, "close", lambda self: None)
    total = R.write_timecoded("clip.mkv", tmp_path / "tc.mkv", config=CONFIG)
    assert total == len(written) == 6
    codes = [R.read_timecode(f, bits=CONFIG.bits, strip_px=CONFIG.strip_px) for f in written]
    assert codes == list(range(6)), "each frame carries its own index"

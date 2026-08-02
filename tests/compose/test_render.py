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
        "syncsummoner.compose.features.read_video", lambda p: (np.zeros((2, 4, 4, 3), np.float32), 30.0)
    )
    assert R._source_frames("clip.mp4", CONFIG).shape == (2, 4, 4, 3)


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
    module("syncsummoner.device.capture", Capture=record("capture"))
    module("syncsummoner.device.playout", Playout=record("playout"))
    module("syncsummoner.device.link", Link=record("link"))
    rig = R.open_rig(CONFIG)
    assert rig.session == (("transport",), {"cc_budget_hz": CONFIG.cc_budget_hz})
    assert built["capture"][1]["width"] == 32 and built["playout"][1]["height"] == 24
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

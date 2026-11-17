"""Doctor: every rig check against fakes built from what the real rig reported."""

# pylint: disable=missing-function-docstring,redefined-outer-name

import shutil
import subprocess
from contextlib import contextmanager

import numpy as np
import pytest
from pyvmancer import LatencyReport, PortBusyError, RateResult, VmancerError
from pyvmancer.latency import DEFAULT_RATES
from pyvmancer.midi import MidiController

from syncsummoner import cli
from syncsummoner.device import doctor as dr
from syncsummoner.device.doctor import Doctor, Format
from syncsummoner.device.host import HostCommandError
from syncsummoner.device.recorder import MJPEG, RecorderError
from syncsummoner.device.transport import ProgramInfo, Transport
from syncsummoner.probe.patterns import generate
from syncsummoner.program.vmprog import ControlMode
from tests.declared import config, decl, toggle

from .conftest import (
    COLORBARS_DECLARED,
    COLORBARS_INFO,
    DECLARED,
    FakeClock,
    FakeDevice,
    FakeMidiPort,
    FakeShell,
)

FMT = Format("1080p30", 1920, 1080, 30.0)
KMSPRINT = """Connector 0 (35) HDMI-A-1 (connected)
  Encoder 0 (34) TMDS
    Crtc 3 (102) 1920x1080@30.00 74.250 1920/88/44/148/- 1080/4/5/36/- 30 (30.00) P|D
      Plane 3 (91) fb-id: 722 (crtcs: 3) 0,0 1920x1080 -> 0,0 1920x1080 (XR24 RG16)
        FB 722 1920x1080 RG16"""
USB3_FORMATS = """Compressed:       mjpeg :          Motion-JPEG : 1920x1080 2560x1440
Raw       :        nv12 :           Y/UV 4:2:0 : 640x480 1280x720 1920x1080
Raw       :     yuyv422 :           YUYV 4:2:2 : 1920x1080
[in#0 @ 0x5d] Error opening input: Immediate exit requested"""
USB2_FORMATS = """Raw       :     yuyv422 :           YUYV 4:2:2 : 640x480 720x480 720x576
Raw       :        nv12 :           Y/UV 4:2:0 : 640x480 720x576"""
VIDEO = {
    "source": "hdmi",
    "timing": "1080p30",
    "locked": True,
    "overridden": False,
    "analog": {"locked": False, "timing": "1080p30"},
    "hdmi": {"locked": True, "connected": False, "timing": "1080p30"},
    "output": {"timing": "1080p30", "hdmi_connected": True, "analog_configured": True},
    "hdmi_tx": {"de": "2FE90F", "vic": 34},
}
FPGA = {"state": "running", "configured": True, "program": "Passthru"}


class Pi:
    """Source host answering the doctor's one read with the sysfs values it holds."""

    def __init__(
        self, blank="0", blink="0", size="1920,1080", depth="16", kms=KMSPRINT, *, down=False, **drm
    ):
        self.values = [blank, blink, size, depth, drm.get("player", ""), drm.get("holders", "")]
        self.kms = kms
        self.down = down
        self.commands = []

    def __call__(self, command, data=None):
        del data
        self.commands.append(command)
        if self.down:
            raise HostCommandError("pi@videopi: `true` exited 255: Connection refused")
        if "cursor_blink" in command and command.startswith("echo 0"):
            self.values[1] = "0"
            return ""
        return "\n".join([*(f"={v}" for v in self.values), self.kms]).strip()


class RigShell(FakeShell):
    """Serial shell reporting the rig's firmware, FPGA and video status."""

    def __init__(self, video=None):
        super().__init__()
        self.video = VIDEO if video is None else video

    def version(self):
        return "1.0.1"

    def fpga_status(self):
        return dict(FPGA)

    def video_status(self):
        return self.video


class FakeLink:
    """Source link whose ``up`` unblanks the fake Pi."""

    def __init__(self, pi):
        self.pi = pi
        self.calls = []

    def up(self):
        self.calls.append("up")
        self.pi.values[0] = "0"
        return "On"

    def down(self):
        self.calls.append("down")
        return "Off"

    @contextmanager
    def quiet(self):
        self.down()
        yield self
        self.up()


class FakePlayout:
    """Keeps the last frame shown, the rate it was held at, and whether it is still up."""

    def __init__(self):
        self.frame = self.fps = None
        self.playing = False

    def show(self, frame, *, fps):
        self.frame, self.fps, self.playing = frame, fps, True

    def stop(self):
        self.playing = False


class FakeRecorder:
    """Records nothing; ``fail`` probes raise as a card with no mode yet does."""

    mode = MJPEG
    ffmpeg = "ffmpeg"

    def __init__(self, fail=0):
        self.fail = fail
        self.takes = 0

    @contextmanager
    def recording(self, path, *, seconds=None):
        del path, seconds
        self.takes += 1
        if self.takes <= self.fail:
            raise RecorderError("the recorder exited 1 for /dev/video0: Device or resource busy")
        yield self


def sysfs(tmp_path, speed, hub="5000"):
    """A sysfs tree with the card's video function behind its own hub."""
    hub_dir = tmp_path / "devices/usb4/4-2/4-2.4"
    function = hub_dir / "4-2.4.3"
    interface = function / "4-2.4.3:1.0"
    interface.mkdir(parents=True)
    (hub_dir / "speed").write_text(f"{hub}\n")
    (function / "speed").write_text(f"{speed}\n")
    node = tmp_path / "class/video4linux/video0"
    node.mkdir(parents=True)
    (node / "device").symlink_to(interface)
    return tmp_path


@pytest.fixture
def rig(tmp_path):
    """Doctor against a healthy rig, every endpoint faked; tweak the parts per test."""

    def build(*, pi=None, speed="5000", formats=USB3_FORMATS, shell=None, **kw):
        pi = Pi() if pi is None else pi
        clock = FakeClock()
        device = FakeDevice(RigShell() if shell is None else shell)
        player = FakePlayout()
        doc = Doctor(
            FMT,
            runner=pi,
            sysfs=sysfs(tmp_path, speed),
            formats=(lambda node: formats) if isinstance(formats, str) else formats,
            open_transport=lambda: Transport(device, DECLARED),
            playout=player,
            link=FakeLink(pi),
            recorder=kw.pop("recorder", FakeRecorder()),
            rng=np.random.default_rng(7),
            sleep=clock.sleep,
            clock=clock,
            **kw,
        )
        doc.pi, doc.dev, doc.player = pi, device, player
        return doc

    return build


def by_check(findings):
    return {f.check: f for f in findings}


def test_healthy_rig_passes_every_check(rig):
    doc = rig()
    found = by_check(doc.run())
    assert all(f.ok and not f.fix for f in found.values())
    assert {"ssh.reachable", "source.mode", "capture.usb_speed", "capture.mode", "fpga.running"} <= set(found)
    assert found["device.format"].detail == "output timing 1080p30, session 1080p30"
    assert found["capture.mode"].detail == "1920x1080 offered as mjpeg, nv12, yuyv422"
    assert doc.dev.closed


def test_blanked_framebuffer_and_cursor_blink_fail_then_fix(rig):
    doc = rig(pi=Pi(blank="4", blink="1"))
    found = by_check(doc.checks())
    assert not found["source.fb_unblanked"].ok and "--fix" in found["source.fb_unblanked"].fix
    assert found["source.fb_unblanked"].detail.endswith("reads 4")
    assert not found["source.cursor_blink"].ok
    fixed = by_check(doc.run(fix=True))
    assert fixed["source.fb_unblanked"].ok and fixed["source.cursor_blink"].ok
    assert doc._link.calls == ["up"]  # pylint: disable=protected-access
    assert any("sudo -n tee /sys/class/graphics/fbcon/cursor_blink" in c for c in doc.pi.commands)


def test_fix_on_a_healthy_rig_applies_nothing(rig):
    doc = rig()
    assert doc.fix(doc.checks()) == []


def test_mode_and_geometry_must_match_the_format(rig):
    kms = KMSPRINT.replace("1920x1080@30.00", "1280x720@60.00")
    found = by_check(rig(pi=Pi(size="1280,720", depth="32", kms=kms)).checks())
    assert not found["source.mode"].ok and "1920x1080@30" in found["source.mode"].fix
    assert not found["source.fb_geometry"].ok
    assert found["source.fb_geometry"].detail == "framebuffer 1280x720 at 32 bpp"


@pytest.mark.parametrize(
    ("player", "holders", "ok", "detail"),
    [
        ("", "", True, "held by nothing besides the player"),
        ("4242", "4242", True, "held by nothing besides the player"),
        ("4242", "4242 977 31", False, "held by 31, 977 besides the player"),
        ("", "977", False, "held by 977 besides the player"),
    ],
)
def test_only_the_player_may_hold_the_drm_card(rig, player, holders, ok, detail):
    found = by_check(rig(pi=Pi(player=player, holders=holders)).checks())["source.drm_free"]
    assert found.ok is ok and found.detail == f"{dr.DRM_CARD} {detail}"
    assert ("977" in found.fix) is not ok


def test_no_kmsprint_skips_the_mode_check(rig):
    found = by_check(rig(pi=Pi(kms="")).checks())
    assert "source.mode" not in found and found["source.fb_geometry"].ok


def test_unreachable_host_skips_host_checks_and_the_loop(rig):
    found = rig(pi=Pi(down=True)).checks(loop=True)
    names = [f.check for f in found]
    assert not by_check(found)["ssh.reachable"].ok
    assert not any(n.startswith("source.") for n in names)
    assert not by_check(found)["loop"].ok


def test_card_on_usb2_loses_mjpeg(rig):
    found = by_check(rig(speed="480", formats=USB2_FORMATS).checks())
    assert not found["capture.usb_speed"].ok and "USB 3" in found["capture.usb_speed"].fix
    assert found["capture.usb_speed"].detail == "/dev/video0 on a 480 Mbit/s link"
    assert not found["capture.mode"].ok
    assert found["capture.mode"].detail == "1920x1080 offered as nothing"


def test_the_cards_own_usb2_hub_is_not_read(tmp_path):
    assert dr.usb_speed(sysfs(tmp_path, "5000", hub="480"), "/dev/video0") == 5000


def test_missing_capture_node(rig, tmp_path):
    doc = rig()
    doc.capture = "/dev/video7"
    found = by_check(doc.checks())
    assert not found["capture.present"].ok and "capture.mode" not in found
    assert dr.usb_speed(tmp_path / "nowhere", "/dev/video0") is None


def test_missing_ffmpeg_fails_the_mode_check(rig):
    def absent(node):
        raise RecorderError(f"'ffmpeg' not found on PATH for {node}")

    found = by_check(rig(formats=absent).checks())
    assert found["capture.usb_speed"].ok and not found["capture.mode"].ok
    assert "not found" in found["capture.mode"].detail


def test_parse_formats_keeps_every_size():
    offered = dr.parse_formats(USB3_FORMATS)
    assert offered["mjpeg"] == {"1920x1080", "2560x1440"}
    assert "1920x1080" in offered["nv12"] and "Error" not in str(offered)


def test_output_held_at_ntsc_fails_the_format(rig):
    held = dict(VIDEO, timing="NTSC", overridden=True, output=dict(VIDEO["output"], timing="NTSC"))
    found = by_check(rig(shell=RigShell(held)).checks())
    assert not found["device.format"].ok and "Pref. Timing" in found["device.format"].fix
    assert not found["video.genlock"].ok


def test_a_device_left_on_another_input_is_flagged(rig):
    """Under Vid Route Mode Auto a dropped source link can leave the device on analog, black."""
    shell = RigShell()
    doc = rig(shell=shell)
    assert by_check(doc.checks())["device.input"].detail == "input hdmi, session hdmi"
    shell.video = dict(VIDEO, source="analog")
    found = by_check(doc.checks())
    assert not found["device.input"].ok and "Vid Route Mode" in found["device.input"].fix
    assert found["device.input"].detail == "input analog, session hdmi"


def test_vid_route_mode_is_checked_against_hdmi_in_by_default(rig):
    """Under Dual In the hdmi input still reads locked; only Vid Route Mode shows the black output."""
    shell = RigShell()
    doc = rig(shell=shell)
    assert by_check(doc.checks())["video.route"].detail == "Vid Route Mode 1 (HDMI In), expected 1 (HDMI In)"
    shell.routed = 4
    found = by_check(doc.checks())
    assert [f.check for f in found.values() if not f.ok] == ["video.route"]
    assert "Vid Route Mode to HDMI In" in found["video.route"].fix
    doc.route = 4
    assert by_check(doc.checks())["video.route"].ok, "the expected value is overridable"


def test_device_unreachable(rig):
    doc = rig()

    def refuse():
        raise OSError("no serial link is open")

    doc._open_transport = refuse  # pylint: disable=protected-access
    found = by_check(doc.checks(loop=True))
    assert not found["device.reachable"].ok and "dialout" in found["device.reachable"].fix
    assert found["ssh.reachable"].ok


def test_a_port_held_elsewhere_is_named_rather_than_read_as_unreachable(rig):
    """The shell serves one client, so a held tty is another process to close, not a cable."""
    doc = rig()

    def busy():
        raise PortBusyError("/dev/ttyACM0")

    doc._open_transport = busy  # pylint: disable=protected-access
    found = by_check(doc.checks())
    assert "device.reachable" not in found
    assert not found["device.port_free"].ok and "fuser /dev/ttyACM0" in found["device.port_free"].detail


def swap_rb(frame):
    return np.ascontiguousarray(frame[:, :, ::-1])


@pytest.fixture
def captured(monkeypatch):
    """Serve the pushed frame back through ``shape`` in place of decoding a take."""
    state = {"shape": lambda frame: frame}

    def install(doc, shape=None):
        state["shape"] = shape or state["shape"]
        monkeypatch.setattr(
            dr, "last_frame", lambda path, width, height, ffmpeg: state["shape"](doc.player.frame)
        )

    return install


def test_loop_passes_on_a_clean_chain(rig, captured):
    doc = rig(recorder=FakeRecorder(fail=2))
    captured(doc)
    found = by_check(doc.run(loop=True))
    assert found["loop.signal"].ok and found["loop.index"].ok and found["loop.bars"].ok
    assert found["loop.bars"].detail == "bars read white yellow cyan green magenta red blue"
    assert doc.dev.shell.loads == ["Passthru"]
    assert doc.player.fps == FMT.fps and not doc.player.playing
    assert doc._link.calls[:2] == ["down", "up"]  # pylint: disable=protected-access


def test_loop_catches_a_red_blue_swap_at_any_gain(rig, captured):
    doc = rig()
    captured(doc, lambda frame: 0.4 * swap_rb(frame) + 0.05)
    found = by_check(doc.loop(Transport(doc.dev, DECLARED)))
    assert found["loop.index"].ok
    assert not found["loop.bars"].ok
    assert found["loop.bars"].detail == "bars read white cyan yellow green magenta blue red"


def test_loop_reports_a_stale_frame(rig, captured):
    doc = rig()
    other = generate("smpte_bars", width=FMT.width, height=FMT.height, rng=None)
    captured(doc, lambda frame: dr.with_state_index(other, 3) if frame is not None else None)
    doc._rng = np.random.default_rng(1)  # pylint: disable=protected-access
    found = by_check(doc.loop(Transport(doc.dev, DECLARED)))
    assert found["loop.signal"].ok and not found["loop.index"].ok


@pytest.mark.parametrize(
    ("transmitter", "detail"),
    [
        ({"de": "000000", "vic": 0}, "the device is not transmitting 1080p30"),
        (None, "the device reports no transmitter state"),
    ],
)
def test_loop_judges_the_signal_by_the_transmitter_not_the_picture(rig, captured, transmitter, detail):
    """A dead transmitter fails the loop even while the card hands back a structured frame."""
    video = {k: v for k, v in VIDEO.items() if k != "hdmi_tx"}
    doc = rig(shell=RigShell(video if transmitter is None else dict(video, hdmi_tx=transmitter)))
    captured(doc)
    found = doc.loop(Transport(doc.dev, DECLARED))
    assert [f.check for f in found] == ["loop.signal"] and not found[0].ok
    assert found[0].detail == detail and "video status" in found[0].fix


def test_loop_takes_achromatic_high_contrast_as_picture(rig, captured):
    """Grey bars carry no chroma at all; with the transmitter up they are picture, only miscoloured."""
    doc = rig()
    captured(doc, lambda frame: np.repeat(frame.mean(axis=2, keepdims=True), 3, axis=2))
    found = by_check(doc.loop(Transport(doc.dev, DECLARED)))
    assert found["loop.signal"].ok and found["loop.signal"].detail == "picture captured"
    assert found["loop.index"].ok and not found["loop.bars"].ok


def test_loop_reports_a_uniform_field_as_no_picture(rig, captured):
    """Video black at 16 with the transmitter up is a program carrying nothing, not the bars."""
    doc = rig()
    captured(doc, lambda frame: np.full((FMT.height, FMT.width, 3), 16 / 255, np.float32))
    found = doc.loop(Transport(doc.dev, DECLARED))
    assert [f.check for f in found] == ["loop.signal"] and not found[0].ok
    assert found[0].detail == "the device outputs a uniform field" and "Passthru" in found[0].fix


def test_loop_with_nothing_captured(rig, captured):
    doc = rig(recorder=FakeRecorder(fail=10**6))
    captured(doc)
    found = doc.loop(Transport(doc.dev, DECLARED))
    assert found[0].detail == "nothing captured" and not found[0].ok


def test_bar_channels_is_gain_and_pedestal_free():
    bars = generate("smpte_bars", width=320, height=180, rng=None)
    want = dr.bar_channels(bars)
    assert np.array_equal(dr.bar_channels(0.3 * bars + 0.2), want)
    assert dr.bar_names(want) == ["white", "yellow", "cyan", "green", "magenta", "red", "blue"]


@pytest.mark.skipif(shutil.which("ffmpeg") is None, reason="ffmpeg is not installed")
def test_last_frame_puts_the_cards_chroma_exchange_back(tmp_path):
    width, height = 320, 180
    bars = generate("smpte_bars", width=width, height=height, rng=None)
    take = tmp_path / "take.mkv"
    raw = (bars * 255).astype(np.uint8).tobytes()
    argv = ["ffmpeg", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "rgb24"]
    argv += ["-s", f"{width}x{height}", "-r", "30", "-i", "-", "-vf", "swapuv", "-c:v", "mjpeg"]
    subprocess.run(argv + ["-q:v", "2", str(take)], input=raw * 30, check=True)
    frame = dr.last_frame(take, width, height)
    assert dr.bar_names(dr.bar_channels(frame)) == dr.bar_names(dr.bar_channels(bars))
    assert dr.last_frame(tmp_path / "missing.mkv", width, height) is None


def test_list_formats_reads_ffmpeg_stderr(monkeypatch):
    seen = {}

    def run(argv, **kwargs):
        seen["argv"], seen["kwargs"] = argv, kwargs
        return subprocess.CompletedProcess(argv, 251, "", USB3_FORMATS)

    monkeypatch.setattr(dr.subprocess, "run", run)
    monkeypatch.setattr(dr, "require_ffmpeg", lambda name: f"/usr/bin/{name}")
    assert dr.list_formats("/dev/video0") == USB3_FORMATS
    assert seen["argv"][-4:] == ["-list_formats", "all", "-i", "/dev/video0"]


def measured(lags=((5, 0.04), (80, 0.05)), *, single=0.033, timing="1080p30", restored=True):
    """A pyvmancer latency report: ``single`` s per write, and a sentinel lag per rate."""
    rates = tuple(RateResult(float(r), 2 * r, 4 * r, 2.0, lag) for r, lag in lags)
    period = 1 / 30 if timing == "1080p30" else None
    return LatencyReport((3,), timing, period, 0.004, np.full(10, single), rates, (40,), restored)


class Measure:
    """Stands in for ``measure_latency``, keeping what it was asked to measure."""

    def __init__(self, result=None):
        self.result = measured() if result is None else result
        self.calls = []

    def __call__(self, device, **kwargs):
        self.calls.append((device, kwargs))
        if isinstance(self.result, Exception):
            raise self.result
        return self.result


def test_latency_is_opt_in_and_measures_the_programs_first_knob(rig):
    measure = Measure()
    doc = rig(measure=measure)
    assert not {"midi.latency", "midi.backlog"} & set(by_check(doc.run()))
    assert not measure.calls
    found = by_check(doc.run(latency=True))
    assert found["midi.latency"].ok and found["midi.backlog"].ok and found["device.format"].ok
    assert "single write median 33.0 ms (0.99 frames)" in found["midi.latency"].detail
    device, kwargs = measure.calls[0]
    assert device is doc.dev and kwargs["params"] == (3,) and kwargs["rates"] == DEFAULT_RATES
    assert doc.dev.closed


def test_latency_backlog_names_the_fastest_clean_rate(rig):
    measure = Measure(measured(((20, 0.04), (40, 0.06), (80, 0.3))))
    found = by_check(
        rig(measure=measure, latency_params=(1, 2), latency_rates=(20, 40, 80)).checks(latency=True)
    )
    assert found["midi.latency"].ok and not found["midi.backlog"].ok
    assert found["midi.backlog"].fix.endswith("send at most 40 sets/s")
    assert measure.calls[0][1]["params"] == (1, 2) and measure.calls[0][1]["rates"] == (20, 40, 80)


def test_pyvmancer_refuses_a_toggle_before_writing(rig):
    doc = rig(latency_params=(7,))
    found = by_check(doc.checks(latency=True))
    assert not found["midi.latency"].ok and "toggles cannot carry a ramp" in found["midi.latency"].detail
    assert "--latency-params" in found["midi.latency"].fix and found["device.format"].ok
    assert not any(call[0] == "set_param" for call in doc.dev.calls) and doc.dev.closed


def test_a_modulated_parameter_is_reported_not_raised(rig):
    refusal = VmancerError("P[3] have modulation sources [7]; disable them first")
    doc = rig(measure=Measure(refusal))
    found = doc.latency(Transport(doc.dev, DECLARED))
    assert [(f.check, f.ok, f.detail) for f in found] == [("midi.latency", False, f"skipped: {refusal}")]


def test_latency_without_midi_or_timing_or_restore(rig):
    doc = rig(measure=Measure(measured(timing=None, restored=False)))
    doc.dev.has_midi = False
    assert (
        by_check(doc.latency(Transport(doc.dev, DECLARED)))["midi.latency"].detail
        == "skipped: no MIDI link is open"
    )
    doc.dev.has_midi = True
    found = by_check(doc.latency(Transport(doc.dev, DECLARED)))
    assert not found["midi.latency"].ok and "midi.backlog" not in found
    assert not found["midi.restored"].ok and found["midi.restored"].detail == "P[3] did not read back [40]"


def test_write_latency_times_single_writes_on_the_programs_first_knob(rig):
    measure = Measure(measured(single=0.031, restored=False))
    transport = Transport(rig().dev, DECLARED)
    assert dr.write_latency(transport, measure=measure) == pytest.approx(0.031)
    assert measure.calls[0][1]["params"] == (3,) and measure.calls[0][1]["rates"] == ()


@pytest.mark.parametrize(
    "result",
    [VmancerError("P[3] have modulation sources [7]"), ValueError("toggles"), measured(single=np.nan)],
)
def test_write_latency_is_none_when_refused_or_unconverged(rig, result):
    assert dr.write_latency(Transport(rig().dev, DECLARED), measure=Measure(result)) is None


def test_write_latency_needs_midi(rig):
    doc, measure = rig(), Measure()
    doc.dev.has_midi = False
    assert dr.write_latency(Transport(doc.dev, DECLARED), measure=measure) is None and not measure.calls


def test_latency_param_is_the_first_declared_continuous_knob():
    assert dr.latency_param(ProgramInfo.from_json(COLORBARS_INFO)) == 1
    assert dr.latency_param(ProgramInfo.from_json(COLORBARS_INFO, COLORBARS_DECLARED)) == 3
    toggles_only = config("Toggles", *(toggle(i) for i in range(7, 12)), decl(2, mode=ControlMode.STEPS_4))
    assert dr.latency_param(ProgramInfo.from_json(COLORBARS_INFO, toggles_only)) == 1
    faded = config("Faded", toggle(7), decl(12, "Blend"))
    assert dr.latency_param(ProgramInfo.from_json(COLORBARS_INFO, faded)) == 12


def test_cli_doctor_reports_json_and_exit_status(monkeypatch, rig, capsys):
    doc = rig(pi=Pi(blank="4"), measure=Measure())
    built = {}

    def make(fmt, **kwargs):
        built.update(kwargs, fmt=fmt)
        return doc

    monkeypatch.setattr(dr, "Doctor", make)
    assert cli.main(["device", "doctor", "--format", "1080p30", "--capture", "/dev/video0"]) == 1
    out = capsys.readouterr()
    assert '"ok": false' in out.out and "FAIL source.fb_unblanked" in out.err
    assert built["fmt"] == FMT and built["host"] == "pi@videopi"
    assert cli.main(["device", "doctor", "--fix", "--source-host", "pi@other"]) == 0
    assert built["host"] == "pi@other"
    assert built["latency_params"] is None and built["latency_rates"] is None
    argv = ["device", "doctor", "--latency", "--latency-params", "1,2", "--latency-rates", "5,40"]
    assert cli.main(argv) == 0
    assert built["latency_params"] == (1, 2) and built["latency_rates"] == (5.0, 40.0)
    assert cli.main(["device", "doctor", "--format", "4k"]) == 1
    assert "unknown format" in capsys.readouterr().err


def test_cli_doctor_fails_midi_cc_map_on_a_corrupted_map(monkeypatch, rig, capsys):
    """A slot that learned P1's CC follows P1, so every take looks like passthrough."""
    doc = rig()
    monkeypatch.setattr(dr, "Doctor", lambda fmt, **kwargs: doc)
    assert cli.main(["device", "doctor"]) == 0
    assert "midi.cc_map" in capsys.readouterr().err
    doc.dev.shell.live[11] = 0
    assert cli.main(["device", "doctor"]) == 1
    err = capsys.readouterr().err
    assert "FAIL midi.cc_map" in err and "P12 on 0/32, want 11/43" in err
    assert "vmancer cc-map --restore --yes" in err


def test_midi_cc_map_is_checked_against_the_controller_in_use(rig):
    doc = rig()
    doc.dev.midi = MidiController(FakeMidiPort(doc.dev.shell), cc_map=range(20, 32))
    found = by_check(doc.checks())
    assert not found["midi.cc_map"].ok and "P1 on 0/32, want 20/52" in found["midi.cc_map"].detail
    doc.dev.shell.live = list(range(20, 32))
    assert by_check(doc.checks())["midi.cc_map"].ok

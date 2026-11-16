"""Doctor: every rig check against fakes built from what the real rig reported."""

# pylint: disable=missing-function-docstring,redefined-outer-name

import shutil
import subprocess
from contextlib import contextmanager

import numpy as np
import pytest

from syncsummoner import cli
from syncsummoner.device import doctor as dr
from syncsummoner.device.doctor import Doctor, Format
from syncsummoner.device.host import HostCommandError
from syncsummoner.device.recorder import MJPEG, RecorderError
from syncsummoner.device.transport import Transport
from syncsummoner.probe.patterns import generate

from .conftest import FakeClock, FakeDevice, FakeShell

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

    def __init__(self, blank="0", blink="0", size="1920,1080", depth="16", kms=KMSPRINT, *, down=False):
        self.values = [blank, blink, size, depth]
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
    """Keeps the last frame shown."""

    def __init__(self):
        self.frame = None

    def show(self, frame):
        self.frame = frame


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
            open_transport=lambda: Transport(device),
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


def test_device_unreachable(rig):
    doc = rig()

    def refuse():
        raise OSError("no serial link is open")

    doc._open_transport = refuse  # pylint: disable=protected-access
    found = by_check(doc.checks(loop=True))
    assert not found["device.reachable"].ok and "dialout" in found["device.reachable"].fix
    assert found["ssh.reachable"].ok


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
    assert ("load", "Passthru", 0) in doc.dev.calls
    assert doc._link.calls[:2] == ["down", "up"]  # pylint: disable=protected-access


def test_loop_catches_a_red_blue_swap_at_any_gain(rig, captured):
    doc = rig()
    captured(doc, lambda frame: 0.4 * swap_rb(frame) + 0.05)
    found = by_check(doc.loop(Transport(doc.dev)))
    assert found["loop.index"].ok
    assert not found["loop.bars"].ok
    assert found["loop.bars"].detail == "bars read white cyan yellow green magenta blue red"


def test_loop_reports_a_stale_frame(rig, captured):
    doc = rig()
    other = generate("smpte_bars", width=FMT.width, height=FMT.height, rng=None)
    captured(doc, lambda frame: dr.with_state_index(other, 3) if frame is not None else None)
    doc._rng = np.random.default_rng(1)  # pylint: disable=protected-access
    found = by_check(doc.loop(Transport(doc.dev)))
    assert found["loop.signal"].ok and not found["loop.index"].ok


def test_loop_reports_the_no_signal_splash(rig, captured):
    doc = rig()
    splash = np.zeros((FMT.height, FMT.width, 3), np.float32)
    splash[400:600, 600:1300] = 1.0
    captured(doc, lambda frame: splash)
    found = doc.loop(Transport(doc.dev))
    assert [f.check for f in found] == ["loop.signal"] and not found[0].ok
    assert found[0].detail == "the card shows its No Signal splash"


def test_loop_reports_a_uniform_field_as_no_picture(rig, captured):
    """Video black at 16 is a link carrying nothing: not the splash, and not the bars."""
    doc = rig()
    captured(doc, lambda frame: np.full((FMT.height, FMT.width, 3), 16 / 255, np.float32))
    found = doc.loop(Transport(doc.dev))
    assert [f.check for f in found] == ["loop.signal"] and not found[0].ok
    assert found[0].detail == "the device outputs a uniform field" and "Passthru" in found[0].fix


def test_loop_with_nothing_captured(rig, captured):
    doc = rig(recorder=FakeRecorder(fail=10**6))
    captured(doc)
    found = doc.loop(Transport(doc.dev))
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


def test_cli_doctor_reports_json_and_exit_status(monkeypatch, rig, capsys):
    doc = rig(pi=Pi(blank="4"))
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
    assert cli.main(["device", "doctor", "--format", "4k"]) == 1
    assert "unknown format" in capsys.readouterr().err

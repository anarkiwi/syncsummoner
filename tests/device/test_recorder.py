"""Recorder: ffmpeg owns the capture, so nothing on the host competes with the rig."""

# pylint: disable=missing-function-docstring

import contextlib
import io
import subprocess
import threading
import types
from pathlib import Path

import numpy as np
import pytest
from pyvmancer.errors import DeviceGoneError, StallError

from syncsummoner.device import recorder as rec, v4l2rec
from syncsummoner.device.recorder import (
    FFV1,
    MJPEG,
    RAW_422,
    BlankTakeError,
    InputRouteError,
    RouteError,
    Recorder,
    RecorderError,
    SignalLostError,
    TakeReport,
    inspect_take,
    require_ffmpeg,
    require_input,
    require_route,
    rig_state,
    require_signal,
    settle,
)
from tests.transmitter import DEAD, UNREPORTED, Transmitter, at, blackout


def fake_popen(started, *, returncode=None, says=b""):
    """Popen stand-in recording argv, with a process that stays up unless told otherwise."""

    def make(argv, *, stderr=None, **kwargs):
        del kwargs
        started.append(argv)
        if says and stderr is not None:
            stderr.write(says)
        return types.SimpleNamespace(
            stdin=types.SimpleNamespace(write=lambda b: None, flush=lambda: None, close=lambda: None),
            poll=lambda: returncode,
            returncode=returncode,
            wait=lambda timeout=None: 0,
            kill=lambda: None,
        )

    return make


def fake_run(calls, stdout=b"", returncode=0):
    """subprocess.run stand-in recording argv and replaying one canned result."""

    def run(argv, **kwargs):
        del kwargs
        calls.append(argv)
        return types.SimpleNamespace(stdout=stdout, returncode=returncode, stderr=b"")

    return run


def flat_stream(levels):
    """Raw RGB frames at the inspection geometry, one uniform level each."""
    width, height = rec.INSPECT_SIZE
    return b"".join(bytes([int(level)]) * (width * height * 3) for level in levels)


class FakeRecorder:
    """Recorder stand-in for settle: every recording leaves a probe file behind."""

    ffmpeg = "ffmpeg"

    def __init__(self):
        self.probes = []

    @contextlib.contextmanager
    def recording(self, path, *, seconds=None, settle_s=1.5):
        del settle_s
        Path(path).write_bytes(b"probe")
        self.probes.append((str(path), seconds))
        yield None


def test_require_ffmpeg_resolves_a_binary_on_path(monkeypatch):
    monkeypatch.setattr(rec.shutil, "which", lambda name: f"/usr/bin/{name}")
    assert require_ffmpeg("ffmpeg") == "/usr/bin/ffmpeg"


def test_require_ffmpeg_names_the_missing_binary(monkeypatch):
    monkeypatch.setattr(rec.shutil, "which", lambda name: None)
    with pytest.raises(RecorderError, match="not-a-real-binary.*PATH"):
        require_ffmpeg("not-a-real-binary")


def option(argv, flag):
    return argv[argv.index(flag) + 1]


def test_the_recorder_runs_the_v4l2_recorder_asking_the_card_for_its_mode():
    argv = Recorder("/dev/video2", fps=29.97, mode=MJPEG, python="py").command("/tmp/take.mkv", seconds=12.5)
    assert argv[:3] == ["py", "-m", "syncsummoner.device.v4l2rec"]
    assert [option(argv, f) for f in ("--device", "--width", "--height", "--fps", "--input-format")] == [
        "/dev/video2",
        "1920",
        "1080",
        "29.97",
        "mjpeg",
    ]
    assert option(argv, "--seconds") == "12.500"
    tail = argv[argv.index("--") - 1 :]
    assert tail[:4] == [
        "/tmp/take.mkv",
        "--",
        "-c:v",
        "copy",
    ], "the card compresses; the host does not re-encode"
    assert (
        option(tail, "-flush_packets") == "1" and int(option(tail, "-cluster_time_limit")) > 0
    ), "readable live"


@pytest.mark.parametrize("mode", [RAW_422, FFV1])
def test_every_output_argument_of_a_mode_reaches_the_encoder_in_order(mode):
    argv = Recorder(mode=mode).command("/tmp/archive.mkv")
    assert option(argv, "--input-format") == "yuyv422" and "--seconds" not in argv
    assert argv[argv.index("--") + 1 :] == list(mode[1:])


def test_the_recorder_command_parses_back_to_what_was_asked():
    argv = Recorder("/dev/video1", width=1280, height=720, fps=60.0, mode=FFV1, ffmpeg="/opt/ffmpeg").command(
        "/tmp/a.mkv", seconds=2.0
    )
    args = v4l2rec.parse_args(argv[3:])
    assert (args.device, args.width, args.height, args.fps, args.seconds) == (
        "/dev/video1",
        1280,
        720,
        60.0,
        2.0,
    )
    assert (args.input_format, args.ffmpeg, args.out, args.out_args) == (
        "yuyv422",
        "/opt/ffmpeg",
        "/tmp/a.mkv",
        list(FFV1[1:]),
    )


def test_recording_stops_the_encoder_and_checks_it_wrote(tmp_path):
    started = []
    take = tmp_path / "take.mkv"
    recorder = Recorder(popen=fake_popen(started), sleep=lambda s: take.write_bytes(b"x" * 16))
    with recorder.recording(take, seconds=1.0):
        pass
    assert len(started) == 1


def test_a_recorder_that_dies_at_once_reports_what_ffmpeg_said(tmp_path):
    """A recorder that dies silently cannot be told apart from a rig gone dark."""
    popen = fake_popen([], returncode=1, says=b"/dev/video0: Device or resource busy")
    recorder = Recorder(popen=popen, sleep=lambda s: None)
    with pytest.raises(RecorderError, match="Device or resource busy"):
        with recorder.recording(tmp_path / "take.mkv"):
            pass


def test_a_recorder_that_says_nothing_still_names_the_device(tmp_path):
    recorder = Recorder(popen=fake_popen([], returncode=1), sleep=lambda s: None)
    with pytest.raises(RecorderError, match="no diagnostic"):
        with recorder.recording(tmp_path / "take.mkv"):
            pass


def test_a_recording_that_wrote_nothing_is_an_error(tmp_path):
    recorder = Recorder(popen=fake_popen([]), sleep=lambda s: None)
    with pytest.raises(RecorderError, match="wrote nothing"):
        with recorder.recording(tmp_path / "empty.mkv"):
            pass


def test_stopping_a_stuck_recorder_kills_it():
    killed = []
    proc = types.SimpleNamespace(
        stdin=types.SimpleNamespace(write=lambda b: None, flush=lambda: None, close=lambda: None),
        wait=lambda timeout=None: (
            (_ for _ in ()).throw(subprocess.TimeoutExpired("ffmpeg", 1)) if timeout else 0
        ),
        kill=lambda: killed.append(True),
    )
    assert Recorder().stop(proc, timeout_s=0.01) == 0 and killed == [True]


def test_stopping_a_recorder_whose_pipe_is_already_gone_still_waits_for_it():
    """A dead pipe is not a failed take: the exit status is still what the caller gets."""
    proc = types.SimpleNamespace(stdin=None, wait=lambda timeout=None: 3, kill=lambda: None)
    assert Recorder().stop(proc) == 3


def test_timestamps_and_frame_log_come_from_the_sidecar(tmp_path):
    take = tmp_path / "take.mkv"
    log = np.zeros(3, dtype=v4l2rec.FRAME)
    log["sequence"], log["t_ns"] = [7, 8, 10], [1_000_500_000_000, 1_000_533_000_000, 1_000_600_000_000]
    v4l2rec.sidecar(take).write_bytes(log.tobytes())
    stamps = Recorder().timestamps(take)
    assert stamps.dtype == np.float64 and stamps == pytest.approx([1000.5, 1000.533, 1000.6])
    assert Recorder().frame_log(take)["sequence"].tolist() == [7, 8, 10]


def test_inspection_point_samples_rgb_so_the_card_s_own_pixels_are_judged(monkeypatch, tmp_path):
    argv = []
    monkeypatch.setattr(rec.subprocess, "run", fake_run(argv, stdout=flat_stream([0, 90, 180, 255])))
    report = inspect_take(tmp_path / "take.mkv")
    assert report.frames == 4 and report.picture == 0, "uniform fields are dead at any brightness"
    assert report.luma == pytest.approx(np.mean([0, 90, 180, 255]) / 255.0, abs=1e-4)
    assert option(argv[0], "-pix_fmt") == "rgb24" and option(argv[0], "-ss") == "0.000"
    assert "flags=neighbor" in option(argv[0], "-vf")
    inspect_take(tmp_path / "take.mkv", start_s=2.5)
    assert option(argv[1], "-ss") == "2.500" and argv[1].index("-ss") < argv[1].index("-i")


def test_a_live_paused_picture_settles_but_is_not_a_usable_take():
    """A load waits for any picture; a take must also not be held still."""
    paused = TakeReport(frames=10, picture=10, spread=255, motion=-0.5, frozen=True, luma=0.1)
    assert paused.mostly_picture and not paused.usable and "UNUSABLE: frozen" in str(paused)
    assert TakeReport(frames=10, picture=5).mostly_picture is False, "half is not most"


def wait(recorder, transport, clock, *, timeout_s=40.0, probe_path, **kw):
    return settle(
        recorder,
        transport,
        program="Teletext",
        timing="1080p30",
        probe_timeout_s=timeout_s,
        probe_path=probe_path,
        clock=clock,
        sleep=clock.sleep,
        **kw,
    )


def test_settle_waits_out_a_load_blackout_before_recording(monkeypatch, tmp_path, clock):
    """Nothing is recorded while the transmitter is off; the first probe after it returns is judged."""
    recorder = FakeRecorder()
    polled = []

    class Watched(Transmitter):
        """Notes how many probes had been recorded at each poll."""

        def video_status(self):
            polled.append(len(recorder.probes))
            return super().video_status()

    held = TakeReport(frames=10, picture=10, frozen=True)
    monkeypatch.setattr(rec, "inspect_take", lambda path, ffmpeg="ffmpeg": held)
    probe = tmp_path / "probe.mkv"
    got = wait(recorder, Watched(*blackout(5)), clock, probe_path=probe, probe_s=0.5, poll_s=0.25)
    assert got is held and len(recorder.probes) == 1, "a still picture is back; it need not move"
    assert polled == [0] * 6 and clock.slept[:5] == [0.25] * 5
    assert recorder.probes[0] == (str(probe), 0.5)
    assert not probe.exists(), "the probe never outlives the wait it was recorded for"


def test_settle_returns_a_uniform_probe_for_the_caller_to_judge(monkeypatch, tmp_path, clock):
    """With the transmitter up a flat field is the program's own output, not a signal still to come."""
    flat = TakeReport(frames=6)
    monkeypatch.setattr(rec, "inspect_take", lambda path, ffmpeg="ffmpeg": flat)
    assert wait(FakeRecorder(), Transmitter(), clock, probe_path=tmp_path / "p.mkv") is flat


def test_settle_waits_out_a_blackout_of_any_length(monkeypatch, tmp_path, clock):
    """The transmitter coming back is the event; however long it takes, nothing gives up on it."""
    monkeypatch.setattr(rec, "inspect_take", lambda path, ffmpeg="ffmpeg": TakeReport(frames=6))
    transmitter = Transmitter(*blackout(2000))
    assert wait(FakeRecorder(), transmitter, clock, timeout_s=1.0, probe_path=tmp_path / "p.mkv").frames
    assert transmitter.polls == 2001


def test_settle_fails_at_once_when_the_device_leaves_usb(tmp_path, clock):
    """Losing the device is what ends the transmitter wait early, and the card is never recorded."""
    recorder, transmitter = FakeRecorder(), Transmitter(DEAD)
    transmitter.attached = (True, True, False)
    with pytest.raises(DeviceGoneError, match="waiting for Teletext to transmit 1080p30 .transmitter down"):
        wait(recorder, transmitter, clock, probe_path=tmp_path / "p.mkv")
    assert transmitter.polls == 3 and not recorder.probes


def test_settle_stalls_only_when_the_caller_opts_in(tmp_path, clock):
    """A transmitter whose state never changes trips the stall guard when one is set."""
    transmitter = Transmitter(at("720p60"))
    transmitter.stall, transmitter.clock = 10.0, clock
    with pytest.raises(StallError, match="nothing changed for 10.0s"):
        wait(FakeRecorder(), transmitter, clock, probe_path=tmp_path / "p.mkv", poll_s=1.0)


def test_settle_fails_on_firmware_that_reports_no_transmitter(tmp_path, clock):
    """The event cannot be seen without transmitter state, so that is an error at once with the status."""
    recorder = FakeRecorder()
    with pytest.raises(
        SignalLostError, match="the firmware reports no transmitter state after the load"
    ) as raised:
        wait(recorder, Transmitter(UNREPORTED), clock, probe_path=tmp_path / "p.mkv")
    assert '"hdmi_connected": true' in str(raised.value) and not recorder.probes


def test_require_input_names_the_input_the_device_fell_back_to():
    """A device left on another input fails the check; None skips it."""
    analog = dict(at("1080p30"), source="analog")
    assert require_input(Transmitter(), program="P", source="hdmi").input_source == "hdmi"
    assert require_input(Transmitter(analog), program="P", source=None).input_source == "analog"
    with pytest.raises(InputRouteError, match="P: the device reads input 'analog', the session feeds 'HDMI'"):
        require_input(Transmitter(analog), program="P", source="HDMI")


def test_settle_gives_up_on_a_card_that_delivers_no_frames(monkeypatch, tmp_path):
    monkeypatch.setattr(rec, "inspect_take", lambda path, ffmpeg="ffmpeg": TakeReport())
    ticks = iter([0.0, 0.0, 99.0])
    probe = tmp_path / "probe.mkv"
    with pytest.raises(
        BlankTakeError, match="transmitting 1080p30 but nothing captured within 1.0s.*0 frames"
    ):
        settle(
            FakeRecorder(),
            Transmitter(),
            program="Dead",
            timing="1080p30",
            probe_timeout_s=1.0,
            probe_path=probe,
            probe_s=0.0,
            clock=lambda: next(ticks),
            sleep=lambda s: None,
        )
    assert not probe.exists(), "a failed wait cleans up after itself as well"


def test_require_signal_passes_a_driving_transmitter_and_names_a_dropped_one():
    assert (
        require_signal(Transmitter(), program="P", timing="1080p30", when="then").output_timing == "1080p30"
    )
    with pytest.raises(SignalLostError, match="P: the transmitter is not driving 1080p30 as it ended"):
        require_signal(Transmitter(DEAD), program="P", timing="1080p30", when="as it ended")


def test_settle_waits_through_a_recorder_that_cannot_open_the_card_yet(tmp_path, monkeypatch, clock):
    """A load drops the source link, so the card has no mode to open until it relocks."""
    attempts = []

    def popen(argv, *, stderr=None, **kwargs):
        del argv, kwargs
        attempts.append(1)
        if len(attempts) < 3:
            if stderr is not None:
                stderr.write(b"/dev/video0: No such device")
            return types.SimpleNamespace(poll=lambda: 1, returncode=1, wait=lambda timeout=None: 1)
        (tmp_path / "probe.mkv").write_bytes(b"x")
        return types.SimpleNamespace(
            stdin=types.SimpleNamespace(write=lambda b: None, flush=lambda: None, close=lambda: None),
            poll=lambda: None,
            returncode=None,
            wait=lambda timeout=None: 0,
            kill=lambda: None,
        )

    live = TakeReport(frames=8, picture=8, frozen=False, luma=0.4)
    monkeypatch.setattr(rec, "inspect_take", lambda *a, **k: live)
    recorder = Recorder(popen=popen, sleep=clock.sleep)
    report = wait(recorder, Transmitter(), clock, probe_path=tmp_path / "probe.mkv")
    assert report is live and len(attempts) == 3


def test_settle_reports_the_recorder_error_when_the_card_never_opens(tmp_path, clock):
    """Giving up on a card that never opened must not read as a black picture."""
    popen = fake_popen([], returncode=1, says=b"/dev/video0: Device or resource busy")
    recorder = Recorder(popen=popen, sleep=clock.sleep)
    with pytest.raises(BlankTakeError, match="Device or resource busy"):
        wait(recorder, Transmitter(), clock, timeout_s=6.0, probe_path=tmp_path / "probe.mkv")


def test_the_recorder_is_asked_to_die_with_the_harness(tmp_path):
    """Killing the harness skips every finally, and the card stays held by nobody's child."""
    seen = {}

    def popen(argv, **kwargs):
        del argv
        seen.update(kwargs)
        (tmp_path / "take.mkv").write_bytes(b"x")
        return types.SimpleNamespace(
            stdin=types.SimpleNamespace(write=lambda b: None, flush=lambda: None, close=lambda: None),
            poll=lambda: None,
            returncode=None,
            wait=lambda timeout=None: 0,
            kill=lambda: None,
        )

    with Recorder(popen=popen, sleep=lambda s: None).recording(tmp_path / "take.mkv"):
        pass
    assert seen["preexec_fn"] is rec.die_with_parent


def test_asking_to_die_with_the_parent_is_survivable():
    """A kernel that will not take the request must not stop a recording."""
    rec.die_with_parent()


def test_rig_state_names_route_input_and_transmitter_and_never_raises():
    """A rig fault report must still be written when the device cannot be read."""
    assert (
        rig_state(Transmitter())
        == "Vid Route Mode 1 (HDMI In), input hdmi unlocked, transmitter vic 34 de 2FE90F"
    )

    class Gone(Transmitter):
        """A device that left USB."""

        def video_status(self):
            raise OSError("device left USB")

    assert rig_state(Gone()) == "rig state unread: OSError: device left USB"


def test_require_route_names_the_reading_and_the_panel_setting():
    """Only a mismatch raises; None skips reading the setting at all."""
    assert require_route(Transmitter(), program="P", route=1) == 1
    assert require_route(types.SimpleNamespace(), program="P", route=None) is None
    port = Transmitter()
    port.routed = 2
    with pytest.raises(RouteError, match=r"P: Vid Route Mode reads 2 \(Dual In\), the session needs 1 "):
        require_route(port, program="P", route=1)


def test_a_roll_asks_the_recorder_for_the_next_file_with_the_captures_both_share():
    said = []
    stdin = types.SimpleNamespace(write=said.append, flush=lambda: None)
    Recorder().roll(types.SimpleNamespace(stdin=stdin), "/t/take-01.mkv", 100, 250)
    assert said == [b"roll 100 250 /t/take-01.mkv\n"]
    v4l2rec.watch_stdin(io.BytesIO(said[0]), threading.Event(), lambda *a: said.append(a))
    assert said[-1] == (100, 250, "/t/take-01.mkv"), "the recorder reads back what was asked"

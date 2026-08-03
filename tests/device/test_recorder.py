"""Recorder: ffmpeg owns the capture, so nothing on the host competes with the rig."""

# pylint: disable=missing-function-docstring

import contextlib
import subprocess
import types
from pathlib import Path

import numpy as np
import pytest

from syncsummoner.device import recorder as rec
from syncsummoner.device.recorder import (
    FFV1,
    MJPEG,
    RAW_422,
    BlankTakeError,
    Recorder,
    RecorderError,
    TakeReport,
    inspect_take,
    settle,
)


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


def gray_stream(levels):
    """Raw gray frames at the inspection geometry, one constant level each."""
    stride = rec.INSPECT_WIDTH * (rec.INSPECT_WIDTH * 9 // 16)
    return b"".join(bytes([int(level)]) * stride for level in levels)


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


def test_the_recorder_asks_the_card_for_what_it_can_actually_deliver():
    """MJPEG measured 59.8fps against 25.8 raw and 7.7 through a per-frame loop."""
    argv = Recorder(width=1920, height=1080, fps=30.0, mode=MJPEG).command("/tmp/take.mkv", seconds=12.5)
    assert argv[argv.index("-input_format") + 1] == "mjpeg"
    assert argv[argv.index("-video_size") + 1] == "1920x1080"
    assert argv[argv.index("-c:v") + 1] == "copy", "the card compresses; the host does not re-encode"
    assert argv[argv.index("-t") + 1] == "12.500" and argv[-1] == "/tmp/take.mkv"


def test_an_archive_recording_keeps_the_cards_own_samples():
    argv = Recorder(mode=RAW_422).command("/tmp/raw.mkv")
    assert argv[argv.index("-input_format") + 1] == "yuyv422"
    assert argv[argv.index("-c:v") + 1] == "rawvideo" and "-t" not in argv


def test_every_output_argument_of_a_mode_reaches_the_encoder_in_order():
    """A lossless mode is a whole argument list, not a codec name."""
    argv = Recorder(mode=FFV1).command("/tmp/archive.mkv")
    assert argv[-len(FFV1) : -1] == list(FFV1[1:])
    assert argv[argv.index("-slices") + 1] == "4" and argv[-1] == "/tmp/archive.mkv"


def test_copyts_replaces_the_duration_rather_than_joining_it():
    """Under copyts ffmpeg reads -t against the card's clock and stops before it starts."""
    argv = Recorder(mode=FFV1, copyts=True).command("/tmp/archive.mkv", seconds=30.0)
    assert "-copyts" in argv and "-t" not in argv
    assert argv.index("-copyts") > argv.index("-i"), "an output flag, after the input it stamps"


def test_without_copyts_a_duration_still_bounds_the_take():
    argv = Recorder(mode=MJPEG).command("/tmp/take.mkv", seconds=4.0)
    assert argv[argv.index("-t") + 1] == "4.000" and "-copyts" not in argv


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


def test_timestamps_are_the_per_packet_presentation_times():
    """Under copyts these are CLOCK_MONOTONIC, which is what the host schedules on."""
    calls = []
    recorder = Recorder(run=fake_run(calls, stdout=b"1000.5\n1000.533\n1000.567\n"))
    stamps = recorder.timestamps("/tmp/archive.mkv")
    assert stamps.dtype == np.float64
    assert stamps == pytest.approx([1000.5, 1000.533, 1000.567])
    argv = calls[0]
    assert argv[0] == "ffprobe" and argv[argv.index("-show_entries") + 1] == "packet=pts_time"
    assert argv[argv.index("-select_streams") + 1] == "v" and argv[-1] == "/tmp/archive.mkv"


def test_timestamps_accept_text_output_too():
    recorder = Recorder(run=fake_run([], stdout="0.0 0.04 0.08"))
    assert recorder.timestamps("/tmp/a.mkv") == pytest.approx([0.0, 0.04, 0.08])


def test_timestamps_of_an_unreadable_file_are_empty():
    """A failed probe is no timeline, not a crash: the caller has nothing to attribute."""
    assert Recorder(run=fake_run([], stdout=b"", returncode=1)).timestamps("/tmp/gone.mkv").size == 0


def test_inspecting_a_take_counts_blank_and_distinct_frames(monkeypatch, tmp_path):
    argv = []
    monkeypatch.setattr(rec.subprocess, "run", fake_run(argv, stdout=gray_stream([0, 90, 180, 255])))
    report = inspect_take(tmp_path / "take.mkv")
    assert (report.frames, report.distinct, report.blank) == (4, 4, 1)
    assert report.usable and report.luma == pytest.approx(np.mean([0, 90, 180, 255]) / 255.0, abs=1e-6)
    assert argv[0][argv[0].index("-pix_fmt") + 1] == "gray"


def test_an_all_black_take_is_reported_unusable(monkeypatch, tmp_path):
    monkeypatch.setattr(rec.subprocess, "run", fake_run([], stdout=gray_stream([0, 0, 0, 0])))
    report = inspect_take(tmp_path / "black.mkv")
    assert report.blank == 4 and not report.usable and "UNUSABLE" in str(report)


def test_a_take_with_no_frames_at_all_is_unusable(monkeypatch, tmp_path):
    monkeypatch.setattr(rec.subprocess, "run", fake_run([], stdout=b""))
    report = inspect_take(tmp_path / "empty.mkv")
    assert report == TakeReport(frames=0, distinct=0, blank=0, luma=0.0) and not report.usable


def test_settling_judges_liveness_the_way_a_take_is_judged(monkeypatch, tmp_path):
    """A guard built on the discarded per-frame loop failed on a perfectly good picture."""
    reports = iter([TakeReport(10, 1, 10, 0.0), TakeReport(10, 9, 0, 0.4)])
    monkeypatch.setattr(rec, "inspect_take", lambda path, ffmpeg="ffmpeg": next(reports))
    ticks = iter([0.0, 0.0, 1.0, 2.0])
    recorder = FakeRecorder()
    probe = tmp_path / "probe.mkv"
    got = settle(
        recorder,
        program="Teletext",
        timeout_s=40.0,
        probe_path=probe,
        probe_s=0.5,
        clock=lambda: next(ticks),
        sleep=lambda s: None,
    )
    assert got.usable and len(recorder.probes) == 2, "it keeps probing until the picture is back"
    assert recorder.probes[0] == (str(probe), 0.5)
    assert not probe.exists(), "the probe never outlives the wait it was recorded for"


def test_settling_gives_up_with_what_it_saw(monkeypatch, tmp_path):
    monkeypatch.setattr(rec, "inspect_take", lambda path, ffmpeg="ffmpeg": TakeReport(6, 1, 6, 0.0))
    ticks = iter([0.0, 0.0, 99.0])
    probe = tmp_path / "probe.mkv"
    with pytest.raises(BlankTakeError, match="no picture within 1.0s"):
        settle(
            FakeRecorder(),
            program="Dead",
            timeout_s=1.0,
            probe_path=probe,
            probe_s=0.0,
            clock=lambda: next(ticks),
            sleep=lambda s: None,
        )
    assert not probe.exists(), "a failed wait cleans up after itself as well"


def test_settling_past_the_deadline_never_records(monkeypatch, tmp_path):
    """A budget already spent buys no probe, and the report says nothing was seen."""
    monkeypatch.setattr(rec, "inspect_take", lambda path, ffmpeg="ffmpeg": TakeReport(9, 9, 0, 0.5))
    recorder = FakeRecorder()
    with pytest.raises(BlankTakeError, match="0 frames"):
        settle(
            recorder,
            program="Dead",
            timeout_s=0.0,
            probe_path=tmp_path / "probe.mkv",
            clock=lambda: 5.0,
            sleep=lambda s: None,
        )
    assert not recorder.probes

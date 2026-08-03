"""Recorder: ffmpeg owns the capture, so nothing on the host competes with the rig."""

# pylint: disable=missing-function-docstring

import subprocess
import types

import pytest

from syncsummoner.device.recorder import MJPEG, RAW_422, Recorder, RecorderError


def fake_popen(started, *, returncode=None):
    """Popen stand-in recording argv, with a process that stays up unless told otherwise."""

    def make(argv, **kwargs):
        del kwargs
        started.append(argv)
        return types.SimpleNamespace(
            stdin=types.SimpleNamespace(write=lambda b: None, flush=lambda: None, close=lambda: None),
            poll=lambda: returncode,
            wait=lambda timeout=None: 0,
            kill=lambda: None,
        )

    return make


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


def test_recording_stops_the_encoder_and_checks_it_wrote(tmp_path):
    started = []
    take = tmp_path / "take.mkv"
    recorder = Recorder(popen=fake_popen(started), sleep=lambda s: take.write_bytes(b"x" * 16))
    with recorder.recording(take, seconds=1.0):
        pass
    assert len(started) == 1


def test_a_recorder_that_dies_at_once_is_an_error(tmp_path):
    recorder = Recorder(popen=fake_popen([], returncode=1), sleep=lambda s: None)
    with pytest.raises(RecorderError, match="exited immediately"):
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

"""Mastering: probed durations, clamped fades, and the argv that trims and muxes."""

# pylint: disable=missing-function-docstring

import json
import subprocess

import pytest

from syncsummoner.compose import master as M


def _fake_run(stdout=b"", returncode=0, seen=None):
    def run(argv, **kwargs):  # pylint: disable=unused-argument
        if seen is not None:
            seen.append(argv)
        return subprocess.CompletedProcess(argv, returncode, stdout, b"broken")

    return run


def test_probe_duration_reads_format_header(monkeypatch):
    payload = json.dumps({"format": {"duration": "12.5"}}).encode()
    monkeypatch.setattr(subprocess, "run", _fake_run(payload))
    assert M.probe_duration("clip.mkv") == pytest.approx(12.5)


def test_probe_duration_raises_on_failure(monkeypatch):
    monkeypatch.setattr(subprocess, "run", _fake_run(returncode=1))
    with pytest.raises(RuntimeError):
        M.probe_duration("clip.mkv")


def test_probe_duration_of_a_stream_with_no_header(monkeypatch):
    monkeypatch.setattr(subprocess, "run", _fake_run(b"{}"))
    assert M.probe_duration("clip.mkv") == 0.0


def test_common_duration_is_the_shorter_input(monkeypatch):
    durations = iter([30.0, 20.0])
    monkeypatch.setattr(M, "probe_duration", lambda path, **kw: next(durations))
    assert M.common_duration("clip.mkv", "track.flac", None) == pytest.approx(20.0)


def test_common_duration_needs_something_probeable(monkeypatch):
    monkeypatch.setattr(M, "probe_duration", lambda path, **kw: 0.0)
    with pytest.raises(ValueError):
        M.common_duration("clip.mkv")


def test_fades_sit_at_the_edges():
    video, audio = M.fade_filters(60.0, fade_in=1.5, fade_out=2.0)
    assert video == "fade=t=in:st=0:d=1.500,fade=t=out:st=58.000:d=2.000"
    assert audio == "afade=t=in:st=0:d=1.500,afade=t=out:st=58.000:d=2.000"


def test_fades_cannot_overlap_on_a_short_clip():
    video, _ = M.fade_filters(2.0, fade_in=5.0, fade_out=5.0)
    assert video == "fade=t=in:st=0:d=1.000,fade=t=out:st=1.000:d=1.000"


def test_no_fade_leaves_the_chain_empty():
    assert M.fade_filters(10.0, fade_in=0.0, fade_out=0.0) == ("", "")


def test_master_command_muxes_and_trims():
    argv = M.master_command("take.mkv", "track.flac", "out.mp4", duration=42.0)
    assert argv[:6] == ["ffmpeg", "-loglevel", "error", "-y", "-i", "take.mkv"]
    assert argv[6:8] == ["-i", "track.flac"]
    assert "-t" in argv and argv[argv.index("-t") + 1] == "42.000"
    assert argv[argv.index("-af") + 1].startswith("afade=t=in")
    assert argv[-1] == "out.mp4"
    assert "-shortest" in argv


def test_master_command_without_audio_is_silent():
    argv = M.master_command("take.mkv", None, "out.mp4", duration=5.0, fade_in=0.0, fade_out=0.0)
    assert "-an" in argv
    assert "-vf" not in argv
    assert "-map" not in argv


def test_master_probes_when_no_duration_is_given(monkeypatch):
    seen = []
    monkeypatch.setattr(M, "probe_duration", lambda path, **kw: 9.0)
    monkeypatch.setattr(subprocess, "run", _fake_run(seen=seen))
    assert M.master("take.mkv", "track.flac", "out.mp4") == pytest.approx(9.0)
    assert seen[0][seen[0].index("-t") + 1] == "9.000"


def test_master_raises_on_ffmpeg_failure(monkeypatch):
    monkeypatch.setattr(subprocess, "run", _fake_run(returncode=1))
    with pytest.raises(RuntimeError, match="mastering failed"):
        M.master("take.mkv", None, "out.mp4", seconds=3.0)


def test_an_excerpt_seeks_the_track_to_where_it_was_taken_from():
    argv = M.master_command("take.mkv", "track.flac", "out.mp4", duration=30.0, audio_start=60.0)
    assert argv[argv.index("track.flac") - 3 : argv.index("track.flac")] == ["-ss", "60.000", "-i"]


def test_the_takes_lead_in_is_trimmed_by_filter_not_by_seek():
    """Capture timestamps are the card's, so a seek lands near the frame, not on it."""
    argv = M.master_command("take.mkv", None, "out.mp4", duration=10.0, video_start=1.27)
    chain = argv[argv.index("-vf") + 1]
    assert chain.startswith("trim=start=1.270,setpts=PTS-STARTPTS,fade=t=in")
    assert "-ss" not in argv


def test_no_lead_in_leaves_the_chain_untrimmed():
    argv = M.master_command("take.mkv", None, "out.mp4", duration=10.0, fade_in=0.0, fade_out=0.0)
    assert "-vf" not in argv

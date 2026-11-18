"""A recording decoded once, segment by segment, to what judging reads, and kept beside it."""

# pylint: disable=missing-function-docstring

import logging
import shutil
import subprocess

import numpy as np
import pytest

from syncsummoner.device.v4l2rec import FRAME, sidecar
from syncsummoner.probe import decoded as D
from syncsummoner.probe.patterns import burn_timecode

pytestmark = pytest.mark.skipif(shutil.which("ffmpeg") is None, reason="ffmpeg is not installed")

SPEC = D.Spec(survival=(16, 9), calibration=(8, 6))


def encode(path, n, *, codec=("-c:v", "ffv1"), fps=30):
    rng = np.random.default_rng(n)
    plates = rng.uniform(0.1, 0.9, (n, 40, 96, 3)).astype(np.float32)
    frames = np.stack([burn_timecode(p, i, bits=16, strip_px=8) for i, p in enumerate(plates)])
    argv = ["ffmpeg", "-loglevel", "error", "-y", "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", "96x40"]
    argv += ["-r", str(fps), "-i", "pipe:0", *codec, str(path)]
    subprocess.run(argv, input=(frames * 255).astype(np.uint8).tobytes(), check=True)
    return path


@pytest.mark.parametrize(
    "codec",
    [("-c:v", "ffv1"), ("-c:v", "mjpeg", "-q:v", "2"), ("-c:v", "libx264", "-g", "7", "-bf", "2")],
    ids=["ffv1", "mjpeg", "h264-gop"],
)
def test_segments_decoded_side_by_side_join_into_the_single_decode(tmp_path, codec):
    path = encode(tmp_path / "rec.mkv", 40, codec=codec)
    whole = D.decode(path, SPEC, workers=1)
    split = D.decode(path, SPEC, workers=5)
    assert whole["codes"].tolist() == list(range(40))
    assert np.allclose(whole["times"], np.arange(40) / 30, atol=1e-3)
    for key, value in whole.items():
        assert np.array_equal(split[key], value), key
    assert whole["luma"].shape == (40, 9, 16) and whole["small"].shape == (40, 3, 6, 8)
    assert "log" not in whole


def test_keyframe_segments_cover_the_recording_once():
    key = np.zeros(20, dtype=bool)
    key[[0, 5, 10, 15]] = True
    assert D.segments(key, 4).tolist() == [0, 5, 10, 15, 20]
    assert D.segments(key, 2).tolist() == [0, 10, 20]
    assert D.segments(np.ones(5, dtype=bool), 9).tolist() == [0, 1, 2, 3, 4, 5]
    assert D.segments(key, 1).tolist() == [0, 20]


def test_ffprobe_is_found_beside_ffmpeg():
    assert D.ffprobe_for("ffmpeg") == "ffprobe"
    assert D.ffprobe_for("/opt/ff/bin/ffmpeg") == "/opt/ff/bin/ffprobe"


def test_a_recording_is_decoded_once_then_read_from_its_cache(tmp_path, monkeypatch):
    path = encode(tmp_path / "rec.mkv", 12)
    log = np.zeros(12, dtype=FRAME)
    log["sequence"] = np.arange(12)
    sidecar(path).write_bytes(log.tobytes())
    first = D.features(path, SPEC, workers=2)
    assert D.cache_path(path).is_file() and first.log["sequence"].tolist() == list(range(12))
    monkeypatch.setattr(D, "decode", lambda *a, **k: pytest.fail("decoded twice"))
    again = D.features(path, SPEC)
    assert again.key == first.key and len(again) == 12
    assert np.array_equal(again.luma, first.luma) and np.array_equal(again.small, first.small)
    assert again.log["sequence"].tolist() == list(range(12)) and np.array_equal(again.times, first.times)
    path.touch()
    assert D.features(path, SPEC).key == first.key, "a recording rewritten unchanged is matched by its hash"


@pytest.mark.parametrize("change", ["content", "spec", "log"])
def test_a_cache_no_longer_matching_is_decoded_afresh(tmp_path, change):
    path = encode(tmp_path / "rec.mkv", 12)
    D.features(path, SPEC)
    spec = SPEC
    if change == "content":
        encode(path, 13)
    elif change == "spec":
        spec = D.Spec(survival=(8, 4), calibration=(8, 6))
    else:
        sidecar(path).write_bytes(np.zeros(12, dtype=FRAME).tobytes())
    found = D.features(path, spec)
    assert len(found) == (13 if change == "content" else 12)
    assert found.luma.shape[1:] == spec.survival[::-1]


def test_an_unreadable_cache_is_said_and_replaced(tmp_path, caplog):
    path = encode(tmp_path / "rec.mkv", 6)
    D.cache_path(path).write_bytes(b"not a cache")
    with caplog.at_level(logging.WARNING, logger="syncsummoner"):
        assert len(D.features(path, SPEC)) == 6
    assert "unreadable feature cache" in caplog.text
    assert len(D.features(path, SPEC)) == 6


def test_a_cache_that_cannot_be_written_is_said_and_skipped(tmp_path, monkeypatch, caplog):
    path = encode(tmp_path / "rec.mkv", 6)

    def refuse(*_, **__):
        raise OSError("read-only")

    monkeypatch.setattr(D.os, "replace", refuse)
    with caplog.at_level(logging.WARNING, logger="syncsummoner"):
        assert len(D.features(path, SPEC)) == 6
    assert "features not cached: read-only" in caplog.text
    assert not D.cache_path(path).exists() and not list(tmp_path.glob("*.partial"))
    assert len(D.features(path, SPEC, store=False)) == 6


def test_frames_the_packets_do_not_account_for_are_said(tmp_path, monkeypatch, caplog):
    path = encode(tmp_path / "rec.mkv", 6)
    probed = D.packets(path)
    monkeypatch.setattr(
        D, "packets", lambda *a, **k: (np.r_[probed[0], 999], np.r_[probed[1], True], *probed[2:])
    )
    with caplog.at_level(logging.WARNING, logger="syncsummoner"):
        assert D.decode(path, SPEC, workers=1)["codes"].size == 6
    assert "6 frames decoded from 7 packets" in caplog.text


def test_a_segment_ffmpeg_cannot_decode_is_an_error(tmp_path):
    with pytest.raises(RuntimeError, match="failed with exit"):
        D._segment(  # pylint: disable=protected-access
            tmp_path / "missing.mkv",
            SPEC,
            start_s=None,
            count=1,
            width=96,
            ffmpeg="ffmpeg",
            done=lambda n: None,
        )


def test_a_recording_with_no_video_is_an_error(tmp_path, monkeypatch):
    path = encode(tmp_path / "rec.mkv", 2)
    monkeypatch.setattr(D, "packets", lambda *a, **k: (np.zeros(0, np.int64), np.zeros(0, bool), 1, 96))
    with pytest.raises(RuntimeError, match="no video packets"):
        D.decode(path, SPEC)


def test_the_spec_of_a_render_config_takes_its_strip():
    config = type("Config", (), {"strip_px": 4, "bits": 12})()
    assert D.Spec.of(config) == D.Spec(strip_px=4, bits=12)
    assert D.Spec.of(config).token()["schema"] == D.SCHEMA


def test_a_deduplicated_take_decodes_to_the_old_takes_features_capture_for_capture(tmp_path):
    # pylint: disable=no-member
    import cv2  # pylint: disable=import-outside-toplevel

    from syncsummoner.probe import dedupe  # pylint: disable=import-outside-toplevel
    from tests import takes  # pylint: disable=import-outside-toplevel

    rng = np.random.default_rng(3)
    shown = np.repeat(np.arange(12), rng.integers(1, 4, 12))
    plates = rng.uniform(0.1, 0.9, (12, 40, 96, 3)).astype(np.float32)
    jpegs = [
        cv2.imencode(".jpg", (burn_timecode(p, i, bits=16, strip_px=8) * 255).astype(np.uint8))[1].tobytes()
        for i, p in enumerate(plates)
    ]
    old = takes.old(tmp_path / "old.mkv", [jpegs[i] for i in shown])
    new = tmp_path / "new.mkv"
    shutil.copy(old, new)
    shutil.copy(sidecar(old), sidecar(new))
    assert dedupe.dedupe(new).status == "deduplicated"
    before = D.decode(old, SPEC, workers=1)
    assert before["codes"].tolist() == shown.tolist()
    for workers in (1, 4):
        after = D.decode(new, SPEC, workers=workers)
        assert after.keys() == before.keys()
        for key, value in before.items():
            assert np.array_equal(after[key], value), key

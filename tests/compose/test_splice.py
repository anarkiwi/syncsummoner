"""Splicing takes frame for frame by their attribution."""

# pylint: disable=missing-function-docstring

import shutil
import subprocess

import numpy as np
import pytest

from syncsummoner.compose import render as R


def test_a_splice_takes_each_clip_frame_from_its_cuts_take_at_the_capture_showing_it():
    cuts = [R.Cut(0.0, 0.2, "X"), R.Cut(0.2, 0.3, "Y"), R.Cut(0.3, 0.4, "X")]
    picked = {"X": np.arange(12) * 2 + 1, "Y": np.r_[np.full(6, -1), 40, 42, 44, -1, -1, -1]}
    owner, wanted = R.splice_indices(cuts, picked, 30.0)
    assert owner.tolist() == [0] * 6 + [1] * 3 + [0] * 3
    assert wanted["X"].tolist() == [1, 3, 5, 7, 9, 11, 19, 21, 23] and wanted["Y"].tolist() == [40, 42, 44]


def test_a_clip_frame_no_capture_shows_is_an_error_naming_the_program_and_time():
    cuts = [R.Cut(0.0, 0.2, "X"), R.Cut(0.2, 0.4, "Y")]
    with pytest.raises(R.SpliceError, match=r"Y: no capture shows clip frames 0.267-0.333s"):
        R.splice_indices(cuts, {"X": np.arange(12), "Y": np.r_[np.arange(8), -1, -1, 10, 11]}, 30.0)
    with pytest.raises(R.SpliceError, match=r"Y: no capture shows clip frames 0.200-0.400s"):
        R.splice_indices(cuts, {"X": np.arange(12)}, 30.0)
    with pytest.raises(R.SpliceError, match="comes before the last"):
        R.splice_indices(cuts, {"X": np.arange(12), "Y": np.r_[np.arange(7), 3, np.arange(8, 12)]}, 30.0)
    with pytest.raises(R.SpliceError, match="no cut covers 0.200s"):
        R.splice_indices([R.Cut(0.0, 0.2, "X"), R.Cut(0.3, 0.4, "X")], {"X": np.arange(12)}, 30.0)
    with pytest.raises(ValueError, match="nothing to splice"):
        R.splice_indices([], {}, 30.0)


def _flat_take(path, levels, *, fps=60):
    """A take whose capture ``i`` is one flat luma level, ``levels[i]``."""
    frames = np.repeat(np.asarray(levels, np.uint8)[:, None, None], 48 * 64, axis=1).tobytes()
    argv = ["ffmpeg", "-loglevel", "error", "-y", "-f", "rawvideo", "-pix_fmt", "gray", "-s", "64x48"]
    argv += ["-r", str(fps), "-i", "pipe:0", "-c:v", "mjpeg", "-q:v", "2", "-pix_fmt", "yuvj420p", str(path)]
    subprocess.run(argv, input=frames, check=True)


def _levels(path):
    argv = ["ffmpeg", "-loglevel", "error", "-i", str(path), "-f", "rawvideo", "-pix_fmt", "gray", "pipe:1"]
    out = subprocess.run(argv, check=True, capture_output=True).stdout
    return np.frombuffer(out, np.uint8).reshape(-1, 48 * 64).mean(axis=1)


@pytest.mark.skipif(shutil.which("ffmpeg") is None, reason="ffmpeg is not installed")
def test_a_spliced_master_is_frame_for_frame_the_captures_picked(tmp_path):
    _flat_take(tmp_path / "x.mkv", 10 + 3 * np.arange(30))
    _flat_take(tmp_path / "y.mkv", 150 + 3 * np.arange(30))
    config = R.RenderConfig(width=64, height=48, fps=30.0)
    cuts = [R.Cut(0.0, 0.2, "X"), R.Cut(0.2, 0.4, "Y"), R.Cut(0.4, 0.5, "X")]
    picked = {"X": np.arange(15) * 2 + 1, "Y": np.r_[np.full(6, -1), np.arange(12, 30, 2)]}
    takes = {"X": tmp_path / "x.mkv", "Y": tmp_path / "y.mkv"}
    assert R.splice(cuts, takes, picked, tmp_path / "out.mp4", config=config) == 15
    shown = np.r_[10 + 3 * picked["X"][:6], 150 + 3 * picked["Y"][6:12], 10 + 3 * picked["X"][12:]]
    assert np.abs(_levels(tmp_path / "out.mp4") - shown).max() < 1.5


@pytest.mark.skipif(shutil.which("ffmpeg") is None, reason="ffmpeg is not installed")
def test_a_take_that_ends_before_its_cut_fails_the_splice_and_names_it(tmp_path):
    _flat_take(tmp_path / "x.mkv", np.arange(8))
    config = R.RenderConfig(width=64, height=48, fps=30.0)
    with pytest.raises(R.SpliceError, match=r"x.mkv ended before capture 8"):
        R.splice(
            [R.Cut(0.0, 0.4, "X")],
            {"X": tmp_path / "x.mkv"},
            {"X": np.arange(12)},
            tmp_path / "o.mp4",
            config=config,
        )
    with pytest.raises(R.SpliceError, match="encoding .* failed"):
        R.splice(
            [R.Cut(0.0, 0.1, "X")],
            {"X": tmp_path / "x.mkv"},
            {"X": np.arange(3)},
            tmp_path / "no/o.mp4",
            config=config,
        )

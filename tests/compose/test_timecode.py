"""A timecoded clip as encoded: its lead-in and clip frames read back from the stored strips."""

# pylint: disable=missing-function-docstring

import subprocess

import numpy as np
import pytest

from syncsummoner.compose import lock as L
from syncsummoner.compose import render as R
from syncsummoner.probe import decoded

CONFIG = R.RenderConfig(width=160, height=48, fps=10.0, strip_px=8, bits=16)


@pytest.fixture(name="clip")
def clip_fixture(monkeypatch, tmp_path):
    rng = np.random.default_rng(3)
    source = [rng.uniform(0.2, 0.8, (CONFIG.height, CONFIG.width, 3)).astype(np.float32) for _ in range(5)]
    monkeypatch.setattr(
        "syncsummoner.compose.features.read_frames",
        lambda p, max_frames=None: iter([(CONFIG.fps, f) for f in source]),
    )
    out = tmp_path / "tc.mkv"
    assert R.write_timecoded("source.mkv", out, config=CONFIG, lead=4) == 5
    return out


def test_an_encoded_clip_reads_back_its_wrapped_lead_in_then_every_frame(clip):
    codes = decoded.features(clip, decoded.Spec.of(CONFIG)).codes
    assert L.signed(codes, CONFIG.bits).tolist() == [-4, -3, -2, -1, 0, 1, 2, 3, 4]


def test_a_program_that_turns_the_picture_over_leaves_no_legible_strip(clip, tmp_path):
    """A geometric program moves the strip off the edge; that is unread, not a frame index."""
    flipped = tmp_path / "flipped.mkv"
    subprocess.run(
        ["ffmpeg", "-loglevel", "error", "-i", str(clip), "-vf", "vflip", "-c:v", "ffv1", str(flipped)],
        check=True,
    )
    assert (decoded.features(flipped, decoded.Spec.of(CONFIG)).codes == -1).all()

"""Every reader of a take gives the same result from a deduplicated take as from the old take."""

# pylint: disable=missing-function-docstring

import shutil
import subprocess

import numpy as np
import pytest

from syncsummoner.compose import lock as LK
from syncsummoner.compose import master as M
from syncsummoner.compose import render as R
from syncsummoner.device import doctor as DR
from syncsummoner.device.recorder import inspect_take
from syncsummoner.device.v4l2rec import sidecar
from syncsummoner.probe.dedupe import dedupe
from tests import takes

pytestmark = pytest.mark.skipif(shutil.which("ffmpeg") is None, reason="needs ffmpeg")

SHADES = np.repeat(np.arange(14), [1, 3, 2, 1, 2, 2, 3, 1, 2, 2, 1, 2, 3, 2])


@pytest.fixture(name="pair")
def _pair(tmp_path):
    old = takes.old(tmp_path / "old.mkv", takes.jpegs(SHADES.tolist()))
    new = tmp_path / "new.mkv"
    shutil.copy(old, new)
    shutil.copy(sidecar(old), sidecar(new))
    assert dedupe(new).status == "deduplicated"
    return old, new


def md5s(path):
    argv = ["ffmpeg", "-loglevel", "error", "-i", str(path), "-f", "framemd5", "pipe:1"]
    out = subprocess.run(argv, capture_output=True, check=True, text=True).stdout
    return [line for line in out.splitlines() if not line.startswith("#")]


def test_a_splice_and_a_master_come_out_the_same(pair, tmp_path):
    config = R.RenderConfig(width=takes.WIDTH, height=takes.HEIGHT, fps=30.0)
    cuts = [R.Cut(0.0, 0.2, "X")]
    out = {}
    for take in pair:
        spliced = tmp_path / f"{take.stem}-cut.mkv"
        assert R.splice(cuts, {"X": take}, {"X": np.arange(1, 13, 2)}, spliced, config=config) == 6
        mastered = tmp_path / f"{take.stem}-master.mkv"
        M.master(take, None, mastered, seconds=0.4, fade_in=0.1, fade_out=0.1, video_start=0.05)
        out[take.stem] = md5s(spliced), md5s(mastered)
    assert out["new"] == out["old"] and len(out["old"][0]) == 6
    assert M.probe_duration(pair[1]) == pytest.approx(M.probe_duration(pair[0]), abs=1e-3), "to the time base"


def test_live_attribution_inspection_and_the_last_frame_read_the_same(pair):
    old, new = pair
    size = (8, 4)
    a, b = (LK.read_live(t, strip_px=4, bits=4, size=size) for t in pair)
    assert all(np.array_equal(x, y) for x, y in zip(a, b)) and a[1].shape[0] == SHADES.size
    assert inspect_take(new, samples=60, start_s=0.05) == inspect_take(old, samples=60, start_s=0.05)
    last = [DR.last_frame(t, takes.WIDTH, takes.HEIGHT) for t in pair]
    assert last[0] is not None and np.array_equal(last[0], last[1])

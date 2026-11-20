"""The synthetic stimulus: set by seed and index, moving as its known flow says, spanning its ranges."""

# pylint: disable=missing-function-docstring,no-member

import math

import cv2
import numpy as np
import pytest

from syncsummoner.compose.stimulus import Coverage, Stimulus, _texture

SMALL = {"width": 160, "height": 96, "texture": 128}


def test_a_frame_depends_only_on_the_seed_the_raster_and_its_index():
    one, two = Stimulus(seed=5, **SMALL), Stimulus(seed=5, **SMALL)
    assert (one.frame(7) == two.frame(7)).all() and (one.frame(-4) == two.frame(-4)).all()
    assert all((a == b).all() for a, b in zip(one.frames(6, 3), [one.frame(i) for i in (6, 7, 8)]))
    assert not (one.frame(7) == Stimulus(seed=6, **SMALL).frame(7)).all()


def test_the_next_frame_is_this_one_moved_by_its_flow():
    """Pull each pixel of the next frame back along the known flow: it reproduces this frame."""
    stim = Stimulus(seed=11, shot_frames=30, **SMALL)
    here, there = stim.frame(10).astype(np.float32), stim.frame(11).astype(np.float32)
    flow = stim.flow(10).astype(np.float32)
    y, x = np.indices((SMALL["height"], SMALL["width"]), dtype=np.float32)
    back = cv2.remap(
        there, x + flow[..., 0], y + flow[..., 1], cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT
    )
    inner = np.s_[12:-12, 12:-12]
    still = np.abs(here - there)[inner].mean()
    assert np.abs(here - back)[inner].mean() < still / 4, "the flow explains the motion"


def test_a_pure_drift_flows_the_same_everywhere_against_its_velocity():
    stim = Stimulus(seed=2, coverage=Coverage(spin=0.0, zoom=0.0, swing=(0.0, 0.0)), **SMALL)
    flow, shot = stim.flow(0), stim.shot(0)
    assert np.allclose(flow, flow[0, 0]), "the same everywhere"
    assert np.allclose(
        flow[0, 0] / np.hypot(*flow[0, 0]), -np.array(shot.velocity) / np.hypot(*shot.velocity)
    )


def test_the_flow_across_a_cut_is_unknown_and_the_shot_changes_there():
    stim = Stimulus(seed=2, shot_frames=5, **SMALL)
    assert np.isnan(stim.flow(4)).all() and not np.isnan(stim.flow(3)).any()
    assert stim.placed(5) == (1, 0) and stim.placed(-1) == (-1, 4)
    assert stim.shot(0) != stim.shot(1)


def test_a_texture_holds_one_band_and_wraps():
    tex = _texture(256, 16.0, 1)
    power = np.abs(np.fft.fft2(tex[..., 0])) ** 2
    f = np.hypot(*np.meshgrid(np.fft.fftfreq(256), np.fft.fftfreq(256), indexing="ij"))
    peak = f.ravel()[np.argmax(power.ravel())]
    assert abs(math.log2(peak * 16.0)) < 0.5
    assert np.allclose(tex.std(axis=(0, 1)), 1.0, atol=1e-3)


def test_shots_span_their_ranges_evenly():
    """A scrambled Sobol run: 64 shots fill each eighth of every range."""
    stim = Stimulus(seed=9, **SMALL)
    shots = [stim.shot(k) for k in range(64)]
    cov = stim.coverage
    speed = [
        math.log(math.hypot(*s.velocity) / cov.speed[0]) / math.log(cov.speed[1] / cov.speed[0])
        for s in shots
    ]
    level = [(s.level - cov.level[0]) / (cov.level[1] - cov.level[0]) for s in shots]
    for u in (speed, level):
        counts = np.bincount(np.minimum((np.asarray(u) * 8).astype(int), 7), minlength=8)
        assert counts.min() >= 6 and counts.max() <= 10


def test_each_frames_ti_is_drawn_afresh_about_the_coverages_level():
    """The analytic TI each step is scaled to is the TI the rendered frames show, white frame to frame."""
    cov = Coverage(swing=(0.8, 0.8))
    stim = Stimulus(seed=4, shot_frames=400, coverage=cov, **SMALL)
    luma = [
        f.astype(np.float32) @ np.array([0.2126, 0.7152, 0.0722], np.float32) for f in stim.frames(0, 400)
    ]
    ti = np.array([(b - a).std() for a, b in zip(luma, luma[1:])])
    steps = np.diff(stim.path(0), axis=0)
    shot = stim.shot(0)
    cross = steps[:, 0] * shot.velocity[1] - steps[:, 1] * shot.velocity[0]
    assert np.allclose(cross, 0.0), "drift keeps its direction"
    assert np.median(ti) == pytest.approx(cov.ti() * math.exp(-0.5 * 0.8**2), rel=0.35)
    assert np.log(ti).std() == pytest.approx(0.8, rel=0.25)
    assert abs(np.corrcoef(ti[:-1], ti[1:])[0, 1]) < 0.2, "each frame an independent sample"
    assert (stim.path(0)[0] == 0).all()


def test_one_texture_holds_every_shot_alike_but_for_its_motion():
    stim = Stimulus(seed=3, coverage=Coverage().one_texture(), **SMALL)
    shots = [stim.shot(k) for k in range(8)]
    looks = {(s.wavelength, s.contrast, s.level, s.saturation) for s in shots}
    assert len(looks) == 1 and len({s.velocity for s in shots}) == 8
    assert Coverage().one_texture().ti() == pytest.approx(Coverage().ti())

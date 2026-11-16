"""Survival: how much of a source's picture and motion an output keeps."""

# pylint: disable=missing-function-docstring

import numpy as np
import pytest

from syncsummoner import aesthetics
import syncsummoner.aesthetics.survival as S


def footage(n=120, shape=(36, 64), seed=3):
    """A moving natural-ish clip: a drifting blurred noise field with cuts."""
    rng = np.random.default_rng(seed)
    base = rng.normal(size=(shape[0] * 2, shape[1] * 2)).astype(np.float32)
    frames = []
    for i in range(n):
        if i % 40 == 0:
            base = rng.normal(size=base.shape).astype(np.float32)
        step = int(round(4 * np.sin(i / 7.0))) + i // 3
        frames.append(0.5 + 0.15 * np.roll(base, step, axis=1)[: shape[0], : shape[1]])
    return np.stack(frames)


def test_si_of_a_uniform_field_is_zero_and_of_a_picture_is_not():
    assert S.spatial_information(np.full((20, 30), 0.4, np.float32)) == 0.0
    assert S.spatial_information(footage(1)[0]) > 0.0


def test_rgb_frames_are_read_as_luma():
    grey = footage(1)[0]
    rgb = np.repeat(grey[:, :, None], 3, axis=2)
    assert S.spatial_information(rgb) == pytest.approx(S.spatial_information(grey), rel=1e-5)


def test_ti_is_zero_for_a_held_frame():
    frame = footage(1)[0]
    assert S.temporal_information(frame, frame) == 0.0


def test_information_leads_with_nan_ti():
    si, ti = S.information(footage(5))
    assert si.shape == ti.shape == (5,)
    assert np.isnan(ti[0]) and np.all(np.isfinite(ti[1:]))


def test_effective_samples_shrink_for_autocorrelated_series():
    rng = np.random.default_rng(0)
    white = rng.normal(size=500)
    smooth = np.convolve(white, np.ones(10), "same")
    assert S.effective_samples(white, white) == pytest.approx(500, rel=0.1)
    assert S.effective_samples(smooth, smooth) < 100


def test_critical_z_grows_with_the_leads_tried():
    assert S.critical_z(1) == pytest.approx(S.SIGMA)
    assert S.critical_z(2) > S.critical_z(1)


def test_follow_z_is_zero_without_enough_samples():
    assert S.follow_z(0.9, 3.0) == 0.0
    assert S.follow_z(float("nan"), 50.0) == 0.0


def test_passthrough_follows_and_keeps_everything():
    si, ti = S.information(footage())
    kept = S.survival_stats(si, ti, si, ti)
    assert kept.si_ratio == pytest.approx(1.0)
    assert kept.ti_ratio == pytest.approx(1.0)
    assert kept.following and kept.lead == 0


def test_a_one_frame_lead_is_found():
    si, ti = S.information(footage())
    late_si, late_ti = np.r_[np.nan, si[:-1]], np.r_[np.nan, ti[:-1]]
    kept = S.survival_stats(si, ti, late_si, late_ti)
    assert kept.lead == 1 and kept.follow == pytest.approx(1.0)


def test_independent_motion_does_not_follow():
    si, ti = S.information(footage(seed=3))
    other_si, other_ti = S.information(footage(seed=9)[::-1])
    assert not S.survival_stats(si, ti, other_si, other_ti).following


def test_quantisation_floors_match_simulated_noise():
    rng = np.random.default_rng(7)
    noise = [rng.uniform(-0.5, 0.5, (400, 400)).astype(np.float32) for _ in range(2)]
    assert S.spatial_information(noise[0]) == pytest.approx(S.quantisation_si(1.0), rel=0.05)
    assert S.temporal_information(*noise) == pytest.approx(S.quantisation_ti(1.0), rel=0.02)


def test_the_live_floor_is_the_geometric_mean_of_source_and_noise():
    assert S.live_floor(100.0, 1.0) == pytest.approx(10.0)
    assert S.live_floor(-1.0, 1.0) == 0.0


def test_unmeasured_spans_report_nothing_kept():
    empty = np.full(10, np.nan)
    kept = S.survival_stats(empty, empty, empty, empty)
    assert kept.frames == 0 and np.isnan(kept.si_ratio) and np.isnan(kept.ti_ratio)


def test_facade_exports_survival():
    assert aesthetics.survival_stats is S.survival_stats

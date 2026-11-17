"""The device's frame line on the capture clock, from frame starts bracketed by captures."""

# pylint: disable=missing-function-docstring

import numpy as np
import pytest

from syncsummoner.compose import frameline as F

CAPTURE = 16_666_670.7


def brackets(ppm, *, frames=12_000, phase=0.37, named=1200):
    """Frame-start brackets of a card capturing twice a frame, ``ppm`` off, the first ``named`` named."""
    period = 2 * CAPTURE * (1 + ppm * 1e-6)
    tau = np.arange(2 * frames + 4) * CAPTURE
    frame = np.floor((tau - phase * CAPTURE) / period).astype(np.int64)
    k = np.flatnonzero(np.diff(frame) > 0) + 1
    truth = frame[k]
    return tau[k - 1], tau[k], np.where(truth < named, truth, -1), truth, period


@pytest.mark.parametrize("ppm", [-72.0, -0.5, 0.0, 45.0])
def test_every_start_is_named_and_the_true_line_admitted(ppm):
    lo, hi, named, truth, period = brackets(ppm)
    line = F.fit(lo, hi, named, 0.0)
    assert line.frame.tolist() == truth.tolist()
    least, most = line.polygon.periods
    assert least <= period <= most and line.residual < 1.0 and line.slack == 0.0


def test_two_slips_pin_the_period():
    lo, hi, named, _, period = brackets(-72.0, frames=24_000)
    line = F.fit(lo, hi, named, 0.0)
    assert line.period == pytest.approx(period, abs=1.0)
    assert np.ptp(line.polygon.periods) < 1.0


def test_a_three_capture_frame_ties_and_hides_no_frame():
    lo, hi, named, truth, _ = brackets(45.0)
    line = F.fit(lo, hi, named, 0.0)
    assert line.tied == 1 and line.frame.tolist() == truth.tolist()


def test_a_line_carried_on_names_later_starts_as_a_fit_does():
    lo, hi, named, truth, _ = brackets(-72.0)
    head = 3000
    line = F.extend(F.fit(lo[:head], hi[:head], named[:head], 0.0), lo[head:], hi[head:], 0.0)
    assert line.frame.tolist() == truth.tolist()
    assert line.period == pytest.approx(F.fit(lo, hi, named, 0.0).period, rel=1e-9)


def test_a_start_a_frame_off_the_line_is_a_stray():
    lo, hi, named, truth, _ = brackets(-72.0, frames=3000)
    lo, hi = lo.copy(), hi.copy()
    lo[2000], hi[2000] = lo[2000] + 2 * CAPTURE, hi[2000] + 2 * CAPTURE
    line = F.fit(lo, hi, named, 1e5)
    assert line.strays.size == 1 and set(line.strays) <= {2000, 2001}
    kept = line.frame >= 0
    assert (line.frame[kept] == truth[kept]).sum() == kept.sum() - (line.strays[0] == 2001)


def test_a_named_start_off_the_line_past_the_jitter_is_a_line_error():
    lo, hi, named, _, _ = brackets(0.0, frames=3000)
    lo, hi = lo.copy(), hi.copy()
    lo[600:1200] += 1.5 * CAPTURE
    hi[600:1200] += 1.5 * CAPTURE
    with pytest.raises(F.LineError, match="past the capture jitter of 0.100 ms") as err:
        F.fit(lo, hi, named, 1e5)
    assert err.value.miss > 1e5


def test_a_miss_within_the_jitter_widens_its_bracket():
    lo, hi, named, truth, _ = brackets(0.0, frames=3000)
    lo, hi = lo.copy(), hi.copy()
    lo[1100] += 1.05 * CAPTURE
    hi[1100] += 1.05 * CAPTURE
    line = F.fit(lo, hi, named, 0.1 * CAPTURE)
    assert line.frame.tolist() == truth.tolist()
    assert line.slack == pytest.approx(0.05 * CAPTURE, rel=1e-3) and line.residual <= line.slack + 1.0


@pytest.mark.parametrize(
    "named,period,match",
    [
        (np.full(4, -1), None, "no bracket is named"),
        (np.array([5, 5, -1, -1]), None, "all one frame's"),
    ],
)
def test_a_line_nothing_fixes_is_a_line_error(named, period, match):
    lo, hi = np.arange(4) * 2 * CAPTURE, np.arange(4) * 2 * CAPTURE + CAPTURE
    with pytest.raises(F.LineError, match=match):
        F.fit(lo, hi, named, 0.0, period=period)


def test_a_run_with_nothing_named_counts_from_its_first_start_within_a_period_range():
    lo, hi, _, truth, period = brackets(-30.0, frames=2000)
    line = F.fit(lo, hi, np.full(lo.size, -1), 0.0, period=(period * 0.999, period * 1.001))
    assert (line.frame - line.frame[0]).tolist() == (truth - truth[0]).tolist()


def test_the_capture_line_measures_the_timestamps_jitter():
    rng = np.random.default_rng(1)
    sequence = np.r_[np.arange(500), np.arange(600, 1000)]
    t = 1e9 + sequence * CAPTURE + np.r_[np.zeros(500), np.full(400, 3e6)]
    late = rng.random(t.size) * 2e5
    seg = F.segments(np.round(t + late).astype(np.int64), sequence)
    assert np.bincount(seg).tolist() == [500, 400]
    period, jitter = F.capture_line(t + late, sequence, seg)
    assert period == pytest.approx(CAPTURE, rel=1e-6) and 0.9e5 < jitter < 2e5


@pytest.mark.parametrize(
    "t,sequence,runs",
    [
        ([0, 10, 20, 30, 40], [0, 1, 2, 3, 4], [0, 0, 0, 0, 0]),
        ([0, 10, 20, 90, 100], [0, 1, 2, 3, 4], [0, 0, 0, 1, 1]),
        ([0, 10, 20, 30, 40], [0, 1, 3, 4, 5], [0, 0, 1, 1, 1]),
        ([0], [0], [0]),
    ],
)
def test_runs_break_at_a_long_interval_or_a_lost_sequence_number(t, sequence, runs):
    assert F.segments(np.array(t, np.int64), np.array(sequence)).tolist() == runs


def test_the_least_squares_line_keeps_inside_the_polygon():
    poly = F.Polygon(0, np.array([(0.0, 10.0), (1.0, 10.0), (1.0, 11.0), (0.0, 11.0)]))
    u, p = poly.least_squares(np.array([0, 1, 2]), np.array([5.0, 25.0, 45.0]))
    assert 0.0 <= u <= 1.0 and 10.0 <= p <= 11.0
    u, p = poly.least_squares(np.array([0, 1, 2]), np.array([0.5, 11.0, 21.5]))
    assert (u, p) == pytest.approx((0.5, 10.5))

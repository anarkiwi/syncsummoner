"""Numbering a processed run between two Passthru runs by the capture clock."""

# pylint: disable=missing-function-docstring

import numpy as np

from syncsummoner.compose import relock as R

T = 33_333_333.0
#: A period range a few parts per million wide, as a 380 s run's change events admit.
PERIODS = (T * (1 - 2e-6), T * (1 + 2e-6))


def run(first, last, latency, phase, base=0):
    """Frames ``first..last`` starting at ``n T + latency``, each bracketed by captures two a frame."""
    n = np.arange(first, last + 1, dtype=np.float64)
    start = n * T + latency
    capture = T / 2
    hi = phase + np.ceil((start - phase) / capture) * capture
    return R.Brackets(hi - capture, hi, n - base)


LEAD_IN = run(0, 299, 0.0, 1e6)


def test_relock_offsets_inside_half_a_frame_number_the_run_once():
    mid, end = run(400, 8999, 2e6, 5e6, base=400), run(9100, 9199, 4e6, 9e6)
    assert R.candidates([LEAD_IN, end], mid, PERIODS) == [400]


def test_latencies_the_captures_leave_a_capture_wide_admit_two_numberings():
    mid, end = run(400, 8999, 9e6, 3e6, base=400), run(9100, 9199, 1e6, 12e6)
    assert R.candidates([LEAD_IN, end], mid, PERIODS) == [400, 401]


def test_end_runs_two_frames_apart_on_the_clock_number_the_run_no_way():
    mid, end = run(400, 8999, 2e6, 5e6, base=400), run(9100, 9199, 0.0, 9e6, base=-2)
    assert not R.candidates([LEAD_IN, end], mid, PERIODS)


def test_a_program_repeating_a_frame_mid_run_still_numbers_it():
    """A delay step shows one frame twice: that start has no change event, and the numbering stands."""
    mid = run(400, 8999, 2e6, 5e6, base=400)
    kept = np.ones(mid.n.size, dtype=bool)
    kept[4000] = False
    stepped = R.Brackets(mid.lo[kept], mid.hi[kept], mid.n[kept])
    assert R.candidates([LEAD_IN, run(9100, 9199, 4e6, 9e6)], stepped, PERIODS) == [400]


def test_latency_folds_the_numbering_in():
    b = R.Brackets.of(np.array([10.0, 50.0, 0.0]), np.array([20.0, 60.0, 5.0]), np.array([0, 1, -1]))
    assert b.n.tolist() == [0.0, 1.0]
    assert R.latency(b, 40.0) == (10.0, 20.0) and R.latency(b, 40.0, shift=1) == (-30.0, -20.0)

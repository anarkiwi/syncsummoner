"""Frame-indexed evidence: ticks, landings, readbacks, receipts and the picture test."""

# pylint: disable=missing-function-docstring

import json
import types

import numpy as np
import pytest
from scipy.stats import chi2, norm

from syncsummoner.compose import evidence as ev
from syncsummoner.device.wire import READ, RECEIPT, SEND

PERIOD = 33_333_333.0
TRUE_TICK = 5_000_000.0
TICKS = ev.Ticks(lo=TRUE_TICK - 400_000.0, hi=TRUE_TICK + 600_000.0, period=PERIOD, count=10)
DELTA = 1.25 * PERIOD
LEAD = 4
FRAMES = 40
TIMES = ev.FrameTimes(
    ns=TRUE_TICK + DELTA + (np.arange(FRAMES + LEAD) - LEAD) * PERIOD,
    lead=LEAD,
    origin=TRUE_TICK + DELTA,
    period=PERIOD,
)
PLAN = np.dtype([("target", "<i8"), ("param", "<i8"), ("value", "<i8")])
WAKE = 100_000.0
WRITE = 200_000.0


def true_tick(n):
    return TRUE_TICK + np.asarray(n, dtype=np.float64) * PERIOD


def sends_at(rows, t1=None):
    """SEND records for ``(target, param, value)`` rows, woken just after :func:`ev.send_time` says."""
    rows = np.array(rows, dtype=PLAN)
    due = ev.send_time(rows["target"], TICKS, TIMES, DELTA)
    t1 = due + WAKE + WRITE if t1 is None else np.asarray(t1, np.float64)
    out = np.zeros(len(rows), dtype=SEND)
    out["t0_ns"], out["t1_ns"], out["due_ns"] = t1 - WRITE, t1, due
    out["param"], out["value"], out["target"] = rows["param"], rows["value"], rows["target"]
    return out


def device_reads(sends, initial, *, start, stop, every=4_000_000, length=2_000_000):
    """Readbacks of a device latching each send on the first true tick after it, sampled mid-request."""
    t0 = np.arange(start, stop, every, dtype=np.float64)
    sample = t0 + length / 2
    lands = true_tick(np.ceil((sends["t1_ns"] - TRUE_TICK) / PERIOD))
    o = np.tile(np.asarray(initial, dtype=np.int64), (t0.size, 1))
    for k in np.argsort(lands, kind="stable"):
        o[sample >= lands[k], sends["param"][k] - 1] = sends["value"][k]
    out = np.zeros(t0.size, dtype=READ)
    out["t0_ns"], out["t1_ns"], out["o"] = t0, t0 + length, o
    return out


def test_frame_times_take_each_frames_first_capture_and_fit_the_line():
    k = np.arange(2 * (FRAMES + LEAD))
    t = 1e9 + k * PERIOD / 2 + np.where(k % 2, 3e5, 0.0)
    shown = k // 2 - LEAD
    runs = [types.SimpleNamespace(first=0, last=k.size - 1, slot=0)]
    times = ev.frame_times(t, shown, runs, 2, LEAD)
    assert np.allclose(times.ns, 1e9 + np.arange(FRAMES + LEAD) * PERIOD)
    assert times.origin == pytest.approx(1e9 + LEAD * PERIOD) and times.period == pytest.approx(PERIOD)
    with pytest.raises(ev.EvidenceError):
        ev.frame_times(t[:1], shown[:1], [types.SimpleNamespace(first=0, last=0, slot=0)], 2, LEAD)


def test_frame_times_fall_back_to_the_line_where_no_capture_started_a_frame():
    ns = TIMES.ns.copy()
    ns[LEAD + 3] = np.nan
    times = ev.FrameTimes(ns=ns + 1e5, lead=LEAD, origin=TIMES.origin, period=PERIOD)
    at = times.at(np.array([2, 3, -LEAD - 1, FRAMES + 5]))
    assert at[0] == pytest.approx(TIMES.origin + 2 * PERIOD + 1e5)
    assert at[1:].tolist() == pytest.approx(
        (TIMES.origin + np.array([3, -LEAD - 1, FRAMES + 5]) * PERIOD).tolist()
    )
    assert times.frame_of(TIMES.origin + 7 * PERIOD + 0.4 * PERIOD) == 7


def test_transitions_bracket_each_change_between_the_reads_either_side():
    reads = np.zeros(3, dtype=READ)
    reads["t0_ns"], reads["t1_ns"] = [0, 10, 20], [5, 15, 25]
    reads["o"][2, 1] = 9
    moves = ev.transitions(reads)
    assert moves.tolist() == [(2, 10.0, 25.0, 0, 9)]


def test_after_finds_the_first_tick_and_its_clearance():
    ticks = ev.Ticks(lo=0.0, hi=1e6, period=33e6, count=1)
    n, clear = ticks.after(np.array([5e6, 0.5e6, 33.5e6, -1e6]))
    assert n.tolist() == [1, 0, 1, 0]
    assert clear.tolist() == [28e6, -0.5e6, -0.5e6, 1e6]
    assert ticks.at(2) == pytest.approx(0.5e6 + 66e6) and ticks.width == 1e6


def test_tick_bracket_intersects_intervals_holding_the_same_tick():
    a = TRUE_TICK - np.array([2e6, 0.5e6, 1e6])
    b = TRUE_TICK + np.array([1e6, 2e6, 0.3e6])
    ticks = ev.tick_bracket(a, b, PERIOD)
    assert (ticks.lo, ticks.hi, ticks.period, ticks.count) == (
        pytest.approx(TRUE_TICK - 0.5e6),
        pytest.approx(TRUE_TICK + 0.3e6),
        PERIOD,
        3,
    )


def test_tick_bracket_fits_a_device_period_a_few_ppm_off_the_given_one():
    rng = np.random.default_rng(3)
    true_period = PERIOD * (1 + 4e-6)
    n = np.sort(rng.choice(30_000, size=60, replace=False))
    tick = TRUE_TICK + n * true_period
    a, b = tick - rng.uniform(0.2e6, 3e6, n.size), tick + rng.uniform(0.2e6, 3e6, n.size)
    fixed = n * PERIOD
    assert (a - fixed).max() > (b - fixed).min()
    ticks = ev.tick_bracket(a, b, PERIOD)
    assert ticks.period == pytest.approx(true_period, rel=5e-7)
    k = np.rint((0.5 * (a + b) - ticks.phase) / ticks.period)
    assert np.all((ticks.lo + k * ticks.period > a - 1) & (ticks.hi + k * ticks.period <= b + 1))
    assert k.tolist() == (n - n[0]).tolist()


def test_tick_bracket_rejects_changes_off_one_grid_and_no_changes():
    a = np.array([0.0, PERIOD + 2e6, 2 * PERIOD])
    with pytest.raises(ev.EvidenceError, match="disagree|grid"):
        ev.tick_bracket(a, a + 1e6, PERIOD)
    with pytest.raises(ev.EvidenceError, match="never seen"):
        ev.tick_bracket(np.zeros(0), np.zeros(0), PERIOD)


def test_send_time_is_midway_between_ticks_so_errors_either_way_still_land():
    frames = np.arange(-LEAD, FRAMES + 10)
    due = ev.send_time(frames, TICKS, TIMES, DELTA)
    true_n = np.ceil((due - TRUE_TICK) / PERIOD)
    assert np.all(true_tick(true_n - 1) < due) and np.all(due < true_tick(true_n))
    assert np.allclose(due - TICKS.at(true_n - 1), PERIOD / 2)
    for off in np.linspace(-(PERIOD - TICKS.width) / 2 + 1.0, (PERIOD - TICKS.width) / 2 - 1.0, 7):
        n, clear = TICKS.after(due + off)
        assert ev.frame_of_tick(n, TICKS, TIMES, DELTA).tolist() == frames.tolist()
        assert np.allclose(clear, (PERIOD - TICKS.width) / 2 - off) and np.all(clear > 0)
    n, _ = TICKS.after(due + PERIOD)
    assert np.all(ev.frame_of_tick(n, TICKS, TIMES, DELTA) == frames + 1)


def test_read_back_changes_recover_the_true_tick_and_each_writes_tick():
    rows = [(2, 1, 100), (5, 1, 200), (5, 2, 50), (8, 1, 200), (9, 3, 7), (9, 3, 8)]
    sends = sends_at(rows + [(t, 7 + t % 3, t) for t in range(12, 38, 2)])
    sends["t1_ns"][5] = sends["t1_ns"][4] + 1e6
    initial = np.zeros(12, dtype=np.int64)
    reads = device_reads(sends, initial, start=TIMES.at(0), stop=TIMES.at(39))
    moves = ev.transitions(reads)
    ticks = ev.tick_bracket(moves["a"], moves["b"], PERIOD)
    assert ticks.period == pytest.approx(PERIOD, rel=1e-3) and ticks.count == moves.size
    base = int(np.rint((ticks.phase - TRUE_TICK) / PERIOD))
    k = np.rint((0.5 * (moves["a"] + moves["b"]) - ticks.phase) / ticks.period)
    assert np.all((moves["a"] < ticks.at(k)) & (ticks.at(k) <= moves["b"]))
    seen = ev.observed_ticks(sends, moves, ticks)
    want = np.ceil((sends["t1_ns"] - TRUE_TICK) / PERIOD).astype(np.int64) - base
    assert seen[[0, 1, 2, 5]].tolist() == want[[0, 1, 2, 5]].tolist()
    assert seen[3] == -1 and seen[4] == -1
    assert ev.observed_ticks(sends[sends["param"] == 4], moves, ticks).size == 0
    four = sends[:1].copy()
    four["param"] = 4
    assert ev.observed_ticks(four, moves, ticks).tolist() == [-1]


def test_deadline_is_bounded_by_late_writes_below_and_latched_writes_above():
    ticks = ev.Ticks(lo=0.0, hi=1e6, period=33e6, count=1)
    sends = np.zeros(3, dtype=SEND)
    sends["t1_ns"] = [32.5e6, 20e6, 10e6]
    lo, hi = ev.deadline(sends, np.array([2, 1, -1]), ticks)
    assert (lo, hi) == (pytest.approx(0.5e6), pytest.approx(13e6))
    assert ev.deadline(sends, np.full(3, -1), ticks) == (-np.inf, np.inf)


def toggle_take(switch, *, flicker=None, count=60):
    k = np.arange(count)
    t = 1000.0 + k * 500.0
    values = np.array([300, 1023] * (len(switch) // 2))
    level = np.full(count, 255, dtype=np.uint8)
    for s, v in zip(switch, values):
        level[s:] = 0 if v == 300 else 255
    if flicker is not None:
        level[flicker] = 255 - level[flicker]
    images = np.broadcast_to(level[:, None, None, None], (count, 1, 2, 2)).copy()
    toggles = np.zeros(len(switch), dtype=SEND)
    toggles["t1_ns"] = t[np.asarray(switch) - 3] + 100
    toggles["t0_ns"] = toggles["t1_ns"] - 50
    toggles["value"], toggles["target"] = values, ev.CALIBRATION
    return images, t, k % 2 == 0, k // 2, toggles


def test_toggle_landings_find_the_frame_each_write_first_shows_in():
    images, t, starts, shown, toggles = toggle_take([6, 20, 34, 48])
    assert ev.toggle_landings(images, t, starts, shown, toggles).tolist() == [3, 10, 17, 24]
    assert ev.toggle_landings(images, t, starts, shown, toggles, until=t[52]).tolist() == [3, 10, 17, 24]


def test_toggle_landings_reject_switches_off_a_frame_start_or_that_flicker():
    images, t, starts, shown, toggles = toggle_take([6, 21, 34, 48], flicker=40)
    assert ev.toggle_landings(images, t, starts, shown, toggles).tolist() == [
        3,
        ev.CALIBRATION,
        ev.CALIBRATION,
        24,
    ]


def test_toggle_landings_need_two_states_each_with_a_picture():
    images, t, starts, shown, toggles = toggle_take([6, 20])
    one = toggles.copy()
    one["value"] = 300
    assert ev.toggle_landings(images, t, starts, shown, one).tolist() == [ev.CALIBRATION] * 2
    late = toggles.copy()
    late["t1_ns"][1] = t[-1] + 10
    late["t0_ns"][1] = t[-1] + 5
    assert ev.toggle_landings(images, t, starts, shown, late).tolist() == [ev.CALIBRATION] * 2


def test_latency_is_the_median_tick_to_capture_delay_and_rejects_disagreement():
    n = np.array([3, 7, 11, 15])
    ticks = ev.Ticks(lo=TRUE_TICK, hi=TRUE_TICK, period=PERIOD, count=1)
    measured = ev.latency(n + 1, n, ticks, TIMES)
    assert measured.delta == pytest.approx(DELTA + PERIOD) and measured.samples == 4
    assert measured.frames == pytest.approx(2.25) and measured.spread == pytest.approx(0.0, abs=1)
    assert measured.as_dict()["frames"] == pytest.approx(2.25)
    with pytest.raises(ev.EvidenceError, match="half a frame"):
        ev.latency(np.array([4, 8, 12, 16]), np.array([3, 7, 11, 14]), ticks, TIMES)
    with pytest.raises(ev.EvidenceError, match="no calibration"):
        ev.latency(np.full(2, ev.CALIBRATION), n[:2], ticks, TIMES)


def test_writes_sent_on_schedule_land_on_their_targets():
    plan = np.array([(2, 1, 10), (2, 2, 20), (5, 1, 30), (9, 3, 40)], dtype=PLAN)
    sends = np.r_[sends_at([(0, 1, 1)]), sends_at(plan.tolist())]
    sends["target"][0] = ev.START
    got = ev.landed(plan, sends, TICKS, TIMES, DELTA)
    assert got.ok and got.rows["landed"].tolist() == [2, 2, 5, 9]
    assert got.rows["sent"].tolist() == [0, 1, 2, 3] and "4 writes scheduled: 0 missing" in got.summary()
    assert got.wake_ms == pytest.approx(WAKE / 1e6) and "0.10 ms after due" in got.summary()
    sends["t0_ns"][3] += 2_500_000
    assert ev.landed(plan, sends, TICKS, TIMES, DELTA).wake_ms == pytest.approx(2.6)
    bare = np.dtype([(name, SEND[name]) for name in SEND.names if name != "due_ns"])
    undue = np.zeros(len(sends), dtype=bare)
    for name in bare.names:
        undue[name] = sends[name]
    assert np.isnan(ev.landed(plan, undue, TICKS, TIMES, DELTA).wake_ms)


def test_late_early_missing_duplicated_and_out_of_order_writes_are_counted():
    plan = np.array([(2, 1, 10), (2, 2, 20), (5, 1, 30), (9, 3, 40), (12, 4, 50)], dtype=PLAN)
    on_time = ev.send_time(plan["target"], TICKS, TIMES, DELTA) + WAKE + WRITE
    t1 = on_time + np.array([0, 0, PERIOD, -PERIOD, 0])
    sends = sends_at(plan.tolist(), t1)
    sends = np.r_[sends[[1, 0, 2, 3]], sends[[3]]]
    got = ev.landed(plan, sends, TICKS, TIMES, DELTA)
    assert (got.missing, got.duplicated, got.out_of_order, got.late, got.early) == (1, 1, 1, 1, 1)
    assert got.ambiguous == 0 and not got.ok
    assert got.rows["landed"].tolist() == [2, 2, 6, 8, ev.CALIBRATION]


def test_a_write_on_a_tick_or_read_back_off_its_predicted_tick_is_ambiguous():
    plan = np.array([(2, 1, 10), (5, 2, 20)], dtype=PLAN)
    sends = sends_at(
        plan.tolist(), [TICKS.at(2 + 0) + 0.0, ev.send_time(np.array([5]), TICKS, TIMES, DELTA)[0]]
    )
    seen = np.array([-1, 6])
    got = ev.landed(plan, sends, TICKS, TIMES, DELTA, seen)
    assert got.ambiguous == 2
    assert got.rows["landed"][1] == 6 and got.late >= 1


def test_readbacks_agree_with_the_writes_landed_before_them():
    sends = sends_at([(2, 1, 100), (5, 1, 200), (5, 2, 50), (8, 1, 200)])
    initial = np.arange(12) * 10
    reads = device_reads(sends, initial, start=TIMES.at(0), stop=TIMES.at(12))
    got = ev.applied(reads, sends, initial, TICKS, TIMES, DELTA)
    assert got.ok and got.reads == len(reads) and got.checked == 12 * len(reads)
    assert (got.confirmed, got.unconfirmed) == (3, 1)
    assert "0 disagree" in got.summary()


def test_a_readback_disagreeing_is_attributed_to_its_slot_and_frame():
    sends = sends_at([(2, 1, 100), (5, 2, 50)])
    initial = np.zeros(12, dtype=np.int64)
    reads = device_reads(sends, initial, start=TIMES.at(0), stop=TIMES.at(10))
    k = int(np.searchsorted(reads["t0_ns"], TIMES.at(7) - DELTA + 3e6))
    reads["o"][k, 1] = 49
    got = ev.applied(reads, sends, initial, TICKS, TIMES, DELTA)
    assert got.disagree.tolist() == [(k, 2, 7, 50, 49)]


def test_a_read_straddling_a_tick_may_show_either_side_of_it():
    sends = sends_at([(3, 1, 100)])
    tick = TICKS.at(np.ceil((sends["t1_ns"][0] - TICKS.hi) / PERIOD))
    reads = np.zeros(3, dtype=READ)
    reads["t0_ns"] = [tick - 10e6, tick - 1e6, tick + 10e6]
    reads["t1_ns"] = reads["t0_ns"] + np.array([1e6, 2e6, 1e6])
    reads["o"][2, 0] = 100
    for value in (0, 100):
        reads["o"][1, 0] = value
        got = ev.applied(reads, sends, np.zeros(12), TICKS, TIMES, DELTA)
        assert got.ok and got.ambiguous == 1
    reads["o"][1, 0] = 7
    assert ev.applied(reads, sends, np.zeros(12), TICKS, TIMES, DELTA).disagree["read"].tolist() == [1]


def test_applied_lands_a_write_on_its_observed_tick():
    sends = sends_at([(3, 1, 100)])
    n = int(TICKS.after(sends["t1_ns"])[0][0])
    reads = np.zeros(2, dtype=READ)
    reads["t0_ns"] = [TICKS.at(n) + 5e6, TICKS.at(n + 1) + 5e6]
    reads["t1_ns"] = reads["t0_ns"] + 1e6
    reads["o"][1, 0] = 100
    assert not ev.applied(reads, sends, np.zeros(12), TICKS, TIMES, DELTA).ok
    assert ev.applied(reads, sends, np.zeros(12), TICKS, TIMES, DELTA, np.array([n + 1])).ok


def test_expected_messages_follow_the_cc_encoding():
    sends = np.zeros(3, dtype=SEND)
    sends["param"], sends["value"] = [1, 2, 2], [1023, 0, 512]
    msgs = ev.expected_messages(sends, [16, 40])
    assert msgs[:2].tolist() == [[16, 127], [48, 127]]
    assert msgs[2].tolist() == [40, 0] and msgs[3].tolist() == [40, 64]
    lo = ev.expected_messages(sends[:1], [16], high_resolution=False)
    assert lo.tolist() == [[16, 127]]
    word = ev.expected_messages(sends[[2]], [3, 3])
    assert abs(((word[0, 1] << 7) | word[1, 1]) / 16383 * 1023 - 512) < 0.5


def receipts(pairs):
    out = np.zeros(len(pairs), dtype=RECEIPT)
    if pairs:
        out["cc"], out["value"] = np.array(pairs).T
    return out


@pytest.mark.parametrize(
    "got,matched,ok",
    [
        ([(1, 10), (33, 5), (2, 7)], 3, True),
        ([(1, 10), (2, 7)], 2, False),
        ([(1, 10), (5, 5), (33, 5), (2, 7)], 3, False),
        ([(9, 9), (1, 10), (33, 5), (2, 7), (9, 9)], 3, False),
        ([], 0, False),
    ],
)
def test_receipts_are_matched_in_order_against_what_was_sent(got, matched, ok):
    result = ev.received(np.array([(1, 10), (33, 5), (2, 7)]), receipts(got))
    assert (result.sent, result.received, result.matched, result.ok) == (3, len(got), matched, ok)
    assert f"{matched} matched" in result.summary()


def test_frame_change_is_the_mean_absolute_difference_from_the_frame_before():
    stack = np.array([0, 2, 2, -1], dtype=np.float32)[:, None, None] * np.ones((1, 2, 3))
    assert np.isnan(ev.frame_change(stack)[0]) and ev.frame_change(stack)[1:].tolist() == [2, 0, 3]


def test_matched_scores_use_the_quantum_where_controls_do_not_vary():
    d_src = np.arange(20, dtype=np.float64)
    d_out = np.ones(20)
    d_out[5] = 3.0
    scores = ev.matched_scores(d_out, d_src, np.array([5, 6]), np.arange(20), 0.25)
    assert scores.tolist() == [8.0, 0.0]


def test_tail_p_is_a_mid_p_strictly_inside_the_unit_interval():
    null = np.arange(10.0)
    assert ev.tail_p(np.array([100.0, -1.0, 5.0]), null).tolist() == pytest.approx(
        [0.5 / 11, 10.5 / 11, 5 / 11]
    )
    p = ev.tail_p(np.random.default_rng(0).normal(size=500), np.random.default_rng(1).normal(size=50))
    assert np.all((p > 0) & (p < 1))


def test_fisher_z_of_uniform_p_is_standard_normal():
    z = np.array([ev.fisher_z(np.random.default_rng(s).uniform(size=40))[0] for s in range(300)])
    assert abs(z.mean()) < 0.2 and 0.8 < z.std() < 1.2
    assert ev.fisher_z(np.zeros(0)) == (0.0, 0.0)


def test_fisher_z_discounts_serially_dependent_p():
    p = np.repeat(np.random.default_rng(3).uniform(0.01, 0.2, size=8), 10)
    z, n_eff = ev.fisher_z(p)
    naive = norm.isf(chi2.sf(-2 * np.log(p).sum(), 2 * p.size))
    assert n_eff < p.size / 3 and z < naive


def test_steps_all_at_the_empirical_floor_count_as_independent():
    p = np.full(15, 0.5 / 180)
    z, n_eff = ev.fisher_z(p)
    assert n_eff == 15.0 and z == pytest.approx(norm.isf(chi2.sf(-2 * np.log(p).sum(), 30)))


def test_effective_samples_of_an_ar1_series():
    rng = np.random.default_rng(7)
    rho, n = 0.8, 20000
    e = rng.normal(size=n)
    x = np.zeros(n)
    for i in range(1, n):
        x[i] = rho * x[i - 1] + e[i]
    assert ev.effective(x) == pytest.approx(n * (1 - rho) / (1 + rho), rel=0.15)
    assert ev.effective(rng.normal(size=n)) == pytest.approx(n, rel=0.05)
    assert ev.effective(np.ones(10)) == 10.0 and ev.effective(np.ones(2)) == 2.0


def test_critical_z_holds_the_family_wise_tail():
    assert ev.critical(1) == pytest.approx(3.0)
    assert 12 * norm.sf(ev.critical(12)) == pytest.approx(norm.sf(3.0))
    assert ev.critical(0) == ev.critical(1)


def picture(seed, steps, *, count=240, jump=40.0, delay=0):
    """A source of random motion, and an output following it that also jumps at each step frame."""
    rng = np.random.default_rng(seed)
    src = np.cumsum(
        rng.normal(0, rng.uniform(0.5, 4.0, size=(count, 1, 1, 1)), size=(count, 1, 4, 4)), axis=0
    )
    level = np.zeros(count)
    steps = np.asarray(steps, dtype=np.int64)
    for i, (frame, size) in enumerate(zip(steps, np.broadcast_to(jump, steps.shape))):
        level[frame + delay :] += size if i % 2 == 0 else -size
    out = src + level[:, None, None, None] + rng.normal(0, 0.3, size=src.shape)
    return out, src


def spaced(rng, count, n, avoid=()):
    pool = np.setdiff1d(np.arange(3, count - 3, 3), np.asarray(avoid, dtype=np.int64))
    return np.sort(rng.choice(pool, size=n, replace=False))


def test_a_parameter_whose_steps_jump_the_picture_is_moved_and_one_without_is_not():
    rng = np.random.default_rng(11)
    real = spaced(rng, 240, 15)
    idle = spaced(rng, 240, 15, avoid=np.r_[real - 3, real, real + 3])
    out, src = picture(11, real, jump=np.where(np.arange(15) == 7, 0.0, 40.0))
    frames, params = np.r_[real, idle], np.r_[np.full(15, 1), np.full(15, 2)]
    moved = ev.acted_upon(out, src, np.full(240, 100.0), frames, params, {1: "Gain"})
    one, two = moved[0], moved[1]
    assert (one.param, one.name, two.name) == (1, "Gain", "P2")
    assert one.moved and one.lag == 0 and one.steps == 15 and one.least_p < 0.05
    assert not two.moved and two.critical == pytest.approx(ev.critical(4)) and one.flat == 0.0


def test_a_jump_one_frame_late_is_moved_at_lag_one():
    real = spaced(np.random.default_rng(5), 240, 15)
    out, src = picture(5, real, jump=np.where(np.arange(15) == 3, 0.0, 40.0), delay=1)
    got = ev.acted_upon(out, src, np.zeros(240), real, np.ones(15, dtype=int), {})[0]
    assert got.moved and got.lag == 1 and got.flat == 1.0


def test_with_no_effect_the_family_wise_false_positive_rate_holds():
    hits = 0
    for seed in range(200):
        rng = np.random.default_rng(1000 + seed)
        steps = spaced(rng, 120, 8)
        out, src = picture(1000 + seed, steps, count=120, jump=0.0)
        hits += any(v.moved for v in ev.acted_upon(out, src, np.zeros(120), steps, np.ones(8, int), {}))
    assert hits / 200 < 0.02


def test_parameters_landing_only_together_are_tested_with_their_partners():
    rng = np.random.default_rng(2)
    joint = spaced(rng, 240, 10)
    solo = spaced(rng, 240, 5, avoid=np.r_[joint - 3, joint, joint + 3])
    out, src = picture(2, joint, jump=np.where(np.arange(10) == 5, 0.0, 40.0))
    frames = np.r_[joint, joint, joint[:4], solo]
    params = np.r_[np.full(10, 3), np.full(10, 4), np.full(4, 5), np.full(5, 5)]
    by = {v.param: v for v in ev.acted_upon(out, src, np.zeros(240), frames, params, {})}
    assert by[3].partners == (4, 5) and by[4].partners == (3, 5) and by[3].moved and by[3].jointly == 10
    assert by[5].partners == () and by[5].jointly == 4 and by[5].steps == 5


def test_steps_outside_the_measured_frames_are_not_tested():
    out, src = picture(4, [], count=60)
    out[20] = np.nan
    moved = ev.acted_upon(out, src, np.zeros(60), np.array([0, 59, 20]), np.array([1, 2, 3]), {})
    assert [v.param for v in moved] == [3]
    assert moved[0].steps == 0 and moved[0].z == 0.0 and np.isnan(moved[0].least_p) and not moved[0].moved


def verdict(**kw):
    base = {"param": 1, "name": "Gain", "steps": 5, "effective": 5.0, "least_p": 0.5, "z": 0.0, "lag": 0}
    return ev.Moved(**{**base, "critical": 3.0, "flat": 0.0, "jointly": 0, **kw})


@pytest.mark.parametrize(
    "moved,name,applied_ok,landed_ok,reason",
    [
        (verdict(), "Gain", True, False, "not sent on time"),
        (verdict(), "Gain", False, True, "did not take it"),
        (verdict(), " - ", True, True, "leaves the slot unused"),
        (verdict(steps=0), "Gain", True, True, "no step landed"),
        (verdict(partners=(2, 3)), "Gain", True, True, "even with P2, P3"),
        (verdict(flat=0.75), "Gain", True, True, "uniform field at 75%"),
        (verdict(), "Gain", True, True, "ignores it in this state"),
    ],
)
def test_why_not_names_the_earliest_broken_link(moved, name, applied_ok, landed_ok, reason):
    assert reason in ev.why_not(moved, name, applied_ok, landed_ok)


def evidence(loaded_ok=True, received=None):
    plan = np.array([(2, 1, 10), (4, 2, 20)], dtype=PLAN)
    sends = sends_at(plan.tolist())
    landed = ev.landed(plan, sends, TICKS, TIMES, DELTA)
    disagree = np.array([(0, 2, 4, 20, 0)], dtype=ev.DISAGREE)
    applied = ev.Applied(reads=1, checked=12, ambiguous=0, disagree=disagree, confirmed=1, unconfirmed=0)
    moved = [verdict(z=5.0, partners=(2,)), verdict(param=2, name="Mix")]
    return ev.Evidence(
        program="Edge",
        latency={"frames": 1.2, "spread": 1e6, "samples": 8},
        ticks={"width_ms": 1.0, "count": 9},
        landed=landed,
        applied=applied,
        received=received,
        moved=moved,
        loaded={"ok": loaded_ok, "summary": "fpga runs Edge"},
    )


def test_evidence_is_ok_only_when_every_link_holds():
    good = evidence()
    good.applied = ev.Applied(1, 12, 0, np.zeros(0, dtype=ev.DISAGREE), 1, 0)
    assert good.ok
    good.loaded = {"ok": False}
    assert not good.ok
    assert not evidence().ok
    bad = evidence(received=ev.Received(sent=2, received=1, matched=1))
    bad.applied = good.applied
    bad.loaded = {"ok": True}
    assert not bad.ok


def test_evidence_reports_per_parameter_verdicts():
    proof = evidence(received=ev.Received(sent=4, received=4, matched=4))
    table = proof.table()
    assert [row["readback"] for row in table] == ["agrees", "disagrees"]
    assert table[0]["acted_upon"] and table[0]["with"] == [2] and table[0]["why_not"] is None
    assert table[1]["why_not"] == "the firmware did not take it" and table[1]["on_target"] == 1
    lines = proof.lines()
    assert any("only together with P2" in line for line in lines)
    assert any("NOT acted upon: the firmware did not take it" in line for line in lines)
    assert any("4 messages sent" in line for line in lines)
    said = json.loads(proof.to_json())
    assert said["ok"] is False and said["disagree"] == [[0, 2, 4, 20, 0]] and len(said["parameters"]) == 2
    assert said["late"] == [] and said["received"].startswith("4 messages")

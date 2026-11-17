"""Frame-indexed evidence: ticks, landings, readbacks, receipts and the picture test."""

# pylint: disable=missing-function-docstring

import json
import types

import numpy as np
import pytest
from scipy.stats import chi2, norm

from syncsummoner.compose import evidence as ev
from syncsummoner.compose import matched as mt
from syncsummoner.device import wire as W
from syncsummoner.device.wire import MESSAGE, READ, RECEIPT, SEND

PERIOD = 33_333_333.0
TRUE_TICK = 5_000_000.0
TICKS = ev.Ticks(lo=TRUE_TICK - 400_000.0, hi=TRUE_TICK + 600_000.0, period=PERIOD, count=10)
DELTA = 1.25 * PERIOD
LEAD = 4
FRAMES = 40
TIMES = ev.FrameTimes(lead=LEAD, origin=TRUE_TICK + DELTA, period=PERIOD)
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


def test_frame_times_are_the_attributions_line_from_clip_frame_0():
    found = types.SimpleNamespace(lead=LEAD, frame0_ns=1e9 + 0.3, period_ns=PERIOD * (1 + 5e-5))
    times = ev.frame_times(found)
    assert (times.lead, times.origin, times.period) == (LEAD, 1e9 + 0.3, PERIOD * (1 + 5e-5))
    n = np.array([-LEAD, 0, 3, FRAMES + 5])
    assert times.at(n).tolist() == pytest.approx((1e9 + 0.3 + n * PERIOD * (1 + 5e-5)).tolist())
    assert times.frame_of(times.at(7) + 0.4 * PERIOD) == 7 and times.frame_of(times.at(7) - 0.4 * PERIOD) == 7


def test_a_latency_bounds_the_devices_own_by_the_lines_its_captures_admit():
    lat = ev.Latency(lo=DELTA - 1e5, hi=DELTA + 2e5, samples=3, period=PERIOD, line_lo=-4e5, line_hi=3e5)
    assert lat.bounds == (DELTA - 5e5, DELTA + 5e5)
    said = lat.as_dict()
    assert (said["line_lo"], said["line_hi"], said["basis"]) == (-4e5, 3e5, ev.BASIS)
    assert ev.Latency(lo=1.0, hi=2.0, samples=0, period=PERIOD).bounds == (1.0, 2.0)


def test_after_finds_the_first_tick_and_its_clearance():
    ticks = ev.Ticks(lo=0.0, hi=1e6, period=33e6, count=1)
    n, clear = ticks.after(np.array([5e6, 0.5e6, 33.5e6, -1e6]))
    assert n.tolist() == [1, 0, 1, 0]
    assert clear.tolist() == [28e6, -0.5e6, -0.5e6, 1e6]
    assert ticks.at(2) == pytest.approx(0.5e6 + 66e6) and ticks.width == 1e6


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


def shown_with(latency, sent):
    """The frame each write sent at ``sent`` first shows in, for a true send-to-capture ``latency``."""
    return np.ceil((np.asarray(sent) + latency - TIMES.origin) / PERIOD).astype(np.int64)


def test_latency_brackets_the_true_send_to_capture_delay_from_where_each_write_showed():
    rng = np.random.default_rng(5)
    sent = TIMES.origin + rng.uniform(0, 30 * PERIOD, 24)
    true = 1.37 * PERIOD
    shown = shown_with(true, sent)
    measured = ev.latency(sent, shown, TIMES)
    lo = (TIMES.at(shown - 1) - sent).max()
    hi = (TIMES.at(shown) - sent).min()
    assert (measured.lo, measured.hi) == (pytest.approx(lo), pytest.approx(hi))
    assert measured.lo < true <= measured.hi and measured.hi - measured.lo < PERIOD / 4
    assert measured.samples == 24 and measured.delta == pytest.approx(0.5 * (lo + hi))
    assert measured.spread == pytest.approx(0.5 * (hi - lo)) and measured.frames == pytest.approx(
        1.37, abs=0.13
    )
    said = measured.as_dict()
    assert said["lo"] == measured.lo and said["delta"] == measured.delta and said["frames"] == measured.frames
    off = np.r_[sent[:1], sent]
    assert ev.latency(off, np.r_[ev.CALIBRATION, shown], TIMES).samples == 24


def test_latency_writes_that_disagree_a_little_give_the_band_they_disagree_over():
    sent = TIMES.origin + np.array([0.5, 0.4]) * PERIOD
    measured = ev.latency(sent, np.array([2, 3]), TIMES)
    assert (measured.lo, measured.hi) == (pytest.approx(1.5 * PERIOD), pytest.approx(1.6 * PERIOD))
    assert measured.samples == 2


def test_latency_rejects_writes_half_a_frame_apart_and_none_shown():
    sent = TIMES.origin + np.array([0.95, 0.4]) * PERIOD
    with pytest.raises(ev.EvidenceError, match="half a frame"):
        ev.latency(sent, np.array([2, 3]), TIMES)
    with pytest.raises(ev.EvidenceError, match="no calibration"):
        ev.latency(sent, np.full(2, ev.CALIBRATION), TIMES)
    with pytest.raises(ev.EvidenceError, match="no calibration"):
        ev.latency(sent, np.full(2, -LEAD - 1), TIMES)


def test_the_grid_puts_each_frames_tick_its_latency_band_before_its_capture():
    lat = ev.Latency(lo=DELTA - 600_000.0, hi=DELTA + 400_000.0, samples=7, period=PERIOD)
    ticks = ev.grid(TIMES, lat)
    assert (ticks.lo, ticks.hi, ticks.period, ticks.count) == (TICKS.lo, TICKS.hi, PERIOD, 7)
    rng = np.random.default_rng(8)
    true = DELTA + rng.uniform(-600_000.0, 400_000.0)
    sent = TIMES.origin + rng.uniform(0, 30 * PERIOD, 200)
    n, clear = ticks.after(sent)
    sure = clear >= 0
    assert sure.mean() > 0.9
    got = ev.frame_of_tick(n, ticks, TIMES, lat.delta)
    assert got[sure].tolist() == shown_with(true, sent)[sure].tolist()


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


def test_a_window_widened_by_the_lines_spread_holds_every_landing_it_admits():
    """A write echoed clear of its tick lands on one frame; the line's spread opens the window either side."""
    plan = np.array([(4, 1, 10)], dtype=PLAN)
    sends = sends_at(plan.tolist())
    tight = ev.landed(plan, sends, TICKS, TIMES, DELTA)
    assert tight.rows[["first", "last"]].tolist() == [(4, 4)] and tight.ambiguous == 0
    wide = ev.landed(plan, sends, TICKS, TIMES, DELTA, spread=(-0.6 * PERIOD, 0.6 * PERIOD))
    assert wide.rows[["first", "last"]].tolist() == [(3, 5)] and wide.ambiguous == 1 and wide.ok
    late = ev.landed(plan, sends_at(plan.tolist(), sends["t1_ns"] + 2 * PERIOD), TICKS, TIMES, DELTA)
    assert late.late == 1 and not late.ok


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
    assert got.ambiguous == 2 and got.ok
    assert got.rows["landed"][1] == 6
    assert got.rows[["first", "last"]].tolist() == [(2, 3), (5, 6)]


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


def test_a_read_within_a_frame_of_a_landing_may_show_either_side_of_it():
    sends = sends_at([(3, 1, 100)])
    tick = TICKS.at(TICKS.after(sends["t1_ns"])[0][0])
    reads = np.zeros(4, dtype=READ)
    reads["t0_ns"] = [tick - PERIOD - 3e6, tick - 1e6, tick + PERIOD - 1e6, tick + PERIOD + 2e6]
    reads["t1_ns"] = reads["t0_ns"] + 1e6
    reads["o"][[2, 3], 0] = 100
    for value in (0, 100):
        reads["o"][1:3, 0] = value
        got = ev.applied(reads, sends, np.zeros(12), TICKS, TIMES, DELTA)
        assert got.ok and got.ambiguous == 2
    reads["o"][1, 0] = 7
    reads["o"][0, 0], reads["o"][3, 0] = 100, 0
    bad = ev.applied(reads, sends, np.zeros(12), TICKS, TIMES, DELTA).disagree
    assert bad[["read", "want", "got"]].tolist() == [(0, 0, 100), (1, 0, 7), (3, 100, 0)]


def test_applied_lands_a_write_on_its_given_tick():
    sends = sends_at([(3, 1, 100)])
    n = int(TICKS.after(sends["t1_ns"])[0][0])
    reads = np.zeros(2, dtype=READ)
    reads["t0_ns"] = [TICKS.at(n) - 2 * PERIOD, TICKS.at(n + 1) + 5e6]
    reads["t1_ns"] = reads["t0_ns"] + 1e6
    reads["o"][1, 0] = 100
    assert ev.applied(reads, sends, np.zeros(12), TICKS, TIMES, DELTA).ok
    late = ev.applied(reads, sends, np.zeros(12), TICKS, TIMES, DELTA, np.array([n + 3]))
    assert late.disagree[["read", "want", "got"]].tolist() == [(1, 0, 100)]


def echo_record(t0s, polls, heard):
    """Sends of P1 at ``t0s`` (1 ms each), polls as ``(t0, t1)``, and each send's messages heard at a poll."""
    sends = np.zeros(len(t0s), dtype=SEND)
    sends["t0_ns"], sends["t1_ns"], sends["param"] = t0s, np.asarray(t0s) + 1_000_000, 1
    sends["value"] = np.arange(len(t0s)) * 100
    reads = np.zeros(len(polls), dtype=READ)
    reads["t0_ns"], reads["t1_ns"] = np.array(polls).T
    expected = W.expected_messages(sends, [0])
    got = [(reads["t1_ns"][k], m["cc"], m["value"]) for m, k in zip(expected, heard) if k is not None]
    return sends, reads, np.array(got, dtype=RECEIPT)


def test_an_echo_brackets_when_the_device_took_a_write_between_two_replies():
    """After the previous reply's request and the write's start, by the reply it came with."""
    polls = [(0, 2_000_000), (10_000_000, 12_000_000), (20_000_000, 22_000_000)]
    sends, reads, got = echo_record([1_000_000, 15_000_000, 16_000_000], polls, [1, 1, 2, 2, 2, None])
    echo = W.echoed(sends, got, reads, [0])
    assert echo[0].tolist() == [1_000_000, 12_000_000]
    assert echo[1].tolist() == [15_000_000, 22_000_000]
    assert np.isnan(echo[2]).all()
    assert np.isnan(W.echoed(sends, got[:3], reads, [0])[1]).all(), "an MSB alone does not take a write"


def test_a_resent_writes_echo_aligns_to_its_first_copy():
    polls = [(0, 2_000_000), (10_000_000, 12_000_000), (20_000_000, 22_000_000)]
    sends, reads, got = echo_record([1_000_000, 15_000_000, 16_000_000], polls, [1, 1, None, None, 2, 2])
    sends["value"][2] = sends["value"][1]
    got["value"][2:] = W.expected_messages(sends[[1]], [0])["value"]
    echo = W.echoed(sends, got, reads, [0])
    assert echo[1].tolist() == [15_000_000, 22_000_000] and np.isnan(echo[2]).all()
    errors = W.echoed(sends, got, reads[:0], [0], read_errors=np.array([12_000_000]))
    assert errors[0].tolist() == [1_000_000, 12_000_000]


def test_latency_from_brackets_uses_their_far_ends():
    t = true_tick(np.arange(3)) - 5e6
    bracket = np.c_[t, t + 2e6]
    shown = TIMES.frame_of(true_tick(np.arange(3)) + DELTA)
    point = ev.latency(t, shown, TIMES)
    wide = ev.latency(bracket, shown, TIMES)
    assert wide.lo == pytest.approx(point.lo - 2e6) and wide.hi == pytest.approx(point.hi)
    assert ev.latency(np.c_[t, np.r_[np.nan, t[1:]]], shown, TIMES).samples == 2


def test_a_write_lands_where_its_echo_says_late_unechoed_or_on_a_tick():
    """Echoed before its tick: on target; after it: late; across it: on a tick; never: unechoed."""
    plan = np.array([(5, 1, 10), (6, 1, 20), (7, 1, 30), (8, 1, 40)], dtype=PLAN)
    sends = sends_at(plan.tolist())
    tick = true_tick(plan["target"])
    echo = np.c_[sends["t0_ns"], [tick[0] - 3e6, tick[1] + 3e6, tick[2] + 3e6, np.nan]]
    echo[1, 0] = tick[1] + 1e6
    echo[3, 0] = np.nan
    got = ev.landed(plan, sends, TICKS, TIMES, DELTA, echo=echo)
    rows = got.rows
    assert rows["landed"].tolist()[:3] == [5, 7, 8] and rows["landed"][3] == ev.CALIBRATION
    assert min(rows["clear_ns"][:2]) > 0 > rows["clear_ns"][2]
    assert rows["echo_ns"].tolist() == [int(v) for v in echo[:3, 1]] + [-1]
    assert rows[["first", "last"]].tolist()[:3] == [(5, 5), (7, 7), (7, 8)]
    assert (got.late, got.early, got.ambiguous, got.unechoed, got.missing) == (1, 0, 1, 1, 0)
    assert got.slack_ms == pytest.approx((TICKS.lo + 6 * PERIOD - tick[1] - 3e6) / 1e6)
    assert not got.ok and "1 never echoed" in got.summary()
    copy = np.r_[sends, sends[[0]]]
    copy[-1]["t0_ns"] += 1
    again = ev.landed(plan[:1], copy[[0, 4]], TICKS, TIMES, DELTA, resent=np.array([1]))
    assert again.duplicated == 0 and again.rows["sent"].tolist() == [0]


def test_readbacks_hold_only_what_was_echoed():
    sends = sends_at([(5, 1, 100)])
    n = int(TICKS.after(sends["t1_ns"])[0][0])
    reads = np.zeros(1, dtype=READ)
    reads["t0_ns"] = TICKS.at(n + 1) + 5e6
    reads["t1_ns"] = reads["t0_ns"] + 1e6
    lost = np.full((1, 2), np.nan)
    assert ev.applied(reads, sends, np.zeros(12), TICKS, TIMES, DELTA, echo=lost).ok
    assert not ev.applied(reads, sends, np.zeros(12), TICKS, TIMES, DELTA).ok
    copy = ev.applied(reads, np.r_[sends, sends], np.zeros(12), TICKS, TIMES, DELTA, resent=np.array([1]))
    assert not copy.ok and copy.confirmed == 0


def test_the_realized_state_holds_each_slot_from_the_frame_its_write_took_effect():
    rows = np.zeros(4, dtype=ev.LANDED)
    rows["param"], rows["wire"], rows["sent"] = [1, 1, 2, 3], [5, 9, 7, 8], [0, 1, 2, 3]
    rows["landed"] = [-3, 2, 3, ev.CALIBRATION]
    rows["sent"][3] = 3
    state = ev.realized(rows, np.arange(12), 5)
    assert state.shape == (5, 12)
    assert state[:, 0].tolist() == [5, 5, 9, 9, 9]
    assert state[:, 1].tolist() == [1, 1, 1, 7, 7]
    assert state[:, 2].tolist() == [2] * 5 and (state[:, 3:] == np.arange(3, 12)).all()


def receipts(pairs):
    out = np.zeros(len(pairs), dtype=RECEIPT)
    if pairs:
        out["cc"], out["value"] = np.array(pairs).T
    return out


def messages(pairs):
    """Expected messages, one send each."""
    out = np.zeros(len(pairs), dtype=MESSAGE)
    out["cc"], out["value"] = np.array(pairs).T
    out["send"] = np.arange(len(pairs))
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
def test_receipts_are_aligned_in_order_against_what_was_sent(got, matched, ok):
    result = W.received(messages([(1, 10), (33, 5), (2, 7)]), receipts(got))
    assert (result.sent, result.received, result.matched, result.ok) == (3, len(got), matched, ok)
    assert result.extra == len(got) - matched
    assert f"{matched} aligned" in result.summary()


def test_a_lost_burst_costs_only_its_own_messages():
    """A ramp lost mid-burst whose values recur later must not drag the rest of the take out of line."""
    ramp = [(0, v) for v in range(40)]
    sent = ramp + ramp[::-1] + [(5, 1), (5, 2)] * 20
    lost = set(range(40, 60))
    got = [m for i, m in enumerate(sent) if i not in lost]
    sends = np.zeros(len(sent), dtype=SEND)
    sends["t0_ns"] = 1000 * np.arange(len(sent))
    sends["param"], sends["target"] = 1, np.arange(len(sent)) // 10
    result = W.received(messages(sent), receipts(got), sends)
    assert result.matched == len(got) and result.sent - result.matched == len(lost)
    assert result.missing["send"].tolist() == sorted(lost)
    assert result.missing["t0_ns"].tolist() == [1000 * i for i in sorted(lost)]
    assert result.gaps() == [
        {"messages": 20, "sends": [40, 59], "params": [1], "targets": [4, 5], "t0_ns": [40000, 59000]}
    ]
    assert "20 missing from sends 40..59 (P1 for frames 4..5, sent over 0.02 ms)" in result.summary()


def test_separate_gaps_are_reported_apart():
    sent = [(1, v) for v in range(10)]
    result = W.received(messages(sent), receipts([m for i, m in enumerate(sent) if i not in (2, 3, 7)]))
    assert [gap["sends"] for gap in result.gaps()] == [[2, 3], [7, 7]]
    assert W.Received(sent=1, received=1, matched=1).gaps() == []


def test_frame_change_is_the_mean_absolute_difference_from_the_frame_before():
    stack = np.array([0, 2, 2, -1], dtype=np.float32)[:, None, None] * np.ones((1, 2, 3))
    assert np.isnan(ev.frame_change(stack)[0]) and ev.frame_change(stack)[1:].tolist() == [2, 0, 3]


def test_matched_scores_use_the_quantum_where_controls_do_not_vary():
    d_src = np.arange(20, dtype=np.float64)
    d_out = np.ones(20)
    d_out[5] = 3.0
    scores = mt.matched_scores(d_out, d_src, np.array([5, 6]), np.arange(20), 0.25)
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
    assert mt.effective(x) == pytest.approx(n * (1 - rho) / (1 + rho), rel=0.15)
    assert mt.effective(rng.normal(size=n)) == pytest.approx(n, rel=0.05)
    assert mt.effective(np.ones(10)) == 10.0 and mt.effective(np.ones(2)) == 2.0


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
        (verdict(effect=0.1, scatter=0.5), "Gain", True, True, "minimum detectable effect"),
        (verdict(effect=0.1, scatter=5.0), "Gain", True, True, "steps would show its observed mean change"),
        (verdict(effect=-0.1, scatter=5.0), "Gain", True, True, "no mean change to build on"),
    ],
)
def test_why_not_names_the_earliest_broken_link(moved, name, applied_ok, landed_ok, reason):
    assert reason in ev.why_not(moved, name, applied_ok, landed_ok)


def test_a_statistic_says_moved_or_not_seen_with_an_absolute_minimum_detectable_effect():
    assert verdict(z=4.0).verdict == "moved" and verdict(z=4.0).steps_needed == 0
    weak = verdict(effect=0.1, scatter=5.0)
    assert weak.verdict == "not seen" and weak.detectable == pytest.approx(3.0 * 5.0 / np.sqrt(5.0))
    assert weak.steps_needed == int(np.ceil((3.0 * 5.0 / 0.1) ** 2))
    assert verdict().steps_needed == 0 and verdict(effect=-1.0, scatter=1.0).steps_needed == 0
    assert verdict(effect=0.1, scatter=1.0, effective=0.0).detectable == np.inf


def test_a_small_real_effect_beside_a_huge_one_is_never_called_no_effect():
    """The lead's Porthole case: one toggle moves the picture hugely, another parameter only a little."""
    rng = np.random.default_rng(21)
    big = spaced(rng, 240, 12)
    small = spaced(rng, 240, 12, avoid=np.r_[big - 3, big, big + 3])
    out, src = picture(21, np.r_[big, small], jump=np.r_[np.full(12, 60.0), np.full(12, 0.4)])
    order = np.argsort(np.r_[big, small])
    frames, params = np.r_[big, small][order], np.r_[np.full(12, 1), np.full(12, 2)][order]
    kinds = np.full(24, ev.LIVE, dtype=object)
    moved = ev.acted_upon(
        out, src, np.full(240, 100.0), frames, params, {1: "Window", 2: "Curve"}, kinds=kinds
    )
    proof = evidence_of(moved, {1: {ev.LIVE: 12}, 2: {ev.LIVE: 12}})
    rows = {row["param"]: row for row in proof.table()}
    assert rows[1]["verdict"] == "live, moved"
    assert rows[2]["verdict"] in ("live, moved", "live, not seen")
    if rows[2]["verdict"] == "live, not seen":
        assert "minimum detectable effect" in rows[2]["why_not"] and rows[2]["mde"] > 0
    assert all("no effect" not in str(row["why_not"]) for row in rows.values())


def kinds_picture(seed):
    """P1's live steps jump the picture; P2's steps are proven no-ops and do not; P3 is dead and jumps it."""
    rng = np.random.default_rng(seed)
    live = spaced(rng, 240, 12)
    noop = spaced(rng, 240, 12, avoid=np.r_[live - 3, live, live + 3])
    bad = spaced(rng, 240, 8, avoid=np.r_[live - 3, live, live + 3, noop - 3, noop, noop + 3])
    out, src = picture(seed, np.r_[live, bad], jump=40.0)
    frames = np.r_[live, live[:4], noop, bad]
    params = np.r_[np.full(12, 1), np.full(4, 2), np.full(12, 2), np.full(8, 3)]
    kinds = np.r_[
        np.full(12, ev.LIVE, dtype=object),
        np.full(4, ev.NOOP, dtype=object),
        np.full(12, ev.NOOP, dtype=object),
        np.full(8, ev.DEAD, dtype=object),
    ]
    return out, src, frames, params, kinds


def test_steps_are_tested_by_pool_and_proven_no_ops_beside_a_step_are_not_its_partners():
    out, src, frames, params, kinds = kinds_picture(3)
    moved = ev.acted_upon(out, src, np.full(240, 100.0), frames, params, {}, kinds=kinds)
    by = {(v.param, v.pool): v for v in moved}
    assert set(by) == {(1, ev.LIVE_POOL), (2, ev.DEAD_POOL), (3, ev.DEAD_POOL)}
    assert (
        by[1, ev.LIVE_POOL].moved and by[1, ev.LIVE_POOL].partners == () and by[1, ev.LIVE_POOL].steps == 12
    )
    assert by[2, ev.DEAD_POOL].steps == 12 and not by[2, ev.DEAD_POOL].moved
    assert by[3, ev.DEAD_POOL].moved
    assert by[1, ev.LIVE_POOL].critical == pytest.approx(ev.critical(3 * len(ev.LAGS)))
    proof = evidence_of(
        moved, {1: {ev.LIVE: 12}, 2: {ev.NOOP: 16}, 3: {ev.DEAD: 8}, 4: {ev.DEAD: 2}, 5: {ev.LIVE: 1}}
    )
    rows = {row["param"]: row for row in proof.table()}
    assert rows[1]["verdict"] == "live, moved" and rows[1]["why_not"] is None
    assert rows[2]["verdict"] == "never live" and "16 proven no-ops" in rows[2]["why_not"]
    assert rows[3]["verdict"] == "dead, moved" and "contradiction" in rows[3]["why_not"]
    assert rows[4]["verdict"] == "never live" and rows[4]["steps_tested"] == 0
    assert rows[5]["verdict"] == "untested" and "outside the clip" in rows[5]["why_not"]
    assert proof.contradictions == [3]
    assert [e["param"] for e in proof.schedule()] == [2, 4, 5]
    assert any("model: 16 no-op" in line for line in proof.lines())


def test_steps_that_change_no_value_are_not_tested():
    out, src, frames, params, kinds = kinds_picture(4)
    kinds = np.where(params == 1, ev.UNCHANGED, kinds)
    assert all(v.param != 1 for v in ev.acted_upon(out, src, np.zeros(240), frames, params, {}, kinds=kinds))


def evidence_of(moved, predicted):
    plan = np.array([(2, 1, 10)], dtype=PLAN)
    landed = ev.landed(plan, sends_at(plan.tolist()), TICKS, TIMES, DELTA)
    applied = ev.Applied(1, 12, 0, np.zeros(0, dtype=ev.DISAGREE), 1, 0)
    return ev.Evidence("Edge", {}, {}, landed, applied, None, moved, {"ok": True}, predicted)


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
    bad = evidence(received=W.Received(sent=2, received=1, matched=1))
    bad.applied = good.applied
    bad.loaded = {"ok": True}
    assert not bad.ok


def test_evidence_reports_per_parameter_verdicts():
    proof = evidence(received=W.Received(sent=4, received=4, matched=4))
    table = proof.table()
    assert [row["readback"] for row in table] == ["agrees", "disagrees"]
    assert table[0]["acted_upon"] and table[0]["with"] == [2] and table[0]["why_not"] is None
    assert table[1]["why_not"] == "the firmware did not take it" and table[1]["on_target"] == 1
    lines = proof.lines()
    assert any("only together with P2" in line for line in lines)
    assert any("the firmware did not take it" in line for line in lines)
    assert any("4 messages sent" in line for line in lines)
    said = json.loads(proof.to_json())
    assert said["ok"] is False and said["disagree"] == [[0, 2, 4, 20, 0]] and len(said["parameters"]) == 2
    assert said["late"] == [] and said["received"].startswith("4 messages")


def test_a_proven_no_op_next_to_a_live_step_is_not_tested_for_a_contradiction():
    """A live step's picture lingers into the next frame, so the no-op after it cannot be judged."""
    rng = np.random.default_rng(8)
    live = spaced(rng, 240, 12)
    out, src = picture(8, live, jump=40.0, delay=1)
    frames = np.r_[live, live + 1]
    params = np.ones(24, dtype=int)
    kinds = np.r_[np.full(12, ev.LIVE, dtype=object), np.full(12, ev.NOOP, dtype=object)]
    pools = {v.pool for v in ev.acted_upon(out, src, np.full(240, 100.0), frames, params, {}, kinds=kinds)}
    assert pools == {ev.LIVE_POOL}


def bypassed_picture(seed, wet=180, dead=10):
    """Output follows the source, its change amplified from ``wet`` on (bypass), where P5's dead steps lie."""
    rng = np.random.default_rng(seed)
    out, src = picture(seed, [])
    gain = np.where(np.arange(240) >= wet, 4.0, 1.0)[:, None, None, None]
    out = src * gain + rng.normal(0, 0.3, size=src.shape)
    dead = spaced(rng, 240, dead, avoid=np.arange(0, wet + 3))
    frames = np.r_[wet, dead]
    params = np.r_[11, np.full(dead.size, 5)]
    kinds = np.r_[np.array([ev.LIVE], dtype=object), np.full(dead.size, ev.DEAD, dtype=object)]
    regime = ev.Regime((np.arange(240) >= wet).astype(np.int64), {11: 1})
    return out, src, frames, params, kinds, regime


def test_dead_steps_are_tested_against_the_step_free_frames_of_their_own_switch_regime():
    out, src, frames, params, kinds, regime = bypassed_picture(31)
    luma = np.full(240, 100.0)
    blind = {v.param: v for v in ev.acted_upon(out, src, luma, frames, params, {}, kinds=kinds)}
    assert blind[5].moved, "against processed frames the bypassed picture's change looks like a step"
    matched = {
        v.param: v for v in ev.acted_upon(out, src, luma, frames, params, {}, kinds=kinds, regime=regime)
    }
    assert not matched[5].moved and matched[5].steps == 10 and matched[5].unmatched == 0
    assert matched[11].critical == blind[11].critical


def test_steps_whose_regime_has_no_null_are_left_untested_and_said_so():
    out, src, frames, params, kinds, _ = bypassed_picture(32, dead=1)
    regime = ev.Regime(np.isin(np.arange(240), frames[1] + np.r_[0, 1]).astype(np.int64), {11: 1})
    got = {
        v.param: v
        for v in ev.acted_upon(out, src, np.full(240, 100.0), frames, params, {}, kinds=kinds, regime=regime)
    }
    assert got[5].steps == 0 and got[5].unmatched > 0 and not got[5].moved
    assert "too few step-free frames" in ev.why_not(got[5], "Posterize", True, True)
    proof = evidence_of(list(got.values()), {5: {ev.DEAD: 1}, 11: {ev.LIVE: 1}})
    row = {r["param"]: r for r in proof.table()}[5]
    assert row["dead_steps_unmatched"] == got[5].unmatched and row["verdict"] == "never live"
    assert any("with no regime null" in line for line in proof.lines())


def test_one_regime_is_the_null_of_every_step_free_frame():
    out, src, frames, params, kinds, _ = bypassed_picture(33)
    luma = np.full(240, 100.0)
    one = ev.Regime(np.zeros(240, dtype=np.int64))
    assert ev.acted_upon(out, src, luma, frames, params, {}, kinds=kinds, regime=one) == ev.acted_upon(
        out, src, luma, frames, params, {}, kinds=kinds
    )


def test_a_step_is_tested_over_its_landing_window_not_on_one_frame():
    """A jump anywhere in the window shows across it; its last frame alone misses an early landing."""
    rng = np.random.default_rng(21)
    last = spaced(rng, 240, 15) + 2
    true = last - rng.integers(1, 3, size=15)
    out, src = picture(21, true)
    spans = np.full(15, 3)
    wide = ev.acted_upon(out, src, np.zeros(240), last, np.ones(15, int), {}, lags=(0,), spans=spans)[0]
    point = ev.acted_upon(out, src, np.zeros(240), last, np.ones(15, int), {}, lags=(0,))[0]
    assert wide.moved and wide.steps == 15 and not point.moved


def test_windows_that_overlap_make_partners_and_clear_ones_stand_alone():
    rng = np.random.default_rng(3)
    a = spaced(rng, 240, 10)
    out, src = picture(3, a)
    frames, params = np.r_[a, a + 1], np.r_[np.ones(10, int), np.full(10, 2)]
    point = {v.param: v for v in ev.acted_upon(out, src, np.zeros(240), frames, params, {})}
    assert point[1].partners == () and point[2].partners == ()
    spans = np.r_[np.ones(10, int), np.full(10, 2)]
    wide = {v.param: v for v in ev.acted_upon(out, src, np.zeros(240), frames, params, {}, spans=spans)}
    assert wide[1].partners == (2,) and wide[2].partners == (1,) and wide[1].jointly == 10

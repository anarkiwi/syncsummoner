import numpy as np
import pytest

from syncsummoner.compose import evidence as ev
from syncsummoner.device.playout import ASKED, SHOWN
from syncsummoner.device.wire import NO_FLOOR, READ, SEND, TICK

PERIOD = 33_333_333


def ticks_of(rows):
    """Frame ticks from ``(t0, t1, field)`` rows, the transport fields left at zero."""
    out = np.zeros(len(rows), dtype=TICK)
    if len(rows):
        out["t0_ns"], out["t1_ns"], out["field"] = np.array(rows, dtype=np.int64).T
    return out


def device(seconds: float, *, phase: int = 0, poll: int = 4_000_000, rtt: int = 1_900_000):
    """Alternate field_count and readback polls against a device ticking every PERIOD from ``phase``."""
    t, ticks, reads = 0, [], []
    while t < seconds * 1e9:
        ticks.append((t, t + rtt, (t + rtt // 2 - phase) // PERIOD))
        t += poll
        reads.append((t, t + rtt))
        t += poll
    return ticks_of(ticks), reads


def readbacks(reads, sends, start, latch_of):
    """Each read reports the values latched by the device frame it sampled."""
    out = np.zeros(len(reads), dtype=READ)
    for k, (t0, t1) in enumerate(reads):
        now = (t0 + (t1 - t0) // 2) // PERIOD
        o = list(start)
        for s, latch in zip(sends, latch_of):
            if latch <= now:
                o[s["param"] - 1] = s["value"]
        out[k] = (t0, t1, tuple(o))
    return out


def sends_at(times, values, floors):
    out = np.zeros(len(times), dtype=SEND)
    out["t0_ns"], out["t1_ns"] = times, np.asarray(times) + 100_000
    out["param"], out["value"], out["target"], out["floor"] = 1, values, np.arange(len(times)), floors
    return out


def test_counts_respect_command_order():
    ticks = ticks_of([(0, 10, 5), (20, 30, 6), (40, 50, 6)])
    assert ev.count_before(ticks, np.array([9, 10, 11, 45])).tolist() == [-1, -1, 5, 6]
    assert ev.count_after(ticks, np.array([0, 19, 40])).tolist() == [6, 6, ev.UNBOUNDED]


def test_a_readback_pins_each_latch_to_the_tick_after_the_device_took_it():
    ticks, reads = device(3.0)
    times = np.array([0.4e9, 1.1e9, 1.85e9, 2.5e9]).astype(np.int64)
    sends = sends_at(times, [100, 900, 100, 900], NO_FLOOR)
    truth = times // PERIOD + 1
    reads = readbacks(reads, sends, [500] * 12, truth)
    bracket = np.c_[times, times + 2_000_000].astype(np.float64)
    latch = ev.latches(sends, bracket, ticks, reads, [500] * 12)
    assert (latch[:, 0] <= truth).all() and (truth <= latch[:, 1]).all()
    assert (latch[:, 0] == latch[:, 1]).all()


def test_an_unchanged_value_keeps_only_the_send_and_echo_bounds():
    ticks, reads = device(1.0)
    times = np.array([0.3e9]).astype(np.int64)
    sends = sends_at(times, [500], NO_FLOOR)
    reads = readbacks(reads, sends, [500] * 12, times // PERIOD + 1)
    bracket = np.array([[times[0], times[0] + 40_000_000]], dtype=np.float64)
    latch = ev.latches(sends, bracket, ticks, reads, [500] * 12)
    assert latch[0, 0] == ev.count_before(ticks, times)[0] + 1
    assert latch[0, 1] == ev.count_after(ticks, bracket[:, 1])[0] + 1 > latch[0, 0]


def test_an_unechoed_write_has_no_upper_bound():
    ticks, reads = device(1.0)
    sends = sends_at(np.array([300_000_000]), [900], NO_FLOOR)
    latch = ev.latches(sends, np.array([[3e8, np.nan]]), ticks, np.zeros(0, dtype=READ), [0] * 12)
    assert latch[0, 1] == ev.UNBOUNDED


def asked_at(times, frames):
    out = np.zeros(len(times), dtype=ASKED)
    out["t0_ns"], out["t1_ns"], out["frame"] = times, np.asarray(times) + 3_000_000, frames
    return out


def pi_frame(t, flip_phase):
    """The clip frame the Pi has flipped by host time ``t``, flipping every PERIOD from ``flip_phase``."""
    return (np.asarray(t) - flip_phase) // PERIOD


@pytest.mark.parametrize("lead", [0, 3])
def test_the_offset_is_pinned_by_pi_replies_and_floors_when_the_device_follows_the_player(lead):
    flip_phase, tick_phase = 5_000_000, 9_000_000
    ticks, _ = device(4.0, phase=tick_phase - lead * PERIOD)
    asks = np.arange(0.2e9, 3.8e9, 0.137e9).astype(np.int64)
    asked = asked_at(asks, pi_frame(asks + 1_000_000, flip_phase))
    times = np.arange(0.3e9, 3.7e9, 0.211e9).astype(np.int64)
    floors = pi_frame(times - 2_000_000, flip_phase)
    latch_true = (times + 2_000_000 - tick_phase + lead * PERIOD) // PERIOD + 1
    latch = np.c_[latch_true, latch_true]
    flips = np.arange(10, 110)
    shown = np.zeros(flips.size, dtype=SHOWN)
    shown["frame"], shown["t_ns"] = flips, flip_phase + flips * PERIOD + 1_000_000
    found = ev.offset(latch, floors, asked, shown, ticks)
    assert found.lo <= lead <= found.hi
    assert found.exact


def shown_at(times, frames):
    out = np.zeros(len(times), dtype=SHOWN)
    out["t_ns"], out["frame"] = times, frames
    return out


def test_crossed_bounds_mean_the_device_and_player_disagree():
    ticks = ticks_of([(0, 10, 100), (30, 40, 101)])
    asked = asked_at(np.array([20]), [90])
    with pytest.raises(ev.EvidenceError, match="not locked"):
        ev.offset(np.array([[105, 105]]), np.array([100]), asked, shown_at([25], [99]), ticks)


def test_a_write_floor_tightens_the_upper_bound():
    ticks = ticks_of([(0, 10, 100), (30, 40, 103)])
    asked = asked_at(np.array([20]), [98])
    found = ev.offset(np.array([[104, 104]]), np.array([102]), asked, shown_at([25], [99]), ticks)
    assert (found.lo, found.hi) == (2, 2) and found.exact


def test_no_replies_or_no_reports_leave_the_offset_unknown():
    ticks = ticks_of([(0, 10, 100)])
    with pytest.raises(ev.EvidenceError, match="both are needed"):
        ev.offset(
            np.zeros((0, 2), np.int64),
            np.zeros(0, np.int64),
            asked_at(np.array([20]), [90]),
            shown_at([25], [99]),
            ticks,
        )


def plan_of(targets):
    plan = np.zeros(len(targets), dtype=[("target", "<i8"), ("param", "<i8"), ("value", "<i8")])
    plan["target"], plan["param"], plan["value"] = targets, 1, np.arange(len(targets))
    return plan


def test_a_write_lands_on_its_latch_less_the_offset_and_never_before_its_floor():
    plan = plan_of([10, 20, 30])
    sends = sends_at(np.array([1, 2, 3]), [0, 1, 2], [9, 17, 31])
    sends["target"] = [10, 20, 30]
    latch = np.array([[112, 112], [119, 121], [133, 133]])
    got = ev.landed(plan, sends, latch=latch, shift=ev.Offset(102, 102))
    assert got.rows["first"].tolist() == [10, 17, 31]
    assert got.rows["last"].tolist() == [10, 19, 31]
    assert (got.late, got.early, got.ambiguous) == (1, 1, 1)


def test_an_inexact_offset_widens_every_window():
    plan = plan_of([10])
    sends = sends_at(np.array([1]), [0], [9])
    sends["target"] = [10]
    got = ev.landed(plan, sends, latch=np.array([[112, 112]]), shift=ev.Offset(101, 102))
    assert (got.rows["first"][0], got.rows["last"][0], got.ambiguous) == (10, 11, 1)

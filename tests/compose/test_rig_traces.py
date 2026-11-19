"""Command-order evidence against event records the rig wrote: what it actually did, replayed offline."""

from pathlib import Path

import numpy as np
import pytest

from syncsummoner.compose import evidence as ev
from syncsummoner.device.wire import echoed

TRACES = Path(__file__).resolve().parents[1] / "fixtures" / "causal"
PERIOD = 1e9 / 30
NONE = (np.zeros((0, 2), np.int64), np.zeros(0, np.int64))


def trace(name):
    """A recorded lead-in: frame ticks, readbacks, receipts, sends, and the Pi's reports and replies."""
    with np.load(TRACES / f"{name}.npz") as held:
        return {k: held[k] for k in held.files}


def test_a_playing_transport_pins_the_offset_within_a_second_of_polls():
    """Boneoh RGB Bit Rotator on the rig: pairwise bounds leave a frame; the vblank phase closes it."""
    w = trace("bitrotator-leadin")
    assert w["ticks"]["playing"].all() and np.ptp(w["ticks"]["t0_ns"]) < 1.1e9
    told = (ev.count_before(w["ticks"], w["asked"]["t0_ns"]) >= 0) & (
        w["asked"]["frame"] < w["shown"]["frame"].max()
    )
    phased = ev.phase_offset(w["asked"][told], w["shown"], w["ticks"])
    assert phased == ev.Offset(50, 50)
    assert ev.offset(*NONE, w["asked"], w["shown"], w["ticks"], PERIOD) == ev.Offset(50, 50)


def test_each_recorded_start_write_latches_on_one_device_frame():
    """Readbacks bracketed by frame ticks pin every start write to a single device frame."""
    w = trace("bitrotator-leadin")
    took = ev.taken(w["sends"], echoed(w["sends"], w["receipts"], w["reads"], w["cc_map"]))
    latch = ev.latches(w["sends"], took, w["ticks"], w["reads"], w["initial"])
    assert (latch[:, 0] == latch[:, 1]).all()
    assert (latch[:, 0] > ev.count_before(w["ticks"], w["sends"]["t0_ns"])).all()


def test_a_stopped_transport_on_the_rig_is_refused():
    """The session that ran with the transport stopped: its field_count never moved."""
    w = trace("bitrotator-stopped")
    assert np.unique(w["ticks"]["field"]).size == 1
    with pytest.raises(ev.StoppedError, match="transport is stopped"):
        ev.offset(*NONE, w["asked"], w["shown"], w["ticks"], PERIOD)


def test_a_six_minute_render_take_pins_its_offset_though_the_device_clock_wanders():
    """Teletect RamBend in a render on rc.9: no dry control, so three player segments, and 350 s of ticks
    too long for one period end to end; the brackets that meet nearest the middle pin it."""
    w = trace("rambend-render")
    assert np.unique(w["shown"]["base"]).size == 3
    found = ev.offset(*NONE, w["asked"], w["shown"], w["ticks"], PERIOD)
    assert found.exact and 302 <= found.lo <= 303

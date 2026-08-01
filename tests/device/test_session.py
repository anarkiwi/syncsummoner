"""Session: additive addressing, parking, CC budget, caching, settle timing."""

# pylint: disable=missing-function-docstring

import numpy as np
import pytest

from syncsummoner.device import session as sess_mod
from syncsummoner.device.profile import PARAM_COUNT, PARAM_MAX
from syncsummoner.device.session import (
    AddressingError,
    ParkError,
    Session,
    to_device,
)

from .conftest import FakeCapture, FakeClock, FakeTransport


def make_session(transport=None, clock=None, **kwargs):
    """Session wired to fakes and a virtual clock."""
    clock = FakeClock() if clock is None else clock
    transport = FakeTransport() if transport is None else transport
    return Session(transport, sleep=clock.sleep, clock=clock, **kwargs), transport, clock


@pytest.mark.parametrize(
    "value,expected",
    [(True, PARAM_MAX), (False, 0), (0.0, 0), (0.5, 512), (1.0, PARAM_MAX), (2.0, PARAM_MAX)],
)
def test_to_device(value, expected):
    assert to_device(value) == expected


def test_to_device_passes_int_units():
    assert to_device(700) == 700
    assert to_device(-5) == 0
    assert to_device(9999) == PARAM_MAX


def test_offsets_are_relative_to_park():
    session, _, _ = make_session(park_values=[300] * PARAM_COUNT)
    assert session.offsets({1: 0.75}) == {1: 767 - 300}


def test_offsets_clip_at_zero_when_unreachable_from_below():
    session, _, _ = make_session(park_values=[600] * PARAM_COUNT)
    assert session.offsets({1: 0.1}) == {1: 0}


def test_zero_park_makes_cc_absolute():
    session, port, _ = make_session()
    session.park()
    session.set_params({1: 0.75, 7: True})
    state = port.program_state()
    assert state[0] == 767
    assert state[6] == PARAM_MAX


def test_park_drives_manual_and_zeroes_midi():
    port = FakeTransport(manual=[512] * PARAM_COUNT)
    session, _, _ = make_session(port)
    port.midi[:] = 400
    session.park()
    assert np.all(port.manual[:-1] == 0)
    assert np.all(port.midi == 0)
    assert np.all(port.program_state()[:-1] == 0)
    assert sum(1 for c in port.calls if c[0] == "manual") == PARAM_COUNT


def test_park_leaves_the_crossfader_open():
    """P12 gates the output; parking it at zero blacks out every measurement."""
    port = FakeTransport(manual=[512] * PARAM_COUNT)
    session, _, _ = make_session(port)
    session.park()
    assert port.manual[sess_mod.CROSSFADER_INDEX - 1] == PARAM_MAX
    assert session.park_values[sess_mod.CROSSFADER_INDEX - 1] == PARAM_MAX


def test_park_verifies_readback():
    port = FakeTransport(bias=200)
    session, _, _ = make_session(port)
    with pytest.raises(ParkError, match="read back"):
        session.park()


def test_set_params_caches_unchanged():
    session, port, _ = make_session()
    session.set_params({1: 0.5, 2: 0.5})
    port.calls.clear()
    session.set_params({1: 0.5})
    assert not port.calls
    session.set_params({1: 0.6})
    assert port.calls == [("cc", 1, 614)]
    assert session.sent[1] == 614


def test_set_params_rejects_bad_index():
    session, _, _ = make_session()
    with pytest.raises(ValueError, match="1..12"):
        session.set_params({13: 0.5})


def test_cc_budget_paces_aggregate_rate():
    clock = FakeClock()
    session, port, _ = make_session(clock=clock, cc_budget_hz=100.0)
    for value in range(1, 21):
        session.set_params({1: value})
    assert len(port.calls) == 20
    assert clock.now == pytest.approx(19 * 2 / 100.0, rel=1e-6)


def test_cc_budget_does_not_sleep_when_idle():
    clock = FakeClock()
    session, _, _ = make_session(clock=clock, cc_budget_hz=100.0)
    session.set_params({1: 0.5})
    clock.now += 10.0
    session.set_params({1: 0.6})
    assert not clock.slept


def test_zero_budget_rejected():
    with pytest.raises(ValueError, match="positive"):
        Session(FakeTransport(), cc_budget_hz=0)


def test_park_values_length_checked():
    with pytest.raises(ValueError, match="12 entries"):
        Session(FakeTransport(), park_values=[0, 0])


def test_verify_addressing_passes_on_absolute_landing():
    session, _, _ = make_session()
    assert session.verify_addressing(1, target=0.75) == 767


def test_verify_addressing_detects_misaddressing():
    port = FakeTransport()
    port.set_param = lambda index, value: port.midi.__setitem__(index % PARAM_COUNT, int(value))
    session, _, _ = make_session(port, tolerance=8)
    with pytest.raises(AddressingError, match="apparent offset"):
        session.verify_addressing(1, target=0.75)


def test_verify_addressing_detects_gain_error():
    session, _, _ = make_session(FakeTransport(gain=0.5))
    with pytest.raises(AddressingError):
        session.verify_addressing(3, target=0.8)


def test_load_program_absorbs_blackout_and_reparks():
    clock = FakeClock()
    session, port, _ = make_session(clock=clock, load_blackout_s=4.0)
    session.set_params({1: 0.5})
    session.load_program("Isotherm")
    assert port.loaded == "Isotherm"
    assert 4.0 in clock.slept
    assert session.sent == {i: 0 for i in range(1, PARAM_COUNT + 1)}


def test_load_program_can_skip_park():
    session, port, _ = make_session()
    session.load_program("Isotherm", park=False)
    assert not [c for c in port.calls if c[0] == "manual"]
    assert not session.sent


def ramp_frames(diffs, size=4):
    """Frames whose successive mean absolute differences follow ``diffs``."""
    frames = [np.zeros((size, size, 3), dtype=np.float32)]
    for diff in diffs:
        frames.append(frames[-1] + np.float32(diff))
    return frames


def test_settle_frames_reports_plateau():
    quiet = [0.0] * 8
    capture = FakeCapture(ramp_frames(quiet + [0.5, 0.4, 0.3] + [0.0] * 10))
    session, _, _ = make_session()
    assert session.settle_frames(capture, {1: 0.5}, baseline=8, hold=4) == 3


def test_settle_frames_flags_non_settling():
    capture = FakeCapture(ramp_frames([0.0] * 8 + [0.5, -0.5] * 30))
    session, _, _ = make_session()
    assert session.settle_frames(capture, {1: 0.5}, baseline=8, max_frames=20, hold=4) is None


def test_framediff_floor_tracks_measured_noise():
    rng = np.random.default_rng(0)
    frames = [rng.random((8, 8, 3), dtype=np.float32) * 0.01 for _ in range(16)]
    session, _, _ = make_session()
    floor = session.framediff_floor(FakeCapture(frames), 12)
    assert 0.0 < floor < 0.02


def test_framediff_floor_without_frames():
    session, _, _ = make_session()
    assert session.framediff_floor(FakeCapture([None]), 4) == pytest.approx(1e-6)


def test_park_values_property_is_a_copy():
    session, _, _ = make_session(park_values=[5] * PARAM_COUNT)
    values = session.park_values
    values[0] = 999
    assert session.park_values[0] == 5

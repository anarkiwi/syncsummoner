"""Session: additive addressing, parking, CC budget, caching, settle timing."""

# pylint: disable=missing-function-docstring

import contextlib
import types

import numpy as np
import pytest

from syncsummoner.device import journal as jn
from syncsummoner.device import session as sess_mod
from syncsummoner.device.profile import PARAM_COUNT, PARAM_MAX, ParamKind, ParamSpec
from syncsummoner.device.session import (
    AddressingError,
    ParkError,
    Session,
    to_device,
)

from .conftest import FakeCapture, FakeClock, FakeTransport

_TICK = iter(float(n) for n in range(1, 100000))


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


class FakeLink:
    """Source link stub recording its transitions into a shared timeline."""

    def __init__(self, log):
        self.log = log

    @contextlib.contextmanager
    def quiet(self):
        self.log.append(("link", "down"))
        try:
            yield self
        finally:
            self.log.append(("link", "up"))


def logging_session(port, clock, **kwargs):
    """Session whose sleeps land in the transport's call timeline."""

    def sleep(seconds):
        port.calls.append(("sleep", seconds))
        clock.sleep(seconds)

    return Session(port, sleep=sleep, clock=clock, **kwargs)


def test_load_program_holds_the_link_down_across_the_whole_reconfiguration():
    port, clock = FakeTransport(), FakeClock()
    session = logging_session(port, clock, load_blackout_s=4.0)
    session.load_program("Isotherm", link=FakeLink(port.calls), park=False)
    assert port.calls == [
        ("link", "down"),
        ("load", "Isotherm"),
        ("sleep", 4.0),
        ("link", "up"),
        ("video_status",),
    ]


def test_load_program_waits_for_source_lock_before_parking():
    port, clock = FakeTransport(), FakeClock()
    session = logging_session(port, clock, load_blackout_s=0.0)
    session.load_program("Isotherm", link=FakeLink(port.calls), park=True)
    lock = port.calls.index(("video_status",))
    writes = [i for i, call in enumerate(port.calls) if call[0] == "manual"]
    assert writes and lock < min(writes)


def test_load_program_without_a_link_does_not_wait_for_lock():
    port, clock = FakeTransport(), FakeClock()
    session = logging_session(port, clock, load_blackout_s=0.0)
    session.load_program("Isotherm", park=False)
    assert ("video_status",) not in port.calls


def test_wait_source_lock_gives_up_when_the_source_never_returns():
    port, clock = FakeTransport(source_locked=False), FakeClock()
    session = logging_session(port, clock)
    assert session.wait_source_lock(polls=3) is False
    assert port.calls.count(("video_status",)) == 3


def test_load_program_restores_the_link_when_the_load_fails():
    port, clock = FakeTransport(), FakeClock()

    def boom(_name):
        raise RuntimeError("serial died")

    port.load_program = boom
    session = logging_session(port, clock)
    with pytest.raises(RuntimeError):
        session.load_program("Isotherm", link=FakeLink(port.calls))
    assert port.calls == [("link", "down"), ("link", "up")]


def test_load_program_without_a_link_touches_nothing_extra():
    port, clock = FakeTransport(), FakeClock()
    logging_session(port, clock, load_blackout_s=4.0).load_program("Isotherm", park=False)
    assert port.calls == [("load", "Isotherm"), ("sleep", 4.0)]


def test_load_program_journals_whether_the_link_was_used():
    log = jn.Journal(clock=lambda: next(_TICK))
    session, port, _ = make_session()
    session.journal = log
    session.load_program("Isotherm", park=False)
    session.load_program("Colorbars", link=FakeLink(port.calls), park=False)
    assert [e["link"] for e in log.events] == [False, True]


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


def _info(specs):
    return types.SimpleNamespace(params=specs)


def _spec(index, name, kind):
    return ParamSpec(index=index, name=name, native_min=0, native_max=100, kind=kind)


def test_working_point_opens_the_output_and_skips_driven_slots():
    info = _info(
        [
            _spec(1, "Posterize", ParamKind.CONTINUOUS),
            _spec(2, "Flag", ParamKind.BOOLEAN),
            _spec(3, "-", ParamKind.UNUSED),
            _spec(12, "Null 12", ParamKind.CONTINUOUS),
        ]
    )
    port = FakeTransport()
    session, _, _ = make_session(port)
    point = session.working_point(info, exclude=[1])
    assert 1 not in point and 3 not in point
    assert point[2] is False
    assert point[sess_mod.CROSSFADER_INDEX] == 1.0


def test_arm_modulation_binds_operators_and_disarm_clears_them():
    info = _info([_spec(i, f"P{i}", ParamKind.CONTINUOUS) for i in range(1, 6)])
    port = FakeTransport()
    session, _, _ = make_session(port)
    armed = session.arm_modulation(info, ["Sync LFO"], np.random.default_rng(0), exclude=[1, 2], count=2)
    assert armed == ["P3<-Sync LFO", "P4<-Sync LFO"]
    assert port.sources[3] == "Sync LFO"
    session.disarm_modulation()
    assert set(port.sources.values()) == {sess_mod.DISABLED_OPERATOR}


def test_ensure_live_resyncs_once_then_raises():
    port = FakeTransport()
    session, _, _ = make_session(port)
    capture = types.SimpleNamespace(wait_for_content=lambda **_kw: False, wait_for_lock=lambda **_kw: False)
    with pytest.raises(sess_mod.DeviceError, match="power-cycle"):
        session.ensure_live(capture)
    assert port.resyncs >= 1


def test_ensure_live_without_motion_accepts_a_still_stimulus():
    """A probe drives a still pattern; no motion is correct, only the splash is failure."""
    port = FakeTransport()
    session, _, _ = make_session(port)
    capture = types.SimpleNamespace(wait_for_content=lambda **_kw: False, wait_for_lock=lambda **_kw: True)
    session.ensure_live(capture, require_motion=False)
    assert port.resyncs == 0


def test_ensure_live_returns_once_content_arrives():
    port = FakeTransport()
    session, _, _ = make_session(port)
    seen = {"n": 0}

    def content(**_kw):
        seen["n"] += 1
        return seen["n"] > 1

    session.ensure_live(types.SimpleNamespace(wait_for_content=content))
    assert port.resyncs == 1


def test_ensure_live_failure_reports_what_preceded_it():
    """The whole point: name the actions since the device was last healthy."""
    log = jn.Journal(clock=lambda: next(_TICK))
    port = FakeTransport()
    port.resync = lambda **_kw: False
    session, _, _ = make_session(port)
    session.journal = log
    log.record("health", ok=True)
    session.load_program("Isotherm", park=False)
    capture = types.SimpleNamespace(wait_for_content=lambda **_kw: False, wait_for_lock=lambda **_kw: False)
    with pytest.raises(sess_mod.DeviceError) as err:
        session.ensure_live(capture)
    text = str(err.value)
    assert "load_program" in text and "Isotherm" in text
    assert "last healthy" in text


def test_health_marks_the_device_good_and_journals_it():
    log = jn.Journal(clock=lambda: next(_TICK))
    port = FakeTransport()
    session, _, _ = make_session(port)
    session.journal = log
    rng = np.random.default_rng(1)
    capture = types.SimpleNamespace(
        frames=lambda n, **_kw: [rng.random((8, 8, 3)).astype(np.float32) for _ in range(n)],
        chroma_fraction=lambda _f: 0.7,
    )
    assert session.health(capture)["ok"] is True
    assert log.since_last_good() == []


def test_health_reports_the_writes_since_the_last_probe():
    """A 171s gap between probes hid whatever preceded a fault."""
    log = jn.Journal(clock=lambda: next(_TICK))
    port = FakeTransport()
    session, _, _ = make_session(port)
    session.journal = log
    session.set_params({1: 0.2, 2: 0.4})
    session.set_params({1: 0.6})
    first = session.health()
    assert first["writes"] == 3
    assert session.health()["writes"] == 0, "the counter resets so each window stands alone"

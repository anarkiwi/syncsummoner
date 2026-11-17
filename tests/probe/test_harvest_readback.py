"""A sweep opens a setpoint's window only once the device reads back exactly what it was sent."""

# pylint: disable=missing-function-docstring

import pytest

from syncsummoner.probe import harvest as H
from tests.device.conftest import FakeClock, Readback
from tests.probe.test_harvest import CONFIG, Sess


@pytest.mark.parametrize(
    "wrong,named",
    [
        ({7: 1023}, r"P7 read back \[1023\], sent \[512\]"),
        ({2: 511, 11: 0}, r"P2, P11 read back \[511, 0\], sent"),
    ],
    ids=["toggle-coerced", "two-slots"],
)
def test_a_setpoint_the_device_did_not_take_opens_no_window(wrong, named):
    """A toggle slot coerced to a rail reads back other than sent, so its frames would be misattributed."""
    clock, session = FakeClock(), Sess(Readback())
    base = {i: 0.5 for i in range(1, 13)}
    session.set_params(base)
    session.transport.wrong = wrong
    with pytest.raises(H.ReadbackError, match=r"setpoint 0: " + named):
        H.sweep(session, base, [{1: 0.25}], config=CONFIG, sleep=clock.sleep, clock=clock)
    assert clock.now == pytest.approx(CONFIG.settle_s), "it fails once settled, before any dwell"


def test_a_readback_equal_to_the_setpoint_passes():
    session = Sess(Readback())
    session.set_params({7: True, 8: False, 9: 0.5})
    H.verify_readback(session, 3, H.raw_params({7: True, 8: False, 9: 0.5}))
    with pytest.raises(H.ReadbackError, match="setpoint 3: P9 read back"):
        H.verify_readback(session, 3, H.raw_params({7: True, 8: False, 9: 0.25}))

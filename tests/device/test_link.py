"""Link: fbdev blanking of the HDMI source, with no source host and no network."""

# pylint: disable=missing-function-docstring

import subprocess

import pytest

from syncsummoner.device import host as dev_host, link as lk
from syncsummoner.device.link import DOWN, POWERDOWN, UNBLANK, UP, Link, LinkError

from .conftest import FakeClock


class FakeHost:
    """Source host stub: a blanking write drives what ``dpms`` reads back, after ``lag`` polls."""

    def __init__(self, *, lag=0, stuck=False):
        self.commands = []
        self.state = UP
        self.lag = lag
        self.stuck = stuck
        self.pending = None

    def __call__(self, command):
        self.commands.append(command)
        if command.startswith("cat "):
            if self.pending is not None and not self.stuck:
                if self.lag > 0:
                    self.lag -= 1
                else:
                    self.state, self.pending = self.pending, None
            return f"  {self.state}\n"
        self.pending = DOWN if int(command.split()[1]) == POWERDOWN else UP
        return ""

    @property
    def writes(self):
        return [c for c in self.commands if not c.startswith("cat ")]


def make_link(host=None, **kwargs):
    """Link wired to a fake host and a virtual clock."""
    host = FakeHost() if host is None else host
    clock = FakeClock()
    return Link(runner=host, sleep=clock.sleep, clock=clock, **kwargs), host, clock


def test_down_powers_the_framebuffer_off_through_sudo():
    link, host, _ = make_link()
    assert link.down() == DOWN
    assert host.writes == [f"echo {POWERDOWN} | sudo -n tee {lk.DEFAULT_BLANK} > /dev/null"]
    assert link.state() == DOWN


def test_up_unblanks_and_restores_the_link():
    link, host, _ = make_link()
    link.down()
    assert link.up() == UP
    assert host.writes[-1] == f"echo {UNBLANK} | sudo -n tee {lk.DEFAULT_BLANK} > /dev/null"


def test_state_reads_dpms_and_strips():
    link, host, _ = make_link()
    assert link.state() == UP
    assert host.commands[0] == f"cat {lk.DEFAULT_DPMS}"


def test_paths_and_host_are_parameters_not_rig_constants():
    link, host, _ = make_link(blank_path="/sys/fbX/blank", dpms_path="/sys/cardX/dpms")
    link.down()
    assert "/sys/fbX/blank" in host.writes[0]
    assert host.commands[1] == "cat /sys/cardX/dpms"


def test_transition_is_polled_until_dpms_follows():
    link, _, clock = make_link(FakeHost(lag=3), poll_s=0.5)
    assert link.down() == DOWN
    assert clock.slept == [0.5] * 3


def test_stuck_link_raises_within_the_timeout():
    link, _, clock = make_link(FakeHost(stuck=True), timeout_s=1.0, poll_s=0.25)
    with pytest.raises(LinkError, match="want 'Off'"):
        link.down()
    assert clock.now == pytest.approx(1.0)


def test_quiet_drops_for_the_block_and_restores_after():
    link, host, _ = make_link()
    seen = []
    with link.quiet() as held:
        seen.append(held.state())
    assert seen == [DOWN]
    assert link.state() == UP
    assert len(host.writes) == 2


def test_quiet_restores_on_exception():
    link, _, _ = make_link()
    with pytest.raises(ZeroDivisionError):
        with link.quiet():
            raise ZeroDivisionError
    assert link.state() == UP


def test_rejects_non_positive_poll():
    with pytest.raises(ValueError, match="positive"):
        Link(runner=FakeHost(), poll_s=0)


def test_default_runner_is_ssh(monkeypatch):
    seen = {}

    def fake_run(argv, **_kwargs):
        seen["argv"] = argv
        return subprocess.CompletedProcess(argv, 0, stdout=DOWN.encode(), stderr=b"")

    monkeypatch.setattr(dev_host.subprocess, "run", fake_run)
    assert Link().down() == DOWN
    assert seen["argv"][:4] == ["ssh", "-o", "BatchMode=yes", lk.DEFAULT_HOST]

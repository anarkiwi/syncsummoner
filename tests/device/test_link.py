"""Link: fbdev blanking of the HDMI source, with no source host and no network."""

# pylint: disable=missing-function-docstring

import re
import signal
import subprocess
import sys

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
        self.pending = DOWN if int(re.search(r"echo (\d) \| sudo", command)[1]) == POWERDOWN else UP
        return ""

    @property
    def writes(self):
        return [c for c in self.commands if not c.startswith("cat ")]


def make_link(host=None, **kwargs):
    """Link wired to a fake host and a virtual clock."""
    host = FakeHost() if host is None else host
    clock = FakeClock()
    return Link(runner=host, sleep=clock.sleep, clock=clock, **kwargs), host, clock


def test_down_then_up_each_issue_one_drive_and_wait_for_dpms():
    link, host, _ = make_link()
    assert link.down() == DOWN and link.state() == DOWN
    assert link.up() == UP and link.state() == UP
    assert "kill -USR1" in host.writes[0] and f"echo {POWERDOWN} | sudo" in host.writes[0]
    assert "kill -USR2" in host.writes[1] and f"echo {UNBLANK} | sudo" in host.writes[1]
    assert len(host.writes) == 2 and all(lk.DEFAULT_PLAYER_PID in w for w in host.writes)


def shell_runner(tmp_path):
    """Runs each command in a real shell, with a ``sudo`` that just runs its command."""
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    (bin_dir / "sudo").write_text('#!/bin/sh\n[ "$1" = -n ] && shift\nexec "$@"\n')
    (bin_dir / "sudo").chmod(0o755)
    env = {"PATH": f"{bin_dir}:/usr/bin:/bin"}
    return lambda command: subprocess.run(
        ["sh", "-c", command], env=env, capture_output=True, text=True, check=True
    ).stdout


SIGNALLED = (
    "import signal, sys, time\n"
    "for s in (signal.SIGUSR1, signal.SIGUSR2):\n"
    "    signal.signal(s, lambda n, _: print(signal.Signals(n).name, flush=True))\n"
    "print('ready', flush=True)\n"
    "time.sleep(30)\n"
)


@pytest.mark.parametrize(("drive", "want"), [("down", DOWN), ("up", UP)])
def test_a_live_player_is_signalled_and_fbdev_left_alone(tmp_path, drive, want):
    pidfile, blank, dpms = tmp_path / "player.pid", tmp_path / "blank", tmp_path / "dpms"
    dpms.write_text(want)
    with subprocess.Popen([sys.executable, "-c", SIGNALLED], stdout=subprocess.PIPE, text=True) as proc:
        assert proc.stdout.readline() == "ready\n"
        pidfile.write_text(f"{proc.pid}\n")
        link = Link(
            runner=shell_runner(tmp_path), blank_path=str(blank), dpms_path=str(dpms), player_pid=str(pidfile)
        )
        assert getattr(link, drive)() == want
        got = proc.stdout.readline().strip()
        proc.send_signal(signal.SIGKILL)
    assert got == f"SIGUSR{1 if want == DOWN else 2}" and not blank.exists()


@pytest.mark.parametrize("pid", [None, "dead", ""])
@pytest.mark.parametrize(("drive", "value"), [("down", POWERDOWN), ("up", UNBLANK)])
def test_without_a_live_player_fbdev_blanking_drives_the_link(tmp_path, pid, drive, value):
    pidfile, blank, dpms = tmp_path / "player.pid", tmp_path / "blank", tmp_path / "dpms"
    dpms.write_text(DOWN if value == POWERDOWN else UP)
    if pid == "dead":
        with subprocess.Popen(["true"]) as gone:
            gone.wait()
        pid = str(gone.pid)
    if pid is not None:
        pidfile.write_text(pid)
    link = Link(
        runner=shell_runner(tmp_path), blank_path=str(blank), dpms_path=str(dpms), player_pid=str(pidfile)
    )
    getattr(link, drive)()
    assert blank.read_text() == f"{value}\n"


def test_state_reads_dpms_and_strips():
    link, host, _ = make_link()
    assert link.state() == UP
    assert host.commands[0] == f"cat {lk.DEFAULT_DPMS}"


def test_paths_and_host_are_parameters_not_rig_constants():
    link, host, _ = make_link(
        blank_path="/sys/fbX/blank", dpms_path="/sys/cardX/dpms", player_pid="/run/p.pid"
    )
    link.down()
    assert "/sys/fbX/blank" in host.writes[0] and "cat /run/p.pid" in host.writes[0]
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

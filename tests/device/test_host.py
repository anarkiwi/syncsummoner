"""The one ssh runner both playout and link control drive the source host with."""

# pylint: disable=missing-function-docstring

import subprocess

import pytest

from syncsummoner.device import host


def fake_ssh(monkeypatch, stdout=b""):
    """Capture the argv and stdin a runner would shell out with."""
    seen = {}

    def fake_run(argv, **kwargs):
        seen.update(argv=argv, kwargs=kwargs)
        return subprocess.CompletedProcess(argv, 0, stdout=stdout, stderr=b"")

    monkeypatch.setattr(host.subprocess, "run", fake_run)
    return seen


def test_reader_returns_stripped_stdout(monkeypatch):
    seen = fake_ssh(monkeypatch, stdout=b"On\n")
    assert host.ssh_runner("pi@videopi")("cat /sys/dpms") == "On"
    assert seen["argv"] == ["ssh", "-o", "BatchMode=yes", "pi@videopi", "cat /sys/dpms"]
    assert seen["kwargs"]["check"] is True
    assert seen["kwargs"]["input"] is None


def test_writer_pipes_stdin_bytes(monkeypatch):
    seen = fake_ssh(monkeypatch)
    assert host.ssh_runner()("cat > /dev/fb0", b"\x00\x01") == ""
    assert seen["argv"][3] == host.DEFAULT_HOST
    assert seen["kwargs"]["input"] == b"\x00\x01"
    assert seen["kwargs"]["timeout"] == host.DEFAULT_TIMEOUT


def _raising(monkeypatch, err):
    """Make the runner's subprocess call raise, as a broken link does."""

    def fake_run(argv, **kwargs):
        del kwargs
        raise err(argv)

    monkeypatch.setattr(host.subprocess, "run", fake_run)


def test_a_failed_command_carries_ssh_own_diagnosis(monkeypatch):
    denied = b"pi@videopi: Permission denied (publickey).\n"
    _raising(monkeypatch, lambda argv: subprocess.CalledProcessError(255, argv, stderr=denied))
    with pytest.raises(host.HostCommandError) as caught:
        host.ssh_runner()("cat > /dev/shm/syncsummoner-clip.mkv", b"x")
    said = str(caught.value)
    assert "Permission denied (publickey)" in said and "exited 255" in said and host.DEFAULT_HOST in said


def test_a_silent_failure_still_names_the_command(monkeypatch):
    _raising(monkeypatch, lambda argv: subprocess.CalledProcessError(1, argv, stderr=b""))
    with pytest.raises(host.HostCommandError, match="no output"):
        host.ssh_runner()("sudo -n tee /sys/class/graphics/fb0/blank")


def test_a_hung_command_says_it_timed_out(monkeypatch):
    _raising(monkeypatch, lambda argv: subprocess.TimeoutExpired(argv, 30.0))
    with pytest.raises(host.HostCommandError, match="timed out after 30s"):
        host.ssh_runner(timeout=30.0)("sleep 600")


def test_a_long_command_is_shortened_in_the_error(monkeypatch):
    _raising(monkeypatch, lambda argv: subprocess.CalledProcessError(1, argv, stderr=b""))
    with pytest.raises(host.HostCommandError) as caught:
        host.ssh_runner()("x" * 200)
    assert "..." in str(caught.value) and len(str(caught.value)) < 160

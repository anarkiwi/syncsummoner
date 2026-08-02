"""The one ssh runner both playout and link control drive the source host with."""

# pylint: disable=missing-function-docstring

import subprocess

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

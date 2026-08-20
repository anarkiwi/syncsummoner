"""Command execution on the stimulus source host.

One Raspberry Pi both drives the framebuffer feeding the device and owns the
HDMI link out of it, so playout and link control address the same host over the
same ssh transport.
"""

from __future__ import annotations

import subprocess
from typing import Protocol

DEFAULT_HOST = "pi@videopi"
#: Sized for the slowest thing sent: a whole framebuffer piped over the link.
DEFAULT_TIMEOUT = 30.0


class HostCommandError(RuntimeError):
    """A command on the source host failed, or never answered."""


class Runner(Protocol):
    """Runs one command on the source host, returning its stdout.

    ``data`` is piped to the command's stdin; callers that only write ignore the
    result, callers that only read pass no data.
    """

    def __call__(self, command: str, data: bytes | None = None) -> str: ...


def _brief(command: str, limit: int = 60) -> str:
    """First line of a command, short enough to sit in an error message."""
    head = command.strip().splitlines()[0] if command.strip() else command
    return head if len(head) <= limit else f"{head[:limit]}..."


def ssh_runner(host: str = DEFAULT_HOST, *, timeout: float = DEFAULT_TIMEOUT) -> Runner:
    """Return a runner executing commands over BatchMode ssh.

    ssh's own diagnosis is what says whether the key, the host key or the command
    is at fault, so it is carried into the raised error rather than dropped.
    """

    def run(command: str, data: bytes | None = None) -> str:
        try:
            done = subprocess.run(
                ["ssh", "-o", "BatchMode=yes", host, command],
                input=data,
                check=True,
                capture_output=True,
                timeout=timeout,
            )
        except subprocess.CalledProcessError as err:
            said = (err.stderr or b"").decode(errors="replace").strip().splitlines()
            raise HostCommandError(
                f"{host}: `{_brief(command)}` exited {err.returncode}: {said[-1] if said else 'no output'}"
            ) from err
        except subprocess.TimeoutExpired as err:
            raise HostCommandError(f"{host}: `{_brief(command)}` timed out after {timeout:g}s") from err
        return done.stdout.decode(errors="replace").strip()

    return run

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


class Runner(Protocol):
    """Runs one command on the source host, returning its stdout.

    ``data`` is piped to the command's stdin; callers that only write ignore the
    result, callers that only read pass no data.
    """

    def __call__(self, command: str, data: bytes | None = None) -> str: ...


def ssh_runner(host: str = DEFAULT_HOST, *, timeout: float = DEFAULT_TIMEOUT) -> Runner:
    """Return a runner executing commands over BatchMode ssh."""

    def run(command: str, data: bytes | None = None) -> str:
        done = subprocess.run(
            ["ssh", "-o", "BatchMode=yes", host, command],
            input=data,
            check=True,
            capture_output=True,
            timeout=timeout,
        )
        return done.stdout.decode(errors="replace").strip()

    return run

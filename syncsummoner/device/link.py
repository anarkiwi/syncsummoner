"""Rig-level control of the HDMI link from the stimulus source into the device.

Under full KMS ``vcgencmd display_power`` is a no-op; the DRM fbdev emulation
(``vc4drmfb``) is the working control. Blanking the source framebuffer drops the
link, which the device reports as loss of source lock, and clears the display.
"""

from __future__ import annotations

import time
from contextlib import contextmanager
from typing import Callable, Iterator

from .host import DEFAULT_HOST, Runner, ssh_runner

DEFAULT_BLANK = "/sys/class/graphics/fb0/blank"
DEFAULT_DPMS = "/sys/class/drm/card1-HDMI-A-1/dpms"
#: ``FB_BLANK_UNBLANK`` and ``FB_BLANK_POWERDOWN`` from the fbdev ioctl interface.
UNBLANK, POWERDOWN = 0, 4
#: Values ``dpms`` reads back as.
UP, DOWN = "On", "Off"


class LinkError(RuntimeError):
    """The source link did not reach the state it was driven to."""


class Link:
    """The HDMI link out of the source host, driven through fbdev blanking.

    ``runner``, ``sleep`` and ``clock`` are injected so the whole class is
    exercisable with no source host and no network. Writing ``blank`` needs
    passwordless sudo; ``dpms`` is world readable.
    """

    def __init__(
        self,
        host: str = DEFAULT_HOST,
        *,
        blank_path: str = DEFAULT_BLANK,
        dpms_path: str = DEFAULT_DPMS,
        timeout_s: float = 10.0,
        poll_s: float = 0.25,
        runner: Runner | None = None,
        sleep: Callable[[float], None] = time.sleep,
        clock: Callable[[], float] = time.monotonic,
    ):
        if poll_s <= 0:
            raise ValueError(f"poll_s must be positive, got {poll_s}")
        self.blank_path = blank_path
        self.dpms_path = dpms_path
        self.timeout_s = float(timeout_s)
        self.poll_s = float(poll_s)
        self._runner = ssh_runner(host) if runner is None else runner
        self._sleep = sleep
        self._clock = clock

    def state(self) -> str:
        """Link state as the connector's ``dpms`` attribute reports it."""
        return self._runner(f"cat {self.dpms_path}").strip()

    def _blank(self, value: int, want: str) -> str:
        """Drive the framebuffer blanking level and wait for ``dpms`` to follow."""
        self._runner(f"echo {value} | sudo -n tee {self.blank_path} > /dev/null")
        deadline = self._clock() + self.timeout_s
        while True:
            state = self.state()
            if state == want:
                return state
            if self._clock() >= deadline:
                raise LinkError(f"link reads {state!r} after {self.timeout_s}s, want {want!r}")
            self._sleep(self.poll_s)

    def down(self) -> str:
        """Drop the link, so the device sees its source disappear."""
        return self._blank(POWERDOWN, DOWN)

    def up(self) -> str:
        """Restore the link; the source framebuffer comes back cleared."""
        return self._blank(UNBLANK, UP)

    @contextmanager
    def quiet(self) -> Iterator["Link"]:
        """Hold the link down for the block, restoring it on the way out or on error.

        The framebuffer is cleared by the drop, so any stimulus the source was
        displaying has to be pushed again afterwards.
        """
        self.down()
        try:
            yield self
        finally:
            self.up()

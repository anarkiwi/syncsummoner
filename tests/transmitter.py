"""``video status`` payloads for the Videomancer's HDMI transmitter, and a transport replaying them."""

import time

from pyvmancer.liveness import Watch
from pyvmancer.video import VideoStatus

#: A transmitter driving 1080p30 into an attached sink.
LIVE = {
    "source": "hdmi",
    "timing": "1080p30",
    "output": {"timing": "1080p30", "hdmi_connected": True},
    "hdmi_tx": {"vic": 34, "de": "2FE90F"},
}
#: The transmitter stopped, as it is for a few seconds of every program load.
DEAD = dict(LIVE, hdmi_tx={"vic": 0, "de": "000000"})
#: Firmware that reports no transmitter state at all.
UNREPORTED = {key: value for key, value in LIVE.items() if key != "hdmi_tx"}


def at(timing: str) -> dict:
    """:data:`LIVE`, driving ``timing`` instead."""
    return dict(LIVE, timing=timing, output=dict(LIVE["output"], timing=timing))


def blackout(polls: int) -> list[dict]:
    """A load's blackout: the transmitter off for ``polls`` reads, then back."""
    return [DEAD] * polls + [LIVE]


class Live:
    """USB presence and pyvmancer liveness watches, as the transport hands them out.

    ``attached`` lists the presence each check reads in turn, holding the last;
    ``stall`` and ``clock`` configure every watch, as the shell's settings do.
    """

    attached = (True,)
    report = None
    stall = None
    clock = staticmethod(time.monotonic)
    watched: list | None = None

    def present(self):
        """The next scripted presence."""
        seen = list(self.attached)
        self.attached = tuple(seen[1:]) or (seen[-1],)
        return seen[0]

    def watch(self, what):
        """A real pyvmancer ``Watch`` on this fake's presence."""
        if self.watched is not None:
            self.watched.append(what)
        return Watch(what, self.present, report=self.report, stall=self.stall, clock=self.clock)


class Transmitter(Live):
    """A transport whose ``video_status`` replays ``payloads`` in order, then holds the last."""

    def __init__(self, *payloads: dict):
        self.payloads = list(payloads) or [LIVE]
        self.polls = 0

    def video_status(self) -> VideoStatus:
        """The next payload, parsed as the transport parses it."""
        payload = self.payloads[min(self.polls, len(self.payloads) - 1)]
        self.polls += 1
        return VideoStatus.from_json(payload)

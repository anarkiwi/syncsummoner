"""``video status`` payloads for the Videomancer's HDMI transmitter, and a transport replaying them."""

from pyvmancer.video import VideoStatus

#: A transmitter driving 1080p30 into an attached sink.
LIVE = {
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


class Transmitter:
    """A transport whose ``video_status`` replays ``payloads`` in order, then holds the last."""

    def __init__(self, *payloads: dict):
        self.payloads = list(payloads) or [LIVE]
        self.polls = 0

    def video_status(self) -> VideoStatus:
        """The next payload, parsed as the transport parses it."""
        payload = self.payloads[min(self.polls, len(self.payloads) - 1)]
        self.polls += 1
        return VideoStatus.from_json(payload)

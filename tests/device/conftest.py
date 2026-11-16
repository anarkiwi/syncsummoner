"""Fakes standing in for the Videomancer, the capture card and the Pi."""

# pylint: disable=missing-function-docstring

import json

import mido
import numpy as np
import pytest
from pyvmancer import Videomancer
from pyvmancer.ccmap import encode_setting
from pyvmancer.const import CC_MAP_SETTING_KEY, CC_MAP_UNASSIGNED, CC_MSB_MASK, DEFAULT_CC_MSB
from pyvmancer.midi import MidiController, lsb_partner
from pyvmancer.transports.base import MidiTransport
from pyvmancer.video import VideoStatus

from syncsummoner.device.profile import PARAM_COUNT, PARAM_MAX
from tests.transmitter import LIVE

COLORBARS_INFO = {
    "name": "Colorbars",
    "id": 3,
    "parameters": [
        {"name": "-", "min": 0, "max": 100},
        {"name": "Posterize", "min": 0, "max": 7},
        {"name": "Mix", "min": 0, "max": 100},
        {"name": "Null 4", "min": 0, "max": 100},
        {"name": "Gain", "min": 0.0, "max": 2.5},
        {"name": "Steps", "min": 1, "max": 4},
        {"name": "Level", "min": 0, "max": 1},
        {"name": "Pattern", "min": 0, "max": 1},
        {"name": "Blue Only", "min": 0, "max": 1},
        {"name": "Mono", "min": 0, "max": 1},
        {"name": "Bypass", "min": 0, "max": 1},
        {"name": "Fader", "min": 0, "max": 100},
    ],
}


class FakeShell:
    """Serial shell recording every verb and modelling the additive parameter sum.

    The CC map has live slots and settings keys that can disagree, as on 1.0.1: a key write
    reaches its slot only when the value changes, unless ``stuck``, and an ``armed`` slot learns
    the next CC received.
    """

    def __init__(self, manual=None, info=None):
        self.manual = np.zeros(PARAM_COUNT, dtype=int) if manual is None else np.asarray(manual, dtype=int)
        self.sources = {}
        self.midi = np.zeros(PARAM_COUNT, dtype=int)
        self.info = COLORBARS_INFO if info is None else info
        self.calls = []
        self.program = "Colorbars"
        self.state_shape = "flat"
        self.timeout = 5.0
        self.live = list(DEFAULT_CC_MSB)
        self.keys = [encode_setting(m) for m in self.live]
        self.armed = None
        self.stuck = False

    def cc_map(self):
        return {
            "assignments": [
                None if m is None else {"msb": m, "lsb": lsb_partner(m) or 255, "auto": True}
                for m in self.live
            ]
        }

    def get_setting(self, key):
        return json.dumps({"key": key, "type": "uint32", "value": self.keys[key - CC_MAP_SETTING_KEY]})

    def set_setting(self, key, value):
        self.calls.append(("setting", key, value))
        slot = key - CC_MAP_SETTING_KEY
        if value != self.keys[slot]:
            self.keys[slot] = value
            if not self.stuck:
                self.live[slot] = None if value == CC_MAP_UNASSIGNED else value & CC_MSB_MASK

    def receive_cc(self, cc):
        self.calls.append(("cc", cc))
        if self.armed is not None:
            self.live[self.armed - 1], self.armed = cc, None

    def load_program(self, name, settle=1.5):
        self.calls.append(("load", name, settle))
        self.program = name

    def close(self):
        self.calls.append(("close",))

    def combined(self):
        return np.clip(self.manual + self.midi, 0, PARAM_MAX)

    def version(self):
        self.calls.append(("version",))
        return "1.0.0-rc.37"

    def status(self):
        return {"program": self.program, "firmware": "1.0.0-rc.37"}

    def program_info(self):
        self.calls.append(("program_info",))
        return self.info

    def program_state(self):
        return {"m": self.manual.tolist(), "t": [512] * PARAM_COUNT}

    def modulation_status(self):
        if self.state_shape == "flat":
            return {"o": self.combined().tolist(), "m": self.manual.tolist()}
        if self.state_shape == "slots":
            return {"modulators": [{"o": int(v)} for v in self.combined()]}
        if self.state_shape == "list":
            return [1, 2]
        if self.state_shape == "odd":
            return {"o": ["x"] * PARAM_COUNT, "mods": [{"z": 1}] * PARAM_COUNT, "combined": [3] * PARAM_COUNT}
        return {}

    def set_modulation(self, index, manual, time_=None, space=None, slope=None):
        self.calls.append(("set_modulation", index, manual, time_, space, slope))
        self.manual[index] = manual

    def set_source(self, index, source):
        self.calls.append(("set_source", index, source))
        self.sources[index] = source

    def operators(self):
        return {"Sync LFO": {"id": 7}, "Disabled": {"id": 0}, "Pendulum": {"id": 10}}

    def video_status(self):
        return {"input": {"source": "analog", "locked": True}, "timing": "720p60"}

    def video_state(self):
        return VideoStatus.from_json(self.video_status())

    def resync(self, **kwargs):
        self.calls.append(("resync", tuple(sorted(kwargs))))
        return True

    def set_video_timing(self, timing):
        self.calls.append(("video_timing", timing))


class FakeMidiPort(MidiTransport):
    """MIDI port delivering every control change to a :class:`FakeShell`."""

    def __init__(self, shell):
        self.shell = shell

    def send(self, data):
        for message in mido.parse_all(data):
            if message.type == "control_change":
                self.shell.receive_cc(message.control)

    def receive(self, timeout=0.0):
        del timeout
        return b""

    def close(self):
        pass


def videomancer(shell=None, cc_map=None):
    """A real pyvmancer facade over a :class:`FakeShell` and a :class:`FakeMidiPort`."""
    shell = FakeShell() if shell is None else shell
    return Videomancer(midi=MidiController(FakeMidiPort(shell), cc_map=cc_map), shell=shell)


class FakeDevice:
    """Stands in for ``pyvmancer.Videomancer``."""

    def __init__(self, shell=None):
        self.shell = FakeShell() if shell is None else shell
        self.midi = MidiController(FakeMidiPort(self.shell))
        self.has_midi = True
        self.has_shell = True
        self.calls = []
        self.closed = False
        self.cc_assignment = None

    def consume_midi_assign(self):
        self.calls.append(("sink",))

    def reassert_cc_map(self, phase="park"):
        self.calls.append(("reassert", phase))

    def verify_cc_map(self):
        self.calls.append(("verify",))

    def programs(self):
        return ["Colorbars", "Isotherm", "Passthru"]

    def load_program(self, name, settle=1.5):
        self.calls.append(("load", name, settle))
        self.shell.program = name

    def set_param(self, index, value):
        self.calls.append(("set_param", index, value))
        self.shell.midi[index - 1] = int(value)

    def play(self):
        self.calls.append(("play",))

    def stop(self):
        self.calls.append(("stop",))

    def set_bpm(self, bpm):
        self.calls.append(("bpm", bpm))

    def close(self):
        self.closed = True


class FakeTransport:
    """Transport stub for session tests: additive parameters, no pyvmancer."""

    msgs_per_param = 2

    def __init__(self, manual=None, gain=1.0, bias=0, *, source_locked=True):
        start = [512] * PARAM_COUNT if manual is None else manual
        self.manual = np.array(start, dtype=int)
        self.midi = np.zeros(PARAM_COUNT, dtype=int)
        self.gain = gain
        self.bias = bias
        self.calls = []
        self.loaded = None
        self.sources = {}
        self.resyncs = 0
        self.source_locked = source_locked
        self.cc_assignment = None
        self.reasserted = None

    def consume_midi_assign(self):
        self.calls.append(("sink",))

    def reassert_cc_map(self):
        self.calls.append(("reassert",))
        return self.reasserted

    def verify_cc_map(self):
        self.calls.append(("verify",))

    def video_status(self):
        self.calls.append(("video_status",))
        return VideoStatus.from_json(dict(LIVE, source="hdmi", hdmi={"locked": self.source_locked}))

    def set_param(self, index, value):
        self.calls.append(("cc", index, int(value)))
        self.midi[index - 1] = int(value)

    def set_modulation_source(self, index, source):
        self.calls.append(("source", index, source))
        self.sources[index] = source

    def resync(self, **_kw):
        self.resyncs += 1
        return True

    def set_manual(self, index, value, *, time_=None, space=None, slope=None):
        del time_, space, slope
        self.calls.append(("manual", index, int(value)))
        self.manual[index - 1] = int(value)

    def program_state(self):
        return np.clip(self.manual + self.midi * self.gain + self.bias, 0, PARAM_MAX).astype(int)

    def load_program(self, name):
        self.loaded = name
        self.calls.append(("load", name))


class FakeCapture:
    """Serves a scripted list of frames, repeating the last one."""

    def __init__(self, frames):
        self.frames = list(frames)
        self.reads = 0

    def read(self):
        frame = self.frames[min(self.reads, len(self.frames) - 1)]
        self.reads += 1
        return frame


class FakeVideoCapture:
    """cv2.VideoCapture stand-in."""

    def __init__(self, device, api=None):
        self.device = device
        self.api = api
        self.props = {}
        self.frames = []
        self.reads = 0
        self.released = False
        self.opened = True

    def isOpened(self):  # pylint: disable=invalid-name
        """cv2 spelling, kept verbatim."""
        return self.opened

    def set(self, prop, value):
        self.props[prop] = value
        return True

    def read(self):
        if self.reads >= len(self.frames):
            return False, None
        frame = self.frames[self.reads]
        self.reads += 1
        return True, frame

    def release(self):
        self.released = True


class FakeClock:
    """Monotonic clock advanced only by the sleeps handed to it."""

    def __init__(self):
        self.now = 0.0
        self.slept = []

    def __call__(self):
        return self.now

    def sleep(self, seconds):
        self.slept.append(seconds)
        self.now += seconds


@pytest.fixture(name="device")
def device_fixture():
    return FakeDevice()


@pytest.fixture(name="clock")
def clock_fixture():
    return FakeClock()

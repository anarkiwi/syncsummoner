"""Transport: program info parsing, state readback, serial verbs."""

# pylint: disable=missing-function-docstring

import types

import numpy as np
import pytest

from syncsummoner.device import transport as tr
from syncsummoner.device.profile import PARAM_COUNT, ParamKind

from .conftest import COLORBARS_INFO, FakeDevice, FakeShell


def test_program_info_derives_kinds(device):
    info = tr.Transport(device).program_info()
    kinds = [p.kind for p in info.params]
    assert info.name == "Colorbars"
    assert info.program_id == 3
    assert kinds[0] is ParamKind.UNUSED
    assert kinds[3] is ParamKind.UNUSED
    assert kinds[1] is ParamKind.QUANTIZED
    assert info.params[1].steps == 8
    assert kinds[2] is ParamKind.CONTINUOUS
    assert kinds[4] is ParamKind.CONTINUOUS
    assert info.params[5].steps == 4
    assert all(k is ParamKind.BOOLEAN for k in kinds[6:11])
    assert [p.index for p in info.params] == list(range(1, PARAM_COUNT + 1))
    assert len(info.used) == 10


def test_quantized_threshold_follows_sample_budget():
    info = tr.ProgramInfo.from_json({"parameters": [{"name": "Steps", "min": 0, "max": 40}]}, steps=64)
    assert info.params[0].kind is ParamKind.QUANTIZED
    assert info.params[0].steps == 41


def test_degenerate_range_is_unused():
    info = tr.ProgramInfo.from_json({"parameters": [{"name": "Odd", "min": 5, "max": 5}]})
    assert info.params[0].kind is ParamKind.UNUSED


@pytest.mark.parametrize(
    "name,kind",
    [
        ("-", ParamKind.UNUSED),
        ("", ParamKind.UNUSED),
        ("  ", ParamKind.UNUSED),
        ("Null 4", ParamKind.UNUSED),
        ("null 11", ParamKind.UNUSED),
        ("None", ParamKind.UNUSED),
        ("Mix", ParamKind.CONTINUOUS),
        ("Nullify", ParamKind.CONTINUOUS),
        ("Null gate", ParamKind.CONTINUOUS),
    ],
)
def test_slot_names(name, kind):
    info = tr.ProgramInfo.from_json({"parameters": [{"name": name, "min": 0, "max": 100}]})
    assert info.params[0].kind is kind


def test_program_info_loads_when_not_current(device):
    port = tr.Transport(device)
    port.program_info("Isotherm")
    assert ("load", "Isotherm", 0) in device.calls


def test_program_info_skips_load_when_current(device):
    port = tr.Transport(device)
    port.load_program("Colorbars")
    device.calls.clear()
    port.program_info("Colorbars")
    assert not device.calls


def test_current_program_falls_back_to_status(device):
    assert tr.Transport(device).current_program() == "Colorbars"


def test_program_state_prefers_combined(device):
    port = tr.Transport(device)
    device.shell.manual[:] = 100
    port.set_param(1, 400)
    assert port.program_state()[0] == 500
    assert port.program_state()[1] == 100


def test_program_state_accepts_per_slot_dicts(device):
    device.shell.state_shape = "slots"
    device.shell.manual[:] = 7
    assert np.all(tr.Transport(device).program_state() == 7)


def test_program_state_skips_unusable_candidates(device):
    device.shell.state_shape = "odd"
    assert np.all(tr.Transport(device).program_state() == 3)


def test_program_state_ignores_non_mapping_payload(device):
    device.shell.state_shape = "list"
    device.shell.manual[:] = 11
    assert np.all(tr.Transport(device).program_state() == 11)


def test_program_state_falls_back_to_manual(device):
    device.shell.state_shape = "empty"
    device.shell.manual[:] = 42
    assert np.all(tr.Transport(device).program_state() == 42)


def test_program_state_raises_without_vector():
    shell = FakeShell()
    shell.state_shape = "empty"
    shell.program_state = lambda: {}
    with pytest.raises(tr.vm.VmancerError, match="12-slot"):
        tr.Transport(FakeDevice(shell)).program_state()


def test_set_manual_is_zero_based_and_clipped(device):
    port = tr.Transport(device)
    port.set_manual(12, 5000)
    assert ("set_modulation", 11, 1023) in device.shell.calls


@pytest.mark.parametrize("index", [0, 13, -1])
def test_index_validation(device, index):
    with pytest.raises(ValueError, match="1..12"):
        tr.Transport(device).set_param(index, 0.5)


def test_video_status_parses_nested(device):
    status = tr.Transport(device).video_status()
    assert status.timing == "720p60"
    assert status.input_source == "analog"
    assert status.locked is True
    assert status.raw["timing"] == "720p60"


def test_video_status_defaults_to_unlocked():
    assert tr.VideoStatus.from_json({}).locked is False


def test_transport_verbs(device):
    port = tr.Transport(device)
    port.set_video_timing("720p60")
    port.transport_play()
    port.transport_stop()
    port.set_bpm(128)
    assert port.programs() == ["Colorbars", "Isotherm", "Passthru"]
    assert port.firmware() == "1.0.0-rc.37"
    assert ("video_timing", "720p60") in device.shell.calls
    assert [c for c in device.calls if c[0] in ("play", "stop", "bpm")] == [
        ("play",),
        ("stop",),
        ("bpm", 128),
    ]


def test_msgs_per_param(device):
    port = tr.Transport(device)
    assert port.msgs_per_param == 2
    device.midi.high_resolution = False
    assert port.msgs_per_param == 1
    device.has_midi = False
    assert port.msgs_per_param == 1


def test_context_manager_closes(device):
    with tr.Transport(device) as port:
        assert port.device is device
    assert device.closed


def test_open_uses_pyvmancer(monkeypatch):
    seen = {}

    def fake_open(serial_number=None):
        seen["serial"] = serial_number
        return FakeDevice()

    monkeypatch.setattr(tr.vm.Videomancer, "open", staticmethod(fake_open))
    port = tr.Transport.open(serial="E464B0605F113625")
    assert seen["serial"] == "E464B0605F113625"
    assert port.program_info().name == COLORBARS_INFO["name"]


def test_video_status_top_level_wins_over_nested_substatus():
    """Real rc.37 payload: nested hdmi.locked=false must not clobber locked=true."""
    payload = {
        "source": "analog",
        "timing": "PAL",
        "locked": True,
        "analog": {"locked": True, "timing": "PAL"},
        "hdmi": {"locked": False, "connected": True, "timing": "NTSC"},
        "output": {"timing": "PAL", "hdmi_connected": True},
    }
    status = tr.VideoStatus.from_json(payload)
    assert status.locked is True
    assert status.timing == "PAL"
    assert status.input_source == "analog"


def test_source_locked_follows_the_selected_input():
    """An unattended sweep must not record a dead link as measurements."""
    live = tr.VideoStatus.from_json(
        {"source": "hdmi", "locked": True, "hdmi": {"locked": True}, "analog": {"locked": False}}
    )
    assert live.source_locked

    dead = tr.VideoStatus.from_json(
        {"source": "hdmi", "locked": True, "hdmi": {"locked": False}, "analog": {"locked": True}}
    )
    assert not dead.source_locked

    forced = tr.VideoStatus.from_json(
        {"source": "hdmi", "locked": False, "overridden": True, "hdmi": {"locked": True}}
    )
    assert forced.source_locked, "top-level tracks genlock; a forced timing must not read as dead"

    assert not tr.VideoStatus.from_json({"source": "hdmi", "locked": False}).source_locked
    assert tr.VideoStatus.from_json({"source": "analog", "locked": True}).source_locked


def test_resync_bounces_timing_and_restores_input():
    """The timing bounce is the strongest reset serial offers; no reboot verb exists."""
    calls = []

    class Shell:
        """Records the verbs resync issues."""

        def video_status(self):
            return {"source": "hdmi", "timing": "1080p30", "locked": True, "hdmi": {"locked": True}}

        def set_video_timing(self, timing):
            calls.append(("timing", timing))

        def set_video_input(self, source):
            calls.append(("input", source))

    port = types.SimpleNamespace(shell=Shell())
    transport = tr.Transport(port)
    assert transport.resync(sleep=lambda _s: calls.append(("sleep", None)))
    assert [c for c in calls if c[0] != "sleep"] == [
        ("timing", "720p60"),
        ("timing", "1080p30"),
        ("input", "hdmi"),
    ]


def test_resync_reports_failure_when_the_input_stays_dead():
    class Shell:
        """Reports an input that never comes back."""

        def video_status(self):
            return {"source": "hdmi", "timing": "1080p30", "locked": True, "hdmi": {"locked": False}}

        def set_video_timing(self, timing):
            pass

        def set_video_input(self, source):
            pass

    transport = tr.Transport(types.SimpleNamespace(shell=Shell()))
    assert not transport.resync(sleep=lambda _s: None)

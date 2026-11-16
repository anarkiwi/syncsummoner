"""Transport: program info parsing, state readback, serial verbs."""

# pylint: disable=missing-function-docstring

import contextlib

import numpy as np
import pytest
from pyvmancer import CcMapError, ReplyShapeError, ShellError, VmancerError
from pyvmancer.ccmap import encode_setting
from pyvmancer.const import CC_MAP_UNASSIGNED, ShellErrorCode
from pyvmancer.shell import Reply

from syncsummoner.device import transport as tr
from syncsummoner.device.profile import PARAM_COUNT, ParamKind, Role
from tests.transmitter import DEAD, LIVE, UNREPORTED, at

from .conftest import COLORBARS_INFO, FakeDevice, FakeShell, GatedLink, videomancer


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


def test_program_info_names_roles_and_keeps_the_wet_control_continuous(device):
    port = tr.Transport(device)
    info = port.program_info()
    assert [p.role for p in info.params if p.role is not Role.EFFECT] == [Role.BYPASS, Role.WET]
    assert info.params[10].name == "Bypass" and info.params[11].kind is ParamKind.CONTINUOUS
    assert port.program_payload() == COLORBARS_INFO, "the reply is archived exactly as the device gave it"
    unnamed = tr.ProgramInfo.from_json(
        {"parameters": [{"name": f"Null {i}", "min": 0, "max": 100} for i in range(1, 13)]}
    )
    assert unnamed.params[11].role is Role.WET and unnamed.params[11].kind is ParamKind.CONTINUOUS
    assert unnamed.params[11].steps is None and unnamed.params[10].kind is ParamKind.UNUSED


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
    assert device.shell.loads == ["Isotherm"]
    assert device.program == "Isotherm"
    assert device.calls[-3:] == [("sink",), ("parameters", True), ("assign", "load")]


def test_program_info_skips_load_when_current(device):
    port = tr.Transport(device)
    port.load_program("Colorbars")
    device.calls.clear()
    port.program_info("Colorbars")
    assert not device.calls


def test_a_device_reporting_the_plain_name_does_not_stand_in_for_another_author(device):
    port = tr.Transport(device)
    assert port.current_program() == "Colorbars"
    port.program_info("czartroniks/Colorbars")
    assert device.shell.loads == ["Colorbars"]
    assert ("enumerate", "czartroniks") in device.shell.calls
    assert port.current_program() == "czartroniks/Colorbars"


def test_qualified_names_load_once_and_reload_when_the_author_changes(device):
    port = tr.Transport(device)
    port.load_program("czartroniks/Teletext")
    device.shell.calls.clear()
    port.program_info("czartroniks/Teletext")
    assert not device.shell.loads
    port.program_info("Teletext")
    assert [c[:2] for c in device.shell.calls if c[0] in ("enumerate", "load")] == [
        ("enumerate", "lzx"),
        ("load", "Teletext"),
    ]


def test_programs_are_passed_through_qualified(device):
    device.programs = lambda: ["Teletext", "czartroniks/Teletext", "boneoh/RGB Bit Crush"]
    assert tr.Transport(device).programs() == ["Teletext", "czartroniks/Teletext", "boneoh/RGB Bit Crush"]


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
    assert ("set_modulation", 11, 1023, None, None, None) in device.shell.calls


def test_set_manual_forwards_the_macro_arguments(device):
    port = tr.Transport(device)
    port.set_manual(3, 512, time_=700, space=400, slope=512)
    assert ("set_modulation", 2, 512, 700, 400, 512) in device.shell.calls


def test_sweepable_skips_booleans_unused_and_the_crossfader():
    info = tr.ProgramInfo.from_json(
        {
            "name": "X",
            "parameters": [
                {"name": "Posterize", "min": 0, "max": 7},
                {"name": "Smoothing", "min": 0, "max": 100},
                {"name": "-", "min": 0, "max": 100},
                {"name": "Flag", "min": 0, "max": 1},
            ]
            + [{"name": f"Null {i}", "min": 0, "max": 100} for i in range(5, 12)]
            + [{"name": "Mix", "min": 0, "max": 100}],
        }
    )
    assert [p.index for p in info.sweepable()] == [1, 2]
    assert [p.index for p in info.sweepable(1)] == [1]
    assert [p.index for p in info.sweepable(exclude=[1])] == [2]


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
    lines = []
    port = tr.Transport.open(serial="E464B0605F113625", report=lines.append, stall=300.0)
    assert seen["serial"] == "E464B0605F113625"
    shell = port.device.shell
    shell.report("still loading X")
    assert lines == ["still loading X"] and shell.stall == 300.0
    assert port.watch("loading X").stall == 300.0
    assert port.program_info().name == COLORBARS_INFO["name"]


def test_resync_delegates_to_the_device_library():
    """The timing bounce is device protocol; pyvmancer owns and tests it."""
    device = FakeDevice()
    assert tr.Transport(device).resync() is True
    assert any(c[0] == "resync" for c in device.shell.calls)


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


def settings_writes(shell):
    return [c[1:] for c in shell.calls if c[0] == "setting"]


def test_every_load_rewrites_the_whole_cc_map_even_when_it_is_already_right():
    """A key can hold its target while its live slot differs, so no load trusts the map."""
    port = tr.Transport(videomancer())
    shell = port.device.shell
    for program in ("Isotherm", "Colorbars"):
        before = len(settings_writes(shell))
        port.load_program(program)
        writes = settings_writes(shell)[before:]
        assert writes[::2] == [(key, CC_MAP_UNASSIGNED) for key in range(200, 200 + PARAM_COUNT)]
        assert writes[1::2] == [(200 + i, encode_setting(i)) for i in range(PARAM_COUNT)]
        assert port.cc_assignment.summary() == "cc-map: assigned P1..P12 -> CC0..11 (was: ok)"
    assert shell.live == list(range(PARAM_COUNT))


def test_a_load_repairs_a_live_map_its_settings_keys_do_not_show():
    port = tr.Transport(videomancer())
    port.device.shell.live[11] = 0
    port.load_program("Isotherm")
    assert port.device.shell.live == list(range(PARAM_COUNT))
    assert port.cc_assignment.before.mismatched == (12,)
    port.verify_cc_map()


def test_a_map_that_does_not_take_after_the_write_raises():
    port = tr.Transport(videomancer())
    shell = port.device.shell
    shell.live[3], shell.stuck = 50, True
    with pytest.raises(CcMapError, match="P4 CC50"):
        port.load_program("Isotherm")
    with pytest.raises(CcMapError):
        port.verify_cc_map()


def test_the_sink_cc_is_one_no_slot_listens_on():
    port = tr.Transport(videomancer())
    port.consume_midi_assign()
    sink = port.device.shell.calls[-1]
    assert sink == ("cc", 119) and 119 not in port.device.shell.live
    assert port.reassert_cc_map() is None


@pytest.mark.parametrize(
    ("payload", "timing", "expected"),
    [
        (LIVE, "1080p30", True),
        (LIVE, "1080P30", True),
        (at("NTSC"), "ntsc", True),
        (DEAD, "1080p30", False),
        (dict(LIVE, hdmi_tx={"vic": 34, "de": "000000"}), "1080p30", False),
        (dict(LIVE, hdmi_tx={"vic": 0, "de": "2FE90F"}), "1080p30", False),
        (at("720p60"), "1080p30", False),
        (dict(LIVE, output={"timing": "1080p30", "hdmi_connected": False}), "1080p30", False),
        (UNREPORTED, "1080p30", None),
        (dict(LIVE, hdmi_tx={"vic": 34}), "1080p30", None),
        ({k: v for k, v in LIVE.items() if k != "output"}, "1080p30", None),
    ],
    ids=[
        "driving",
        "case-blind",
        "named-timing",
        "dead",
        "no-data-enable",
        "no-vic",
        "wrong-timing",
        "no-sink",
        "no-hdmi-tx",
        "partial-hdmi-tx",
        "no-output",
    ],
)
def test_transmitting_needs_an_active_transmitter_at_the_session_timing(payload, timing, expected):
    """Missing fields leave the signal unknown; it is never assumed to be there."""
    assert tr.transmitting(tr.VideoStatus.from_json(payload), timing) is expected


def test_a_link_gated_load_restores_the_link_on_the_reply_then_waits_for_the_run():
    device = FakeDevice()
    link = GatedLink(device.shell)
    port = tr.Transport(device)
    port.load_program("Passthru", hold=link.quiet)
    assert ("load", "Passthru", False) in device.shell.calls, "the command runs with the link down"
    assert link.calls == ["down", "up"] and device.shell.program == "Passthru"
    assert port.current_program() == "Passthru"
    assert device.calls[-2:] == [("parameters", True), ("assign", "load")]


def test_waiting_for_the_run_inside_the_hold_never_ends():
    """The fake's dependency is real: a run awaited with the link down is a deadlock."""
    shell = FakeShell()
    with GatedLink(shell).quiet():
        shell.command("program", "load", "Passthru")
        with pytest.raises(AssertionError, match="link_up=False: deadlock"):
            shell.wait_loaded("Passthru", watch=shell.watch("loading Passthru"))


def test_a_liveness_wait_cannot_run_while_the_link_is_held_down():
    device = FakeDevice()
    port = tr.Transport(device)
    early = port.watch("waiting for the transmitter")

    @contextlib.contextmanager
    def hold():
        with GatedLink(device.shell).quiet():
            yield
            with pytest.raises(tr.HeldLinkError, match="held down for Passthru"):
                early.observe("dead")
            port.watch("waiting for picture")

    with pytest.raises(tr.HeldLinkError, match="waiting for picture"):
        port.load_program("Passthru", hold=hold)
    assert device.shell.link_up
    port.watch("waiting for picture").observe("live")


def test_a_load_refused_with_the_link_down_still_restores_it():
    device = FakeDevice()
    device.shell.load_error = 7
    link = GatedLink(device.shell)
    port = tr.Transport(device)
    with pytest.raises(ShellError):
        port.load_program("Passthru", hold=link.quiet)
    assert link.calls == ["down", "up"] and device.shell.link_up
    assert port.current_program() == "Colorbars"
    port.watch("waiting for picture").observe("live")


def test_a_load_answering_other_than_ok_raises_after_restoring_the_link():
    device = FakeDevice()
    command = device.shell.command
    device.shell.command = lambda *parts, **kw: (
        Reply("program", "busy") if parts[1] == "load" else command(*parts, **kw)
    )
    link = GatedLink(device.shell)
    with pytest.raises(VmancerError, match="busy"):
        tr.Transport(device).load_program("Passthru", hold=link.quiet)
    assert link.calls == ["down", "up"]


def test_progress_lines_say_when_the_link_is_down():
    device = FakeDevice()
    port = tr.Transport(device)
    lines = []
    report = port.annotate(lines.append)

    @contextlib.contextmanager
    def hold():
        report("still loading Passthru")
        yield

    report("still loading Passthru")
    port.load_program("Passthru", hold=hold)
    assert lines == ["still loading Passthru", "still loading Passthru, link down"]
    assert port.annotate(None) is None


def test_firmware_without_authors_loads_plain_names_without_enumerating():
    device = FakeDevice()

    def authors():
        device.shell.calls.append(("authors",))
        raise ShellError(ShellErrorCode.PARSE_ERROR, "usage: programs authors")

    device.shell.authors = authors
    port = tr.Transport(device)
    port.load_program("Passthru")
    port.load_program("Colorbars")
    assert device.shell.loads == ["Passthru", "Colorbars"]
    assert [c for c in device.shell.calls if c[0] in ("authors", "enumerate")] == [("authors",)]


@pytest.mark.parametrize(
    "error",
    [ShellError(ShellErrorCode.SERVICE_UNAVAILABLE, "busy"), ReplyShapeError("programs", "{}", ["authors"])],
)
def test_an_authors_failure_is_never_read_as_firmware_without_authors(error):
    """Loading a plain name unenumerated would load the active author's program of that name."""
    device = FakeDevice()

    def authors():
        raise error

    device.shell.authors = authors
    with pytest.raises(type(error)):
        tr.Transport(device).load_program("Passthru")
    assert not device.shell.loads


def test_authors_and_route_are_read_from_the_device():
    """The authors listing and Vid Route Mode, as the render's listing error and route check use them."""
    device = FakeDevice()
    port = tr.Transport(device)
    assert port.authors() == ["lzx", "czartroniks"] and port.route() == 1
    device.shell.routed = 4
    assert port.route() == 4


def test_firmware_without_authors_reports_none():
    device = FakeDevice()

    def authors():
        raise ShellError(ShellErrorCode.UNKNOWN_COMMAND, "unknown command")

    device.shell.authors = authors
    assert tr.Transport(device).authors() is None

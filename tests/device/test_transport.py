"""Transport: program info parsing, state readback, serial verbs."""

# pylint: disable=missing-function-docstring

import contextlib

import numpy as np
import pytest
from pyvmancer import CcMapError, ReplyShapeError, ShellError, VmancerError
from pyvmancer.const import ShellErrorCode
from pyvmancer.shell import Reply

from syncsummoner.device import transport as tr
from syncsummoner.device.declared import UndeclaredProgramError
from syncsummoner.device.profile import PARAM_COUNT, ParamKind
from tests.transmitter import DEAD, LIVE, UNREPORTED, at

from .conftest import (
    COLORBARS_DECLARED,
    COLORBARS_INFO,
    DECLARED,
    FakeDevice,
    FakeShell,
    GatedLink,
    videomancer,
)


def test_program_info_derives_kinds(device):
    info = tr.Transport(device, DECLARED).program_info()
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


def test_program_info_takes_labels_from_the_device_and_kinds_from_the_declaration(device):
    port = tr.Transport(device, DECLARED)
    info = port.program_info()
    assert [p.name for p in info.params] == [p["name"] for p in COLORBARS_INFO["parameters"]]
    assert info.declared is COLORBARS_DECLARED and DECLARED.asked[-1] == "Colorbars"
    assert info.params[11].kind is ParamKind.CONTINUOUS and info.params[11].steps is None
    assert (info.params[4].native_min, info.params[4].native_max) == (0.0, 25.0)
    assert port.program_payload() == COLORBARS_INFO, "the reply is archived exactly as the device gave it"


def test_a_program_loaded_by_author_is_declared_by_that_name(device):
    port = tr.Transport(device, DECLARED)
    port.load_program("czartroniks/Colorbars")
    port.program_info()
    assert DECLARED.asked[-1] == "czartroniks/Colorbars"


def test_an_undeclared_program_is_an_error_not_a_guess(device):
    port = tr.Transport(device, DECLARED)
    with pytest.raises(UndeclaredProgramError, match="'Isotherm'"):
        port.describe({**COLORBARS_INFO, "name": "Isotherm"})


def test_without_a_declaration_every_slot_is_unused():
    info = tr.ProgramInfo.from_json(COLORBARS_INFO)
    assert not info.used and not info.initial and info.params[10].name == "Bypass"


def test_initial_is_the_declared_state_in_device_units(device):
    info = tr.Transport(device, DECLARED).program_info()
    assert info.initial == {2: 0, 3: 1023, 5: 512, 6: 682, 7: 0, 8: 0, 9: 0, 10: 0, 11: 0, 12: 0}


def test_program_info_loads_when_not_current(device):
    port = tr.Transport(device, DECLARED)
    port.program_info("Isotherm")
    assert device.shell.loads == ["Isotherm"]
    assert device.program == "Isotherm"
    assert device.calls[-2:] == [("parameters", True), ("verify",)]


def test_program_info_skips_load_when_current(device):
    port = tr.Transport(device, DECLARED)
    port.load_program("Colorbars")
    device.calls.clear()
    port.program_info("Colorbars")
    assert not device.calls


def test_a_device_reporting_the_plain_name_does_not_stand_in_for_another_author(device):
    port = tr.Transport(device, DECLARED)
    assert port.current_program() == "Colorbars"
    port.program_info("czartroniks/Colorbars")
    assert device.shell.loads == ["Colorbars"]
    assert ("enumerate", "czartroniks") in device.shell.calls
    assert port.current_program() == "czartroniks/Colorbars"


def test_qualified_names_load_once_and_reload_when_the_author_changes(device):
    port = tr.Transport(device, DECLARED)
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
    assert tr.Transport(device, DECLARED).programs() == [
        "Teletext",
        "czartroniks/Teletext",
        "boneoh/RGB Bit Crush",
    ]


def test_current_program_falls_back_to_status(device):
    assert tr.Transport(device, DECLARED).current_program() == "Colorbars"


def test_program_state_prefers_combined(device):
    port = tr.Transport(device, DECLARED)
    device.shell.manual[:] = 100
    port.set_param(1, 400)
    assert port.program_state()[0] == 500
    assert port.program_state()[1] == 100


def test_program_state_accepts_per_slot_dicts(device):
    device.shell.state_shape = "slots"
    device.shell.manual[:] = 7
    assert np.all(tr.Transport(device, DECLARED).program_state() == 7)


def test_program_state_skips_unusable_candidates(device):
    device.shell.state_shape = "odd"
    assert np.all(tr.Transport(device, DECLARED).program_state() == 3)


def test_program_state_ignores_non_mapping_payload(device):
    device.shell.state_shape = "list"
    device.shell.manual[:] = 11
    assert np.all(tr.Transport(device, DECLARED).program_state() == 11)


def test_program_state_falls_back_to_manual(device):
    device.shell.state_shape = "empty"
    device.shell.manual[:] = 42
    assert np.all(tr.Transport(device, DECLARED).program_state() == 42)


def test_program_state_raises_without_vector():
    shell = FakeShell()
    shell.state_shape = "empty"
    shell.program_state = lambda: {}
    with pytest.raises(tr.vm.VmancerError, match="12-slot"):
        tr.Transport(FakeDevice(shell)).program_state()


def test_set_manual_is_zero_based_and_clipped(device):
    port = tr.Transport(device, DECLARED)
    port.set_manual(12, 5000)
    assert ("set_modulation", 11, 1023) in device.shell.calls


def test_sweepable_is_every_declared_continuous_or_quantized_slot():
    info = tr.ProgramInfo.from_json(COLORBARS_INFO, COLORBARS_DECLARED)
    assert [p.index for p in info.sweepable()] == [2, 3, 5, 6, 12]
    assert [p.index for p in info.sweepable(1)] == [2]
    assert [p.index for p in info.sweepable(exclude=[2, 12])] == [3, 5, 6]


@pytest.mark.parametrize("index", [0, 13, -1])
def test_index_validation(device, index):
    with pytest.raises(ValueError, match="1..12"):
        tr.Transport(device, DECLARED).set_param(index, 0.5)


def test_video_status_parses_nested(device):
    status = tr.Transport(device, DECLARED).video_status()
    assert status.timing == "720p60"
    assert status.input_source == "analog"
    assert status.locked is True
    assert status.raw["timing"] == "720p60"


def test_video_status_defaults_to_unlocked():
    assert tr.VideoStatus.from_json({}).locked is False


def test_transport_verbs(device):
    port = tr.Transport(device, DECLARED)
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
    port = tr.Transport(device, DECLARED)
    assert port.msgs_per_param == 2
    device.midi.high_resolution = False
    assert port.msgs_per_param == 1
    device.has_midi = False
    assert port.msgs_per_param == 1


def test_context_manager_closes(device):
    with tr.Transport(device, DECLARED) as port:
        assert port.device is device
    assert device.closed


def test_open_uses_pyvmancer(monkeypatch):
    seen = {}

    def fake_open(serial_number=None):
        seen["serial"] = serial_number
        return FakeDevice()

    monkeypatch.setattr(tr.vm.Videomancer, "open", staticmethod(fake_open))
    lines = []
    port = tr.Transport.open(serial="E464B0605F113625", report=lines.append, stall=300.0, library=DECLARED)
    assert seen["serial"] == "E464B0605F113625"
    shell = port.device.shell
    shell.report("still loading X")
    assert lines == ["still loading X"] and shell.stall == 300.0
    assert port.watch("loading X").stall == 300.0
    assert port.program_info().name == COLORBARS_INFO["name"]


def test_resync_delegates_to_the_device_library():
    """The timing bounce is device protocol; pyvmancer owns and tests it."""
    device = FakeDevice()
    assert tr.Transport(device, DECLARED).resync() is True
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


def sent_ccs(shell):
    return [c[1] for c in shell.calls if c[0] == "cc"]


def test_a_load_reads_the_cc_map_back_and_never_rewrites_it_or_sends_a_sink_cc():
    """The map is persistent settings and CC119 is a MIDI reset, so a load only checks."""
    port = tr.Transport(videomancer())
    shell = port.device.shell
    for program in ("Isotherm", "Colorbars"):
        port.load_program(program)
    assert not settings_writes(shell)
    assert not [cc for cc in sent_ccs(shell) if 102 <= cc <= 119]
    assert shell.live == list(range(PARAM_COUNT)) and not port.cc_faults


def test_a_live_map_off_the_sent_ccs_fails_the_load_and_is_recorded():
    port = tr.Transport(videomancer())
    shell = port.device.shell
    shell.live[11] = 0
    with pytest.raises(CcMapError, match="P12 CC0"):
        port.load_program("Isotherm")
    assert shell.live[11] == 0 and not settings_writes(shell)
    assert [f["when"] for f in port.cc_faults] == ["after loading Isotherm"]
    assert "P12 CC0" in port.cc_faults[0]["error"] and port.cc_faults[0]["time"] > 0


def test_the_take_end_check_is_recorded_as_such():
    port = tr.Transport(videomancer())
    port.verify_cc_map()
    assert not port.cc_faults
    port.device.shell.live[3] = 50
    with pytest.raises(CcMapError, match="P4 CC50"):
        port.verify_cc_map()
    assert [f["when"] for f in port.cc_faults] == ["at the end of a take"]


def test_check_cc_map_returns_what_the_device_reads_back():
    device = FakeDevice()
    port = tr.Transport(device)
    assert port.check_cc_map("now") is None and device.calls == [("verify",)]
    device.cc_error = CcMapError("device CC map differs (P2 CC9)")
    with pytest.raises(CcMapError):
        port.check_cc_map("after parking")
    assert port.cc_faults[0]["when"] == "after parking"


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
    port = tr.Transport(device, DECLARED)
    port.load_program("Passthru", hold=link.quiet)
    assert ("load", "Passthru", False) in device.shell.calls, "the command runs with the link down"
    assert link.calls == ["down", "up"] and device.shell.program == "Passthru"
    assert port.current_program() == "Passthru"
    assert device.calls[-2:] == [("parameters", True), ("verify",)]


def test_waiting_for_the_run_inside_the_hold_never_ends():
    """The fake's dependency is real: a run awaited with the link down is a deadlock."""
    shell = FakeShell()
    with GatedLink(shell).quiet():
        shell.command("program", "load", "Passthru")
        with pytest.raises(AssertionError, match="link_up=False: deadlock"):
            shell.wait_loaded("Passthru", watch=shell.watch("loading Passthru"))


def test_a_liveness_wait_cannot_run_while_the_link_is_held_down():
    device = FakeDevice()
    port = tr.Transport(device, DECLARED)
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
    port = tr.Transport(device, DECLARED)
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
        tr.Transport(device, DECLARED).load_program("Passthru", hold=link.quiet)
    assert link.calls == ["down", "up"]


def test_progress_lines_say_when_the_link_is_down():
    device = FakeDevice()
    port = tr.Transport(device, DECLARED)
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
    port = tr.Transport(device, DECLARED)
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
        tr.Transport(device, DECLARED).load_program("Passthru")
    assert not device.shell.loads


def test_authors_and_route_are_read_from_the_device():
    """The authors listing and Vid Route Mode, as the render's listing error and route check use them."""
    device = FakeDevice()
    port = tr.Transport(device, DECLARED)
    assert port.authors() == ["lzx", "czartroniks"] and port.route() == 1
    device.shell.routed = 4
    assert port.route() == 4


def test_firmware_without_authors_reports_none():
    device = FakeDevice()

    def authors():
        raise ShellError(ShellErrorCode.UNKNOWN_COMMAND, "unknown command")

    device.shell.authors = authors
    assert tr.Transport(device, DECLARED).authors() is None


def _enumerations(device):
    return [c[1] for c in device.shell.calls if c[0] == "enumerate"]


def test_a_program_of_the_active_author_loads_without_rebuilding_its_index(device):
    port = tr.Transport(device, DECLARED)
    port.load_program("Passthru")
    port.load_program("Colorbars")
    assert not _enumerations(device) and device.shell.loads == ["Passthru", "Colorbars"]


@pytest.mark.parametrize(
    "name,listing,enumerated",
    [
        ("czartroniks/Fray", ["Passthru", "czartroniks/Fray"], []),
        ("czartroniks/Fray", ["Passthru", "czartroniks/fray", "boneoh/FRAY"], ["czartroniks"]),
        ("czartroniks/Fray", ["Passthru", "czartroniks/Quilt"], ["czartroniks"]),
        ("czartroniks/YUV_Amp", ["czartroniks/YUV Amp", "boneoh/Quilt"], []),
    ],
    ids=["unique", "shared-with-another-author", "unlisted", "underscore-matches-space"],
)
def test_another_authors_program_skips_enumeration_only_when_its_name_is_unique(
    device, name, listing, enumerated
):
    """The firmware searches every author for a name the active index lacks; a shared name needs one."""
    device.programs = lambda: listing
    port = tr.Transport(device, DECLARED)
    port.programs()
    port.load_program(name)
    assert _enumerations(device) == enumerated


def test_an_unlisted_session_enumerates_another_author(device):
    tr.Transport(device, DECLARED).load_program("czartroniks/Fray")
    assert _enumerations(device) == ["czartroniks"]


@pytest.mark.parametrize(
    "index, value, sent",
    [(8, 511, 0), (8, 512, 1023), (8, 1, 0), (8, True, 1023), (8, False, 0), (1, 300, 300)],
    ids=["below", "threshold", "least", "true", "false", "knob"],
)
def test_a_toggle_slot_is_sent_as_the_firmware_reads_it(device, index, value, sent):
    """The firmware switches at 512, so a toggle below it goes out off however nonzero it is."""
    tr.Transport(device, DECLARED).set_param(index, value)
    assert device.calls[-1] == ("set_param", index, sent)

"""Passes are bracketed by the device's HDMI transmitter; a dark take is controlled before it is judged."""

# pylint: disable=missing-function-docstring

import dataclasses

import pytest

from syncsummoner.compose import lock as L
from syncsummoner.compose import render as R
from syncsummoner.device.recorder import InputRouteError, RouteError
from syncsummoner.probe import harvest as H
from tests import transmitter as tx
from tests.compose import PROBED, rig_transport
from tests.compose.locked import LOCKED, locked_pass, render_locked

LIVE = tx.at(LOCKED.timing)
DARK = R.TakeReport(frames=4)


@pytest.mark.parametrize(
    ("video", "when", "recorded"),
    [
        ([tx.UNREPORTED], "Teletext: the firmware reports no transmitter state after the load", 0),
        (
            [LIVE] * 2 + [tx.DEAD],
            f"Teletext: the transmitter is not driving {LOCKED.timing} as the pass ended",
            1,
        ),
    ],
    ids=["unreported", "dropped-during"],
)
def test_a_transmitter_lost_at_a_pass_boundary_fails_the_pass(monkeypatch, tmp_path, video, when, recorded):
    """A pass whose transmitter was not driving at both ends recorded the card, not the program."""
    rig = locked_pass(monkeypatch, transport=rig_transport(*video))
    with pytest.raises(R.BlankTakeError, match=when):
        render_locked(rig, tmp_path / "take.mkv")
    assert rig.steps.count("record") == recorded and "inspect" not in rig.steps


def test_a_device_left_on_another_input_fails_the_take_before_it_is_kept(monkeypatch, tmp_path):
    """The selected input is verified once the program is up, before anything is judged or locked."""
    rig = locked_pass(monkeypatch, transport=rig_transport(dict(LIVE, source="analog")))
    with pytest.raises(InputRouteError, match="Teletext: the device reads input 'analog'"):
        render_locked(rig, tmp_path / "take.mkv")
    assert "inspect" not in rig.steps and not L.lock_path(tmp_path / "take.mkv").exists()


def controlled(monkeypatch, *takes, passthru=PROBED, transport=None):
    """A locked pass whose takes judge as ``takes`` in turn, holding the last; its control as ``passthru``."""
    rig = locked_pass(monkeypatch, transport=transport)
    seen = list(takes)
    monkeypatch.setattr(R, "inspect_take", lambda *a, **k: seen.pop(0) if len(seen) > 1 else seen[0])
    monkeypatch.setattr(
        H, "settle", lambda *a, program, **k: rig.log.append(("control", program)) or passthru
    )
    return rig


def loads(rig):
    return [step[1] for step in rig.log if step[0] in ("load", "control")]


def test_a_program_driving_a_uniform_field_is_judged_only_after_a_passing_control(monkeypatch, tmp_path):
    """A dark take is the program's fault only once Passthru on the same rig shows a picture."""
    rig = controlled(monkeypatch, DARK)
    with pytest.raises(R.BlankTakeError, match="Teletext: the program outputs no picture .* Passthru"):
        render_locked(rig, tmp_path / "take.mkv")
    assert loads(rig) == ["Teletext", "Passthru", "Passthru"] * (R.PASS_RETRIES + 1)
    assert rig.steps.count("record") == R.PASS_RETRIES + 1


def test_a_dark_passthrough_is_a_rig_fault_not_a_program_verdict(monkeypatch, tmp_path):
    """Passthru dark too blames the rig, naming the route and the transmitter; no retry is spent."""
    rig = controlled(monkeypatch, DARK, passthru=DARK)
    with pytest.raises(R.RigFaultError) as caught:
        render_locked(rig, tmp_path / "take.mkv")
    assert not isinstance(caught.value, R.BlankTakeError)
    assert loads(rig) == ["Teletext", "Passthru", "Passthru"]
    said = str(caught.value)
    assert "black output from a live source" in said and "outputs no picture" not in said
    assert "Vid Route Mode 1 (HDMI In)" in said and "transmitter vic 34 de 2FE90F" in said


def test_a_dark_take_is_controlled_and_its_pass_retried(monkeypatch, tmp_path):
    """Unattended renders recover a one-off dark take: the control passes and the pass runs again."""
    rig = controlled(monkeypatch, DARK, PROBED)
    assert render_locked(rig, tmp_path / "take.mkv") == PROBED
    assert loads(rig) == ["Teletext", "Passthru", "Passthru", "Teletext"]
    assert rig.steps.count("record") == 2 and rig.steps.count("drive") == 2


def test_a_wrong_route_stops_the_render_before_anything_loads(monkeypatch, tmp_path):
    """Vid Route Mode is checked before the run, and its error says what to set on the panel."""
    rig = controlled(monkeypatch, DARK, transport=rig_transport(route=lambda: 4))
    with pytest.raises(RouteError, match=r"reads 4 \(Auto\), the session needs 1 \(HDMI In\); set SETTINGS"):
        render_locked(rig, tmp_path / "take.mkv")
    assert not rig.log


def test_a_route_changed_by_the_load_is_never_a_dead_program(monkeypatch, tmp_path):
    """A route that goes wrong with the program's load stops the take there, before any picture verdict."""
    routes = [1]
    rig = controlled(monkeypatch, DARK, transport=rig_transport(route=lambda: routes.pop(0) if routes else 5))
    with pytest.raises(RouteError, match="Teletext: Vid Route Mode reads 5"):
        render_locked(rig, tmp_path / "take.mkv")
    assert loads(rig) == ["Teletext"] and "record" not in rig.steps


def test_the_route_check_can_be_skipped(monkeypatch, tmp_path):
    """``route=None`` never reads the setting, for a rig on another route by design."""
    rig = controlled(
        monkeypatch, PROBED, transport=rig_transport(route=lambda: pytest.fail("the route was read"))
    )
    config = dataclasses.replace(LOCKED, route=None)
    assert render_locked(rig, tmp_path / "take.mkv", config=config) == PROBED


def test_a_pass_closes_the_transport_it_opened(monkeypatch, tmp_path):
    """The tty is held exclusively, so a transport left open fails the next pass's open."""
    closed = []
    rig = locked_pass(
        monkeypatch, transport=rig_transport(route=lambda: 4, close=lambda: closed.append(True))
    )
    opened, rig.rig = rig.rig, None
    monkeypatch.setattr(R, "open_rig", lambda *a, **k: opened)
    with pytest.raises(RouteError):
        render_locked(rig, tmp_path / "take.mkv")
    assert closed == [True]

"""Render passes are bracketed by the device's HDMI transmitter, and a flat probe stops them early."""

# pylint: disable=missing-function-docstring

import contextlib
import dataclasses
import types

import pytest

from syncsummoner.compose import render as R
from syncsummoner.compose.score import Layer, Score, Section
from syncsummoner.device.recorder import InputRouteError, RouteError
from syncsummoner.probe import harvest as H
from tests import transmitter as tx
from tests.compose import PROBED, rig_transport
from tests.compose.test_render import CONFIG


def bare_rig(transport, loads=None):
    """A rig whose session, playout and recorder do nothing, around ``transport``; ``loads`` records loads."""
    loads = [] if loads is None else loads
    session = types.SimpleNamespace(
        load_program=lambda program, link=None, park=True: loads.append(program),
        working_point=lambda info: {},
        set_params=lambda v: None,
    )
    playout = types.SimpleNamespace(upload=lambda path: 1, playing=lambda fps: contextlib.nullcontext())
    return R.Rig(session=session, capture=None, playout=playout, transport=transport)


def played(rig, tmp_path, recorded):
    score = Score(duration=1.0, sections=[Section(0.0, 1.0, "A")], layers=[Layer("Teletext", 0, [])])
    recorder = types.SimpleNamespace(
        recording=lambda path, seconds=None: recorded.append(path) or contextlib.nullcontext()
    )
    return R.render_played(
        score,
        "clip.mkv",
        tmp_path / "take.mkv",
        profiles={},
        rig=rig,
        config=CONFIG,
        prepared=True,
        recorder=recorder,
    )


@pytest.mark.parametrize(
    ("video", "when", "takes"),
    [
        ([tx.DEAD], "as the pass began", 0),
        ([tx.at(CONFIG.timing), tx.at(CONFIG.timing), tx.DEAD], "as the pass ended", 1),
        ([tx.at(CONFIG.timing)] * 2 + [tx.UNREPORTED], "the firmware reports no transmitter state", 1),
    ],
    ids=["dropped-before", "dropped-during", "unreported"],
)
def test_a_transmitter_lost_at_a_pass_boundary_fails_the_pass(monkeypatch, tmp_path, video, when, takes):
    """A pass whose transmitter was not driving at both ends recorded the card, not the program."""
    monkeypatch.setattr(R, "settle", lambda *a, **k: PROBED)
    monkeypatch.setattr(R, "drive", lambda *a, **k: 0)
    monkeypatch.setattr(R, "plan_automation", lambda *a, **k: {})
    monkeypatch.setattr(R, "inspect_take", lambda *a, **k: pytest.fail("a failed pass is never judged"))
    recorded = []
    with pytest.raises(R.BlankTakeError, match=when):
        played(bare_rig(rig_transport(*video)), tmp_path, recorded)
    assert len(recorded) == takes


def test_a_device_left_on_another_input_fails_the_pass_before_it_runs(monkeypatch, tmp_path):
    """The selected input is verified after the load, before any picture is judged or recorded."""
    monkeypatch.setattr(R, "settle", lambda *a, **k: R.TakeReport(frames=4))
    monkeypatch.setattr(R, "plan_automation", lambda *a, **k: {})
    recorded = []
    analog = dict(tx.at(CONFIG.timing), source="analog")
    with pytest.raises(InputRouteError, match="Teletext: the device reads input 'analog'"):
        played(bare_rig(rig_transport(analog)), tmp_path, recorded)
    assert not recorded


DARK = R.TakeReport(frames=4)


def probes(monkeypatch, *reports, passthru=PROBED):
    """Load probes: the program's ``reports`` in turn, holding the last, and ``passthru`` for the control."""
    seen = list(reports)

    def settle(recorder, transport, *, program, **kwargs):
        del recorder, transport, kwargs
        if program == "Passthru":
            return passthru
        return seen.pop(0) if len(seen) > 1 else seen[0]

    monkeypatch.setattr(R, "settle", settle)
    monkeypatch.setattr(H, "settle", settle)
    monkeypatch.setattr(R, "plan_automation", lambda *a, **k: {})
    monkeypatch.setattr(R, "drive", lambda *a, **k: 0)


def test_a_program_driving_a_uniform_field_is_judged_only_after_a_passing_control(monkeypatch, tmp_path):
    """A flat probe is the program's fault only once Passthru on the same rig shows a picture."""
    probes(monkeypatch, DARK)
    recorded, loads = [], []
    with pytest.raises(R.BlankTakeError, match="Teletext: the program outputs no picture .* Passthru"):
        played(bare_rig(rig_transport(), loads), tmp_path, recorded)
    assert loads == ["Teletext", "Passthru"] * (R.PASS_RETRIES + 1) and not recorded


def test_a_dark_passthrough_is_a_rig_fault_not_a_program_verdict(monkeypatch, tmp_path):
    """Passthru dark too blames the rig, naming the route and the transmitter; no retry is spent."""
    probes(monkeypatch, DARK, passthru=DARK)
    recorded, loads = [], []
    with pytest.raises(R.RigFaultError) as caught:
        played(bare_rig(rig_transport(), loads), tmp_path, recorded)
    assert not isinstance(caught.value, R.BlankTakeError) and loads == ["Teletext", "Passthru"]
    said = str(caught.value)
    assert "black output from a live source" in said and "outputs no picture" not in said
    assert "Vid Route Mode 1 (HDMI In)" in said and "transmitter vic 34 de 2FE90F" in said


def test_a_dark_probe_is_reloaded_and_retried_after_a_passing_control(monkeypatch, tmp_path):
    """Unattended renders recover a one-off dark load: the program is reloaded and its pass runs."""
    probes(monkeypatch, DARK, PROBED)
    monkeypatch.setattr(R, "inspect_take", lambda *a, **k: PROBED)
    recorded, loads = [], []
    assert played(bare_rig(rig_transport(), loads), tmp_path, recorded) == PROBED
    assert loads == ["Teletext", "Passthru", "Teletext"] and len(recorded) == 1


def test_a_dark_take_is_controlled_and_its_pass_retried(monkeypatch, tmp_path):
    """A take with no picture gets the same control before its pass is run again."""
    probes(monkeypatch, PROBED)
    takes = [DARK, PROBED]
    monkeypatch.setattr(R, "inspect_take", lambda *a, **k: takes.pop(0))
    recorded, loads = [], []
    assert played(bare_rig(rig_transport(), loads), tmp_path, recorded) == PROBED
    assert loads == ["Teletext", "Passthru", "Teletext"] and len(recorded) == 2


def test_a_wrong_route_stops_the_render_before_anything_loads(monkeypatch, tmp_path):
    """Vid Route Mode is checked before the run, and its error says what to set on the panel."""
    probes(monkeypatch, DARK)
    recorded, loads = [], []
    with pytest.raises(RouteError, match=r"reads 4 \(Auto\), the session needs 1 \(HDMI In\); set SETTINGS"):
        played(bare_rig(rig_transport(route=lambda: 4), loads), tmp_path, recorded)
    assert not loads and not recorded


def test_a_route_changed_by_the_load_is_never_a_dead_program(monkeypatch, tmp_path):
    """A route that goes wrong after the load stops the render at the load, before any picture verdict."""
    probes(monkeypatch, DARK)
    routes = [1]
    recorded, loads = [], []
    with pytest.raises(RouteError, match="Teletext: Vid Route Mode reads 5"):
        played(
            bare_rig(rig_transport(route=lambda: routes.pop(0) if routes else 5), loads), tmp_path, recorded
        )
    assert loads == ["Teletext"] and not recorded


def test_the_route_check_can_be_skipped(monkeypatch, tmp_path):
    """``route=None`` never reads the setting, for a rig on another route by design."""
    probes(monkeypatch, PROBED)
    monkeypatch.setattr(R, "inspect_take", lambda *a, **k: PROBED)
    config = dataclasses.replace(CONFIG, route=None)
    rig = bare_rig(rig_transport(route=lambda: pytest.fail("the route was read")))
    score = Score(duration=1.0, sections=[Section(0.0, 1.0, "A")], layers=[Layer("Teletext", 0, [])])
    recorder = types.SimpleNamespace(recording=lambda path, seconds=None: contextlib.nullcontext())
    out = tmp_path / "take.mkv"
    assert (
        R.render_played(
            score, "c", out, profiles={}, rig=rig, config=config, prepared=True, recorder=recorder
        )
        == PROBED
    )

"""Render passes are bracketed by the device's HDMI transmitter, and a flat probe stops them early."""

# pylint: disable=missing-function-docstring

import contextlib
import types

import pytest

from syncsummoner.compose import render as R
from syncsummoner.compose.score import Layer, Score, Section
from tests import transmitter as tx
from tests.compose import PROBED, rig_transport
from tests.compose.test_render import CONFIG


def bare_rig(transport):
    """A rig whose session, playout and recorder do nothing, around ``transport``."""
    session = types.SimpleNamespace(
        load_program=lambda program, link=None: None, working_point=lambda info: {}, set_params=lambda v: None
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
        ([tx.at(CONFIG.timing), tx.DEAD], "as the pass ended", 1),
        ([tx.at(CONFIG.timing), tx.UNREPORTED], "the firmware reports no transmitter state", 1),
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


def test_a_program_driving_a_uniform_field_fails_before_the_pass(monkeypatch, tmp_path):
    """With the transmitter up, a flat probe is the program drawing nothing: no pass is worth it."""
    monkeypatch.setattr(R, "settle", lambda *a, **k: R.TakeReport(frames=4))
    monkeypatch.setattr(R, "plan_automation", lambda *a, **k: {})
    recorded = []
    with pytest.raises(R.BlankTakeError, match="Teletext: the program outputs no picture"):
        played(bare_rig(rig_transport()), tmp_path, recorded)
    assert not recorded

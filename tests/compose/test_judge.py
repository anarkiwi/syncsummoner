"""A finished take judged offline: one decode and one attribution handed to every judgement."""

# pylint: disable=missing-function-docstring

import json
import types

import numpy as np
import pytest

from syncsummoner.compose import judge as J
from syncsummoner.compose import locktake
from syncsummoner.compose import takecheck as T
from syncsummoner.compose.cut import Cut
from syncsummoner.compose.score import Layer, Score, Section
from syncsummoner.device.recorder import TakeReport
from syncsummoner.probe import decoded

CONFIG = types.SimpleNamespace(strip_px=8, bits=16, ffmpeg="ffmpeg")
SCORE = Score(
    duration=4.0,
    fps=30.0,
    sections=[Section(0.0, 2.0, "A"), Section(2.0, 4.0, "B")],
    layers=[Layer("Other", 0, []), Layer("Teletext", 1, [])],
)


class Proof:
    """Evidence that holds, as far as judging reads it."""

    ok = True

    def lines(self):
        return ["proven"]

    def to_json(self):
        return '{"ok": true}'


@pytest.fixture(name="stubbed")
def stubbed_fixture(monkeypatch):
    seen = {"decoded": [], "check": None}
    luma = np.arange(20, dtype=np.uint8).reshape(20, 1, 1).repeat(4, axis=1)
    found = types.SimpleNamespace(
        picked=np.array([5, 7, 9]), per_frame=2, summary=lambda: "3/3 clip frames attributed"
    )

    def features(path, spec, **kw):
        seen["decoded"].append((str(path), spec, kw.get("workers")))
        return decoded.Features({"luma": luma, "codes": np.zeros(20, np.int64)}, str(path))

    def evidence(take, **kw):
        seen["evidence"] = (str(take), kw)
        return Proof()

    def check(score, cuts, takes, clip, **kw):
        seen["check"] = (score, cuts, dict(takes), clip, kw)
        return T.CheckReport(cuts=list(cuts))

    monkeypatch.setattr(J.decoded, "features", features)
    monkeypatch.setattr(J, "attribute_take", lambda take, **kw: seen.update(attributed=kw) or found)
    monkeypatch.setattr(locktake, "take_evidence", evidence)
    monkeypatch.setattr(J, "check_takes", check)
    monkeypatch.setattr(
        J, "inspect_luma", lambda frames, start: seen.update(start=start) or TakeReport(frames=3)
    )
    return seen, found


def test_a_take_is_decoded_attributed_and_proven_once(tmp_path, stubbed):
    seen, found = stubbed
    take = tmp_path / "take.mkv"
    judged = J.judge_take(take, clip="clip.mkv", config=CONFIG, program="Teletext", workers=3)
    assert [(p, w) for p, _, w in seen["decoded"]] == [(str(take), 3), ("clip.mkv", 3)]
    assert seen["attributed"]["features"].key == str(take)
    _, kw = seen["evidence"]
    assert kw["found"] is found and kw["features"].key == str(take) and kw["source"].key == "clip.mkv"
    assert seen["start"] == 4 and seen["check"] is None and judged.check is None
    assert json.loads(locktake.evidence_path(take).read_text(encoding="utf-8")) == {"ok": True}
    assert not J.check_path(take).exists()
    assert judged.lines() == [
        "3/3 clip frames attributed",
        "proven",
        f"Teletext: picture {TakeReport(frames=3)}",
    ]


def test_a_take_judged_with_its_score_is_checked_on_every_section(tmp_path, stubbed, monkeypatch):
    seen, found = stubbed
    take = tmp_path / "take.mkv"
    monkeypatch.setattr("syncsummoner.device.wire.load_wire", lambda path: {"program": np.array("Teletext")})
    source = decoded.Features({}, "given")
    judged = J.judge_take(take, clip="clip.mkv", config=CONFIG, score=SCORE, source=source)
    assert judged.program == "Teletext" and len(seen["decoded"]) == 1
    played, cuts, takes, clip, kw = seen["check"]
    assert [layer.program for layer in played.layers] == ["Teletext"]
    assert cuts == [Cut(0.0, 2.0, "Teletext"), Cut(2.0, 4.0, "Teletext")]
    assert takes == {"Teletext": take} and clip == "clip.mkv"
    assert kw["attributions"]["Teletext"] is found.picked and kw["source"] is source
    assert json.loads(J.check_path(take).read_text(encoding="utf-8")) == {"lines": [], "failed": []}


def test_one_program_plays_its_own_layer_or_the_first():
    played, cuts = J.one_program(SCORE, "Pinwheel")
    assert played.layers[0].program == "Pinwheel" and played.layers[0].index == 0
    bare, whole = J.one_program(Score(duration=3.0, fps=30.0), "Pinwheel")
    assert bare.layers[0].gestures == [] and whole == [Cut(0.0, 3.0, "Pinwheel")]
    assert len(cuts) == 2

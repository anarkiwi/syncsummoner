"""A take's dry lead-in judged against its clip: which codes are dry, and the captures that show them."""

# pylint: disable=missing-function-docstring

import json
from types import SimpleNamespace

import numpy as np
import pytest

from syncsummoner.compose import chroma as C
from syncsummoner.compose import timecode
from syncsummoner.compose.locktake import LeadIn
from syncsummoner.device.playout import SHOWN
from syncsummoner.probe.chroma import AS_PLAYED, ORDERS, variants
from syncsummoner.probe.decoded import Spec

FPS = 30.0
SPEC = Spec()


def shown(codes, t0=10**9, step=33_000_000):
    record = np.zeros(len(codes), dtype=SHOWN)
    record["frame"] = codes
    record["t_ns"] = t0 + np.arange(len(codes)) * step
    return record


def wire(*, dry, loads=(), codes=range(-105, 10)):
    return {
        "dry": np.array(json.dumps(dry)),
        "loads": np.array(json.dumps(list(loads))),
        "shown": shown(list(codes)),
    }


def test_a_dry_take_is_judged_on_the_bound_before_its_wet_stretch():
    codes = C.dry_codes(wire(dry={"11": 1023}), FPS)
    lead_in = LeadIn()
    wet = -round(lead_in.wet_s * FPS)
    assert codes.tolist() == list(range(wet - round(lead_in.bound_s * FPS), wet))


def test_a_take_with_no_dry_control_is_judged_on_passthru_before_its_load():
    said = wire(dry=None, codes=range(-500, 0))
    load = int(said["shown"]["t_ns"][10])
    said["loads"] = np.array(json.dumps([{"program": "P", "t0_ns": load, "t1_ns": load + 1}]))
    assert C.dry_codes(said, FPS).tolist() == [-494, -493, -492]
    assert C.dry_codes(wire(dry=None), FPS).size == 0


def strip_codes(codes, bits=SPEC.bits):
    return np.asarray(codes) & ((1 << bits) - 1)


@pytest.fixture(name="take")
def fixture_take(monkeypatch, tmp_path):
    """A take whose dry codes show the clip in ``state['order']``, its strip lost on ``state['lost']``."""
    rng = np.random.default_rng(3)
    lead = 120
    clip = rng.uniform(16, 235, (lead, 3, 6, 8))
    codes = np.arange(-lead, 0)
    said = wire(dry={"11": 1023}, codes=codes)
    state = {"order": 0, "lost": set()}
    take, source = tmp_path / "take.mkv", tmp_path / "clip.mkv"

    def head(path, spec, count, width, ffmpeg):
        assert spec == SPEC and width == 1920 and ffmpeg == "ffmpeg"
        if path == source:
            assert count == lead
            return {"codes": strip_codes(codes), "small": clip}
        seen = np.repeat(codes, 2)[:count]
        small = variants(clip[seen + lead])[state["order"]] + np.array([0, 128, 128])[:, None, None]
        legible = np.array([c not in state["lost"] for c in seen])
        return {"codes": np.where(legible, strip_codes(seen), 1 << 15), "small": small}

    log = np.zeros(4 * lead, dtype=[("t_ns", "<i8")])
    log["t_ns"] = said["shown"]["t_ns"][0] + np.arange(log.size) * 16_500_000
    monkeypatch.setattr(C, "head", head)
    monkeypatch.setattr(C, "load_wire", lambda path: said)
    monkeypatch.setattr(C, "read_log", lambda path: log)
    return state, take, source, lead, said


def judge(take, source, lead):
    return C.lead_in_order(take, source, spec=SPEC, fps=FPS, width=1920, lead=lead)


def test_a_dry_lead_in_shown_as_played_reads_as_played(take):
    _, path, source, lead, _ = take
    found = judge(path, source, lead)
    assert found.order == AS_PLAYED and found.ok and found.frames == 2 * 15


def test_an_exchanged_lead_in_fails(take):
    state, path, source, lead, _ = take
    state["order"] = 1
    found = judge(path, source, lead)
    assert found.order == ORDERS[1] and not found.ok


def test_illegible_strips_are_left_out_and_none_legible_is_an_error(take):
    state, path, source, lead, _ = take
    state["lost"] = set(range(-75, -66))
    assert judge(path, source, lead).frames == 2 * 6
    state["lost"] = set(range(-75, -60))
    with pytest.raises(ValueError, match="legible strip"):
        judge(path, source, lead)


def test_a_lead_in_that_never_showed_its_dry_frames_is_an_error(take):
    _, path, source, lead, said = take
    said["shown"] = said["shown"][:5]
    with pytest.raises(ValueError, match="none of its dry frames"):
        judge(path, source, lead)


CONFIG = SimpleNamespace(fps=FPS, width=1920, strip_px=8, bits=16, ffmpeg="ffmpeg")


def test_a_judged_verdict_is_kept_beside_the_take_and_read_back(take, monkeypatch):
    state, path, source, lead, _ = take
    monkeypatch.setattr(timecode, "clip_layout", lambda clip, ffprobe: SimpleNamespace(lead=lead))
    said = C.judge(path, source, CONFIG)
    assert said["ok"] and said["judged"] and said["dry"]
    assert json.loads(C.record_path(path).read_text(encoding="utf-8")) == said
    assert not C.exchanged(path) and C.swapped(path) == []
    state["order"] = 1
    assert not C.judge(path, source, CONFIG, lead=lead)["ok"]
    assert C.exchanged(path) and C.swapped(path) == ["swapuv"]


def test_an_unreadable_take_keeps_why_and_reads_as_recorded(take, tmp_path):
    state, path, source, lead, _ = take
    state["lost"] = set(range(-lead, 0))
    said = C.judge(path, source, CONFIG, lead=lead)
    assert said == {"error": f"ValueError: {path}: no capture of its dry lead-in carries a legible strip"}
    assert not C.exchanged(path)
    assert not C.exchanged(tmp_path / "never-judged.mkv")

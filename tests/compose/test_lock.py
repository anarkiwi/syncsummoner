"""Exact attribution of a locked take from its strips, the card's repeats and the Pi's flip log."""

# pylint: disable=missing-function-docstring

import subprocess
import types

import numpy as np
import pytest

from syncsummoner.compose import lock as L
from syncsummoner.device.v4l2rec import sidecar
from syncsummoner.probe import decoded
from syncsummoner.probe.patterns import burn_timecode, read_timecode
from tests.compose.locked import CARD_NS, CLEAN, PI_T0, STANDARD, VBLANK_NS, WRAP, take

CONFIG = types.SimpleNamespace(width=160, strip_px=8, bits=16, fps=30.0, ffmpeg="ffmpeg")


def strips(codes, *, width=160, strip_px=8, gain=1.0):
    plate = np.full((strip_px, width, 3), 0.5, np.float32)
    return np.stack([gain * burn_timecode(plate, c, bits=16, strip_px=strip_px) for c in codes])


@pytest.mark.parametrize("gain", [1.0, 0.1])
def test_read_codes_agrees_with_the_scalar_reader(gain):
    codes = [0, 1, 7, 1234, 32767, 32768, 65533, 65535]
    stack = np.concatenate([strips(codes, gain=gain), np.full((1, 8, 160, 3), 0.5)])
    scalar = [read_timecode(s, bits=16, strip_px=8) for s in stack]
    assert L.read_codes(stack, bits=16).tolist() == [-1 if c is None else c for c in scalar]
    assert L.read_codes(stack, bits=16)[:-1].tolist() == (codes if gain == 1.0 else [-1] * len(codes))


def test_a_strip_turned_over_is_unreadable():
    assert (L.read_codes(strips([3, 900])[:, :, ::-1], bits=16) == -1).all()


@pytest.mark.parametrize("code,clip", [(0, 0), (32767, 32767), (32768, -32768), (65533, -3), (65535, -1)])
def test_signed_puts_the_wrapped_lead_in_below_zero(code, clip):
    assert L.signed(np.array([code]), 16)[0] == clip


def encode(path, frames, fps=30):
    raw = (np.clip(frames, 0, 1) * 255).astype(np.uint8)
    argv = ["ffmpeg", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "rgb24"]
    argv += [
        "-s",
        f"{raw.shape[2]}x{raw.shape[1]}",
        "-r",
        str(fps),
        "-i",
        "pipe:0",
        "-c:v",
        "ffv1",
        str(path),
    ]
    subprocess.run(argv, input=raw.tobytes(), check=True)


def test_every_stored_frame_has_its_strip_read(tmp_path):
    codes = [(i & WRAP) for i in range(-3, 5)]
    plate = np.full((24, 160, 3), 0.3, np.float32)
    frames = np.stack([burn_timecode(plate, c, bits=16, strip_px=8) for c in codes])
    frames[2] = plate
    encode(tmp_path / "take.mkv", frames)
    got = decoded.features(tmp_path / "take.mkv", decoded.Spec(), workers=3).codes
    assert got.tolist() == codes[:2] + [-1] + codes[3:]
    assert L.signed(got, 16).tolist() == [-3, -2, -1, 0, 1, 2, 3, 4]


@pytest.mark.parametrize(
    "t,sequence,runs",
    [
        ([0, 10, 20, 30, 40], [0, 1, 2, 3, 4], [(0, 4)]),
        ([0, 10, 20, 90, 100], [0, 1, 2, 3, 4], [(0, 2), (3, 4)]),
        ([0, 10, 20, 30, 40], [0, 1, 3, 4, 5], [(0, 1), (2, 4)]),
        ([0], [0], [(0, 0)]),
        ([], [], []),
    ],
)
def test_runs_break_at_a_long_interval_or_a_lost_sequence_number(t, sequence, runs):
    assert L.runs_of(np.array(t, np.int64), np.array(sequence))[0] == runs


def test_repeat_phase_is_where_the_card_changes_frame():
    assert L.repeat_phase(np.array([False, True, False, True, False, True]), 2) == (1, 1.0)
    assert L.repeat_phase(np.array([True, False, False, True, True, False]), 3) == (0, 2 / 3)
    assert L.repeat_phase(np.zeros(4, bool), 2)[0] is None


@pytest.mark.parametrize("anchored", [0, 1], ids=["lead-in", "post-gap"])
def test_every_capture_is_attributed_across_an_outage_and_a_lost_frame(anchored):
    played = take(STANDARD, strips=(anchored,), misread=(3, 2) if anchored == 0 else None)
    found = L.attribute(played.codes, played.frames, played.flips, **played.kw)
    assert found.picked.tolist() == played.picked().tolist() and found.complete
    assert found.shown.tolist() == played.shown().tolist()
    assert (found.per_frame, found.anchor, len(found.runs), found.drops) == (2, anchored, 3, 1)
    assert [r.slot for r in found.runs] == [0, 13, 21]
    assert all(r.agree == 1.0 and r.repeats == 1.0 for r in found.runs)
    assert found.run == found.runs[1] and found.period_ns == 2 * VBLANK_NS
    assert found.frame0_ns == pytest.approx(PI_T0 + CARD_NS + 8 * 2 * VBLANK_NS, abs=VBLANK_NS * 0.3 + 1)
    assert (found.read, found.agreed) == ((8, 7) if anchored == 0 else (7, 7))


def test_the_summary_says_what_was_attributed():
    played = take(STANDARD)
    found = L.attribute(played.codes, played.frames, played.flips, **played.kw)
    said = found.summary()
    assert said.startswith("6/6 clip frames attributed at 2 captures a frame, 3 runs, clip in captures 8..14")
    assert "strips agree 8/8, 1 frames lost by the driver" in said
    assert found.margin_ms == pytest.approx(VBLANK_NS / 1e6)


def test_a_clip_frame_in_the_outage_has_no_capture():
    played = take(((0, 7, 0.0), (21, 27, 0.3)), count=6)
    found = L.attribute(played.codes, played.frames, played.flips, **played.kw)
    assert found.picked.tolist() == played.picked().tolist() == [-1, -1, 8, 10, 12, 14]
    assert not found.complete
    assert found.run == found.runs[-1]


@pytest.mark.parametrize(
    "kwargs,match",
    [
        ({"runs": STANDARD, "shift": 1}, "the strips put frame starts at copy 0 .* repeats at copy 1"),
        ({"runs": STANDARD, "strips": ()}, "no run of captures carries a legible strip"),
        (
            {"runs": ((0, 3, 0.0), (6, 13, 0.3)), "per_frame": 1, "lead": 4},
            "one capture a frame, so nothing bridges",
        ),
    ],
    ids=["strips-against-repeats", "no-strips", "one-capture-a-frame"],
)
def test_a_take_that_cannot_be_attributed_exactly_is_a_lock_error(kwargs, match):
    played = take(kwargs.pop("runs"), **kwargs)
    with pytest.raises(L.LockError, match=match):
        L.attribute(played.codes, played.frames, played.flips, **played.kw)


def test_an_anchor_showing_no_logged_frame_is_a_lock_error():
    played = take(CLEAN)
    flips = played.flips[played.flips["frame"] >= 4]
    flips["flags"][0] = 1
    with pytest.raises(L.LockError, match="no frame the Pi logged a flip for"):
        L.attribute(played.codes, played.frames, flips, **played.kw)


def saved(tmp_path, played, name="take.mkv"):
    path = tmp_path / name
    L.save_lock(path, played.flips, **played.kw)
    sidecar(path).write_bytes(played.frames.tobytes())
    return path


def test_a_take_is_attributed_from_its_lock_record_and_its_picks_kept(tmp_path, monkeypatch):
    played = take(STANDARD)
    path = saved(tmp_path, played)
    asked = []
    monkeypatch.setattr(
        L.decoded,
        "features",
        lambda take_, spec, **kw: asked.append((take_, spec, kw))
        or decoded.Features({"codes": played.codes}, ""),
    )
    found = L.attribute_take(path, config=CONFIG)
    assert asked == [(path, decoded.Spec(strip_px=8, bits=16), {"ffmpeg": "ffmpeg"})]
    assert np.load(L.picked_path(path)).tolist() == played.picked().tolist() == found.picked.tolist()
    first = int(np.flatnonzero(played.slots == 2 * played.lead)[0])
    assert L.take_start(path, config=CONFIG) == pytest.approx(first / (2 * CONFIG.fps))


def test_a_take_with_no_lock_record_is_lost(tmp_path):
    with pytest.raises(L.LostTimecodeError, match="no lock record"):
        L.attribute_take(tmp_path / "take.mkv", config=CONFIG, codes=np.zeros(0, np.int64))


@pytest.mark.parametrize(
    "runs,count,match",
    [
        (((0, 7, 0.0), (21, 27, 0.3)), 6, r"4/6 clip frames attributed"),
        (CLEAN, 8, r"6/8 clip frames attributed"),
        (((0, 7, 0.0), (13, 19, 0.6), (20, 20, 1.05), (21, 27, 0.6)), 6, "disagree on its cadence"),
    ],
    ids=["outage", "short", "off-cadence"],
)
def test_an_inexact_take_is_refused_and_nothing_kept(tmp_path, runs, count, match):
    played = take(runs, count=count)
    path = saved(tmp_path, played)
    with pytest.raises(L.LockError, match=match):
        L.attribute_take(path, config=CONFIG, codes=played.codes)
    assert not L.picked_path(path).exists()


def test_the_anchor_rests_on_the_lead_in_when_the_clip_misreads_its_strips():
    """Legible wrong clip codes, a program's doing, outnumber the lead-in's yet must not move the anchor."""
    played = take(((0, 59, 0.0),), lead=8, count=22)
    codes = played.codes.copy()
    clip = played.slots // 2 >= 8
    codes[clip] = ((played.slots[clip] + 1) // 2 - 8) & WRAP
    found = L.attribute(codes, played.frames, played.flips, bits=16, **played.kw)
    assert found.shown.tolist() == played.shown().tolist()

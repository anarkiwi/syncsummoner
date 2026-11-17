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


def attribute(played, codes=None):
    return L.attribute(played.codes if codes is None else codes, played.frames, played.flips, **played.kw)


#: A full take's length: a 40 s lead-in at 30 frames a second, then the clip.
FULL = {"lead": 1200, "count": 10500, "hold": 1}


def test_the_yuv_amplifier_chunk_0_pattern_is_attributed_through_its_slip():
    """One gap-free run whose frame starts move from one capture copy to the other mid-take."""
    played = take(((0, 23809, 0.0),), ppm=-72.0, jitter_ns=900_000, **FULL)
    starts = np.flatnonzero(np.diff(played.frame) > 0) + 1
    assert np.unique((starts - 2 * played.frame[starts]) % 2).size == 2
    found = attribute(played)
    assert found.shown.tolist() == played.shown().tolist() and played.shows(found.picked)
    assert found.slips >= 1 and found.drift_ppm == pytest.approx(-72.0, abs=1.0)
    assert (found.read, found.agreed, found.faults, found.strays) == (found.read, found.read, 0, 0)
    assert found.line.residual <= found.jitter_ns


def test_a_program_destroying_the_strips_is_attributed_from_its_lead_in_and_lead_out():
    played = take(((0, 23999, 0.6),), ppm=55.0, jitter_ns=500_000, wet=(1200, 11700), tail=150, **FULL)
    found = attribute(played)
    assert found.shown.tolist() == played.shown().tolist() and played.shows(found.picked)
    assert found.read < 3000 and found.agreed == found.read


def test_a_strip_less_run_after_an_outage_is_numbered_through_the_pis_lag():
    played = take(
        ((0, 299, 0.2), (400, 23999, 0.7, 50)), ppm=-50.0, jitter_ns=500_000, wet=(1200, 11700), **FULL
    )
    found = attribute(played)
    assert found.runs == 2 and found.shown.tolist() == played.shown().tolist()
    assert played.shows(found.picked) and found.drops == 50


def test_strips_a_program_delays_off_the_pis_lag_do_not_number_their_run():
    played = take(((0, 299, 0.2), (400, 23999, 0.7, 50)), ppm=-50.0, wet=(300, 11700), **FULL)
    codes = played.codes.copy()
    late = np.flatnonzero((played.frame >= 260) & (played.frame < 300) & (np.arange(codes.size) >= 300))
    codes[late] = (played.frame[late] - 30 - played.lead) & WRAP
    found = attribute(played, codes)
    assert found.shown.tolist() == played.shown().tolist()


def test_identical_captures_over_a_slip_measure_alike_whichever_frame_they_are_given():
    played = take(((0, 23999, 0.0),), ppm=-60.0, static=(3000, 6000), wet=(1200, 11700), tail=150, **FULL)
    found = attribute(played)
    digest = played.frames["digest"]
    first = {int(f): int(digest[np.flatnonzero(played.frame == f)[0]]) for f in np.unique(played.frame)}
    assert all(digest[k] == first[j + played.lead] for j, k in enumerate(found.picked))
    assert (
        found.tied > 0
        and found.shown[played.frame < 3000].tolist() == played.shown()[played.frame < 3000].tolist()
    )


@pytest.mark.parametrize("frame", [600, 3000], ids=["lead-in", "clip"])
def test_a_frame_line_that_bends_past_the_jitter_is_refused(tmp_path, frame):
    played = take(
        ((0, 11999, 0.0),),
        jitter_ns=300_000,
        ppm=-500.0,
        kink=(frame, 4_166_667),
        lead=1200,
        count=4500,
        hold=1,
    )
    found = attribute(played)
    assert found.faults + found.strays > 0
    with pytest.raises(L.LockError, match="contradict the frame line"):
        L.attribute_take(saved(tmp_path, played), config=CONFIG, codes=played.codes)


def test_the_cards_settling_bytes_before_playout_are_strays_off_the_clip():
    played = take(((0, 3999, 0.0),), lead=1200, count=600, hold=1)
    played.frames["digest"][:6] = np.arange(100, 106)
    found = attribute(played)
    assert found.shown[10:].tolist() == played.shown()[10:].tolist() and found.strays == 0


def test_the_summary_and_record_say_what_was_attributed():
    played = take(STANDARD)
    found = attribute(played)
    said = found.summary()
    assert said.startswith("6/6 clip frames attributed over 3 runs; frame line")
    assert "strips agree 22/22, 0 ahead, 0 behind, 0 a boundary off, 0 lead strips" in said
    assert said.endswith("1 frames lost by the driver")
    record = found.record()
    assert record["runs"] == 3 and record["faults"] == 0 and record["strips_agree"] == 22
    assert found.first_capture == int(np.flatnonzero(played.shown() == 0)[0])


@pytest.mark.parametrize("runs", [STANDARD, CLEAN])
def test_every_capture_is_attributed_across_an_outage_and_a_lost_frame(runs):
    played = take(runs, misread=(20, 2))
    found = attribute(played)
    assert found.shown.tolist() == played.shown().tolist() and played.shows(found.picked)
    assert found.period_ns == pytest.approx(2 * VBLANK_NS, rel=1e-3)
    assert found.frame0_ns == pytest.approx(PI_T0 + CARD_NS + (8 * 2 - 0.5) * VBLANK_NS, abs=VBLANK_NS)


def test_a_clip_frame_in_the_outage_has_no_capture():
    played = take(((0, 7, 0.0), (21, 27, 0.3)), count=6)
    found = attribute(played)
    assert (found.picked < 0).tolist() == [True, True, False, False, False, False]
    assert not found.complete


def test_a_take_with_no_strips_is_a_lock_error():
    played = take(STANDARD, strips=())
    with pytest.raises(L.LockError, match="no run of captures carries two legible strip-coded frames"):
        attribute(played)


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
    assert np.load(L.picked_path(path)).tolist() == found.picked.tolist() and played.shows(found.picked)
    first = int(np.flatnonzero(played.shown() == 0)[0])
    assert L.take_start(path, config=CONFIG) == pytest.approx(first / (2 * CONFIG.fps))


def test_a_lock_record_without_a_lead_out_reads_as_none(tmp_path):
    played = take(STANDARD)
    path = tmp_path / "take.mkv"
    np.savez(L.lock_path(path), flips=played.flips, hold=2, lead=8, count=6)
    sidecar(path).write_bytes(played.frames.tobytes())
    assert L.attribute_take(path, config=CONFIG, codes=played.codes).complete


def test_a_take_with_no_lock_record_is_lost(tmp_path):
    with pytest.raises(L.LostTimecodeError, match="no lock record"):
        L.attribute_take(tmp_path / "take.mkv", config=CONFIG, codes=np.zeros(0, np.int64))


@pytest.mark.parametrize(
    "runs,count,codes,match",
    [
        (((0, 7, 0.0), (21, 27, 0.3)), 6, None, r"4/6 clip frames attributed"),
        (CLEAN, 8, None, r"6/8 clip frames attributed"),
        (((0, 59, 0.0),), 22, "stray", "contradict the frame line"),
    ],
    ids=["outage", "short", "stray"],
)
def test_an_inexact_take_is_refused_and_nothing_kept(tmp_path, runs, count, codes, match):
    played = take(runs, count=count)
    if codes == "stray":
        frame = played.frame
        k = next(i for i in range(40, frame.size - 1) if frame[i] == frame[i - 1] != frame[i + 1])
        played.frames["digest"][k] = played.frames["digest"][k + 1]
    path = saved(tmp_path, played)
    with pytest.raises(L.LockError, match=match):
        L.attribute_take(path, config=CONFIG, codes=played.codes)
    assert not L.picked_path(path).exists()


def test_the_anchor_rests_on_the_lead_in_when_the_clip_misreads_its_strips():
    """Legible wrong clip codes, a program's doing, outnumber the lead-in's yet must not move the line."""
    played = take(((0, 59, 0.0),), lead=8, count=22)
    codes = played.codes.copy()
    clip = played.frame >= 8
    codes[clip] = (played.frame[clip] + 1 - 8) & WRAP
    found = attribute(played, codes)
    assert found.shown.tolist() == played.shown().tolist() and found.ahead > 0

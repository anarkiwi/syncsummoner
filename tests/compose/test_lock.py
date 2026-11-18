"""Exact attribution of a locked take from its strips, the card's repeats and the Pi's flip log."""

# pylint: disable=missing-function-docstring

import json
import subprocess
import types

import numpy as np
import pytest

from syncsummoner.compose import evidence as ev
from syncsummoner.compose import lock as L
from syncsummoner.device.v4l2rec import sidecar
from syncsummoner.device.wire import READ, RECEIPT, SEND
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


def test_strips_a_program_processed_neither_anchor_nor_contradict_the_line():
    """A clip strip a frame behind on a boundary capture faults when trusted, and is no evidence otherwise."""
    clean = take(STANDARD)
    played = take(STANDARD, misread=(14, (clean.frame[14] - 1 - clean.lead) & WRAP))
    trusted = np.ones(played.frame.size, dtype=bool)
    trusted[12:20] = False
    seen = L.attribute(played.codes, played.frames, played.flips, **played.kw, trusted=trusted)
    assert attribute(played).faults == 1 and seen.faults == 0
    assert seen.read == attribute(played).read - int(np.count_nonzero(played.codes[12:20] >= 0))
    assert seen.shown.tolist() == attribute(clean).shown.tolist()


def test_a_run_the_flips_cannot_number_is_left_out_and_the_take_kept_when_the_clip_needs_none(monkeypatch):
    runs = ((0, 29, 0.0), (35, 43, 0.3), (45, 47, 0.3))
    played = take(runs, lead=8, count=6, tail=8, strips=(0, 1))

    def refused(*_a, **_k):
        raise L.LockError("numbered 0 ways")

    monkeypatch.setattr(L, "_bridge", refused)
    found = attribute(played)
    assert (
        found.complete
        and found.picked.tolist() == attribute(take(runs, lead=8, count=6, tail=8)).picked.tolist()
    )
    assert (found.shown[-3:] == -1 - played.lead).all()


def wire_for(path, **arrays):
    np.savez(f"{path}.wire.npz", **arrays)


def test_a_program_processed_the_picture_from_its_first_wet_write_to_its_dry_writes_landing(tmp_path):
    played = take(STANDARD)
    path = tmp_path / "take.mkv"
    L.save_lock(path, played.flips, **played.kw)
    assert L.processed(path, np.load(L.lock_path(path)), played.frames) is None
    L.save_lock(path, played.flips, **played.kw, wet=True)
    record, t = np.load(L.lock_path(path)), played.frames["t_ns"]
    seg = L.frameline.segments(t, played.frames["sequence"])
    assert L.processed(path, record, played.frames) == (float(t[seg == 0][-1]), np.inf)
    L.save_lock(path, played.flips, **{**played.kw, "tail": 2}, wet=True)
    record = np.load(L.lock_path(path))
    assert L.processed(path, record, played.frames) == (float(t[seg == 0][-1]), float(t[seg == seg[-1]][0]))
    sends = np.zeros(4, dtype=SEND)
    sends["target"] = [ev.START, ev.CALIBRATION, 2, played.count]
    sends["t0_ns"], sends["t1_ns"] = [10, 20, 30, 40], [11, 21, 31, 41]
    latency = json.dumps({"hi": 5.0, "period": 100.0})
    common = {"sends": sends, "latency": np.array(latency), "cc_map": np.arange(1, 13)}
    common |= {"receipts": np.zeros(0, RECEIPT), "reads": np.zeros(0, READ)}
    wire_for(path, dry=np.array(json.dumps({"1": 1023})), **common)
    assert L.processed(path, record, played.frames) == (20.0, 41.0 + 105.0)
    loads = [{"program": "p", "t0_ns": 7, "t1_ns": 8}, {"program": "Passthru", "t0_ns": 50, "t1_ns": 60}]
    wire_for(path, dry=np.array("null"), loads=np.array(json.dumps(loads)), **common)
    assert L.processed(path, record, played.frames) == (7.0, 165.0)
    wire_for(path, dry=np.array("null"), **common)
    assert L.processed(path, record, played.frames)[0] == float(t[seg == 0][-1])


#: Passthru lead-in, a processed run with no legible strip, Passthru lead-out, each after a load's outage.
PROCESSED = ((0, 29, 0.0), (35, 69, 0.3), (75, 99, 0.3))


def test_a_processed_run_between_passthru_runs_is_numbered_by_the_capture_clock():
    played = take(PROCESSED, lead=18, count=17, tail=15, strips=(0, 2))
    found = attribute(played)
    (said,) = found.clock
    assert said["run"] == 1 and said["candidates"] and found.complete
    middle = np.arange(30, 65)
    assert (found.ambiguity[middle] == max(said["candidates"]) - min(said["candidates"])).all()
    assert not np.delete(found.ambiguity, middle).any()
    truth, shown = played.shown(), found.shown
    assert ((truth >= shown - found.ambiguity) & (truth <= shown)).all()
    assert found.record()["clock"] == [said]


def test_a_processed_run_the_clock_numbers_no_way_fails_the_take(monkeypatch):
    played = take(PROCESSED, lead=18, count=17, tail=15, strips=(0, 2))
    monkeypatch.setattr(L.relock, "candidates", lambda *a, **k: [])
    with pytest.raises(L.LockError, match="numbers run 1 no way"):
        attribute(played)


def test_a_processed_run_is_numbered_by_its_highest_candidate_and_spans_to_its_lowest(monkeypatch):
    played = take(PROCESSED, lead=18, count=17, tail=15, strips=(0, 2))
    middle = np.arange(30, 65)
    monkeypatch.setattr(L.relock, "candidates", lambda *a, **k: [19])
    highest = attribute(played)
    monkeypatch.setattr(L.relock, "candidates", lambda *a, **k: [17, 19])
    found = attribute(played)
    assert found.shown[middle].tolist() == highest.shown[middle].tolist()
    assert (found.ambiguity[middle] == 2).all() and not highest.ambiguity.any()


def test_the_clocks_numbering_round_trips_through_the_lock_record(tmp_path):
    played = take(STANDARD)
    path = tmp_path / "take.mkv"
    L.save_lock(path, played.flips, **played.kw, wet=True)
    assert L.read_clock(path) == []
    said = [{"run": 2, "candidates": [7, 8], "selection": {"shift": 1, "numbering": 7, "margin_z": 4.5}}]
    L.save_clock(path, said)
    L.save_clock(path, said)
    record = np.load(L.lock_path(path))
    assert L.read_clock(path) == said and record["flips"].tolist() == played.flips.tolist()
    assert bool(record["wet"]) and int(record["count"]) == played.count


def session_played():
    """Two takes of six clip frames played back to back between an 8-frame lead-in and lead-out."""
    return take(((0, 59, 0.0),), lead=8, count=12, tail=8, wet=(8, 20))


def test_a_view_is_one_takes_share_of_a_session_attribution():
    played = session_played()
    whole = attribute(played)
    mine = np.flatnonzero((played.frame >= 13) & (played.frame < 20))
    found = L.attribute(played.codes, played.frames, played.flips, **played.kw, view=L.View(6, 6, mine))
    assert found.lead == 8 + 6 and found.frame0_ns == pytest.approx(
        whole.line.origin + 14 * whole.line.period
    )
    assert found.shown.tolist() == (played.shown()[mine] - 6).tolist()
    assert (mine[found.picked] == whole.picked[6:]).all(), "each frame is picked from the take's own captures"
    assert found.read == 0 and whole.read > 0 and found.anchored, "the session's lead-in anchors the take"
    assert found.span == (0, mine.size - 1)


def test_a_view_holding_none_of_a_frames_captures_leaves_it_unpicked():
    view = L.View(0, 3, np.array([4, 5, 9]))
    assert view.picked(np.array([4, 7, 9, 2])).tolist() == [0, -1, 2]
    assert L.View(0, 2, np.zeros(0, np.int64)).picked(np.array([1, 2])).tolist() == [-1, -1]
    assert view.span(np.array([5, 6, 9])) == (1, 2) and view.span(np.zeros(0, np.int64)) == (0, -1)


def test_a_lead_in_that_fixes_no_line_is_an_anchor_error():
    played = take(((0, 5, 0.0), (10, 59, 0.5)), lead=8, count=12, tail=8, strips={1})
    found = attribute(played)
    assert not found.anchored, "only the lead-out's strips fix a line"
    with pytest.raises(L.AnchorError, match=f"name no two frames by strip.*{L.ANCHOR_FRAMES} consecutive"):
        L.require_anchor(found, "t.mkv")
    good = attribute(session_played())
    assert L.require_anchor(good, "t.mkv") is good


def test_a_session_log_keeps_each_capture_once_in_the_cards_order():
    played = session_played()
    first, second = played.frames[:30], played.frames[26:]
    assert L.session_log([first, second]).tolist() == played.frames.tolist()
    assert L.session_log([]).size == 0


def test_a_session_take_is_attributed_by_the_session_line_from_its_record(tmp_path, monkeypatch):
    """The second take holds no strip: the session's lead-in and its renumbered lead-out number it."""
    played = session_played()
    mine = np.flatnonzero(played.frame >= 13)
    codes = played.codes.copy()
    after = played.frame >= 20
    codes[after] = (played.frame[after] - 8 - 6) & WRAP
    anchors = np.where(played.frame < 13, codes, -1)
    t = played.frames["t_ns"]
    wet = (played.frame >= 8) & (played.frame < 20)
    span = (float(t[wet][0]), float(t[wet][-1]))
    path = tmp_path / "take-1.mkv"
    sidecar(path).write_bytes(played.frames[mine].tobytes())
    said = {"id": "s1", "index": 1, "offsets": [0, 6], "counts": [6, 6], "source_lead": 30}
    L.save_lock(
        path, played.flips, **played.kw, wet=True, **L.session_record(said, played.frames, anchors, span)
    )
    monkeypatch.setattr(L.decoded, "features", lambda *a, **k: decoded.Features({"codes": codes[mine]}, ""))
    found = L.attribute_take(path, config=CONFIG)
    assert found.source_offset == 30 and found.lead == 14 and found.picked.size == 6
    assert (played.frame[mine][found.picked] == 14 + np.arange(6)).all()
    assert json.loads(str(np.load(L.lock_path(path))["session"]))["processed"] == list(span)
    sidecar(path).write_bytes(
        played.frames[mine][:-1].tobytes() + np.array([(999, 1, 1, 0, 1)], played.frames.dtype).tobytes()
    )
    with pytest.raises(L.LockError, match="not the session's s1"):
        L.attribute_take(path, config=CONFIG)

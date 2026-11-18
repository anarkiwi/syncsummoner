"""Take check: every bar of every take judged against the source frames it was fed."""

# pylint: disable=missing-function-docstring,protected-access

import shutil
from dataclasses import astuple
import subprocess

import numpy as np
import pytest
from scipy import ndimage

from syncsummoner import aesthetics
from syncsummoner.compose import render as R
from syncsummoner.compose import takecheck as T
from syncsummoner.compose.score import Score, Section
from syncsummoner.device import recorder as rec
from syncsummoner.probe import footage as F

FPS = 30.0
SIZE = (32, 18)


def texture(rng, shape):
    """Spatially correlated noise in unit range, as natural pictures are."""
    field = ndimage.gaussian_filter(rng.random(shape), 2.0, mode="wrap")
    return (field - field.min()) / np.ptp(field)


def moving(n, seed=0):
    """A drifting field cut at irregular shots, at survival size, as 8-bit luma."""
    rng = np.random.default_rng(seed)
    base, out = texture(rng, (SIZE[1], SIZE[0] * 4)), []
    cuts = set(np.cumsum(rng.integers(20, 70, n)))
    for i in range(n):
        if i in cuts:
            base = texture(rng, base.shape)
        out.append(np.roll(base, i, axis=1)[:, : SIZE[0]])
    return (np.stack(out) * 255).astype(np.uint8)


def recording(frames, *, origin=0.5, rate=60.0):
    """Captures at the card's rate of a clip shown from ``origin``, each frame captured as shown."""
    times = 0.004 + np.arange(int((origin + len(frames) / FPS + 0.5) * rate)) / rate
    shown = np.floor((times - origin) * FPS).astype(np.int64)
    inside = (shown >= 0) & (shown < len(frames))
    held = np.where(inside, np.clip(shown, 0, len(frames) - 1), 0)
    out = frames[held].copy()
    out[~inside] = 0
    return F.Recording(times=times, codes=np.where(inside, shown, -1), frames=out)


def series(frames):
    return aesthetics.information(F.unit_frames(frames))


def test_a_passthrough_lives_and_a_held_or_blank_take_dies():
    clip = moving(270)
    src_si, src_ti = series(clip)
    edges = T.bar_edges(9.0, 2.0, FPS)
    live = T.trace(recording(clip), src_si, src_ti, fps=FPS)
    assert live.how == "strip" and live.start == pytest.approx(0.5, abs=1 / FPS)
    assert all(v.fault is None for v in T.judge(live, src_si, src_ti, edges, fps=FPS))
    frozen = clip.copy()
    frozen[90:] = clip[90]
    held = T.judge(T.trace(recording(frozen), src_si, src_ti, fps=FPS), src_si, src_ti, edges, fps=FPS)
    assert held[0].fault is None and {v.fault for v in held[2:]} == {T.Fault.FROZEN}
    blank = clip.copy()
    blank[90:] = 40
    flat = T.judge(T.trace(recording(blank), src_si, src_ti, fps=FPS), src_si, src_ti, edges, fps=FPS)
    assert {v.fault for v in flat[2:]} == {T.Fault.FLAT}


def test_motion_the_source_never_had_is_unfollowed():
    clip = moving(270)
    src_si, src_ti = series(clip)
    other = moving(270, seed=5)
    out = T.trace(recording(other), src_si, src_ti, fps=FPS)
    verdicts = T.judge(out, src_si, src_ti, T.bar_edges(9.0, 3.0, FPS), fps=FPS)
    faults = [v.fault for v in verdicts]
    assert set(faults) <= {T.Fault.UNFOLLOWED, None} and 2 * faults.count(T.Fault.UNFOLLOWED) > len(faults)


def test_a_take_without_a_strip_is_aligned_by_its_motion():
    clip = moving(300)
    src_si, src_ti = series(clip)
    taken = recording(clip, origin=1.2)
    blind = F.Recording(taken.times, np.full_like(taken.codes, -1), taken.frames)
    out = T.trace(blind, src_si, src_ti, fps=FPS)
    assert out.how == "motion" and out.start == pytest.approx(1.2, abs=1 / FPS)


def _spliced(clip, src_si, src_ti, *, late=0):
    """A master cut from two takes by their attribution, the second ``late`` clip frames behind."""
    first, second = recording(clip, origin=0.0), recording(clip, origin=0.4)
    a, b = (T.trace(r, src_si, src_ti, fps=FPS).picked for r in (first, second))
    half = len(clip) // 2
    frames = np.concatenate([first.frames[a[:half]], second.frames[b[half - late : len(clip) - late]]])
    return F.Recording(np.arange(len(frames)) / FPS, np.full(len(frames), -1), frames)


def test_a_master_spliced_by_attribution_is_alive_on_the_identity_clock():
    clip = moving(300)
    src_si, src_ti = series(clip)
    out = T.trace(_spliced(clip, src_si, src_ti), src_si, src_ti, fps=FPS, model=F.ClockModel.identity(FPS))
    assert out.how == "given" and np.array_equal(out.picked, np.arange(300))
    assert all(v.fault is None for v in T.judge(out, src_si, src_ti, T.bar_edges(10.0, 2.5, FPS), fps=FPS))


def test_a_master_spliced_from_a_late_take_is_dead_where_it_trails():
    clip = moving(300)
    src_si, src_ti = series(clip)
    late = _spliced(clip, src_si, src_ti, late=17)
    out = T.trace(late, src_si, src_ti, fps=FPS, model=F.ClockModel.identity(FPS))
    faults = [v.fault for v in T.judge(out, src_si, src_ti, T.bar_edges(10.0, 2.5, FPS), fps=FPS)]
    assert faults[:2] == [None, None] and all(f is not None for f in faults[2:])


def test_a_given_attribution_is_measured_as_it_is():
    clip = moving(90)
    src_si, src_ti = series(clip)
    taken = recording(clip, origin=0.5)
    fitted = T.trace(taken, src_si, src_ti, fps=FPS)
    given = T.trace(
        taken, src_si, src_ti, fps=FPS, picked=np.r_[fitted.picked[:60], -1, taken.frames.shape[0]]
    )
    assert given.how == "given" and given.start is None
    assert np.array_equal(given.picked, np.r_[fitted.picked[:60], np.full(30, -1)])
    assert np.allclose(given.si[:60], fitted.si[:60]) and np.isnan(given.si[60:]).all()


def test_an_unaligned_recording_is_flat_everywhere():
    clip = moving(60)
    src_si, src_ti = series(clip)
    empty = F.Recording(np.zeros(0), np.zeros(0, np.int64), np.zeros((0, SIZE[1], SIZE[0]), np.uint8))
    out = T.trace(empty, src_si, src_ti, fps=FPS)
    assert out.how == "none" and out.start is None
    assert {v.fault for v in T.judge(out, src_si, src_ti, [0, 60], fps=FPS)} == {T.Fault.FLAT}


def verdict(start, fault=None, z=5.0):
    kept = aesthetics.Survival(10, 1.0, 1.0, 1.0, 1.0, 0.9, 0, z)
    return T.Verdict(start, start + 1.0, kept, fault)


def test_cover_replaces_a_dead_run_with_the_take_that_follows_best():
    verdicts = {
        "a": [verdict(0), verdict(1, T.Fault.FLAT), verdict(2, T.Fault.FLAT), verdict(3)],
        "b": [verdict(0), verdict(1, z=4.0), verdict(2, z=4.0), verdict(3)],
        "c": [verdict(0), verdict(1, z=9.0), verdict(2, T.Fault.FROZEN), verdict(3)],
    }
    cuts, spans = T.cover([R.Cut(0.0, 4.0, "a")], verdicts)
    assert cuts == [R.Cut(0.0, 1.0, "a"), R.Cut(1.0, 3.0, "b"), R.Cut(3.0, 4.0, "a")]
    assert [s[:5] for s in map(astuple, spans)] == [("a", 1.0, 3.0, T.Fault.FLAT, "b")]
    assert spans[0].si_ratio == pytest.approx(1.0) and "keeps SI 100% TI 100%" in str(spans[0])
    assert "covered by b" in str(spans[0])


def test_cover_spares_one_destroyed_bar_and_reports_what_it_cannot_cover():
    verdicts = {
        "a": [verdict(0, T.Fault.FROZEN), verdict(1), verdict(2, T.Fault.FLAT)],
        "b": [verdict(0)] * 3,
    }
    verdicts["b"] = [verdict(0), verdict(1), verdict(2, T.Fault.FLAT)]
    cuts, spans = T.cover([R.Cut(0.0, 3.0, "a")], verdicts, destroy=lambda t: t < 1.0)
    assert cuts == [R.Cut(0.0, 3.0, "a")]
    assert [s[:5] for s in map(astuple, spans)] == [("a", 2.0, 3.0, T.Fault.FLAT, None)]
    assert "no take is alive there" in str(spans[0])


def test_a_bar_is_judged_in_the_cut_its_middle_falls_in():
    kept = aesthetics.Survival(10, 1.0, 1.0, 1.0, 1.0, 0.9, 0, 5.0)
    sliver = T.Verdict(2.0, 2.1, kept, T.Fault.FROZEN)
    verdicts = {"a": [verdict(0), verdict(1), sliver], "b": [verdict(0), verdict(1), verdict(2.0)]}
    cuts, spans = T.cover([R.Cut(0.0, 2.04, "a")], verdicts)
    assert cuts == [R.Cut(0.0, 2.04, "a")] and not spans


def test_the_report_names_alignments_and_failures():
    model = F.ClockModel(1.0 + 1 / 60, 1 / 30)
    traces = {
        "a": T.Trace(None, None, None, None, model, "strip"),
        "b": T.Trace(None, None, None, None, None, "none"),
    }
    report = T.CheckReport(cuts=[], spans=[T.DeadSpan("a", 0.0, 1.0, T.Fault.FLAT, None)], traces=traces)
    lines = report.lines()
    assert "a: aligned by strip, picture starts 1.000s" in lines[0] and "never aligned" in lines[1]
    assert report.failed == report.spans


def test_bar_edges_end_on_the_duration():
    assert list(T.bar_edges(5.0, 2.0, FPS)) == [0, 60, 120, 150]


def test_bars_are_the_planners_split_at_every_cut():
    cuts = [R.Cut(0.0, 1.5, "a"), R.Cut(1.5, 5.0, "b")]
    assert list(T.bar_edges(5.0, 2.0, FPS, cuts=cuts)) == [0, 45, 60, 120, 150]
    assert list(T.bar_edges(5.0, 2.0, FPS, downbeats=[0.5, 2.4, 4.3], cuts=cuts)) == [0, 15, 45, 72, 129, 150]


@pytest.mark.skipif(shutil.which("ffmpeg") is None, reason="ffmpeg is not installed")
def test_takes_are_read_and_checked_from_files(tmp_path):
    rng = np.random.default_rng(1)
    plates = [np.repeat(texture(rng, (48, 64 * 3))[:, :, None], 3, axis=2) for _ in range(3)]
    source = [np.roll(plates[i // 20], i, axis=1)[:, :64].astype(np.float32) for i in range(60)]
    clip = tmp_path / "clip.mkv"
    raw = np.stack([R.burn_timecode(f, i, bits=16, strip_px=8) for i, f in enumerate(source)])
    argv = [
        "ffmpeg",
        "-loglevel",
        "error",
        "-y",
        "-f",
        "rawvideo",
        "-pix_fmt",
        "rgb24",
        "-s",
        "64x48",
        "-r",
        "30",
    ]
    argv += ["-i", "pipe:0", "-c:v", "ffv1", str(clip)]
    subprocess.run(argv, input=(raw * 255).astype(np.uint8).tobytes(), check=True)
    decoded = F.decode_recording(clip, strip_px=8, bits=16)
    assert np.array_equal(decoded.codes, np.arange(60)) and decoded.frames.shape[0] == 60
    score = Score(duration=2.0, fps=FPS, sections=[Section(0.0, 2.0, "A")])
    report = T.check_takes(score, [R.Cut(0.0, 2.0, "p")], {"p": clip}, clip, strip_px=8, bar=0.5)
    assert report.traces["p"].how == "strip" and not report.failed
    with pytest.raises(FileNotFoundError):
        T.check_takes(score, [], {}, tmp_path / "missing.mkv", strip_px=8)


def test_a_passthrough_of_a_grainy_still_is_alive_and_a_still_without_its_grain_is_frozen():
    """A still under grain is one picture plus independent noise, which ``held`` cannot tell from a hold."""
    rng = np.random.default_rng(3)
    plate = texture(rng, SIZE[::-1]) * 160 + 40
    grain = rng.normal(0.0, 1.0, (90, *SIZE[::-1])) * rng.uniform(1.0, 12.0, (90, 1, 1))
    clip = np.clip(plate + grain, 0, 255).astype(np.uint8)
    src_si, src_ti = series(clip)
    out = T.trace(recording(clip, origin=0.5), src_si, src_ti, fps=FPS)
    kept = aesthetics.survival_stats(src_si, src_ti, out.si, out.ti)
    assert rec.held(out.frames[out.picked].astype(np.float32))[1] and kept.following
    assert T.fault(out, 0, 90, kept, src_ti) is None
    even = np.clip(plate + rng.normal(0.0, 6.0, clip.shape), 0, 255).astype(np.uint8)
    stuck = T.trace(recording(even, origin=0.5), src_si, src_ti, fps=FPS, model=out.model)
    lost = aesthetics.survival_stats(src_si, src_ti, stuck.si, stuck.ti)
    assert not lost.following and T.fault(stuck, 0, 90, lost, src_ti) is T.Fault.FROZEN

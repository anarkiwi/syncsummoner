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
from syncsummoner.device.v4l2rec import FRAME
from syncsummoner.probe import decoded
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


def test_motion_the_source_never_had_is_alive():
    clip = moving(270)
    src_si, src_ti = series(clip)
    out = T.trace(recording(moving(270, seed=5)), src_si, src_ti, fps=FPS)
    verdicts = T.judge(out, src_si, src_ti, T.bar_edges(9.0, 3.0, FPS), fps=FPS)
    astray = sum(not v.survival.following for v in verdicts)
    assert 2 * astray > len(verdicts) and all(v.fault is None for v in verdicts)


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


def test_a_master_spliced_from_a_late_take_is_alive_where_it_trails():
    clip = moving(300)
    src_si, src_ti = series(clip)
    late = _spliced(clip, src_si, src_ti, late=17)
    out = T.trace(late, src_si, src_ti, fps=FPS, model=F.ClockModel.identity(FPS))
    verdicts = T.judge(out, src_si, src_ti, T.bar_edges(10.0, 2.5, FPS), fps=FPS)
    assert not verdicts[-1].survival.following and all(v.fault is None for v in verdicts)


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


def test_cover_never_spares_a_bar_the_card_lost_the_signal_over():
    verdicts = {"a": [verdict(0), verdict(1, T.Fault.SIGNAL), verdict(2)], "b": [verdict(0)] * 3}
    cuts, spans = T.cover([R.Cut(0.0, 3.0, "a")], verdicts, destroy=lambda _t: True)
    assert cuts == [R.Cut(0.0, 1.0, "a"), R.Cut(1.0, 2.0, "b"), R.Cut(2.0, 3.0, "a")]
    assert [s[:5] for s in map(astuple, spans)] == [("a", 1.0, 2.0, T.Fault.SIGNAL, "b")]
    assert "signal lost" in str(spans[0])


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


def test_a_cover_reaching_a_cut_edge_off_the_frame_grid_leaves_no_sliver():
    plan = [R.Cut(0.0, 0.99, "a"), R.Cut(0.99, 2.01, "b"), R.Cut(2.01, 3.0, "a")]
    edges = T.bar_edges(3.0, 1.0, FPS, cuts=plan)
    kept = aesthetics.Survival(10, 1.0, 1.0, 1.0, 1.0, 0.9, 0, 5.0)

    def bars(dead):
        return [T.Verdict(a / FPS, b / FPS, kept, dead(a)) for a, b in zip(edges[:-1], edges[1:])]

    verdicts = {"a": bars(lambda _a: None), "b": bars(lambda _a: T.Fault.FROZEN), "c": bars(lambda _a: None)}
    cuts, spans = T.cover(plan, verdicts)
    assert [(s.start, s.end) for s in spans] == [(0.99, 2.01)]
    assert cuts[1] == R.Cut(0.99, 2.01, "a") and len(cuts) == 3
    owners = R.frame_owners(cuts, FPS)
    assert np.array_equal(np.unique(owners), np.arange(len(cuts)))


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
    read = F.decode_recording(clip, strip_px=8, bits=16)
    assert np.array_equal(read.codes, np.arange(60)) and read.frames.shape[0] == 60
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
    assert T.fault(out, 0, 90, kept) is None
    even = np.clip(plate + rng.normal(0.0, 6.0, clip.shape), 0, 255).astype(np.uint8)
    stuck = T.trace(recording(even, origin=0.5), src_si, src_ti, fps=FPS, model=out.model)
    lost = aesthetics.survival_stats(src_si, src_ti, stuck.si, stuck.ti)
    assert not lost.following and T.fault(stuck, 0, 90, lost) is T.Fault.FROZEN


def drifting(n, seed=4):
    """A colour texture drifting a pixel a frame, in unit-range RGB at 64x48."""
    plate = np.stack([texture(np.random.default_rng(seed + c), (48, 64 + n)) for c in range(3)], -1)
    return np.stack([plate[:, i : i + 64] for i in range(n)]).astype(np.float32)


def encode(path, frames):
    """Frames under their timecode strip, losslessly, as a take that showed each clip frame once."""
    raw = np.stack([R.burn_timecode(f, i, bits=16, strip_px=8) for i, f in enumerate(frames)])
    argv = ["ffmpeg", "-loglevel", "error", "-y", "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", "64x48"]
    argv += ["-r", "30", "-i", "pipe:0", "-c:v", "ffv1", str(path)]
    subprocess.run(argv, input=(np.clip(raw, 0, 1) * 255).astype(np.uint8).tobytes(), check=True)
    return path


def five_hz(frames, a, b, dark, bright):
    """Frames ``a:b`` alternated between ``dark`` and ``bright`` of themselves every 3 frames, dark first."""
    out = frames.copy()
    up = ((np.arange(b - a) // 3) % 2).astype(bool)
    out[a:b] = np.where(up[:, None, None, None], bright(frames[a:b]), dark(frames[a:b]))
    return out


@pytest.fixture(name="flashes")
def fixture_flashes(tmp_path):
    """A clip, a take flashing at 5 Hz over the whole screen in its second bar, and one as played."""
    source = drifting(120)
    return {
        "clip": encode(tmp_path / "clip.mkv", source),
        "flash": encode(tmp_path / "flash.mkv", five_hz(source, 30, 60, lambda f: 0.2 * f, lambda f: f)),
        "plain": encode(tmp_path / "plain.mkv", source),
        "red": encode(
            tmp_path / "red.mkv",
            five_hz(source, 30, 60, lambda f: np.full_like(f, 0.5), lambda f: f * 0 + [1.0, 0.0, 0.0]),
        ),
    }


SCORE = Score(duration=4.0, fps=FPS, sections=[Section(0.0, 4.0, "A")])


@pytest.mark.skipif(shutil.which("ffmpeg") is None, reason="ffmpeg is not installed")
def test_a_bar_flashing_at_5hz_over_the_screen_is_covered_from_a_take_that_does_not(flashes):
    takes = {"a": flashes["flash"], "b": flashes["plain"]}
    report = T.check_takes(SCORE, [R.Cut(0.0, 4.0, "a")], takes, flashes["clip"], strip_px=8, bar=1.0)
    assert [s[:5] for s in map(astuple, report.spans)] == [("a", 1.0, 2.0, T.Fault.FLASHING, "b")]
    assert [(c.start, c.end, c.program) for c in report.cuts] == [
        (0.0, 1.0, "a"),
        (1.0, 2.0, "b"),
        (2.0, 4.0, "a"),
    ]
    assert not report.traces["b"].flashing.any() and report.traces["a"].flashing[30:60].any()
    assert (
        "a: aligned by strip, picture starts" in report.lines()[0]
        and "5.0 flashes/s, 0.0 red" in report.lines()[0]
    )


@pytest.mark.skipif(shutil.which("ffmpeg") is None, reason="ffmpeg is not installed")
def test_a_red_flash_no_take_can_cover_fails_even_inside_a_destroy_section(flashes):
    """The red test alone catches it: saturated red against a grey of the same luminance."""
    score = Score(duration=4.0, fps=FPS, sections=[Section(0.0, 4.0, "A", destroy=True)])
    report = T.check_takes(
        score, [R.Cut(0.0, 4.0, "r")], {"r": flashes["red"]}, flashes["clip"], strip_px=8, bar=1.0
    )
    assert [s[:5] for s in map(astuple, report.failed)] == [("r", 1.0, 2.0, T.Fault.FLASHING, None)]
    with pytest.raises(
        T.UnsafeOutputError, match=r"\d\.\d flashes/s, 5\.0 red flashes/s, at 0\.\d\d-2\.\d\ds"
    ):
        T.gate(flashes["red"], strip_px=8)


@pytest.mark.skipif(shutil.which("ffmpeg") is None, reason="ffmpeg is not installed")
def test_the_gate_passes_a_safe_picture_and_names_where_an_unsafe_one_flashes(flashes):
    assert T.gate(flashes["plain"], strip_px=8).safe
    with pytest.raises(T.UnsafeOutputError, match="flash.mkv flashes harmfully: 5.0 flashes/s"):
        T.gate(flashes["flash"], strip_px=8)
    with pytest.raises(RuntimeError, match="cannot read the colour"):
        T.gate(flashes["clip"].with_name("missing.mkv"))


def test_a_take_with_too_few_captures_is_not_screened():
    colour = np.zeros((1, 3, 2, 2), np.uint8)
    marked, risk = T.flashing(colour, np.array([-1, 0, -1]), fps=FPS)
    assert not marked.any() and risk.safe


def test_a_clip_frame_no_capture_shows_holds_the_one_before():
    """A capture missing as a flash turns holds its predecessor, so the flash is seen a frame late."""
    colour = np.stack([np.full((2, 2), v) for v in (0, 128, 128)])[None].repeat(60, 0).astype(np.uint8)
    colour[:, 0] = np.where((np.arange(60) // 3) % 2, 235, 16)[:, None, None]
    picked = np.r_[-1, -1, np.arange(2, 60)]
    picked[30] = -1
    marked, _ = T.flashing(colour, picked, fps=FPS, full=False)
    assert np.flatnonzero(marked).tolist() == sorted(set(range(3, 60, 3)) - {30} | {31})


def with_log(path, log):
    """A take's decoded features, holding ``log`` as its V4L2 frame log."""
    found = decoded.features(path, decoded.Spec(strip_px=8, bits=16))
    return decoded.Features(
        {"times": found.times, "codes": found.codes, "luma": found.luma, "small": found.small, "log": log}, ""
    )


def broken_log(n, at, held):
    """``n`` captures, the card delivering nothing before ``at`` and its own pictures up to ``held``."""
    log = np.zeros(n, FRAME)
    log["sequence"] = np.arange(n)
    log["t_ns"] = (np.arange(n) + 10 * (np.arange(n) >= at)) * 33_333_333
    log["digest"] = np.arange(n)
    log["digest"][held] = log["digest"][held - 1]
    return log


@pytest.mark.skipif(shutil.which("ffmpeg") is None, reason="ffmpeg is not installed")
def test_a_bar_the_card_lost_the_signal_over_is_covered_and_one_without_a_log_judged_as_before(flashes):
    takes = {"a": flashes["plain"], "b": flashes["plain"]}
    given = {"a": np.arange(120), "b": np.arange(120)}
    report = T.check_takes(
        SCORE,
        [R.Cut(0.0, 4.0, "a")],
        takes,
        flashes["clip"],
        strip_px=8,
        bar=1.0,
        attributions=given,
        features={"a": with_log(takes["a"], broken_log(120, 45, 50))},
    )
    assert [s[:5] for s in map(astuple, report.spans)] == [("a", 1.0, 2.0, T.Fault.SIGNAL, "b")]
    assert [(c.start, c.end, c.program) for c in report.cuts] == [
        (0.0, 1.0, "a"),
        (1.0, 2.0, "b"),
        (2.0, 4.0, "a"),
    ]
    assert np.flatnonzero(report.traces["a"].lost).tolist() == list(range(45, 50))
    assert report.traces["b"].lost is None
    short = T.check_takes(
        SCORE,
        [R.Cut(0.0, 4.0, "a")],
        {"a": takes["a"]},
        flashes["clip"],
        strip_px=8,
        bar=1.0,
        attributions=given,
        features={"a": with_log(takes["a"], broken_log(60, 45, 50))},
    )
    assert short.traces["a"].lost is None and not short.spans

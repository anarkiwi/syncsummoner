"""Footage identity: which clip frame every capture shows, from its strip or its motion."""

# pylint: disable=missing-function-docstring

import numpy as np
import pytest

from syncsummoner.probe import footage as F

FPS = 30.0


def captures(n, *, origin=1.3, interval=1 / FPS, rate=60.0, phase=0.004, period=None, misread=0.0, seed=1):
    """Capture times at the card's rate, and the strip each reads, for a player started at ``origin``."""
    rng = np.random.default_rng(seed)
    times = phase + np.arange(n) / rate
    shown = np.floor((times - origin) / interval + 0.5).astype(np.int64)
    codes = shown % period if period else shown.copy()
    codes[shown < 0] = -1
    bad = rng.random(n) < misread
    codes[bad] = rng.integers(0, 1 << 16, bad.sum())
    return times, codes, shown


def test_the_line_through_legible_strips_is_the_player_clock():
    times, codes, _ = captures(3000, misread=0.2)
    model = F.clock_model(times, codes, fps=FPS)
    assert model.origin == pytest.approx(1.3, abs=0.01)
    assert model.interval == pytest.approx(1 / FPS, rel=1e-4)
    assert model.support > 2000


def picks_right(times, shown, model):
    """Whether the capture each frame is attributed to really shows it."""
    picked = F.attribute(times, model)
    return np.array_equal(picked[picked >= 0], shown[picked >= 0])


def test_a_drifting_player_clock_still_attributes_every_frame():
    times, codes, shown = captures(6000, interval=1.0001 / FPS, misread=0.1)
    model = F.clock_model(times, codes, fps=FPS)
    assert np.mean(model.index(times) == shown) > 0.99
    assert picks_right(times, shown, model)


def test_a_startup_burst_off_the_line_is_trimmed():
    times, codes, shown = captures(3000)
    codes[(shown >= 0) & (shown < 20)] = np.arange(((shown >= 0) & (shown < 20)).sum()) // 4
    model = F.clock_model(times, codes, fps=FPS)
    assert np.mean(model.index(times) == shown) > 0.99
    assert picks_right(times, shown, model)


def test_loop_codes_are_unwrapped():
    times, codes, shown = captures(4000, period=500)
    model = F.clock_model(times, codes, fps=FPS, period=500)
    assert model.period == 500
    seen = model.index(times[shown >= 0])
    assert np.mean(seen == shown[shown >= 0] % 500) > 0.99


def test_no_legible_strip_gives_no_model():
    times, codes, _ = captures(100)
    assert F.clock_model(times, np.full_like(codes, -1), fps=FPS) is None
    rng = np.random.default_rng(4)
    assert F.clock_model(times, rng.integers(0, 1 << 16, times.size), fps=FPS) is None


def test_a_held_strip_is_not_a_clock():
    times, codes, _ = captures(3000)
    assert F.clock_model(times, np.full_like(codes, 892), fps=FPS) is None


def test_each_frame_is_attributed_to_the_capture_nearest_its_centre():
    times, codes, shown = captures(600)
    model = F.clock_model(times, codes, fps=FPS)
    picked = F.attribute(times, model, frames=200)
    kept = picked[picked >= 0]
    assert np.array_equal(np.sort(kept), np.unique(kept))
    assert set(kept) == set(range(0, min(200, shown.max() + 1)))
    assert np.all(picked[shown >= 200] == -1)


def test_motion_finds_the_lag_of_a_take_with_no_strip():
    rng = np.random.default_rng(2)
    source_ti = rng.gamma(2.0, size=900)
    times, _, shown = captures(3000, origin=2.0)
    on = (shown >= 0) & (shown < source_ti.size)
    output_ti = np.zeros(times.size)
    change = np.r_[False, np.diff(shown) != 0] & on
    output_ti[change] = source_ti[shown[change]]
    model = F.motion_model(times, output_ti, source_ti, fps=FPS)
    assert model.origin == pytest.approx(2.0, abs=1.5 / FPS)


def test_motion_wraps_over_a_loop():
    rng = np.random.default_rng(5)
    source_ti = rng.gamma(2.0, size=300)
    times, _, shown = captures(2000, origin=-3.0, period=300)
    output_ti = np.zeros(times.size)
    change = np.r_[False, np.diff(shown) != 0]
    output_ti[change] = source_ti[shown[change] % 300]
    model = F.motion_model(times, output_ti, source_ti, fps=FPS, period=300)
    frame = np.rint((times - model.origin) / model.interval).astype(int) % 300
    assert np.mean(np.abs(((frame - shown % 300) + 150) % 300 - 150) <= 1) > 0.95


def test_motion_needs_a_few_frames():
    assert F.motion_model(np.arange(2) / FPS, np.zeros(2), np.zeros(2), fps=FPS) is None


def _recorded(monkeypatch, codes, frames):
    monkeypatch.setattr(
        F,
        "decode_recording",
        lambda clip, **kw: F.Recording(
            times=np.arange(len(codes)) / FPS, codes=np.array(codes), frames=frames
        ),
    )
    return F.source_series("clip.mkv", strip_px=8)


@pytest.mark.parametrize(
    "lead",
    [[65532, 65533, 65534, 65535], [-1, 65534, -1, 65535], []],
    ids=["lead-in", "misread-lead-in", "none"],
)
def test_the_source_series_starts_at_clip_frame_0_after_any_lead_in(monkeypatch, lead):
    rng = np.random.default_rng(4)
    clip = rng.integers(0, 256, (6, 36, 64), dtype=np.uint8)
    held = np.repeat(clip[:1], len(lead), axis=0)
    si, ti = _recorded(monkeypatch, lead + list(range(6)), np.concatenate([held, clip]))
    want_si, want_ti = _recorded(monkeypatch, list(range(6)), clip)
    assert len(si) == 6 and np.array_equal(si, want_si) and np.array_equal(ti, want_ti, equal_nan=True)


def test_a_clip_with_no_lead_in_keeps_an_unread_first_frame(monkeypatch):
    clip = np.random.default_rng(5).integers(0, 256, (5, 36, 64), dtype=np.uint8)
    assert len(_recorded(monkeypatch, [-1, 1, 2, 3, 4], clip)[0]) == 5

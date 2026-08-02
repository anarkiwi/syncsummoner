"""Flicker versus structure: what cycling a parameter actually does to the picture."""

# pylint: disable=missing-function-docstring

import numpy as np

from syncsummoner.aesthetics import flicker


def field(shape=(48, 64), seed=0):
    """Mid-range field, so a level shift does not clip and add spatial structure."""
    rng = np.random.default_rng(seed)
    return (0.3 + 0.4 * rng.random(shape)).astype(np.float32)


def test_whole_frame_level_change_is_pure_flicker():
    """A value remap can only brighten and darken: the blinking-light case.

    The field is kept mid-range because clipping is itself spatial and would
    show up, correctly, as structure.
    """
    base = field()
    frames = np.stack([base + np.float32(0.15 * np.sin(i)) for i in range(16)])
    split = flicker.temporal_split(frames)
    assert split.flicker > 0.95 and split.structure < 0.05


def test_displacement_is_pure_structure():
    """Rolling the picture moves content without changing its level."""
    base = field()
    frames = np.stack([np.roll(base, 3 * i, axis=1) for i in range(16)])
    split = flicker.temporal_split(frames)
    assert split.flicker < 0.05 and split.structure > 0.95


def test_shares_sum_to_one():
    frames = np.stack([field(seed=i) for i in range(8)])
    split = flicker.temporal_split(frames)
    assert abs(split.flicker + split.structure - 1.0) < 1e-6


def test_a_still_clip_is_quiet_not_flickering():
    frames = np.stack([field()] * 8)
    split = flicker.temporal_split(frames)
    assert split.quiet and split.flicker == 0.0


def test_single_frame_is_quiet():
    assert flicker.temporal_split(field()[None]).quiet


def test_rgb_frames_are_reduced_to_luma():
    base = np.repeat(field()[:, :, None], 3, axis=2)
    frames = np.stack([base + np.float32(0.12 * np.sin(i)) for i in range(12)])
    assert flicker.flicker_ratio(frames) > 0.9


def test_flicker_dominated_response_is_held_to_the_slow_rate():
    """The stated failure: an analog effect cycled fast is an expensive blinking light."""
    blink = flicker.TemporalSplit(flicker=0.9, structure=0.1, energy=0.01)
    assert flicker.usable_rate_hz(blink) == 1.0


def test_structured_response_earns_a_faster_rate():
    motion = flicker.TemporalSplit(flicker=0.02, structure=0.98, energy=0.01)
    assert flicker.usable_rate_hz(motion) > 10.0


def test_rate_falls_monotonically_as_flicker_rises():
    rates = [flicker.usable_rate_hz(flicker.TemporalSplit(f, 1 - f, 0.01)) for f in (0.0, 0.1, 0.3, 0.5, 0.9)]
    assert all(a >= b for a, b in zip(rates, rates[1:]))


def test_a_quiet_response_earns_nothing():
    assert flicker.usable_rate_hz(flicker.TemporalSplit(0.0, 0.0, 0.0)) == 1.0

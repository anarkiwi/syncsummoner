"""Clip descriptor assembly and the aggregate score."""

import dataclasses

import numpy as np
import pytest

from syncsummoner.aesthetics import __version__
from syncsummoner.aesthetics.dynamics import StabilityClass
from syncsummoner.aesthetics.score import (
    DEFAULT_WEIGHTS,
    ClipDescriptor,
    ScoreWeights,
    describe_clip,
    score_clip,
)
from tests.aesthetics import drifting, grating, texture

FPS = 30.0


def clip(rng, n_frames=24, shift=8):
    """Drifting texture clip with a frame period of size/shift frames."""
    return drifting(texture(rng), n_frames, shift=shift)


def test_descriptor_is_complete_and_stamped():
    """Every contract field is populated and the analyzer version is recorded."""
    rng = np.random.default_rng(0)
    descriptor = describe_clip(clip(rng), fps=FPS, rng=rng)
    assert isinstance(descriptor, ClipDescriptor)
    assert descriptor.analyzer_version == __version__
    assert descriptor.n_frames == 24 and descriptor.fps == FPS
    assert descriptor.channel_energy.shape == (5, 4)
    assert descriptor.channel_energy.sum() == pytest.approx(1.0, abs=1e-5)
    assert descriptor.information_content.shape == (24,)
    assert descriptor.passthrough_distance is None
    assert 0.0 <= descriptor.concentration <= 1.0
    assert 1.0 <= descriptor.fractal_dimension <= 2.0
    populated = dataclasses.asdict(descriptor)
    assert set(populated) == {f.name for f in dataclasses.fields(ClipDescriptor)}
    assert populated["levels"]["luma_mean"] > 0.0 and populated["motion"]["flow_magnitude"] > 0.0


def test_describe_clip_is_deterministic():
    """Equal inputs and seeds give an identical descriptor."""
    frames = clip(np.random.default_rng(1))
    left, right = (describe_clip(frames, fps=FPS, rng=np.random.default_rng(2)) for _ in range(2))
    assert left.information_content.tolist() == right.information_content.tolist()
    assert score_clip(left) == score_clip(right)


def test_drifting_clip_is_periodic():
    """A clip that returns to its first frame every 8 frames reads as periodic."""
    descriptor = describe_clip(clip(np.random.default_rng(3)), fps=FPS, rng=np.random.default_rng(4))
    assert descriptor.dynamics.stability is StabilityClass.PERIODIC
    assert descriptor.dynamics.winding_number == pytest.approx(0.125, rel=0.05)


def test_still_clip_is_static_and_motionless():
    """A repeated frame has no motion and a static dynamics class."""
    rng = np.random.default_rng(5)
    frames = np.repeat(texture(rng)[None], 6, axis=0)
    descriptor = describe_clip(frames, fps=FPS, rng=rng)
    assert descriptor.dynamics.stability is StabilityClass.STATIC
    assert descriptor.motion.framediff_energy == 0.0


def test_single_frame_clip():
    """A one-frame clip has no motion pairs but still describes."""
    rng = np.random.default_rng(6)
    descriptor = describe_clip(texture(rng)[None], fps=FPS, rng=rng)
    assert descriptor.n_frames == 1
    assert descriptor.motion.flow_magnitude == 0.0


def test_passthrough_distance_against_a_source():
    """A source stack yields the mean per-frame distance; identity is zero."""
    rng = np.random.default_rng(7)
    frames = clip(rng)
    assert describe_clip(frames, fps=FPS, rng=rng, source=frames).passthrough_distance == 0.0
    single = describe_clip(frames, fps=FPS, rng=rng, source=frames[0])
    assert single.passthrough_distance == 0.0


@pytest.mark.parametrize("frames", [np.zeros((0, 4, 4, 3), np.float32), np.zeros((4, 4, 3), np.float32)])
def test_bad_frame_stacks_are_rejected(frames):
    """Empty stacks and bare frames are not clips."""
    with pytest.raises(ValueError):
        describe_clip(frames, fps=FPS, rng=np.random.default_rng(8))


def test_score_is_bounded_and_weight_driven():
    """The aggregate stays in [-1, 1] and collapses to zero with zero weights."""
    rng = np.random.default_rng(9)
    descriptor = describe_clip(clip(rng), fps=FPS, rng=rng)
    assert -1.0 <= score_clip(descriptor) <= 1.0
    assert score_clip(descriptor, ScoreWeights(0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)) == 0.0
    assert score_clip(descriptor, DEFAULT_WEIGHTS) == score_clip(descriptor)


def test_mud_is_penalized():
    """A single-orientation grating clip scores worse on mud than broadband texture."""
    rng = np.random.default_rng(10)
    weights = ScoreWeights(mud=1.0, spectrum=0.0, fractal=0.0, legality=0.0, motion=0.0)
    mud = describe_clip(drifting(grating(), 8, shift=1), fps=FPS, rng=rng)
    varied = describe_clip(drifting(texture(rng), 8, shift=1), fps=FPS, rng=rng)
    assert mud.concentration > varied.concentration
    assert score_clip(mud, weights) < score_clip(varied, weights)


def test_illegal_levels_are_penalized():
    """Clipped, out-of-gamut content loses against legal content."""
    rng = np.random.default_rng(12)
    weights = ScoreWeights(mud=0.0, spectrum=0.0, fractal=0.0, legality=1.0, motion=0.0, boredom=0.0)
    legal = describe_clip(drifting(texture(rng) * 0.5 + 0.25, 8, shift=1), fps=FPS, rng=rng)
    illegal = describe_clip(drifting(np.round(texture(rng)), 8, shift=1), fps=FPS, rng=rng)
    assert score_clip(illegal, weights) < score_clip(legal, weights)

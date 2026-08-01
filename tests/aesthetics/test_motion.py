"""Frame difference and optical flow statistics."""

import numpy as np
import pytest

from syncsummoner.aesthetics.motion import MotionStats, motion_stats
from tests.aesthetics import texture


@pytest.mark.parametrize("shift", [1, 2, 4])
def test_flow_recovers_rigid_translation(shift):
    """Farneback flow returns the applied displacement with near-perfect coherence."""
    frame = texture(np.random.default_rng(0))
    stats = motion_stats(frame, np.roll(frame, shift, axis=1))
    assert stats.flow_magnitude == pytest.approx(shift, abs=0.1)
    assert stats.flow_coherence > 0.99
    assert stats.framediff_energy > 0.0


def test_identical_frames_are_still():
    """No change between frames means no difference energy and no flow."""
    stats = motion_stats(*(texture(np.random.default_rng(1)),) * 2)
    assert stats.framediff_energy == 0.0
    assert stats.flow_magnitude == pytest.approx(0.0, abs=1e-2)
    assert 0.0 <= stats.flow_coherence <= 1.0
    flat = np.full((32, 32, 3), 0.5, dtype=np.float32)
    assert motion_stats(flat, flat) == MotionStats(0.0, 0.0, 0.0)


def test_incoherent_motion_scores_below_rigid_motion():
    """Independent random frames give less coherent flow than a rigid shift."""
    rng = np.random.default_rng(2)
    frame = texture(rng)
    rigid = motion_stats(frame, np.roll(frame, 2, axis=1))
    random = motion_stats(frame, texture(rng))
    assert random.flow_coherence < rigid.flow_coherence


def test_shape_mismatch_is_rejected():
    """Frames of different geometry cannot be compared."""
    rng = np.random.default_rng(3)
    with pytest.raises(ValueError):
        motion_stats(texture(rng, size=64), texture(rng, size=32))

"""The lock statistic, on synthetic sequences and on captured Videomancer angles.

The fixture holds the decoded hue and ramp angles of five programs, 120 captures each,
played at 12 fps into a 30 fps capture, which is what the statistic was fixed against.
"""

from pathlib import Path

import numpy as np
import pytest

from syncsummoner.probe import codeframes as cf

FIXTURE = Path(__file__).parent / "data" / "videomancer_angles.npz"
LENGTH = 120
JITTER_DEG = 5.0
#: One 24 frame loop played at 12 fps into a 30 fps capture.
RATE = 2.0 * np.pi * 0.4 / 24
LOCK_TOL = 0.05
ALIVE, MIRRORED = cf.ChannelState.ALIVE, cf.ChannelState.MIRRORED
FROZEN, DESTROYED = cf.ChannelState.FROZEN, cf.ChannelState.DESTROYED

#: Program: per channel (state, forward, reverse, constant), then the verdict it implies.
HARDWARE = {
    "Passthru": ((ALIVE, 0.996, 0.033, 0.025), (ALIVE, 0.998, 0.007, 0.020), cf.Verdict.PASSTHROUGH),
    "Sabattier": ((ALIVE, 0.987, 0.034, 0.098), (ALIVE, 0.997, 0.017, 0.024), cf.Verdict.PASSTHROUGH),
    "Scramble": ((ALIVE, 0.996, 0.033, 0.026), (FROZEN, 0.036, 0.037, 0.997), cf.Verdict.GEOMETRIC),
    "Keystone": ((MIRRORED, 0.093, 0.988, 0.025), (FROZEN, 0.010, 0.010, 1.000), cf.Verdict.GEOMETRIC),
    "YUV_Phaser": ((ALIVE, 0.996, 0.035, 0.025), (ALIVE, 0.998, 0.006, 0.008), cf.Verdict.PASSTHROUGH),
}


@pytest.fixture(name="captured", scope="module")
def captured_fixture():
    """Measured angles."""
    with np.load(FIXTURE) as data:
        return {key: np.asarray(data[key], dtype=np.float64) for key in data.files}


def hardware(captured, program):
    """Both channels of one program, locked at the rate estimated from them."""
    hue, ramp = captured[f"{program}__hue"], captured[f"{program}__ramp"]
    rate = cf.estimate_rate((hue, ramp), expected_rate(captured))
    return cf.channel_lock(hue, rate), cf.channel_lock(ramp, rate)


def expected_rate(captured):
    """The rate the two frame rates imply, with no reference to the angles."""
    return cf.step_rate(int(captured["count"]), captured["loop_fps"], captured["capture_fps"])


def advancing(rate, length=LENGTH, phase=0.7, jitter=0.0, seed=0):
    """A sequence advancing at a constant rate, optionally with gaussian phase jitter."""
    noise = np.random.default_rng(seed).normal(0.0, np.deg2rad(jitter), length)
    return phase + rate * np.arange(length) + noise


def test_a_clean_advance_locks():
    """A clean advance locks."""
    lock = cf.channel_lock(advancing(RATE), RATE)
    assert lock.state is ALIVE
    assert lock.forward > 0.999
    assert lock.reverse < 0.1 and lock.constant < 0.1
    assert lock.phase == pytest.approx(0.7, abs=1e-6)


def test_decode_jitter_does_not_break_the_lock():
    """Decode jitter does not break the lock."""
    angles = advancing(RATE, jitter=JITTER_DEG)
    lock = cf.channel_lock(angles, RATE)
    assert lock.state is ALIVE
    assert lock.forward > 0.9
    assert abs(np.exp(1j * 24 * angles).mean()) < 0.5


def test_a_constant_sequence_reads_frozen():
    """A constant sequence reads frozen."""
    lock = cf.channel_lock(np.full(LENGTH, -1.2), RATE)
    assert lock.state is FROZEN
    assert lock.constant > 0.99
    assert lock.lock < cf.SURVIVAL_MIN


def test_a_reversed_sequence_reads_mirrored():
    """A reversed sequence reads mirrored."""
    lock = cf.channel_lock(advancing(-RATE, jitter=JITTER_DEG), RATE)
    assert lock.state is MIRRORED
    assert lock.orientation == -1
    assert lock.reverse > 0.9 and lock.forward < cf.SURVIVAL_MIN


def test_uniform_angles_read_destroyed():
    """Uniform angles read destroyed."""
    angles = np.random.default_rng(3).uniform(-np.pi, np.pi, LENGTH)
    lock = cf.channel_lock(angles, RATE)
    assert lock.state is DESTROYED
    assert max(lock.forward, lock.reverse, lock.constant) < cf.SURVIVAL_MIN


def test_no_captures_read_destroyed():
    """No captures read destroyed."""
    lock = cf.channel_lock([], RATE)
    assert lock.state is DESTROYED
    assert lock.lock == 0.0


def test_two_frozen_channels_are_a_frozen_output():
    """Two frozen channels are a frozen output."""
    frozen = cf.channel_lock(np.full(LENGTH, 0.4), RATE)
    verdict, confidence = cf.classify(frozen, frozen)
    assert verdict is cf.Verdict.FROZEN
    assert confidence > 0.99


def test_two_channels_that_track_but_disagree_have_decoded_nothing():
    """Two channels that track but disagree have decoded nothing."""
    hue = advancing(RATE, jitter=66.0, seed=1)
    ramp = advancing(RATE, jitter=66.0, seed=2)
    locks = (cf.channel_lock(hue, RATE), cf.channel_lock(ramp, RATE))
    agreement = max(abs(np.exp(1j * (hue - ramp)).mean()), abs(np.exp(1j * (hue + ramp)).mean()))
    assert all(lock.state is ALIVE for lock in locks)
    assert cf.classify(*locks, agreement=agreement)[0] is cf.Verdict.DESTRUCTIVE


def test_rate_follows_from_the_frame_rates(captured):
    """Rate follows from the frame rates."""
    assert expected_rate(captured) == pytest.approx(0.10472, abs=1e-4)
    assert cf.step_rate(16, 30, 30) == pytest.approx(2.0 * np.pi / 16)


def test_rate_is_refined_within_tolerance_of_the_expected_one():
    """Rate is refined within tolerance of the expected one."""
    true_rate = RATE * 1.03
    angles = advancing(true_rate, jitter=JITTER_DEG)
    assert cf.estimate_rate((angles, angles), RATE) == pytest.approx(true_rate, rel=0.005)


def test_two_dead_channels_fall_back_to_the_expected_rate():
    """Two dead channels fall back to the expected rate."""
    rng = np.random.default_rng(9)
    dead = (np.full(LENGTH, 0.3), rng.uniform(-np.pi, np.pi, LENGTH))
    assert cf.estimate_rate(dead, RATE) == RATE


@pytest.mark.parametrize("program", sorted(HARDWARE))
def test_hardware_channel_states(captured, program):
    """Hardware channel states."""
    expected_hue, expected_ramp, _ = HARDWARE[program]
    for lock, (state, forward, reverse, constant) in zip(
        hardware(captured, program), (expected_hue, expected_ramp)
    ):
        assert lock.state is state, program
        assert lock.forward == pytest.approx(forward, abs=LOCK_TOL)
        assert lock.reverse == pytest.approx(reverse, abs=LOCK_TOL)
        assert lock.constant == pytest.approx(constant, abs=LOCK_TOL)


@pytest.mark.parametrize("program", sorted(HARDWARE))
def test_hardware_verdicts(captured, program):
    """Hardware verdicts."""
    hue_lock, ramp_lock = hardware(captured, program)
    hue, ramp = captured[f"{program}__hue"], captured[f"{program}__ramp"]
    agreement = max(abs(np.exp(1j * (hue - ramp)).mean()), abs(np.exp(1j * (hue + ramp)).mean()))
    linearity = float(np.median(captured[f"{program}__lin"]))
    rotation = float(np.angle(np.exp(1j * (hue_lock.phase - ramp_lock.phase))))
    paired = hue_lock.state is ALIVE and ramp_lock.state is ALIVE
    verdict, confidence = cf.classify(
        hue_lock,
        ramp_lock,
        agreement=agreement,
        linearity=linearity,
        hue_rotation=rotation if paired else float("nan"),
    )
    assert verdict is HARDWARE[program][2], program
    assert confidence > 0.9


def test_the_chain_leaves_hue_and_ramp_within_a_step_of_each_other(captured):
    """The chain leaves hue and ramp within a step of each other."""
    for program in ("Passthru", "Sabattier", "YUV_Phaser"):
        hue_lock, ramp_lock = hardware(captured, program)
        offset = np.degrees(np.angle(np.exp(1j * (hue_lock.phase - ramp_lock.phase))))
        assert abs(offset) < np.degrees(cf.HUE_ROTATION_MIN)


def test_the_old_grid_metric_was_anticorrelated_with_survival(captured):
    """The old grid metric was anticorrelated with survival."""
    count = int(captured["count"])
    live = abs(np.exp(1j * count * captured["Passthru__hue"]).mean())
    dead = abs(np.exp(1j * count * captured["Keystone__ramp"]).mean())
    assert live < dead
    hue_lock, ramp_lock = hardware(captured, "Keystone")
    assert hardware(captured, "Passthru")[0].lock > ramp_lock.lock
    assert hue_lock.state is MIRRORED

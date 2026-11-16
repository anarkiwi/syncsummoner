"""Picture and motion judged by structure, on real MJPEG clips and on the statistics alone."""

# pylint: disable=missing-function-docstring,no-member  ; cv2 defeats pylint introspection

import shutil
import subprocess

import cv2
import numpy as np
import pytest

from syncsummoner.device.capture import structured
from syncsummoner.device.recorder import TakeReport, held, inspect_take

SIZE = (360, 640)
FRAMES = 30
needs_ffmpeg = pytest.mark.skipif(shutil.which("ffmpeg") is None, reason="ffmpeg is not installed")


def mjpeg(path, frames, fps=30):
    """Encode uint8 RGB frames through the card's codec, so its compression noise is real."""
    height, width = frames[0].shape[:2]
    subprocess.run(
        ["ffmpeg", "-loglevel", "error", "-y", "-f", "rawvideo", "-pix_fmt", "rgb24"]
        + ["-video_size", f"{width}x{height}", "-framerate", str(fps), "-i", "pipe:0"]
        + ["-c:v", "mjpeg", "-q:v", "5", "-pix_fmt", "yuvj422p", str(path)],
        input=b"".join(np.ascontiguousarray(f, np.uint8).tobytes() for f in frames),
        check=True,
    )
    return path


def judged(tmp_path, frames):
    return inspect_take(mjpeg(tmp_path / "take.mkv", frames), samples=60)


def flat(level):
    return [np.full(SIZE + (3,), level, np.uint8)] * FRAMES


def dark_footage(rng, still=False):
    """Mostly near-black grain, a faint drifting gradient and a few full-range highlights."""
    y, x = np.indices(SIZE)
    frames = []
    for t in range(FRAMES):
        shift = 0 if still else 6 * t
        base = 1.5 + 1.5 * np.sin((x + shift) / 40.0) + rng.normal(0, 1.5, SIZE)
        spots = ((x - 100 - shift) % 640 < 10) & (np.abs(y - 180) < 10)
        luma = np.clip(np.where(spots, 255, base), 0, 255)
        frames.append(np.repeat(luma[..., None], 3, axis=2).astype(np.uint8))
    return frames


def no_signal_card():
    """High-contrast achromatic slide with text and a QR-like block, as the card synthesizes."""
    card = np.full(SIZE + (3,), 255, np.uint8)
    cv2.putText(card, "No Signal", (150, 120), cv2.FONT_HERSHEY_SIMPLEX, 2.5, (0, 0, 0), 8)
    modules = np.random.default_rng(5).random((21, 21)) < 0.5
    card[180:327, 250:397] = np.where(np.kron(modules, np.ones((7, 7), bool)), 0, 255)[..., None]
    return card


@needs_ffmpeg
@pytest.mark.parametrize("level", [0, 16, 128], ids=["black", "limited-black", "mid-grey"])
def test_a_uniform_field_is_dead_at_any_level(tmp_path, level):
    """Roll Shutter's dead output is video black at 16 on every frame, and is no picture."""
    report = judged(tmp_path, flat(level))
    assert report.frames == FRAMES and report.picture == 0 and report.spread == 0
    assert not report.live and not report.usable and "no picture" in str(report)


@needs_ffmpeg
def test_dark_footage_with_highlights_is_a_moving_picture(tmp_path):
    """The case a brightness threshold called blank: dark, but full range and moving."""
    report = judged(tmp_path, dark_footage(np.random.default_rng(0)))
    assert report.luma < 0.02, "dark enough that the old mean-level test called it blank"
    assert report.picture == FRAMES and report.spread > 200
    assert report.live and not report.frozen and report.usable and report.motion > -0.5
    assert "UNUSABLE" not in str(report)


@needs_ffmpeg
def test_a_paused_picture_with_grain_is_live_but_frozen(tmp_path):
    """Fresh noise every frame makes each MJPEG frame differ, which is not motion."""
    report = judged(tmp_path, dark_footage(np.random.default_rng(1), still=True))
    assert report.live and report.frozen and not report.usable
    assert abs(report.motion + 0.5) < 0.1 and "frozen" in str(report)


@needs_ffmpeg
def test_an_exactly_repeated_picture_is_live_but_frozen(tmp_path):
    report = judged(tmp_path, dark_footage(np.random.default_rng(2))[:1] * FRAMES)
    assert report.live and report.frozen and report.motion is None and "motion n/a" in str(report)


@needs_ffmpeg
def test_the_no_signal_splash_is_no_picture_despite_its_structure(tmp_path):
    report = judged(tmp_path, [no_signal_card()] * FRAMES)
    assert report.spread > 200, "it has more structure than any dark program"
    assert report.splash == FRAMES and report.picture == 0 and not report.live


@needs_ffmpeg
def test_snow_through_mjpeg_is_a_changing_picture_not_a_frozen_one(tmp_path):
    rng = np.random.default_rng(4)
    report = judged(tmp_path, [rng.integers(0, 256, SIZE + (3,), np.uint8) for _ in range(FRAMES)])
    assert report.usable and abs(report.motion + 0.5) < 0.1


@needs_ffmpeg
def test_inspection_is_sized_by_the_take_not_by_an_assumed_aspect(tmp_path):
    frames = [f[:288, :360] for f in dark_footage(np.random.default_rng(3))]
    assert judged(tmp_path, frames).frames == FRAMES


def test_an_unreadable_take_has_no_frames(tmp_path):
    report = inspect_take(tmp_path / "missing.mkv")
    assert report == TakeReport() and not report.live and "0 frames" in str(report)


def test_structure_is_range_beyond_one_quantization_step():
    field = np.full((8, 8, 3), 0.25, np.float32)
    assert not structured(field)
    assert not structured(np.full((8, 8, 3), [1.0, 0.0, 0.0], np.float32)), "a uniform colour is uniform"
    field[0, 0, 1] += 1 / 255
    assert not structured(field), "one step is rounding"
    field[0, 0, 1] += 1 / 255
    assert structured(field)
    assert structured(np.eye(4, dtype=np.float32))


def held_picture(rng, count=40, noise=2.0):
    picture = rng.random((30, 40)) * 255
    return picture + rng.normal(0, noise, (count, 30, 40))


def test_a_held_picture_correlates_its_differences_at_minus_a_half():
    motion, frozen = held(held_picture(np.random.default_rng(0)))
    assert frozen and abs(motion + 0.5) < 0.05


def test_the_null_holds_whatever_the_noise_level():
    for noise in (0.1, 1.0, 20.0):
        assert held(held_picture(np.random.default_rng(1), noise=noise))[1]


def test_smooth_motion_is_not_held():
    y, x = np.indices((30, 40))
    frames = np.stack([128 + 100 * np.sin((x + 2 * t) / 5.0) + 0 * y for t in range(40)])
    motion, frozen = held(frames + np.random.default_rng(2).normal(0, 2.0, frames.shape))
    assert not frozen and motion > 0.5


def test_flicker_is_not_held():
    frames = held_picture(np.random.default_rng(3))
    frames[::2] += 40
    motion, frozen = held(frames)
    assert not frozen and motion < -0.9


def test_snow_shares_no_picture_so_is_not_held():
    """Independent frames also correlate at -1/2; only the shared picture tells them apart."""
    motion, frozen = held(np.random.default_rng(4).random((40, 30, 40)) * 255)
    assert abs(motion + 0.5) < 0.05 and not frozen


def test_too_few_distinct_frames_cannot_show_motion():
    frames = held_picture(np.random.default_rng(5), count=3)
    assert held(np.concatenate([frames, frames[-1:].repeat(10, axis=0)]))[1]
    assert held(np.empty((0, 4, 4)))[1]

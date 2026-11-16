"""Aligning a take: the lag a played run of timecodes holds, against held frames and misreads."""

# pylint: disable=missing-function-docstring

import dataclasses
import shutil
import subprocess
import types

import numpy as np
import pytest

from syncsummoner.compose import render as R

CONFIG = R.RenderConfig(width=32, height=24, fps=10.0, strip_px=4, bits=16)


def test_picture_start_finds_where_the_played_clip_begins(monkeypatch, tmp_path):
    """A take opens on the previous play's last frame, which carries its own timecode."""
    lead, config = 5, dataclasses.replace(CONFIG, lead_s=1.0)
    stale = R.burn_timecode(
        np.full((CONFIG.strip_px, CONFIG.width, 3), 0.5, np.float32),
        299,
        bits=CONFIG.bits,
        strip_px=CONFIG.strip_px,
    )
    strips = [stale for _ in range(lead)]
    strips += [
        R.burn_timecode(
            np.full((config.strip_px, config.width, 3), 0.5, np.float32),
            i,
            bits=config.bits,
            strip_px=config.strip_px,
        )
        for i in range(4)
    ]
    raw = np.clip(np.stack(strips), 0, 1).astype(np.float32)
    monkeypatch.setattr(
        R.subprocess,
        "run",
        lambda argv, **kw: types.SimpleNamespace(stdout=(raw * 255).astype(np.uint8).tobytes()),
    )
    assert R.picture_start(tmp_path / "take.mkv", config=config) == pytest.approx(lead / config.fps)


def test_picture_start_survives_a_misread_strip(monkeypatch, tmp_path):
    """One strip decoding to a wrong low value would otherwise move the whole clip."""
    config = dataclasses.replace(CONFIG, lead_s=1.0)
    plate = np.full((config.strip_px, config.width, 3), 0.5, np.float32)
    codes = [299, 299, 1, 0, 1, 2, 3, 4, 5, 6]
    raw = np.stack([R.burn_timecode(plate, c, bits=config.bits, strip_px=config.strip_px) for c in codes])
    monkeypatch.setattr(
        R.subprocess,
        "run",
        lambda argv, **kw: types.SimpleNamespace(
            stdout=(np.clip(raw, 0, 1) * 255).astype(np.uint8).tobytes()
        ),
    )
    assert R.picture_start(tmp_path / "take.mkv", config=config) == pytest.approx(3 / config.fps)


def test_picture_start_of_a_take_ffmpeg_cannot_read_is_unknown(monkeypatch, tmp_path):
    monkeypatch.setattr(R.subprocess, "run", lambda argv, **kw: types.SimpleNamespace(stdout=b""))
    assert R.picture_start(tmp_path / "take.mkv", config=CONFIG) is None


@pytest.mark.parametrize("flipped", [False, True])
def test_a_program_that_turns_the_picture_over_leaves_its_start_unknown(tmp_path, flipped):
    """A geometric program moves the strip off the edge; that is not a measured zero lag."""
    if shutil.which("ffmpeg") is None:
        pytest.skip("ffmpeg not installed")
    lead = 4
    plate = np.full((CONFIG.height, CONFIG.width, 3), 0.5, np.float32)
    codes = [299] * lead + list(range(10))
    frames = np.stack([R.burn_timecode(plate, i, bits=CONFIG.bits, strip_px=CONFIG.strip_px) for i in codes])
    frames = frames[:, ::-1] if flipped else frames
    take = tmp_path / "take.mkv"
    subprocess.run(
        ["ffmpeg", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "rgb24"]
        + ["-s", f"{CONFIG.width}x{CONFIG.height}", "-r", str(CONFIG.fps), "-i", "pipe:0"]
        + ["-c:v", "ffv1", "-pix_fmt", "bgr0", str(take)],
        input=(np.clip(frames, 0, 1) * 255).astype(np.uint8).tobytes(),
        check=True,
    )
    start = R.picture_start(take, config=dataclasses.replace(CONFIG, lead_s=1.0))
    assert start is None if flipped else start == pytest.approx(lead / CONFIG.fps)


def test_picture_start_reads_the_take_at_session_rate(monkeypatch, tmp_path):
    """The card returns its own frame rate; an index only means a time at the session's."""
    seen = {}

    def fake_run(argv, **kw):
        del kw
        seen["argv"] = argv
        return types.SimpleNamespace(stdout=b"")

    monkeypatch.setattr(R.subprocess, "run", fake_run)
    R.picture_start(tmp_path / "take.mkv", config=CONFIG)
    assert seen["argv"][seen["argv"].index("-vf") + 1].startswith(f"fps={CONFIG.fps},crop=")


def _noisy(run_at, lag, noise, n):
    """Codes for ``n`` frames: a played run from frame ``run_at`` plus misreads at lag 0."""
    codes = [None] * n
    for i in range(run_at, n):
        codes[i] = i - lag
    for i in noise:
        codes[i] = i
    return codes


def test_steady_lag_prefers_a_run_over_scattered_agreement():
    """Scattered misreads agreeing on lag 0 outnumber no single run, and lose to one."""
    codes = _noisy(40, 37, range(0, 36, 3), 50)
    assert sum(c == i for i, c in enumerate(codes)) > sum(c == i - 37 for i, c in enumerate(codes))
    assert R.steady_lag(codes, bits=16) == 37


def test_steady_lag_of_scattered_noise_alone_is_unknown():
    assert R.steady_lag(_noisy(50, 0, range(0, 36, 3), 50), bits=16) is None


def test_a_held_frame_is_no_alignment():
    """The lead-in's held frame keeps one index while its lag grows: no two frames agree."""
    assert R.steady_lag([412] * 30 + [None] * 90, bits=16) is None


def test_steady_lag_of_nothing_is_unknown():
    assert R.steady_lag([], bits=16) is None


def test_a_run_must_outlast_chance_among_the_frames_searched():
    """With few bits, a short run is as likely by chance as by alignment."""
    codes = [None, 0, 1, None] + [None] * 60
    assert R.steady_lag(codes, bits=16) == 1
    assert R.steady_lag(codes, bits=2) is None


def test_runs_split_by_a_dropout_add_up():
    codes = [None] * 10 + [i - 5 for i in range(10, 14)] + [None] + [i - 5 for i in range(15, 18)]
    codes += list(range(18, 23))
    assert R.steady_lag(codes, bits=16) == 5


def test_picture_start_ignores_a_held_frame_and_scattered_misreads(monkeypatch, tmp_path):
    """Spurious decodes that agree on lag 0 must not win over the played run."""
    config = dataclasses.replace(CONFIG, lead_s=3.0)
    plate = np.full((config.strip_px, config.width, 3), 0.5, np.float32)
    dark = np.zeros_like(plate)
    codes = _noisy(30, 20, [0, 2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 22], 50)
    raw = np.stack(
        [
            dark if c is None else R.burn_timecode(plate, c, bits=config.bits, strip_px=config.strip_px)
            for c in codes
        ]
    )
    monkeypatch.setattr(
        R.subprocess,
        "run",
        lambda argv, **kw: types.SimpleNamespace(
            stdout=(np.clip(raw, 0, 1) * 255).astype(np.uint8).tobytes()
        ),
    )
    assert R.picture_start(tmp_path / "take.mkv", config=config) == pytest.approx(20 / config.fps)
    raw[30:] = dark
    assert R.picture_start(tmp_path / "take.mkv", config=config) is None

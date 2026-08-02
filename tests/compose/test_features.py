"""Audio and video feature extraction against synthetic signals with known answers."""

# pylint: disable=missing-function-docstring

import types

import numpy as np
import pytest
from scipy.io import wavfile

from syncsummoner.compose import features as F

from . import SEGMENT_KW, SR, click_track


def test_beat_track_recovers_known_tempo():
    y, sr = click_track(seconds=6.0, bpm=120.0, contrast=False)
    env = F.onset_strength(y, sr)
    beats, tempo = F.beat_track(env, sr / 512)
    assert tempo == pytest.approx(120.0, rel=0.08)
    assert np.diff(beats).mean() == pytest.approx(0.5, rel=0.08)


def test_onset_strength_peaks_on_clicks():
    y, sr = click_track(seconds=3.0, bpm=120.0, contrast=False)
    env = F.onset_strength(y, sr)
    rate = sr / 512
    peaks = np.flatnonzero(env > 0.5) / rate
    assert peaks.size >= 4
    assert np.abs(np.round(peaks / 0.5) * 0.5 - peaks).max() < 0.1


def test_tempo_period_falls_back_outside_range():
    env = np.zeros(8)
    assert F.tempo_period(env, 2.0, bpm_range=(1000.0, 2000.0)) > 0


def test_band_envelopes_route_energy_to_the_right_input():
    t = np.arange(SR) / SR
    low = np.sin(2 * np.pi * 60 * t).astype(np.float32)
    high = np.sin(2 * np.pi * 6000 * t).astype(np.float32)
    assert int(np.argmax(F.band_envelopes(low, SR)[:, SR // 2 :].mean(axis=1))) == 0
    assert int(np.argmax(F.band_envelopes(high, SR)[:, SR // 2 :].mean(axis=1))) == 3


def test_write_cv_wav_is_four_channel(tmp_path):
    bands = F.band_envelopes(click_track(seconds=1.0)[0], SR)
    path = tmp_path / "cv.wav"
    F.write_cv_wav(path, bands, SR)
    sr, data = wavfile.read(path)
    assert sr == SR and data.shape[1] == 4


def test_information_content_ranks_random_above_periodic():
    rng = np.random.default_rng(0)
    periodic = np.tile([0.0, 1.0], 256)
    noise = rng.normal(size=512)
    ic = F.markov_information_content
    assert ic(noise, rng=rng).mean() > ic(periodic, rng=rng).mean()
    assert F.information_content(noise, rng=rng).shape == (512,)


def test_markov_information_content_short_series_is_zero():
    ic = F.markov_information_content(np.arange(2.0), rng=np.random.default_rng(0))
    assert ic.shape == (2,) and not ic.any()


def test_information_content_prefers_the_aesthetics_analyzer(monkeypatch):
    import sys

    import syncsummoner

    fake = types.ModuleType("syncsummoner.aesthetics")
    fake.information_content = lambda series, *, rng, order, n_bins: np.full(len(series), 7.0)
    monkeypatch.setitem(sys.modules, "syncsummoner.aesthetics", fake)
    monkeypatch.setattr(syncsummoner, "aesthetics", fake, raising=False)
    out = F.information_content(np.arange(8.0), rng=np.random.default_rng(0))
    assert out.tolist() == [7.0] * 8


def test_segment_finds_the_spectral_boundary():
    y, sr = click_track(seconds=6.0)
    sections = F.segment(F.band_spectrogram(y, hop=512), sr / 512, **SEGMENT_KW)
    assert len(sections) >= 2
    assert min(abs(s.start - 3.0) for s in sections) < 0.6
    assert sections[0].start == 0.0 and sections[-1].duration > 0


def test_segment_degenerate_input_is_one_section():
    sections = F.segment(np.zeros((4, 2)), 2.0)
    assert len(sections) == 1 and sections[0].label == "A"


def test_load_audio_resamples_and_normalizes(tmp_path):
    path = tmp_path / "a.wav"
    data = (np.sin(2 * np.pi * 220 * np.arange(11025) / 11025) * 20000).astype(np.int16)
    wavfile.write(path, 11025, np.stack([data, data], axis=1))
    y, sr = F.load_audio(path, sr=SR)
    assert sr == SR and y.dtype == np.float32
    assert np.abs(y).max() < 1.01 and y.size == pytest.approx(22050, rel=0.01)


def test_librosa_paths_are_used_when_importable(monkeypatch):
    fake = types.SimpleNamespace(
        load=lambda path, sr, mono: (np.ones(16, np.float32), sr),
        onset=types.SimpleNamespace(onset_strength=lambda y, sr, n_fft, hop_length: np.arange(4.0)),
    )
    monkeypatch.setattr(F, "librosa", fake)
    y, sr = F.load_audio("ignored.wav", sr=8000)
    assert sr == 8000 and y.size == 16
    assert F.onset_strength(np.zeros(8, np.float32), 8000).max() == pytest.approx(1.0)


def test_downbeats_and_tempo_curve_edge_cases():
    beats = np.arange(8) * 0.5
    env = np.zeros(64)
    env[np.round(beats[::4] / (1 / 8)).astype(int)] = 1.0
    downs = F.downbeat_times(beats, env, 8.0, meter=4)
    assert downs.size == 2 and downs[0] == pytest.approx(0.0)
    assert F.downbeat_times(np.array([0.1]), env, 8.0).size == 1
    assert F.tempo_curve(np.array([0.0]), np.zeros(3), 90.0).tolist() == [90.0] * 3
    assert F.tempo_curve(beats, np.array([0.75]), 90.0)[0] == pytest.approx(120.0)


def test_analyze_frames_detects_a_hard_cut():
    rng = np.random.default_rng(3)
    frames = np.concatenate([np.full((6, 8, 8, 3), 0.2, np.float32), np.full((6, 8, 8, 3), 0.9, np.float32)])
    frames += rng.normal(0, 0.005, frames.shape).astype(np.float32)
    vf = F.analyze_frames(frames, fps=30.0)
    assert vf.n_frames == 12 and vf.fps == 30.0
    assert 6 in vf.shot_boundaries.tolist()
    assert vf.motion_energy.size == 12 and vf.luma.size == 12 and vf.chroma.size == 12


def test_analyze_frames_single_frame():
    vf = F.analyze_frames(np.zeros((1, 4, 4, 3), np.float32), fps=25.0)
    assert vf.shot_boundaries.size == 0 and vf.motion_energy.size == 1


def test_read_video_converts_to_rgb(monkeypatch):
    import cv2

    frames = [np.full((4, 4, 3), c, np.uint8) for c in (10, 200)]

    class FakeCap:
        """Minimal cv2.VideoCapture stand-in."""

        def __init__(self, _path):
            self.i = 0

        def get(self, _prop):
            return 24.0

        def read(self):
            if self.i >= len(frames):
                return False, None
            self.i += 1
            return True, frames[self.i - 1]

        def release(self):
            pass

    monkeypatch.setattr(cv2, "VideoCapture", FakeCap)
    stack, fps = F.read_video("fake.mp4")
    assert fps == 24.0 and stack.shape == (2, 4, 4, 3)
    assert stack.max() <= 1.0
    assert F.read_video("fake.mp4", max_frames=0)[0].shape[0] == 0


def test_analyze_end_to_end(tmp_path, monkeypatch):
    y, sr = click_track(seconds=3.0)
    path = tmp_path / "t.wav"
    wavfile.write(path, sr, (y * 20000).astype(np.int16))
    monkeypatch.setattr(
        F,
        "read_frames",
        lambda p, max_frames=None: iter([(30.0, np.zeros((8, 8, 3), np.float32))] * 4),
    )
    feats = F.analyze("clip.mp4", path, rng=np.random.default_rng(1), sr=sr, **SEGMENT_KW)
    assert feats.audio.duration == pytest.approx(3.0, rel=0.01)
    assert feats.audio.bar_duration > 0 and feats.audio.bands.shape[0] == 4
    assert feats.video.n_frames == 4
    empty = F.analyze(None, None, rng=np.random.default_rng(1))
    assert empty.audio is None and empty.video is None


def test_bar_duration_falls_back_to_tempo():
    audio = F.AudioFeatures(
        sr=SR,
        duration=1.0,
        rate=43.0,
        tempo=120.0,
        times=np.zeros(1),
        beats=np.zeros(1),
        downbeats=np.zeros(1),
        tempo_curve=np.zeros(1),
        onset_strength=np.zeros(1),
        onset_ic=np.zeros(1),
        bands=np.zeros((4, 1)),
        sections=(),
    )
    assert audio.bar_duration == pytest.approx(2.0)


def test_streaming_video_analysis_matches_the_whole_stack(monkeypatch):
    """A 3 minute 576p clip costs 21GB as one stack, so the streamed path must agree exactly."""
    import cv2

    rng = np.random.default_rng(3)
    frames = [(rng.random((6, 8, 3)) * 255).astype(np.uint8) for _ in range(5)]
    frames[3] = np.full((6, 8, 3), 250, np.uint8)

    class FakeCap:
        """Minimal cv2.VideoCapture stand-in."""

        def __init__(self, _path):
            self.i = 0

        def get(self, _prop):
            return 25.0

        def read(self):
            if self.i >= len(frames):
                return False, None
            self.i += 1
            return True, frames[self.i - 1]

        def release(self):
            pass

    monkeypatch.setattr(cv2, "VideoCapture", FakeCap)
    stack, fps = F.read_video("fake.mkv")
    want = F.analyze_frames(stack, fps=fps)
    got = F.analyze_video("fake.mkv")
    assert got.n_frames == want.n_frames and got.fps == want.fps
    for name in ("motion_energy", "luma", "chroma", "shot_boundaries"):
        assert np.allclose(getattr(got, name), getattr(want, name)), name


def test_streaming_video_analysis_of_an_empty_clip(monkeypatch):
    import cv2

    class EmptyCap:
        """Capture that decodes nothing."""

        def __init__(self, _path):
            pass

        def get(self, _prop):
            return 0.0

        def read(self):
            """Read."""
            return False, None

        def release(self):
            pass

    monkeypatch.setattr(cv2, "VideoCapture", EmptyCap)
    vf = F.analyze_video("fake.mkv")
    assert vf.n_frames == 0 and vf.fps == 30.0 and vf.motion_energy.size == 0

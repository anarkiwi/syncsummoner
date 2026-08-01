"""Audio/visual lag estimation."""

import numpy as np
import pytest

from syncsummoner.aesthetics.sync import av_correlation

FPS = 60.0


def pulse_train(n, period, offset=0):
    """Impulse-like series with a pulse every ``period`` samples."""
    series = np.zeros(n)
    series[(np.arange(offset % period, n, period))] = 1.0
    return np.convolve(series, np.hanning(7), mode="same")


@pytest.mark.parametrize("lag_frames", [-6, 0, 6])
def test_lag_sign_and_magnitude(lag_frames):
    """Positive lag means the visual series trails the audio series."""
    audio = pulse_train(600, 40)
    visual = np.roll(audio, lag_frames)
    result = av_correlation(visual, audio, visual_fps=FPS, audio_fps=FPS)
    assert result.lag_s == pytest.approx(lag_frames / FPS, abs=1.0 / FPS)
    assert result.correlation > 0.9


def test_resamples_to_a_common_rate():
    """Series at different rates are compared on the finer grid."""
    t_audio = np.arange(1000) / 200.0
    t_visual = np.arange(150) / 30.0
    audio = np.sin(2.0 * np.pi * 1.0 * t_audio)
    visual = np.sin(2.0 * np.pi * 1.0 * (t_visual - 0.1))
    result = av_correlation(visual, audio, visual_fps=30.0, audio_fps=200.0)
    assert result.lag_s == pytest.approx(0.1, abs=0.02)


def test_lag_search_is_bounded():
    """The reported lag never exceeds the requested search window."""
    audio = pulse_train(400, 25)
    result = av_correlation(np.roll(audio, 60), audio, visual_fps=FPS, audio_fps=FPS, max_lag_s=0.1)
    assert abs(result.lag_s) <= 0.1 + 1e-9


def test_uncorrelated_series_score_low():
    """Independent noise gives a much weaker peak than an aligned pair."""
    rng = np.random.default_rng(0)
    noise = av_correlation(rng.standard_normal(500), rng.standard_normal(500), visual_fps=FPS, audio_fps=FPS)
    aligned = pulse_train(500, 30)
    matched = av_correlation(aligned, aligned, visual_fps=FPS, audio_fps=FPS)
    assert noise.correlation < 0.5 < matched.correlation


def test_constant_series_are_handled():
    """A flat series has no variance and correlates at zero."""
    result = av_correlation(np.ones(100), pulse_train(100, 10), visual_fps=FPS, audio_fps=FPS)
    assert result.correlation == pytest.approx(0.0, abs=1e-9)


def test_short_series_are_rejected():
    """Cross-correlation needs at least two samples per series."""
    with pytest.raises(ValueError):
        av_correlation(np.ones(1), np.ones(10), visual_fps=FPS, audio_fps=FPS)

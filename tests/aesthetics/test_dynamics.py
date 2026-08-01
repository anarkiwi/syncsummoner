"""Periodicity, winding number and stability classification."""

import numpy as np
import pytest

from syncsummoner.aesthetics.dynamics import (
    StabilityClass,
    acf_critical,
    analyze_dynamics,
    autocorrelation,
    dominant_period,
    fisher_g_critical,
    fisher_g_pvalue,
    harmonic_bins,
    winding_number,
)

FPS = 30.0


def logistic(n, r=3.99, x0=0.4):
    """Deterministic chaotic series from the logistic map."""
    out = np.empty(n)
    value = x0
    for i in range(n):
        value = r * value * (1.0 - value)
        out[i] = value
    return out


@pytest.mark.parametrize("period", [8, 16, 32])
def test_sine_is_periodic_with_the_right_winding_number(period):
    """A sine reads as periodic with winding number 1/period cycles per frame."""
    series = np.sin(2.0 * np.pi * np.arange(512) / period)
    result = analyze_dynamics(series, fps=FPS)
    assert result.stability is StabilityClass.PERIODIC
    assert result.period_frames == pytest.approx(period, rel=0.02)
    assert result.winding_number == pytest.approx(1.0 / period, rel=0.02)
    assert result.periodicity_strength > 0.9


def test_harmonic_rich_waveforms_stay_periodic():
    """Square and sawtooth waves spread energy over harmonics but remain periodic."""
    t = np.arange(512)
    for series in (np.sign(np.sin(2.0 * np.pi * t / 16)), (t % 13) / 13.0):
        assert analyze_dynamics(series, fps=FPS).stability is StabilityClass.PERIODIC


def test_incommensurate_pair_is_quasiperiodic():
    """Two tones at an irrational ratio, resolvable at this length, are quasiperiodic."""
    t = np.arange(512)
    series = np.sin(2.0 * np.pi * t * 0.1) + np.sin(2.0 * np.pi * t * 0.1618)
    assert analyze_dynamics(series, fps=FPS).stability is StabilityClass.QUASIPERIODIC


def test_locked_pair_is_periodic():
    """A simple 3:2 frequency ratio is a lock, not a detuning."""
    t = np.arange(512)
    series = np.sin(2.0 * np.pi * t * 0.125) + np.sin(2.0 * np.pi * t * 0.1875)
    assert analyze_dynamics(series, fps=FPS).stability is StabilityClass.PERIODIC


@pytest.mark.parametrize("name", ["white", "brown", "logistic", "ramp"])
def test_unstructured_series_are_chaotic(name):
    """Noise, random walks, deterministic chaos and drifts have no significant period."""
    rng = np.random.default_rng(0)
    series = {
        "white": rng.standard_normal(256),
        "brown": np.cumsum(rng.standard_normal(256)),
        "logistic": logistic(256),
        "ramp": np.arange(256, dtype=float),
    }[name]
    result = analyze_dynamics(series, fps=FPS)
    assert result.stability is StabilityClass.CHAOTIC
    assert result.winding_number is None


def test_constant_series_is_static():
    """A series that never changes is static with no period."""
    result = analyze_dynamics(np.full(64, 0.25), fps=FPS)
    assert result.stability is StabilityClass.STATIC
    assert (result.period_frames, result.winding_number) == (None, None)
    assert result.periodicity_strength == 0.0


def test_short_series_are_not_classified():
    """Series too short for a significant autocorrelation peak fall back to chaotic."""
    assert analyze_dynamics(np.array([0.0, 1.0, 0.0, 1.0, 0.0]), fps=FPS).stability is (
        StabilityClass.CHAOTIC
    )


def test_winding_number_matches_analysis():
    """The standalone winding number agrees with the full analysis."""
    series = np.sin(2.0 * np.pi * np.arange(256) / 10.0)
    assert winding_number(series, fps=FPS) == pytest.approx(0.1, rel=0.02)
    assert winding_number(np.zeros(64), fps=FPS) is None


def test_autocorrelation_is_normalized():
    """Autocorrelation starts at 1.0 and is zero for a constant series."""
    acf = autocorrelation(np.sin(np.arange(64) / 3.0))
    assert acf[0] == pytest.approx(1.0)
    assert np.all(autocorrelation(np.ones(64)) == 0.0)


def test_monotone_series_has_no_period():
    """A series whose autocorrelation only decays has no candidate period."""
    assert dominant_period(np.arange(8.0)) == (None, 0.0)


@pytest.mark.parametrize("series", [np.empty(0), np.array([np.nan, 1.0])])
def test_invalid_series_are_rejected(series):
    """Empty or non-finite series raise."""
    with pytest.raises(ValueError):
        analyze_dynamics(series, fps=FPS)


def test_significance_thresholds_are_monotone():
    """Longer series admit weaker peaks as significant."""
    assert acf_critical(64, 32, 0.05) > acf_critical(1024, 512, 0.05)
    assert fisher_g_critical(16, 0.05) > fisher_g_critical(256, 0.05)
    assert fisher_g_pvalue(fisher_g_critical(64, 0.05), 64) == pytest.approx(0.05)
    assert fisher_g_pvalue(0.0, 64) == 1.0 and fisher_g_pvalue(0.5, 1) == 1.0


def test_harmonic_bins_cover_the_comb():
    """Comb bins sit on the fundamental and its harmonics, one lobe wide."""
    bins = harmonic_bins(32, 0.25, 1.0 / 32.0)
    assert set(np.arange(6, 9)).issubset(bins)
    assert bins.max() < 32

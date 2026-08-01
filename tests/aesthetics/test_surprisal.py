"""Markov surprisal, entropy rate and model reuse."""

import numpy as np
import pytest

from syncsummoner.aesthetics.surprisal import SurprisalModel, entropy_rate, information_content

N_BINS = 16


def rng():
    """Seeded generator for every test in this module."""
    return np.random.default_rng(11)


def test_information_content_shape_and_dtype():
    """Surprisal is one non-negative float32 value per sample."""
    series = np.sin(np.arange(200) / 4.0)
    result = information_content(series, rng=rng())
    assert result.shape == (200,) and result.dtype == np.float32
    assert np.all(result >= 0.0) and np.all(np.isfinite(result))


def test_predictable_series_is_less_surprising_than_noise():
    """A repeating pattern has a lower entropy rate than white noise."""
    repeating = np.tile(np.arange(4, dtype=float), 64)
    noise = rng().standard_normal(256)
    assert entropy_rate(repeating, rng=rng()) < entropy_rate(noise, rng=rng())


def test_higher_order_predicts_longer_patterns_better():
    """A pattern needing three symbols of context is cheaper at order 3 than order 1."""
    series = np.tile([0.0, 1.0, 2.0, 1.0, 0.0, 3.0], 60)
    assert entropy_rate(series, rng=rng(), order=3) < entropy_rate(series, rng=rng(), order=1)


def test_constant_series_collapses_to_one_symbol():
    """A degenerate range quantizes to a single bin and is nearly free to predict."""
    result = information_content(np.zeros(64), rng=rng())
    assert np.all(result[1:] < np.log2(N_BINS) / 4.0)


def test_model_scores_an_unseen_series():
    """Unseen contexts back off to the marginal instead of failing."""
    model = SurprisalModel(order=2, n_bins=8, rng=rng()).fit(np.tile([0.0, 1.0], 64))
    scored = model.information_content(np.tile([1.0, 0.0, 1.0, 1.0], 8))
    assert scored.shape == (32,) and np.all(np.isfinite(scored))


def test_unfitted_model_falls_back_to_uniform():
    """Without a fit every symbol costs log2(n_bins) bits."""
    model = SurprisalModel(order=2, n_bins=4, rng=rng())
    assert model.information_content(np.arange(8, dtype=float)) == pytest.approx(2.0)


def test_series_shorter_than_the_context_fits_the_marginal_only():
    """A series with no complete context still yields a usable model."""
    model = SurprisalModel(order=3, n_bins=4, rng=rng()).fit(np.array([0.0, 1.0]))
    assert np.all(np.isfinite(model.information_content(np.array([0.0, 1.0]))))


def test_sampling_is_seeded_and_reproducible():
    """Sampling uses the caller's generator, so equal seeds give equal draws."""
    series = np.tile([0.0, 1.0, 2.0], 40)
    draws = [SurprisalModel(rng=np.random.default_rng(5)).fit(series).sample(16) for _ in range(2)]
    assert np.array_equal(*draws)
    assert draws[0].shape == (16,)


@pytest.mark.parametrize("kwargs", [{"order": 0}, {"n_bins": 1}])
def test_invalid_configuration_is_rejected(kwargs):
    """Order and bin count must be usable."""
    with pytest.raises(ValueError):
        SurprisalModel(rng=rng(), **kwargs)

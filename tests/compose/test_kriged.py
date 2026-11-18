"""Kriged survival: a gated response, the bound as prior, and the records an archive leaves."""

# pylint: disable=missing-function-docstring

import numpy as np
import pytest

from syncsummoner.compose import kriged as K
from syncsummoner.device.profile import MeasurementRecord, Source
from syncsummoner.probe.fit import save_measurements


def _gated(rng, n=96):
    """A response one parameter gates: P5 below the middle flattens the picture whatever P6 adds."""
    settings = np.zeros((n, 12))
    settings[:, 4] = rng.integers(0, 1024, n)
    settings[:, 5] = rng.integers(0, 1024, n)
    keep = np.where(settings[:, 4] < 512, 0.01, 0.4 + settings[:, 5] / 1023.0)
    return settings, keep * np.exp(rng.normal(0.0, 0.05, n))


def _state(p5, p6):
    state = np.zeros((1, 12))
    state[0, 4], state[0, 5] = p5, p6
    return state


def test_a_gated_response_is_kriged_dead_where_one_parameter_closes_it():
    settings, keep = _gated(np.random.default_rng(0))
    model = K.Response.fit("si_ratio", settings, keep)
    assert model is not None and model.columns.tolist() == [4, 5]
    closed, open_ = model.at(_state(100, 1000)), model.at(_state(900, 1000))
    bound = K.boundary("si_ratio", 0.1)
    assert closed.mean[0] < 0.1 < open_.mean[0]
    one = np.ones(1, bool)
    assert closed.margin(one, bound) < 0 < open_.margin(one, bound)


def test_a_first_order_sum_cannot_carry_the_gate():
    settings, keep = _gated(np.random.default_rng(0))
    high = settings[:, 5] > 900
    mean = keep.mean()
    p5_low, p6_high = keep[settings[:, 4] < 200].mean() - mean, keep[high].mean() - mean
    assert mean + p5_low + p6_high > 0.1
    assert K.Response.fit("si_ratio", settings, keep).at(_state(100, 1000)).mean[0] < 0.1


def test_a_setting_no_record_informs_sits_on_the_bound_not_above_it():
    rng = np.random.default_rng(1)
    settings = np.zeros((80, 12))
    settings[:, 4] = rng.integers(0, 400, 80)
    keep = np.exp(np.sin(settings[:, 4] / 20.0) + rng.normal(0.0, 0.05, 80))
    model = K.Response.fit("si_ratio", settings, keep)
    near, far = model.at(_state(200, 0)), model.at(_state(1023, 0))
    assert near.weight[0] > 0.9 and far.weight[0] < 0.01 and far.sd[0] > 10 * near.sd[0]
    one = np.ones(1, bool)
    for bound in (-1.0, 0.0, 1.0):
        assert abs(far.margin(one, bound)) < 0.01 < abs(near.margin(one, near.centre + near.pull[0] + 0.5))


def test_repeated_settings_krige_once_and_map_back_in_order():
    settings, keep = _gated(np.random.default_rng(2))
    model = K.Response.fit("ti_ratio", settings, keep)
    raw = np.vstack([_state(100, 1000), _state(900, 1000), _state(100, 1000)])
    found = model.at(raw)
    assert found.pull[0] == found.pull[2] != found.pull[1] and found.sd.shape == (3,)


def test_bounds_outside_a_response_range_are_infinite():
    assert K.boundary("si_ratio", 0.0) == -np.inf and K.boundary("si_ratio", np.inf) == np.inf
    assert K.boundary("motion_follow", 1.0) == np.inf and K.boundary("motion_follow", -1.0) == -np.inf
    assert K.boundary("motion_follow", 0.5) == pytest.approx(np.arctanh(0.5))
    assert K.boundary("ti_ratio", 2.0) == pytest.approx(np.log(2.0))


def test_values_outside_a_transform_domain_clip_at_the_nearest_measured():
    assert K.transformed("si_ratio", np.array([0.0, 0.5, 2.0])).tolist() == pytest.approx(
        np.log([0.5, 0.5, 2.0]).tolist()
    )
    assert K.transformed("motion_follow", np.array([1.0, 0.5, -0.8])).tolist() == pytest.approx(
        np.arctanh([0.8, 0.5, -0.8]).tolist()
    )
    assert np.isnan(K.transformed("si_ratio", np.zeros(3))).all()


def test_a_response_that_never_varied_is_not_kriged():
    settings = np.random.default_rng(3).integers(0, 1024, (20, 12)).astype(float)
    assert K.Response.fit("si_ratio", settings, np.full(20, 0.5)) is None
    assert K.Response.fit("si_ratio", np.zeros((20, 12)), np.linspace(0.1, 1.0, 20)) is None
    assert K.Response.fit("si_ratio", settings, np.zeros(20)) is None
    assert set(K.fit(settings, {"si_ratio": np.linspace(0.1, 1.0, 20), "flicker": np.ones(20)})) == {
        "si_ratio"
    }


def test_an_archive_directory_gives_each_program_its_own_models(tmp_path):
    rng = np.random.default_rng(4)
    for program in ("alpha", "beta"):
        settings, keep = _gated(rng, 24)
        records = [
            MeasurementRecord(
                program=program,
                firmware="1",
                analyzer="a",
                source=Source.HW,
                params=tuple(int(v) for v in row),
                state_index=i,
                stimulus="footage",
                metrics={"si_ratio": float(k), **({"motion_follow": 0.5 + 0.4 * (i % 2)} if i else {})},
            )
            for i, (row, k) in enumerate(zip(settings, keep))
        ]
        save_measurements(records, tmp_path / f"{program}.parquet")
    found = K.from_directory(tmp_path)
    assert set(found) == {"alpha", "beta"} and set(found["alpha"]) == {"si_ratio", "motion_follow"}
    assert not K.from_directory(None)

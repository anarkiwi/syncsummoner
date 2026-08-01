"""Sweep plans: type-aware sampling, Sobol corners, raster order, hysteresis paths."""

import numpy as np
import pytest

from syncsummoner.device.profile import PARAM_MAX, ParamKind, ParamSpec
from syncsummoner.probe import plans

SPEC = [
    ParamSpec(index=1, name="amount", native_min=0, native_max=100),
    ParamSpec(index=2, name="rate", native_min=0, native_max=100),
    ParamSpec(index=3, name="mode", native_min=0, native_max=7, kind=ParamKind.QUANTIZED, steps=8),
    ParamSpec(index=4, name="bypass", native_min=0, native_max=1, kind=ParamKind.BOOLEAN, steps=2),
    ParamSpec(index=5, name="mono", native_min=0, native_max=1, kind=ParamKind.BOOLEAN, steps=2),
    ParamSpec(index=6, name="-", native_min=0, native_max=100, kind=ParamKind.UNUSED),
]


def test_defaults_skip_unused_and_park_booleans():
    """Defaults skip unused and park booleans."""
    base = plans.defaults(SPEC)
    assert set(base) == {1, 2, 3, 4, 5}
    assert base[1] == 0.5
    assert base[4] is False


def test_oat_is_type_aware():
    """OAT is type aware."""
    vectors = list(plans.oat(SPEC, steps=32))
    assert len(vectors) == 32 + 32 + 8 + 2 + 2
    assert all(6 not in v for v in vectors)
    visited = {round(v[3], 6) for v in vectors}
    assert {round(x, 6) for x in SPEC[2].sample_values() / PARAM_MAX} <= visited
    assert {v[4] for v in vectors} == {True, False}


def test_oat_values_are_in_range_and_typed():
    """OAT values are in range and typed."""
    for vector in plans.oat(SPEC, steps=8):
        assert isinstance(vector[4], bool) and isinstance(vector[5], bool)
        assert all(0.0 <= vector[i] <= 1.0 for i in (1, 2, 3))


def test_oat_varies_one_parameter_at_a_time():
    """OAT varies one parameter at a time."""
    base = plans.defaults(SPEC)
    for vector in plans.oat(SPEC, steps=8):
        moved = [i for i, value in vector.items() if value != base[i]]
        assert len(moved) <= 1


def test_sobol_shape_and_determinism():
    """Sobol shape and determinism."""
    first = list(plans.sobol(SPEC, n=16, rng=np.random.default_rng(0)))
    again = list(plans.sobol(SPEC, n=16, rng=np.random.default_rng(0)))
    assert len(first) == 16
    assert first == again
    assert first != list(plans.sobol(SPEC, n=16, rng=np.random.default_rng(1)))


def test_sobol_booleans_are_corners_and_balanced():
    """Sobol booleans are corners and balanced."""
    vectors = list(plans.sobol(SPEC, n=16, rng=np.random.default_rng(3)))
    for index in (4, 5):
        values = [v[index] for v in vectors]
        assert all(isinstance(x, bool) for x in values)
        assert set(values) == {True, False}
        assert sum(values) == len(values) // 2
    flips = [sum(a[i] != b[i] for i in (4, 5)) for a, b in zip(vectors[:-1], vectors[1:])]
    assert max(flips) == 1


def test_sobol_quantized_snaps_to_native_steps():
    """Sobol quantized snaps to native steps."""
    allowed = set(np.round(SPEC[2].sample_values() / PARAM_MAX, 6))
    for vector in plans.sobol(SPEC, n=32, rng=np.random.default_rng(5)):
        assert round(vector[3], 6) in allowed


def test_sobol_covers_the_cube():
    """Sobol covers the cube."""
    values = np.array([v[1] for v in plans.sobol(SPEC, n=64, rng=np.random.default_rng(2))])
    assert values.min() < 0.15 and values.max() > 0.85
    assert np.histogram(values, bins=4, range=(0, 1))[0].min() >= 8


def test_sobol_empty_plan():
    """Sobol empty plan."""
    assert not list(plans.sobol(SPEC, n=0, rng=np.random.default_rng(0)))


def test_sobol_without_continuous_parameters():
    """Sobol without continuous parameters."""
    booleans = [p for p in SPEC if p.kind is ParamKind.BOOLEAN]
    vectors = list(plans.sobol(booleans, n=4, rng=np.random.default_rng(0)))
    assert len(vectors) == 4 and all(set(v) == {4, 5} for v in vectors)


def test_tongue_raster_is_dense_and_snakes():
    """Tongue raster is dense and snakes."""
    vectors = list(plans.tongue_raster(SPEC, (1, 2), n=5))
    assert len(vectors) == 25
    first_row = [v[2] for v in vectors[:5]]
    second_row = [v[2] for v in vectors[5:10]]
    assert first_row == sorted(first_row)
    assert second_row == sorted(second_row, reverse=True)
    assert len({v[1] for v in vectors}) == 5


def test_tongue_raster_unknown_index():
    """Tongue raster unknown index."""
    with pytest.raises(KeyError):
        list(plans.tongue_raster(SPEC, (1, 99), n=3))


def test_hysteresis_covers_both_directions_and_ramp_rates():
    """Hysteresis covers both directions and ramp rates."""
    vectors = [v[1] for v in plans.hysteresis(SPEC, 1, n=6, micro=3)]
    setpoints = SPEC[0].sample_values(6) / PARAM_MAX
    assert np.allclose(vectors[:6], setpoints)
    assert np.allclose(vectors[6:12], setpoints[::-1])
    assert len(vectors) == 2 * 6 + 2 * (6 + 3 * 5)
    assert set(np.round(setpoints, 6)) <= set(np.round(vectors, 6))


def test_hysteresis_slow_ramp_is_monotone_within_a_pass():
    """Hysteresis slow ramp is monotone within a pass."""
    vectors = [v[1] for v in plans.hysteresis(SPEC, 1, n=4, micro=2)]
    slow_up = vectors[8 : 8 + 4 + 2 * 3]
    assert all(b >= a for a, b in zip(slow_up[:-1], slow_up[1:]))
    assert len(slow_up) > 4


def test_hysteresis_boolean_parameter_has_no_micro_steps():
    """Hysteresis boolean parameter has no micro steps."""
    vectors = [v[4] for v in plans.hysteresis(SPEC, 4, n=8)]
    assert set(vectors) == {True, False}


def test_spec_from_info():
    """Spec from info."""
    info = [
        {"name": "Level", "min": 0, "max": 100},
        {"name": "Pattern", "min": 0, "max": 1},
        {"name": "Mode", "min": 0, "max": 7},
        {"name": "-", "min": 0, "max": 100},
    ]
    spec = plans.spec_from_info(info)
    kinds = [p.kind for p in spec]
    assert kinds == [ParamKind.CONTINUOUS, ParamKind.BOOLEAN, ParamKind.QUANTIZED, ParamKind.UNUSED]
    assert [p.index for p in spec] == [1, 2, 3, 4]
    assert spec[2].steps == 8
    assert len(list(plans.oat(spec, steps=32))) == 32 + 2 + 8

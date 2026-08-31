"""Fitting: response, cliffs, dead zones, hysteresis, interactions, tongues, axes."""

import numpy as np
import pytest
import yaml

from syncsummoner.device.profile import (
    PARAM_MAX,
    Axis,
    MeasurementRecord,
    ParamKind,
    ParamSpec,
    ProgramProfile,
    Source,
)
from syncsummoner.probe import fit

DEFAULT = 512
GRID = np.round(np.linspace(0, 1023, 32)).astype(int)


def record(values, metrics, settle=0):
    """Record."""
    params = [DEFAULT] * 12
    for index, value in values.items():
        params[index - 1] = int(value)
    return MeasurementRecord(
        program="bitcrush_displace",
        firmware="1.0.0-rc.37",
        analyzer="aesthetics 0.1.0",
        source=Source.HW,
        params=tuple(params),
        state_index=0,
        stimulus="zoneplate",
        metrics=metrics,
        settle_frames=settle,
    )


def oat_records():
    """Synthetic device: one distinct behaviour per parameter, one axis each."""
    rows = []
    for value in GRID:
        unit = value / 1023
        rows.append(record({1: value}, {"peak_scale": 4 * unit, "concentration": unit}))
        rows.append(record({2: value}, {"chroma_mean": unit, "colourfulness": 0.5 * unit}))
        rows.append(record({3: value}, {"spectral_slope": -1.4 + max(value - 600, 0) / 423}))
        wobble = abs(value - 512) / 512
        rows.append(record({4: value}, {"flow_magnitude": wobble, "flow_coherence": 0.6 * wobble}))
        rows.append(record({5: value}, {"clip_frac": float(value >= 700), "luma_mean": 0.5 + 0.4 * unit}))
        rows.append(record({6: value}, {"winding_number": unit / 3, "period_frames": 3 + 10 * unit}))
    return rows


@pytest.fixture(name="profile", scope="module")
def profile_fixture():
    """Profile fixture."""
    return fit.fit_profile(oat_records())


def param(profile, index):
    """Param."""
    return profile.params[index - 1]


def test_profile_carries_provenance(profile):
    """Profile carries provenance."""
    assert profile.program == "bitcrush_displace"
    assert profile.firmware == "1.0.0-rc.37"
    assert profile.analyzer == "aesthetics 0.1.0"
    assert profile.source is Source.HW
    assert len(profile.params) == 12


def test_response_curve_and_sensitivity(profile):
    """Sensitivity is a signal-to-noise ratio, so live parameters clear 1 and inert ones do not."""
    spec = param(profile, 1)
    assert len(spec.response) == len(GRID) == len(spec.values)
    assert spec.response[0] == pytest.approx(0.0) and spec.response[-1] == pytest.approx(1.0)
    assert spec.sensitivity > 1.0
    assert param(profile, 7).sensitivity == 0.0


def test_measured_proxy_metrics_are_kept_in_measured_units(profile):
    """Proxy metrics are stored raw, per setpoint, for the metrics the records carry."""
    spec = param(profile, 1)
    assert set(spec.measured) == {"concentration", "spectral_slope", "clip_frac"}
    assert all(len(v) == len(spec.values) for v in spec.measured.values())
    assert spec.measured["concentration"] == pytest.approx([v / 1023 for v in GRID])
    assert param(profile, 5).measured["clip_frac"] == pytest.approx([float(v >= 700) for v in GRID])
    assert param(profile, 7).measured == {}


def test_sensitivity_ranks_parameters_instead_of_saturating(profile):
    """Span-normalising pinned every sole owner of a metric at exactly 1.0."""
    live = [s.sensitivity for s in profile.params if s.sensitivity > 0]
    assert len(live) >= 3
    assert len(set(round(v, 6) for v in live)) > 1, "sensitivities must not all tie"


def test_monotonicity_uses_rank_correlation(profile):
    """Monotonicity uses rank correlation."""
    assert param(profile, 1).monotonic is True
    assert param(profile, 3).monotonic is True
    assert param(profile, 4).monotonic is False


def test_dead_zone_is_the_longest_flat_run(profile):
    """Dead zone is the longest flat run."""
    low, high = param(profile, 3).dead_zone
    assert low == 0
    assert 560 < high < 640
    assert param(profile, 1).dead_zone is None


def test_cliff_atlas_finds_the_glitch_seed(profile):
    """Cliff atlas finds the glitch seed."""
    cliffs = param(profile, 5).cliffs
    assert len(cliffs) == 1
    assert 660 < cliffs[0].at < 760
    assert cliffs[0].jump > 0.5
    assert "clip_frac" in cliffs[0].metrics
    assert not param(profile, 1).cliffs
    assert profile.cliff_atlas()[0][0] == 5


def test_axis_assignment_is_metric_driven(profile):
    """Axis assignment is metric driven."""
    assigned = {p.index: p.axis for p in profile.params}
    assert assigned[1] is Axis.TEXTURE_SCALE
    assert assigned[2] is Axis.COLOR_DESTRUCTION
    assert assigned[3] is Axis.NOISE
    assert assigned[4] is Axis.DISPLACEMENT
    assert assigned[5] is Axis.KEY_THRESHOLD
    assert assigned[6] is Axis.MOTION_RATE
    assert assigned[12] is Axis.UNASSIGNED
    assert [p.index for p in profile.by_axis(Axis.DISPLACEMENT)] == [4]


def test_kinds_inferred_when_no_spec_supplied(profile):
    """Kinds inferred when no spec supplied."""
    assert param(profile, 1).kind is ParamKind.CONTINUOUS
    assert param(profile, 12).kind is ParamKind.UNUSED


def test_kinds_inferred_for_boolean_and_quantized_sweeps():
    """Kinds inferred for boolean and quantized sweeps."""
    rows = [record({7: v}, {"luma_mean": 0.2 + 0.6 * (v > 0)}) for v in (0, 1023)] * 2
    rows += [record({8: v}, {"chroma_mean": v / 1023}) for v in np.round(np.linspace(0, 1023, 8)).astype(int)]
    profile = fit.fit_profile(rows)
    assert profile.params[6].kind is ParamKind.BOOLEAN
    assert profile.params[7].kind is ParamKind.QUANTIZED
    assert profile.params[7].steps == 8
    assert profile.params[6].sensitivity > 0


def test_supplied_specs_keep_device_metadata():
    """Supplied specs keep device metadata."""
    specs = [ParamSpec(index=1, name="threshold", native_min=0, native_max=100)]
    profile = fit.fit_profile(oat_records(), specs=specs)
    assert profile.params[0].name == "threshold"
    assert profile.params[0].native_max == 100
    assert profile.params[0].sensitivity > 0


def test_no_records_is_an_error():
    """No records is an error."""
    with pytest.raises(ValueError, match="no measurement records"):
        fit.fit_profile([])


def test_constant_metrics_do_not_break_the_fit():
    """Constant metrics do not break the fit."""
    rows = [record({1: v}, {"luma_mean": 0.5}) for v in GRID]
    profile = fit.fit_profile(rows)
    assert profile.params[0].sensitivity == 0.0
    assert profile.params[0].axis is Axis.UNASSIGNED


def interaction_records(coupling, *, n=200, seed=4):
    """Interaction records."""
    rng = np.random.default_rng(seed)
    levels = np.round(np.linspace(0, 1023, 8)).astype(int)
    rows = []
    for _ in range(n):
        a, b, c = rng.choice(levels, 3)
        value = a / 1023 + b / 1023 + coupling * (a * b) / 1023**2 + 0.3 * c / 1023
        rows.append(record({1: a, 2: b, 3: c}, {"luma_mean": value, "chroma_mean": 0.4 * value}))
    return rows


def test_additive_response_has_no_interactions():
    """Additive response has no interactions."""
    assert not fit.fit_profile(interaction_records(0.0)).interactions


def test_non_additivity_is_detected_on_the_coupled_pair():
    """Non additivity is detected on the coupled pair."""
    interactions = fit.fit_profile(interaction_records(3.0)).interactions
    assert interactions
    assert interactions[0][:2] == (1, 2)
    assert interactions[0][2] > 0.05


def test_one_at_a_time_records_yield_no_interactions(profile):
    """One at a time records yield no interactions."""
    assert profile.interactions == []


def staircase(unit):
    """Staircase."""
    if 0.25 <= unit <= 0.5:
        return 0.5
    if 0.65 <= unit <= 0.8:
        return 1.0 / 3.0
    return 0.55 + 0.05 * unit


def tongue_records():
    """Tongue records."""
    rows = []
    for a in np.round(np.linspace(0, 1023, 5)).astype(int):
        for b in np.round(np.linspace(0, 1023, 16)).astype(int):
            winding = staircase(b / 1023 + 0.02 * (a / 1023 - 0.5))
            locked = winding in (0.5, 1.0 / 3.0)
            rows.append(
                record(
                    {5: a, 6: b},
                    {
                        "winding_number": winding,
                        "periodicity_strength": 0.9 if locked else 0.2,
                        "stability": 1.0 if locked else 2.0,
                    },
                )
            )
    return rows


@pytest.fixture(name="lock_profile", scope="module")
def lock_profile_fixture():
    """Lock profile fixture."""
    return fit.fit_profile(tongue_records())


def test_lock_map_finds_simple_ratio_tongues(lock_profile):
    """Lock map finds simple ratio tongues."""
    assert len(lock_profile.lock_maps) == 1
    lock_map = lock_profile.lock_maps[0]
    assert lock_map.b == 6
    strong = [t for t in lock_map.tongues if t.strength > 0.5]
    assert {t.ratio for t in strong} == {(1, 2), (1, 3)}
    assert strong[0].ratio == (1, 2)
    assert strong[0].width > strong[1].width


def test_tongue_centres_sit_inside_their_plateau(lock_profile):
    """Tongue centres sit inside their plateau."""
    half = next(t for t in lock_profile.lock_maps[0].tongues if t.ratio == (1, 2))
    assert 0.25 * 1023 <= half.center <= 0.5 * 1023


def test_stability_regions_are_bounded_by_parameter(lock_profile):
    """Stability regions are bounded by parameter."""
    regions = {r["stability"]: r for r in lock_profile.stability_by_region}
    assert set(regions) == {"periodic", "quasiperiodic"}
    assert regions["periodic"]["n"] > 0
    assert regions["periodic"]["bounds"][6][0] > 0


def test_unlocked_drift_is_not_reported_as_a_wide_tongue(lock_profile):
    """Unlocked drift is not reported as a wide tongue."""
    drift = [t for t in lock_profile.lock_maps[0].tongues if t.strength < 0.5]
    widest_locked = max(t.width for t in lock_profile.lock_maps[0].tongues if t.strength > 0.5)
    assert all(t.width < widest_locked for t in drift)


def hysteresis_records(offset):
    """Hysteresis records."""
    setpoints = np.round(np.linspace(0, 1023, 12)).astype(int)
    rows = []
    for direction, path in ((1, setpoints), (-1, setpoints[::-1])):
        for value in path:
            rows.append(record({1: value}, {"luma_mean": value / 1023 + (offset if direction < 0 else 0.0)}))
    return rows


def test_path_dependence_is_catalogued():
    """Path dependence is catalogued."""
    assert fit.fit_profile(hysteresis_records(0.3)).params[0].hysteresis is True


def test_path_independent_parameter_is_not_flagged():
    """Path independent parameter is not flagged."""
    assert fit.fit_profile(hysteresis_records(0.0)).params[0].hysteresis is False


def test_settle_frames_and_non_settling():
    """Settle frames and non settling."""
    rows = [record({1: v}, {"luma_mean": v / 1023}, settle=7) for v in GRID]
    profile = fit.fit_profile(rows)
    assert profile.settle_frames[1] == 7
    assert profile.non_settling is True
    assert fit.fit_profile(oat_records()).non_settling is False


def test_profile_yaml_roundtrip(tmp_path, profile):
    """Profile YAML round-trips."""
    path = tmp_path / "profile.yaml"
    fit.save_profile(profile, path)
    assert "schema_version" in path.read_text(encoding="utf-8")
    loaded = fit.load_profile(path)
    assert loaded.program == profile.program
    assert [p.axis for p in loaded.params] == [p.axis for p in profile.params]
    assert loaded.params[4].cliffs == profile.params[4].cliffs
    assert loaded.params[2].dead_zone == profile.params[2].dead_zone
    assert loaded.source is profile.source
    assert loaded.params[0].measured == profile.params[0].measured
    assert loaded.registered == profile.registered == 0.0


def test_yaml_round_trip_preserves_measured_and_registered(tmp_path):
    """Measured proxy values and the registration score survive a save/load cycle."""
    spec = ParamSpec(
        index=1, name="x", native_min=0, native_max=1, values=[0, 512], measured={"clip_frac": [0.0, 0.75]}
    )
    written = ProgramProfile(
        program="p", firmware="1.0", analyzer="a", source=Source.HW, params=[spec], registered=0.42
    )
    path = tmp_path / "profile.yaml"
    fit.save_profile(written, path)
    loaded = fit.load_profile(path)
    assert loaded.params[0].measured == {"clip_frac": [0.0, 0.75]}
    assert loaded.registered == 0.42


def test_a_profile_written_without_the_new_fields_still_loads(tmp_path):
    """Older profiles carry neither field, so both fall back to empty."""
    path = tmp_path / "old.yaml"
    payload = {
        "schema_version": 1,
        "profile": {
            "program": "p",
            "firmware": "1.0",
            "analyzer": "a",
            "source": "hw",
            "params": [{"index": 1, "name": "x", "native_min": 0.0, "native_max": 1.0}],
        },
    }
    path.write_text(yaml.safe_dump(payload), encoding="utf-8")
    loaded = fit.load_profile(path)
    assert loaded.params[0].measured == {}
    assert loaded.registered == 0.0


def test_profile_rejects_a_newer_schema(tmp_path, profile):
    """Profile rejects a newer schema."""
    path = tmp_path / "profile.yaml"
    fit.save_profile(profile, path)
    path.write_text(path.read_text(encoding="utf-8").replace("schema_version: 1", "schema_version: 99"))
    with pytest.raises(ValueError, match="newer than"):
        fit.load_profile(path)


def test_measurements_parquet_roundtrip(tmp_path):
    """Measurements Parquet round-trips."""
    records = oat_records()
    path = tmp_path / "measurements.parquet"
    fit.save_measurements(records, path)
    rows = fit.load_measurements(path)
    assert len(rows) == len(records)
    assert {"program", "firmware", "analyzer", "source", "state_index", "stimulus"} <= set(rows[0])
    assert {f"p{i}" for i in range(1, 13)} <= set(rows[0])
    assert rows[0]["source"] == "hw"
    assert any(row.get("winding_number") is not None for row in rows)
    assert any(row.get("winding_number") is None for row in rows)


def a_record(*, params, metrics):
    """A measurement record carrying just the metrics a signature reads."""
    return MeasurementRecord(
        program="P",
        firmware="1.0",
        analyzer="aesthetics 0",
        source=Source.HW,
        params=params,
        state_index=0,
        stimulus="codeframes",
        metrics=metrics,
        settle_frames=1,
    )


def test_a_sweep_that_never_isolates_a_parameter_still_separates_them():
    """A Sobol design varies everything at once, so the one-at-a-time subset is empty."""
    rng = np.random.default_rng(5)
    records = []
    for _ in range(48):
        raw = rng.integers(0, PARAM_MAX, size=12)
        chroma = raw[0] / PARAM_MAX
        motion = raw[1] / PARAM_MAX
        records.append(
            a_record(
                params=tuple(int(v) for v in raw),
                metrics={
                    "chroma_mean": chroma,
                    "colourfulness": 0.9 * chroma,
                    "chroma_std": 0.7 * chroma,
                    "winding_number": motion,
                    "period_frames": 0.8 * motion,
                    "periodicity_strength": 0.5 * motion,
                    "peak_scale": 0.02 * rng.random(),
                },
            )
        )
    profile = fit.fit_profile(records)
    axes = {s.index: s.axis for s in profile.params}
    assert axes[1] is Axis.COLOR_DESTRUCTION, "the chroma driver is not a texture parameter"
    assert axes[2] is Axis.MOTION_RATE, "the motion driver is not a texture parameter"

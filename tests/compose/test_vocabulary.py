"""Gesture vocabulary: the design 4.3 table, and what each primitive emits."""

# pylint: disable=missing-function-docstring

import numpy as np
import pytest

from syncsummoner.compose.vocabulary import (
    GESTURES,
    Anchor,
    Automation,
    GestureContext,
    monotonic_segment,
    safe_range,
    step_margin,
    targets,
)
from syncsummoner.device.profile import (
    PARAM_MAX,
    Axis,
    Cliff,
    ParamKind,
    ParamSpec,
    ProgramProfile,
    Source,
)

from . import make_profile

TABLE = {
    "hold": (Anchor.BAR, "safe_range"),
    "ramp": (Anchor.PHRASE, "monotonic_segments"),
    "cliff_cross": (Anchor.DOWNBEAT, "cliff_atlas"),
    "detune_drift": (Anchor.MULTI_BAR, "lock_map"),
    "lock_snap": (Anchor.SECTION, "tongue_centers"),
    "punch": (Anchor.ONSET, "effect_buttons"),
    "hysteresis_loop": (Anchor.TWO_BAR, "hysteresis"),
}


def ctx(**kw):
    base = {"arrival": 4.0, "duration": 2.0, "rate": 15.0, "intensity": 0.7, "axis": Axis.MOTION_RATE}
    base.update(kw)
    return GestureContext(rng=np.random.default_rng(5), **base)


def test_vocabulary_matches_the_design_table():
    assert set(GESTURES) == set(TABLE)
    for name, (anchor, field) in TABLE.items():
        assert (GESTURES[name].anchor, GESTURES[name].profile_field) == (anchor, field)


@pytest.mark.parametrize("name", sorted(TABLE))
def test_every_gesture_is_legal_and_lands_on_its_anchor(name):
    auto = GESTURES[name](make_profile(), ctx())
    assert len(auto) > 0
    assert auto.values.min() >= 0 and auto.values.max() <= PARAM_MAX
    assert np.abs(auto.times - 4.0).min() < 1e-9
    assert auto.times.max() <= 4.0 + 2.0 + 1e-9


@pytest.mark.parametrize("name", ["cliff_cross", "detune_drift", "lock_snap", "punch"])
def test_gestures_without_their_profile_field_emit_nothing(name):
    bare = ProgramProfile(program="bare", firmware="f", analyzer="a", source=Source.HW, params=[])
    assert len(GESTURES[name](bare, ctx())) == 0


def test_cliff_cross_straddles_the_strongest_cliff():
    profile = make_profile()
    auto = GESTURES["cliff_cross"](profile, ctx(intensity=0.0))
    index, cliff = profile.cliff_atlas()[0]
    assert set(auto.indices.tolist()) == {index}
    assert auto.values[:-1].max() < cliff.at <= auto.values[-1]


def test_detune_drift_starts_at_the_lock_edge_and_widens():
    tongue = make_profile().lock_maps[0].tongues[0]
    auto = GESTURES["detune_drift"](make_profile(), ctx())
    assert auto.values[0] == pytest.approx(tongue.center + tongue.width / 2, abs=1)
    assert auto.values[-1] > auto.values[0]


def test_lock_snap_hits_the_tongue_centre():
    auto = GESTURES["lock_snap"](make_profile(), ctx())
    assert auto.values.tolist() == [make_profile().lock_maps[0].tongues[0].center]


def test_punch_is_momentary_on_a_boolean():
    profile = make_profile()
    auto = GESTURES["punch"](profile, ctx(duration=0.1))
    spec = next(p for p in profile.params if p.kind is ParamKind.BOOLEAN)
    assert set(auto.indices.tolist()) == {spec.index}
    assert auto.values.tolist() == [PARAM_MAX, 0]


def test_hysteresis_loop_returns_to_its_start():
    auto = GESTURES["hysteresis_loop"](make_profile(), ctx(duration=4.0))
    assert auto.values[0] == pytest.approx(auto.values[-1], abs=40)
    assert auto.values.max() > auto.values[0]


def test_explicit_targets_override_axis_selection():
    auto = GESTURES["hold"](make_profile(), ctx(targets=(3,)))
    assert auto.indices.tolist() == [3]
    auto = GESTURES["hold"](make_profile(), ctx(targets=(99,)))
    assert auto.indices.tolist() == [2]


def test_safe_range_avoids_the_dead_zone():
    profile = make_profile()
    low = next(p for p in profile.params if p.index == 1)
    assert safe_range(low) == (96, PARAM_MAX)
    high = ParamSpec(index=1, name="x", native_min=0, native_max=1, dead_zone=(900, 1023))
    assert safe_range(high) == (0, 900)
    assert safe_range(ParamSpec(index=7, name="b", native_min=0, native_max=1, kind=ParamKind.BOOLEAN)) == (
        0,
        PARAM_MAX,
    )


def test_monotonic_segment_finds_the_longest_run():
    spec = ParamSpec(
        index=1,
        name="x",
        native_min=0,
        native_max=1,
        values=[0, 100, 200, 300, 400, 500],
        response=[1.0, 0.5, 0.1, 0.2, 0.4, 0.9],
    )
    assert monotonic_segment(spec) == (200, 500)
    assert monotonic_segment(ParamSpec(index=1, name="x", native_min=0, native_max=1)) == (0, PARAM_MAX)


def test_step_margin_tracks_the_probe_resolution():
    spec = next(p for p in make_profile().params if p.index == 2)
    assert step_margin(spec) == round(PARAM_MAX / 62)
    assert step_margin(ParamSpec(index=1, name="x", native_min=0, native_max=1)) >= 1


def test_targets_falls_back_when_the_axis_is_absent():
    profile = make_profile()
    assert targets(profile, Axis.FEEDBACK_GAIN)[0].index == 12, "the most sensitive drivable slot"
    assert targets(profile, Axis.MOTION_RATE)[0].index == 2
    assert targets(profile, Axis.NOISE, kinds=(ParamKind.BOOLEAN,))[0].index == 7


def test_automation_algebra():
    a = Automation.of([0.0, 1.0], 3, np.array([10, 20]))
    b = Automation.of([0.5], 4, 30)
    merged = Automation.concat([a, b, Automation.empty()])
    assert merged.times.tolist() == [0.0, 0.5, 1.0]
    assert merged.indices.tolist() == [3, 4, 3]
    assert Automation.concat([]).times.size == 0
    assert merged.shift(2.0).times.tolist() == [2.0, 2.5, 3.0]
    assert len(merged.within(0.4, 1.0)) == 1
    assert Automation.empty().max_rate() == 0.0
    dense = Automation.of(np.linspace(0, 0.5, 5), 1, np.arange(5))
    assert dense.max_rate() == 5.0
    assert dense.max_rate(window_s=0.25) == 8.0


def test_automation_hold_is_zero_order():
    auto = Automation.of([0.0, 1.0], 3, np.array([10, 20]))
    held = auto.hold(np.array([-0.5, 0.0, 0.5, 1.5]))
    assert held[:, 2].tolist() == [PARAM_MAX // 2, 10, 10, 20]
    assert held[:, 0].tolist() == [PARAM_MAX // 2] * 4


def test_gesture_values_clip_to_the_parameter_range():
    auto = Automation.of([0.0], 1, np.array([5000.0]))
    assert auto.values.tolist() == [PARAM_MAX]


def cliffed(**kinds):
    """The shared profile with some slots declared as other kinds, and a cliff on its boolean."""
    profile = make_profile()
    for spec in profile.params:
        spec.kind = kinds.get(f"p{spec.index}", spec.kind)
        if spec.index == 7:
            spec.cliffs = [Cliff(at=1023, jump=0.99, metrics=("clip_frac",))]
    return profile


@pytest.mark.parametrize("name", sorted(TABLE))
def test_no_gesture_ever_writes_an_undeclared_slot(name):
    profile = cliffed(p2=ParamKind.UNUSED, p7=ParamKind.UNUSED)
    for intensity in np.linspace(0.0, 1.0, 11):
        for chosen in ((), (2,), (7,)):
            auto = GESTURES[name](profile, ctx(intensity=intensity, targets=chosen))
            assert not set(auto.indices.tolist()) & {2, 7}


def test_a_declared_boolean_is_only_ever_pulsed():
    profile = cliffed()
    pulse = GESTURES["punch"](profile, ctx())
    assert pulse.indices.tolist() == [7, 7] and pulse.values.tolist() == [PARAM_MAX, 0]
    for name in sorted(set(TABLE) - {"punch"}):
        assert 7 not in GESTURES[name](profile, ctx(targets=(7,))).indices.tolist(), name


def test_cliff_cross_never_latches_a_toggle_across_its_cliff():
    profile = cliffed()
    for intensity in np.linspace(0.0, 1.0, 11):
        assert 7 not in GESTURES["cliff_cross"](profile, ctx(intensity=intensity)).indices.tolist()

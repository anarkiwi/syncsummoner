"""Score audit: what each layer drives, read off the plan the renderer plays."""

# pylint: disable=missing-function-docstring,protected-access

import numpy as np

from syncsummoner.compose import audit as A
from syncsummoner.compose.score import GestureInstance, Layer, Score, Section
from syncsummoner.device.profile import ParamKind, ParamSpec

from . import make_profile


def score():
    gestures = [
        GestureInstance("punch", 2.0, 0.25, "noise", 0.5, (7,), 1),
        GestureInstance("hold", 6.0, 2.0, "motion_rate", 0.5, (12,), 2),
        GestureInstance("detune_drift", 7.0, 1.0, "motion_rate", 0.5, (), 3),
    ]
    return Score(
        duration=8.0,
        fps=30.0,
        sections=[Section(0.0, 8.0, "A")],
        layers=[Layer("glitch", 0, gestures), Layer("lost", 1, gestures[:1])],
    )


def test_every_parameter_is_reported_by_name_kind_and_what_drove_it():
    profile = make_profile()
    profile.lock_maps = []
    layers = A.audit(score(), {"glitch": profile})
    glitch, lost = layers
    used = {p.index: p for p in glitch.params}
    assert used[7].kind is ParamKind.BOOLEAN and used[7].gestures == {"punch": 1}
    assert (used[7].writes, used[7].low, used[7].high, used[7].final) == (2, 0, 1023, 0)
    assert used[12].kind is ParamKind.CONTINUOUS and used[12].gestures == {"hold": 1}
    assert glitch.silent == {"detune_drift": 1}, "no lock map, so that gesture renders nothing"
    assert not any(p.flags for p in glitch.params) and not glitch.flags
    assert lost.flags and not lost.params
    text = A.format_audit(layers)
    assert "layer 1: lost  ! no profile" in text and "gestures that write nothing: detune_drift x1" in text
    assert "P7  p7               boolean" in text and "punch x1" in text and "undriven" in text


def test_only_a_toggle_left_on_is_flagged():
    toggle = ParamSpec(11, "Bypass", 0, 1, kind=ParamKind.BOOLEAN, steps=2)
    assert not A._flags(toggle, np.array([1023, 0]))
    assert A._flags(toggle, np.array([0, 1023])) == ("latched on",)
    level = ParamSpec(1, "Level", 0, 100)
    assert not A._flags(level, np.array([0, 1023]))


def test_a_layer_that_drives_nothing_is_flagged():
    quiet = Score(duration=4.0, layers=[Layer("glitch", 0, [GestureInstance("lock_snap", 9.0, 1.0)])])
    profile = make_profile()
    profile.lock_maps = []
    (layer,) = A.audit(quiet, {"glitch": profile})
    assert layer.flags == ("drives nothing",)
    assert "! drives nothing" in A.format_audit([layer])

"""Style labelling from descriptions, and validating measurement against it."""

# pylint: disable=missing-function-docstring

import types

from syncsummoner.device.profile import ProgramStyle
from syncsummoner.probe import style


def test_descriptions_label_addressing_as_digital():
    text = "Kaleidoscope N-fold reflective symmetry mirror with line-buffered coordinate remapping"
    assert style.label_from_description(text) is ProgramStyle.DIGITAL


def test_descriptions_label_value_remapping_as_analog():
    text = "Bleach bypass silver retention with desaturated highlights and raised contrast"
    assert style.label_from_description(text) is ProgramStyle.ANALOG


def test_silent_description_is_unknown_not_a_guess():
    assert style.label_from_description("") is ProgramStyle.UNKNOWN
    assert style.label_from_description("Ouroboros video processing program") is ProgramStyle.UNKNOWN


def test_balanced_description_is_mixed():
    assert style.label_from_description("colour with offset") is ProgramStyle.MIXED


def test_labels_read_a_manifest_mapping():
    entry = types.SimpleNamespace(program_name="Kaledos", description="coordinate remapping mirror")
    manifest = types.SimpleNamespace(entries={"kaledos": entry})
    assert style.labels_from_manifest(manifest) == {"Kaledos": ProgramStyle.DIGITAL}
    assert style.labels_from_manifest(None) == {}


def test_thresholds_come_from_the_labelled_populations():
    scores = {"a1": 0.95, "a2": 0.92, "d1": 0.30, "d2": 0.25}
    labels = {
        "a1": ProgramStyle.ANALOG,
        "a2": ProgramStyle.ANALOG,
        "d1": ProgramStyle.DIGITAL,
        "d2": ProgramStyle.DIGITAL,
    }
    high, low = style.thresholds_from_labels(scores, labels)
    assert low < high, "the undecided band must sit between the populations"
    measured, _ = style.measured_styles(scores, labels)
    assert measured["a1"] is ProgramStyle.ANALOG and measured["d2"] is ProgramStyle.DIGITAL


def test_overlapping_populations_collapse_the_band():
    scores = {"a": 0.5, "d": 0.6}
    labels = {"a": ProgramStyle.ANALOG, "d": ProgramStyle.DIGITAL}
    high, low = style.thresholds_from_labels(scores, labels)
    assert high == low, "overlap must not invent a decision band"


def test_agreement_reports_support_not_just_accuracy():
    measured = {"a": ProgramStyle.ANALOG, "d": ProgramStyle.ANALOG, "u": ProgramStyle.DIGITAL}
    labels = {"a": ProgramStyle.ANALOG, "d": ProgramStyle.DIGITAL, "u": ProgramStyle.UNKNOWN}
    check = style.agreement(measured, labels)
    assert check["support"] == 2 and check["agree"] == 1 and check["disagree"] == ["d"]


def test_unlabelled_lists_what_only_measurement_covers():
    labels = {"a": ProgramStyle.ANALOG, "u": ProgramStyle.UNKNOWN}
    assert style.unlabelled({"a": 0.9, "u": 0.5, "x": 0.4}, labels) == ["u", "x"]

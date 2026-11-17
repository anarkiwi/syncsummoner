"""Kinds, register fields and control curves, checked against the SDK's stated behaviour."""

# pylint: disable=missing-function-docstring

import numpy as np
import pytest

from syncsummoner.program import declared as D
from syncsummoner.program.vmprog import ControlMode, read_package
from tests.declared import decl, toggle

ALL = np.arange(1024)


def test_the_abi_register_map():
    assert str(D.field(1)) == "r0[9:0]" and str(D.field(6)) == "r5[9:0]"
    assert [str(D.field(p)) for p in range(7, 12)] == [f"r6[{b}]" for b in range(5)]
    assert str(D.field(12)) == "r7[9:0]"
    assert D.field(1).flat[9] == 319 and D.field(7).flat == (250,)
    with pytest.raises(ValueError):
        D.field(13)


def test_kinds_come_from_control_mode_and_labels_only(quilt_bytes):
    cfg = read_package(quilt_bytes).config
    assert D.kind(cfg.param(1)) == (D.Kind.QUANTIZED, 4)
    assert D.kind(cfg.param(7)) == (D.Kind.BOOLEAN, 2)
    assert D.kind(cfg.param(12)) == (D.Kind.CONTINUOUS, None)
    assert D.kind(None) == (D.Kind.UNUSED, None)
    assert D.kind(decl(1, mode=ControlMode.STEPS_16)) == (D.Kind.QUANTIZED, 16)


@pytest.mark.parametrize("mode,levels", [(ControlMode.STEPS_4 + i, 4 << i) for i in range(7)])
def test_steps_modes_write_exactly_their_level_count(mode, levels):
    assert len(np.unique(D.curve(ALL, mode))) == levels


@pytest.mark.parametrize("mode", [m for m in ControlMode if m >= ControlMode.QUAD_IN])
def test_easing_curves_are_monotonic_and_span_the_range(mode):
    out = D.curve(ALL, mode)
    assert out[0] == 0 and out[-1] == 1023 and np.all(np.diff(out) >= 0) and out.max() <= 1023


def test_polar_wraps_and_linear_scales():
    assert D.curve(1024 + 5, ControlMode.POLAR_DEGS_360) == 5
    assert D.curve(1023, ControlMode.POLAR_DEGS_90) == 255 and D.curve(600, ControlMode.POLAR_DEGS_720) == 176
    assert D.curve(700, ControlMode.LINEAR_DOUBLE) == 1023 and D.curve(700, ControlMode.LINEAR_QUARTER) == 175
    assert D.curve(-5, ControlMode.LINEAR) == 0 and D.curve(511, ControlMode.BOOLEAN) == 0


def test_switches_write_one_bit_at_the_boolean_threshold():
    sw = toggle(8)
    assert D.register_value(sw, 511) == 0 and D.register_value(sw, 512) == 1
    words = D.registers((sw, decl(2), decl(12)), np.array([0, 300, 0, 0, 0, 0, 0, 900, 0, 0, 0, 77]), 7)
    assert words[6] == 0b10 and words[1] == 300 and words[7] == 77 and words[8] == 7


def test_classes_group_values_by_the_bits_the_program_sees():
    knob = decl(5)
    groups = D.classes(knob, (6, 7, 8, 9))
    assert len(groups) == 16 and all(len(g) == 64 for g in groups)
    assert len(D.classes(knob, ())) == 1
    assert len(D.classes(toggle(7), (0,))) == 2 and len(D.classes(toggle(7), ())) == 1

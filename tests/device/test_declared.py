"""Parameter specs from a program's own declaration, and the error when a program has none."""

# pylint: disable=missing-function-docstring

import pytest

from syncsummoner.device import declared as D
from syncsummoner.device.profile import PARAM_COUNT, ParamKind
from syncsummoner.program.vmprog import ControlMode
from tests.declared import Declarations, config, decl, toggle


@pytest.mark.parametrize(
    "param, kind, steps",
    [
        (decl(1, "Size"), ParamKind.CONTINUOUS, None),
        (decl(2, "Grid", high=3, labels=("1", "2", "4", "8")), ParamKind.QUANTIZED, 4),
        (decl(3, "Levels", mode=ControlMode.STEPS_16), ParamKind.QUANTIZED, 16),
        (toggle(7, "Mono"), ParamKind.BOOLEAN, 2),
        (decl(4, "Pair", high=1, labels=("A", "B")), ParamKind.BOOLEAN, 2),
        (None, ParamKind.UNUSED, None),
    ],
)
def test_kind_and_steps_are_the_declared_ones(param, kind, steps):
    spec = D.spec_from_decl(param, 5)
    assert (spec.index, spec.kind, spec.steps) == (5, kind, steps)


def test_a_spec_keeps_the_declared_display_range_and_takes_the_device_label_when_given():
    param = decl(5, "Gain", display=(-10, 25))
    spec = D.spec_from_decl(param, 5)
    assert (spec.name, spec.native_min, spec.native_max) == ("Gain", -10.0, 25.0)
    assert D.spec_from_decl(param, 5, "Gain 5").name == "Gain 5"
    assert D.spec_from_decl(None, 4, "Null 4").name == "Null 4"


@pytest.mark.parametrize(
    "param, value",
    [
        (decl(1, initial=700), 700),
        (decl(2, high=3, initial=2, labels=("a", "b", "c", "d")), 682),
        (toggle(7, initial=1), 1023),
        (toggle(8), 0),
        (decl(3, low=10, high=20, initial=15), 512),
    ],
)
def test_initial_values_are_in_device_units(param, value):
    assert D.initial_value(param) == value


def test_specs_cover_every_slot_and_undeclared_ones_are_unused():
    out = D.specs(config("P", decl(3, "Mix")), {1: "-", 3: "Mix"})
    assert [s.index for s in out] == list(range(1, PARAM_COUNT + 1))
    assert [s.index for s in out if s.kind is not ParamKind.UNUSED] == [3]
    assert out[0].name == "-" and out[1].name == ""
    assert all(s.kind is ParamKind.UNUSED for s in D.specs(None))


def test_a_declaration_resolves_by_device_name_and_names_where_to_put_a_missing_one():
    lib = Declarations(config("Pinwheel"))
    assert D.declaration("czartroniks/Pinwheel", lib).name == "Pinwheel"
    with pytest.raises(D.UndeclaredProgramError, match=r"'Quilt'.*SYNCSUMMONER_CARD.*SYNCSUMMONER_FIRMWARE"):
        D.declaration("Quilt", lib)


def test_the_default_library_is_scanned_once(monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)
    for name in ("SYNCSUMMONER_CARD", "SYNCSUMMONER_FIRMWARE", "SYNCSUMMONER_SDK", "SYNCSUMMONER_COMMUNITY"):
        monkeypatch.delenv(name, raising=False)
    D.default_library.cache_clear()
    try:
        assert D.default_library() is D.default_library()
        with pytest.raises(D.UndeclaredProgramError):
            D.declaration("Anything")
    finally:
        D.default_library.cache_clear()

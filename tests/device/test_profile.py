"""Parameter roles: what a program's own labels say each slot is for, and old profiles loading."""

# pylint: disable=missing-function-docstring

import pytest

from syncsummoner.device.profile import ParamKind, ParamSpec, ProgramProfile, Role, Source, classify_role


@pytest.mark.parametrize(
    "index, name, role",
    [
        (11, "Bypass", Role.BYPASS),
        (10, "Bypass", Role.BYPASS),
        (9, "Reset", Role.MOMENTARY),
        (10, "Retrig", Role.MOMENTARY),
        (7, "Serve", Role.MOMENTARY),
        (9, "Freeze", Role.EFFECT),
        (4, "Clr Step", Role.EFFECT),
        (12, "Mix", Role.WET),
        (12, "Global Blend", Role.WET),
        (12, "Mix Step", Role.WET),
        (12, "Null 12", Role.WET),
        (12, "-", Role.WET),
        (12, "p12", Role.WET),
        (12, "Chroma Crush", Role.EFFECT),
        (6, "Mix", Role.EFFECT),
        (1, "p1", Role.EFFECT),
    ],
)
def test_roles_come_from_whole_words_of_the_label_and_the_slot(index, name, role):
    assert classify_role(index, name) is role


def test_a_profile_without_roles_derives_them_and_one_with_them_keeps_them():
    profile = ProgramProfile(
        program="Old",
        firmware="1.0.1",
        analyzer="aesthetics 0.1.0",
        source=Source.HW,
        params=[
            ParamSpec(11, "Bypass", 0.0, 1.0, kind=ParamKind.BOOLEAN, role=Role.EFFECT),
            ParamSpec(12, "p12", 0.0, 1.0),
        ],
    )
    data = profile.to_dict()
    assert ProgramProfile.from_dict(data).params[0].role is Role.EFFECT, "a stored role is not re-derived"
    for param in data["params"]:
        del param["role"]
    old = ProgramProfile.from_dict(data)
    assert [p.role for p in old.params] == [Role.BYPASS, Role.WET]
    assert old.by_role(Role.WET) == [old.params[1]]

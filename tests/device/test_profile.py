"""Profiles round-trip, and older profiles carrying fields since dropped still load."""

# pylint: disable=missing-function-docstring

from syncsummoner.device.profile import ParamKind, ParamSpec, ProgramProfile, Source


def test_a_profile_round_trips_and_a_stored_role_key_is_ignored():
    profile = ProgramProfile(
        program="Old",
        firmware="1.0.1",
        analyzer="aesthetics 0.1.0",
        source=Source.HW,
        params=[
            ParamSpec(11, "Bypass", 0.0, 1.0, kind=ParamKind.BOOLEAN, steps=2),
            ParamSpec(12, "Blend", 0.0, 100.0),
        ],
    )
    data = profile.to_dict()
    assert ProgramProfile.from_dict(data) == profile
    for param, role in zip(data["params"], ("bypass", "wet")):
        param["role"] = role
    assert ProgramProfile.from_dict(data) == profile

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


def test_toggle_slots_snap_at_the_firmware_threshold_and_knobs_pass():
    from syncsummoner.device.profile import BOOL_THRESHOLD, PARAM_MAX, snap_toggles, toggle_slots

    assert toggle_slots() == (7, 8, 9, 10, 11)
    params = [1, 7, 7, 11, 12]
    values = [BOOL_THRESHOLD - 1, BOOL_THRESHOLD - 1, BOOL_THRESHOLD, 1, 600]
    assert snap_toggles(params, values).tolist() == [BOOL_THRESHOLD - 1, 0, PARAM_MAX, 0, 600]
    assert snap_toggles(params, values, toggles=[12]).tolist() == [511, 511, 512, 1, PARAM_MAX]

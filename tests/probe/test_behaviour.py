"""Native-resolution loop behaviour: known transforms in, the three profile fields out."""

# pylint: disable=missing-function-docstring
# pylint: disable=no-member  ; cv2 is a compiled extension pylint cannot introspect
# pylint: disable=protected-access  ; the dwell picker is the unit worth testing directly

import cv2
import numpy as np
import pytest

from syncsummoner.device.profile import ProgramStyle
from syncsummoner.probe import behaviour as B
from syncsummoner.probe import codeframes as cf
from syncsummoner.probe import style as S
from syncsummoner.probe.archive import FrameRow, GAP

WIDTH, HEIGHT, COUNT = 160, 128, 16
DWELL = 32


@pytest.fixture(name="loop", scope="module")
def loop_fixture():
    """Loop."""
    return cf.build_loop(width=WIDTH, height=HEIGHT, rng=np.random.default_rng(11), count=COUNT)


def bytes_of(frame):
    """One frame as the decoder hands it back: native size, RGB, 8 bit."""
    return np.clip(np.rint(np.asarray(frame) * 255.0), 0, 255).astype(np.uint8)


class Capture:
    """A program's native frames and the rows the archive holds for them."""

    def __init__(self, frames, setpoints):
        self.frames = [bytes_of(f) for f in frames]
        self.rows = [
            FrameRow(frame=i, program="p", params=tuple([max(s, 0)] * 12), setpoint=s)
            for i, s in enumerate(setpoints)
        ]

    def behaviour(self, loop):
        """Behaviour read off just the positions the dwells ask for."""
        spread = B.dwells(self.rows)
        return B.behaviour(spread, {i: self.frames[i] for i in B.positions(spread)}, loop=loop)


def played(loop, dwells=2):
    """The loop as it would be shown, one dwell of captures per setpoint."""
    repeats = -(-DWELL // loop.count)
    return [list(loop.play(repeats))[:DWELL] for _ in range(dwells)]


def capture(loop, transform, dwells=2):
    """A reader over the loop played through a transform, with a gap between dwells."""
    frames, setpoints = [], []
    for setpoint, dwell in enumerate(played(loop, dwells)):
        frames += [np.zeros_like(dwell[0])] + [np.clip(transform(f), 0.0, 1.0) for f in dwell]
        setpoints += [GAP] + [setpoint] * len(dwell)
    return Capture(frames, setpoints)


def measured(loop, transform):
    """Behaviour of one program whose archive holds the transformed loop."""
    return capture(loop, transform).behaviour(loop)


def tile(frame):
    """Two by two tiling: a rearrangement that puts one source value in two places."""
    height, width = frame.shape[:2]
    return np.tile(cv2.resize(frame, (width // 2, height // 2)), (2, 2, 1))


def test_identity_registers_and_is_explained_pointwise(loop):
    found = measured(loop, lambda f: f)
    assert found.registered > 0.9 and found.pointwise > 0.9


def test_a_value_remap_stays_pointwise_and_still_registers(loop):
    found = measured(loop, lambda f: 1.0 - f)
    assert found.pointwise > 0.9
    assert found.registered > 0.9, "inversion still registers; the sign is not the strength"


def test_a_displacement_breaks_the_pointwise_fit(loop):
    found = measured(loop, tile)
    assert found.pointwise < 0.5 and found.registered < 0.5


def test_the_fits_classify_the_transforms_they_were_measured_from(loop):
    """The style is a decision over the library's fits, which is what ``style.py`` takes."""
    scores = {name: measured(loop, t).pointwise for name, t in (("remap", lambda f: 1.0 - f), ("move", tile))}
    styles, band = S.measured_styles(scores, {})
    assert styles == {"remap": ProgramStyle.ANALOG, "move": ProgramStyle.DIGITAL}
    assert band == (0.9, 0.7), "with nothing labelled the band is style.py's own fallback"


def test_a_passthrough_classifies_analog_which_the_enum_cannot_say_otherwise(loop):
    """``ProgramStyle`` has no passthrough member, so an identity is a value map of one."""
    styles, _ = S.measured_styles({"p": measured(loop, lambda f: f).pointwise}, {})
    assert styles["p"] is ProgramStyle.ANALOG


def test_each_dwell_reads_only_the_captures_its_fits_score(loop):
    rows = capture(loop, lambda f: f).rows
    spread = B.dwells(rows)
    assert [int(d.pointwise[0]) for d in spread] == [1, DWELL + 2], "a dwell starts after its gap"
    for dwell in spread:
        assert dwell.pointwise.size == B.POINTWISE_SAMPLES and dwell.registered.size == cf.REGISTER_SAMPLES
        assert set(dwell.registered) <= set(range(dwell.pointwise[0], dwell.pointwise[0] + B.FRAME_SAMPLES))
    assert len(B.positions(spread)) < 2 * B.FRAME_SAMPLES


def test_dwells_are_spread_across_the_sweep_and_bounded():
    rows = [FrameRow(frame=i, program="p", params=(0,) * 12, setpoint=i // 30) for i in range(300)]
    spans = B._dwells(rows)
    assert len(spans) == B.DWELL_SAMPLES and all(count == B.FRAME_SAMPLES for _, count in spans)
    assert [start for start, _ in spans] == [0, 60, 120, 180, 270]


def test_dwells_pass_over_runs_the_capture_cut_short():
    """A run of dropped frames is not a dwell the loop can be tracked across."""
    setpoints = [0] * 30 + [GAP] + [1] * 4 + [2] * 30
    rows = [FrameRow(frame=i, program="p", params=(0,) * 12, setpoint=s) for i, s in enumerate(setpoints)]
    assert B._dwells(rows) == [(0, B.FRAME_SAMPLES), (35, B.FRAME_SAMPLES)]


def test_dwells_of_an_archive_of_nothing_but_gaps():
    rows = [FrameRow(frame=i, program="p", params=(0,) * 12, setpoint=GAP) for i in range(4)]
    assert not B._dwells(rows)


def test_an_archive_of_nothing_but_gaps_has_no_dwell_to_read(loop):
    rows = Capture([np.zeros((HEIGHT, WIDTH, 3), np.float32)] * 4, [GAP] * 4).rows
    with pytest.raises(LookupError, match="no measured dwell"):
        B.behaviour(B.dwells(rows), {}, loop=loop)


def test_dwells_whose_frames_were_never_decoded_read_nothing(loop):
    reader = capture(loop, lambda f: f)
    with pytest.raises(LookupError, match="no frames decoded"):
        B.behaviour(B.dwells(reader.rows), {}, loop=loop)


def test_behaviour_unpacks_into_the_profile_fields(loop):
    registered, pointwise = measured(loop, lambda f: f)
    assert registered > 0.9 and pointwise > 0.9


def test_a_capture_at_another_resolution_is_matched_by_resampling_the_loop(loop):
    """The stimulus is resampled onto the capture; the capture is never scaled down."""
    found = measured(loop, lambda f: cv2.resize(f, (2 * WIDTH, 2 * HEIGHT)))
    assert found.registered > 0.9 and found.pointwise > 0.9


def test_registration_runs_at_the_width_the_carrier_survives_at():
    """Native is where the chain has lowpassed it away; the capture's own where that is smaller."""
    assert B._target((1080, 1920, 3)) == (B.REGISTER_WIDTH, 360)
    assert B._target((HEIGHT, WIDTH, 3)) == (WIDTH, HEIGHT)


def test_the_reference_is_resampled_to_that_width_not_regenerated(loop):
    small = B._scaled(loop, WIDTH // 2)
    assert (small.width, small.height) == (WIDTH // 2, HEIGHT // 2)
    assert abs(float(small.texture.mean())) < 1e-6 and float(np.abs(small.texture).max()) == 1.0
    assert B._scaled(loop, 2 * WIDTH) is loop


def test_a_capture_wider_than_registration_is_area_averaged_onto_it():
    frame = np.zeros((1080, 1920, 3), np.uint8)
    frame[..., 1] = 255
    small = B._reduced(frame, B._target(frame.shape))
    assert small.shape == (360, 640, 3) and small.dtype == np.float32
    assert float(small[..., 1].min()) == 1.0 and float(small[..., 0].max()) == 0.0

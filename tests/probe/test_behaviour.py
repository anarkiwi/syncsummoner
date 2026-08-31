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
from syncsummoner.probe.archive import ArchiveError, FrameRow, GAP

WIDTH, HEIGHT, COUNT = 160, 128, 16
DWELL = 32
FPS = 12.0


@pytest.fixture(name="loop", scope="module")
def loop_fixture():
    """Loop."""
    return cf.build_loop(width=WIDTH, height=HEIGHT, rng=np.random.default_rng(11), count=COUNT)


def bytes_of(frame):
    """One frame as the decoder hands it back: native size, RGB, 8 bit."""
    return np.clip(np.rint(np.asarray(frame) * 255.0), 0, 255).astype(np.uint8)


class Reader:
    """Frame reader stub over a rendered dwell, streaming native frames in order."""

    def __init__(self, frames, setpoints, *, fps=FPS):
        self.frames = [bytes_of(f) for f in frames]
        self.fps = fps
        self.meta = {"width": WIDTH, "height": HEIGHT, "frames": len(self.frames)}
        self.rows = [
            FrameRow(frame=i, program="p", params=tuple([max(s, 0)] * 12), setpoint=s)
            for i, s in enumerate(setpoints)
        ]
        self.spans = []

    def stream(self, *, start=0, count=None, width=None):
        self.spans.append((start, count, width))
        end = len(self.frames) if count is None else start + count
        return iter(self.frames[start:end])


class Archive:
    """Frame archive stub holding one reader per program."""

    def __init__(self, readers):
        self.readers = dict(readers)

    def reader(self, program, key=None):
        del key
        return self.readers.get(program)


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
    return Reader(frames, setpoints)


def measured(loop, transform):
    """Behaviour of one program whose archive holds the transformed loop."""
    return B.program_behaviour(Archive({"p": capture(loop, transform)}), "p", loop=loop)


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


def test_the_pass_reads_native_frames_not_scaled_ones(loop):
    reader = capture(loop, lambda f: f)
    B.program_behaviour(Archive({"p": reader}), "p", loop=loop)
    assert reader.spans and all(width is None for _, _, width in reader.spans)
    assert [count for _, count, _ in reader.spans] == [B.FRAME_SAMPLES] * 2


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


def test_a_program_the_archive_does_not_hold_keeps_the_defaults():
    notes = []
    assert B.program_behaviour(Archive({}), "absent", log=notes.append) == B.Behaviour()
    assert notes and "absent" in notes[0]


def test_an_archive_of_nothing_but_gaps_keeps_the_defaults(loop):
    reader = Reader([np.zeros((HEIGHT, WIDTH, 3), np.float32)] * 4, [GAP] * 4)
    notes = []
    found = B.program_behaviour(Archive({"p": reader}), "p", loop=loop, log=notes.append)
    assert found == B.Behaviour() and "no measured dwell" in notes[0]


def test_a_program_that_will_not_decode_does_not_end_the_refit(loop):
    class Broken(Reader):
        """A reader whose decoder fails part way through the archive."""

        def stream(self, *, start=0, count=None, width=None):
            del start, count, width
            raise ArchiveError("decoding gave 0 bytes")

    reader = Broken([np.zeros((HEIGHT, WIDTH, 3), np.float32)] * DWELL, [0] * DWELL)
    notes = []
    found = B.program_behaviour(Archive({"p": reader}), "p", loop=loop, log=notes.append)
    assert found == B.Behaviour() and "ArchiveError" in notes[0]


def test_behaviour_unpacks_into_the_profile_fields(loop):
    registered, pointwise = measured(loop, lambda f: f)
    assert registered > 0.9 and pointwise > 0.9


def test_every_program_asked_for_is_keyed_even_where_it_could_not_be_read(loop):
    archive = Archive({"a": capture(loop, lambda f: f)})
    found = B.behaviours(archive, ["a", "b"], loop=loop, jobs=2)
    assert sorted(found) == ["a", "b"] and found["b"] == B.Behaviour()
    assert found["a"].pointwise > 0.9


def test_a_capture_at_another_resolution_is_matched_by_resampling_the_loop(loop):
    """The stimulus is resampled onto the capture; the capture is never scaled down."""
    found = measured(loop, lambda f: cv2.resize(f, (2 * WIDTH, 2 * HEIGHT)))
    assert found.registered > 0.9 and found.pointwise > 0.9


def test_a_dwell_the_decoder_gives_no_frames_for_keeps_the_defaults(loop):
    reader = Reader([], [0] * DWELL)
    notes = []
    found = B.program_behaviour(Archive({"p": reader}), "p", loop=loop, log=notes.append)
    assert found == B.Behaviour() and "no frames decoded" in notes[0]


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

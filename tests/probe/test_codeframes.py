"""Redundant loop coding: known transforms in, the right class and residual out."""

# pylint: disable=no-member  ; cv2 is a compiled extension pylint cannot introspect

import cv2
import numpy as np
import pytest

from syncsummoner.device import playout
from syncsummoner.device.profile import ProgramStyle
from syncsummoner.probe import codeframes as cf

WIDTH, HEIGHT, COUNT = 160, 128, 16
REPEATS = 2
SHIFT = (7, -4)
ZOOM = 1.25
ROTATION_DEG = 15.0
HUE_DEG = 40.0
DELAY = 3


@pytest.fixture(name="loop", scope="module")
def loop_fixture():
    """Loop."""
    return cf.build_loop(width=WIDTH, height=HEIGHT, rng=np.random.default_rng(11), count=COUNT)


def chain(frame, seed=5, noise=0.004):
    """Model the playout chain: limited range, 565 quantisation, capture noise."""
    words = np.frombuffer(playout.to_fb565(frame), "<u2").reshape(frame.shape[:2])
    levels = np.stack([words & 31, (words >> 5) & 63, (words >> 11) & 31], -1) / playout.LEVELS
    noisy = levels + np.random.default_rng(seed).normal(0.0, noise, levels.shape)
    return np.clip(noisy, 0.0, 1.0).astype(np.float32)


def warp(image, degrees=0.0, scale=1.0):
    """Rotate and scale about the frame centre; positive degrees is anticlockwise on screen."""
    height, width = image.shape[:2]
    matrix = cv2.getRotationMatrix2D(((width - 1) / 2.0, (height - 1) / 2.0), degrees, scale)
    return cv2.warpAffine(image, matrix, (width, height), borderMode=cv2.BORDER_REFLECT_101)


def hue_rotate(image, degrees):
    """Rotate the opponent chroma pair, leaving luma untouched."""
    luma = image @ cf.LUMA
    cb, cr = image[..., 2] - luma, image[..., 0] - luma
    angle = np.deg2rad(degrees)
    cb2 = cb * np.cos(angle) - cr * np.sin(angle)
    cr2 = cb * np.sin(angle) + cr * np.cos(angle)
    green = luma - (0.299 * cr2 + 0.114 * cb2) / 0.587
    return np.clip(np.stack([luma + cr2, green, luma + cb2], -1), 0.0, 1.0).astype(np.float32)


def tile(image):
    """Two by two tiling: a rearrangement no similarity transform can undo."""
    height, width = image.shape[:2]
    return np.tile(cv2.resize(image, (width // 2, height // 2)), (2, 2, 1))


def luma_key(image, level=0.5):
    """Hard two-colour key: the mean chroma stops moving, the ramp direction does not."""
    luma = (image @ cf.LUMA)[..., None]
    return np.where(luma > level, np.float32([1, 0.2, 0]), np.float32([0, 0.2, 1])).astype(np.float32)


TRANSFORMS = {
    "passthrough": (lambda f, i: f, cf.Verdict.PASSTHROUGH),
    "translate": (lambda f, i: np.roll(f, SHIFT[::-1], axis=(0, 1)), cf.Verdict.GEOMETRIC),
    "zoom": (lambda f, i: warp(f, scale=ZOOM), cf.Verdict.GEOMETRIC),
    "rotate": (lambda f, i: warp(f, degrees=-ROTATION_DEG), cf.Verdict.GEOMETRIC),
    "tile": (lambda f, i: tile(f), cf.Verdict.GEOMETRIC),
    "mirror": (lambda f, i: np.ascontiguousarray(f[:, ::-1]), cf.Verdict.GEOMETRIC),
    "posterize": (lambda f, i: np.floor(f * 4) / 3.0, cf.Verdict.PASSTHROUGH),
    "invert": (lambda f, i: 1.0 - f, cf.Verdict.VALUE),
    "hue_rotate": (lambda f, i: hue_rotate(f, HUE_DEG), cf.Verdict.VALUE),
    "gain": (lambda f, i: np.clip(0.75 * f + 0.08, 0.0, 1.0), cf.Verdict.PASSTHROUGH),
    "rotate_and_tint": (
        lambda f, i: hue_rotate(warp(f, degrees=-ROTATION_DEG), HUE_DEG),
        cf.Verdict.MIXED,
    ),
    "key": (lambda f, i: luma_key(f), cf.Verdict.VALUE),
    "noise": (
        lambda f, i: np.random.default_rng(i).random(f.shape).astype(np.float32),
        cf.Verdict.DESTRUCTIVE,
    ),
}
STYLES = {
    cf.Verdict.PASSTHROUGH: ProgramStyle.UNKNOWN,
    cf.Verdict.FROZEN: ProgramStyle.UNKNOWN,
    cf.Verdict.GEOMETRIC: ProgramStyle.DIGITAL,
    cf.Verdict.VALUE: ProgramStyle.ANALOG,
    cf.Verdict.DESTRUCTIVE: ProgramStyle.MIXED,
    cf.Verdict.MIXED: ProgramStyle.MIXED,
}


def wrapped(angle):
    """Signed angle in (-pi, pi]."""
    return float(np.arctan2(np.sin(angle), np.cos(angle)))


def played(loop, repeats=REPEATS):
    """The loop as it would be shown, in order."""
    return list(loop.play(repeats))


def captured(loop, transform, repeats=REPEATS):
    """Play the loop through a transform and the playout chain."""
    return [chain(np.clip(transform(f, i), 0.0, 1.0)) for i, f in enumerate(played(loop, repeats))]


def test_frame_contract(loop):
    """Frame contract."""
    for index in range(loop.count):
        frame = loop.frame(index)
        assert frame.shape == (HEIGHT, WIDTH, 3)
        assert frame.dtype == np.float32
        assert 0.0 <= frame.min() and frame.max() <= 1.0


def test_frames_clear_of_clipping(loop):
    """Frames clear of clipping."""
    body = np.stack([loop.frame(k)[: loop.body_height] for k in range(loop.count)])
    assert body.min() > 0.1 and body.max() < 0.9


def test_strip_round_trips(loop):
    """Strip round trips."""
    assert [cf.read_strip(chain(loop.frame(k)), loop) for k in range(loop.count)] == list(range(loop.count))


def test_both_channels_carry_the_index(loop):
    """Both channels carry the index."""
    for index in range(loop.count):
        code = cf.decode_frame(chain(loop.frame(index)), loop)
        expected = loop.angle(index)
        assert abs(wrapped(code.hue_angle - expected)) < np.deg2rad(2.0)
        assert abs(wrapped(code.ramp_angle - expected)) < np.deg2rad(2.0)
        assert code.strip_index == index


def test_play_repeats(loop):
    """Play repeats."""
    assert len(played(loop, 3)) == 3 * loop.count


@pytest.mark.parametrize("name", sorted(TRANSFORMS))
def test_classification(loop, name):
    """Classification."""
    transform, expected = TRANSFORMS[name]
    report = cf.analyse(captured(loop, transform), loop)
    assert report.verdict is expected, name
    assert report.style is STYLES[expected]
    assert report.confidence > 0.5
    assert report.frames == REPEATS * loop.count


def test_translation_recovered(loop):
    """Translation recovered."""
    report = cf.analyse(captured(loop, TRANSFORMS["translate"][0]), loop)
    assert report.shift == pytest.approx(SHIFT, abs=0.25)
    assert report.scale == pytest.approx(1.0, abs=0.02)


def test_zoom_recovered(loop):
    """Zoom recovered."""
    report = cf.analyse(captured(loop, TRANSFORMS["zoom"][0]), loop)
    assert report.scale == pytest.approx(ZOOM, rel=0.05)
    assert abs(np.degrees(report.image_rotation)) < 2.0


def test_rotation_recovered_and_separated_from_hue(loop):
    """Rotation recovered and separated from hue."""
    report = cf.analyse(captured(loop, TRANSFORMS["rotate"][0]), loop)
    assert np.degrees(report.image_rotation) == pytest.approx(ROTATION_DEG, abs=2.0)
    assert np.degrees(report.hue_rotation) == pytest.approx(0.0, abs=2.0)
    assert report.scale == pytest.approx(1.0, abs=0.02)


def test_hue_rotation_recovered_and_separated_from_geometry(loop):
    """Hue rotation recovered and separated from geometry."""
    report = cf.analyse(captured(loop, TRANSFORMS["hue_rotate"][0]), loop)
    assert np.degrees(report.hue_rotation) == pytest.approx(HUE_DEG, abs=2.0)
    assert abs(np.degrees(report.image_rotation)) < 2.0
    assert not report.inverting


def test_both_residuals_recovered_together(loop):
    """Both residuals recovered together."""
    report = cf.analyse(captured(loop, TRANSFORMS["rotate_and_tint"][0]), loop)
    assert np.degrees(report.image_rotation) == pytest.approx(ROTATION_DEG, abs=2.0)
    assert np.degrees(report.hue_rotation) == pytest.approx(HUE_DEG, abs=3.0)


def test_single_capture_is_undecidable(loop):
    """Single capture is undecidable."""
    report = cf.analyse([chain(loop.frame(2))], loop)
    assert report.frames == 1
    assert report.coverage == pytest.approx(1.0 / loop.count)
    assert report.repeat_fraction == 0.0


def test_inversion_recovered(loop):
    """Inversion recovered."""
    report = cf.analyse(captured(loop, TRANSFORMS["invert"][0]), loop)
    assert report.inverting
    assert report.correlation < -0.5
    assert report.ramp_survival > cf.SURVIVAL_MIN


def test_mirror_reads_as_reversed_orientation(loop):
    """Mirror reads as reversed orientation."""
    report = cf.analyse(captured(loop, TRANSFORMS["mirror"][0]), loop)
    assert report.orientation == -1
    assert report.agreement > 0.9


def test_tiling_breaks_the_ramp_but_not_the_hue(loop):
    """Tiling breaks the ramp but not the hue."""
    report = cf.analyse(captured(loop, TRANSFORMS["tile"][0]), loop)
    assert report.hue_survival > 0.9
    assert report.linearity < cf.LINEARITY_MIN


def test_saturation_is_measured_not_classified(loop):
    """Saturation is measured not classified."""

    def desaturate(frame, _index):
        grey = (frame @ cf.LUMA)[..., None]
        return grey + 0.2 * (frame - grey)

    report = cf.analyse(captured(loop, desaturate), loop)
    baseline = cf.analyse(captured(loop, TRANSFORMS["passthrough"][0]), loop)
    assert report.hue_survival > 0.9
    assert report.chroma_gain < 0.3 * baseline.chroma_gain


def test_delay_reads_as_index_lag(loop):
    """Delay reads as index lag."""
    frames = played(loop)
    captures = [chain(frames[(i - DELAY) % len(frames)]) for i in range(len(frames))]
    report = cf.analyse(captures, loop)
    assert report.delay_frames == DELAY
    assert report.verdict is cf.Verdict.PASSTHROUGH


def test_freeze_reads_as_repeats(loop):
    """Freeze reads as repeats."""
    report = cf.analyse([chain(loop.frame(7))] * (REPEATS * loop.count), loop)
    assert report.verdict is cf.Verdict.FROZEN
    assert report.style is ProgramStyle.UNKNOWN
    assert report.repeat_fraction == 1.0
    assert report.coverage == pytest.approx(1.0 / loop.count)
    assert report.hue_state is cf.ChannelState.FROZEN
    assert report.ramp_state is cf.ChannelState.FROZEN
    assert report.hue_survival < cf.SURVIVAL_MIN


def test_capture_outrunning_playout_is_not_a_pathology(loop):
    """Capture outrunning playout is not a pathology."""
    fps = (12.0, 30.0)
    captures = [chain(loop.frame(int(n * fps[0] / fps[1]))) for n in range(120)]
    report = cf.analyse(captures, loop, fps=fps)
    assert report.verdict is cf.Verdict.PASSTHROUGH
    assert report.rate == pytest.approx(cf.step_rate(loop.count, *fps), rel=0.01)
    assert report.repeat_fraction < 0.05
    assert report.scramble_fraction == 0.0
    assert report.coverage == 1.0
    assert report.strip_agreement == 1.0


def test_scramble_reads_as_disorder(loop):
    """Scramble reads as disorder."""
    frames = played(loop)
    order = np.random.default_rng(2).permutation(len(frames))
    report = cf.analyse([chain(frames[i]) for i in order], loop)
    assert report.scramble_fraction > 0.4
    assert report.coverage == 1.0


def test_dropped_and_duplicated_frames_tolerated(loop):
    """Dropped and duplicated frames tolerated."""
    rng = np.random.default_rng(4)
    frames = played(loop, 4)
    captures = [chain(f) for f in frames if rng.random() > 0.25]
    report = cf.analyse(captures, loop)
    assert report.coverage == 1.0
    assert report.verdict is cf.Verdict.PASSTHROUGH
    assert report.strip_agreement == 1.0


def test_capture_at_another_resolution(loop):
    """Capture at another resolution."""
    size = (WIDTH * 3 // 4, HEIGHT * 3 // 4)
    captures = [chain(cv2.resize(f, size)) for f in played(loop)]
    report = cf.analyse(captures, loop)
    assert report.verdict is cf.Verdict.PASSTHROUGH
    assert report.hue_survival > 0.9


def test_loop_without_strip():
    """Loop without strip."""
    loop = cf.build_loop(width=WIDTH, height=HEIGHT, rng=np.random.default_rng(3), count=8, strip_px=0)
    assert loop.body_height == HEIGHT
    code = cf.decode_frame(chain(loop.frame(3)), loop)
    assert code.strip_index is None
    assert abs(wrapped(code.hue_angle - loop.angle(3))) < np.deg2rad(2.0)


def test_verdicts_map_onto_program_style():
    """Verdicts map onto program style."""
    assert cf.Verdict.MIXED.style is ProgramStyle.MIXED
    assert {v.style for v in cf.Verdict} <= set(ProgramStyle)

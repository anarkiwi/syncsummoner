"""Stimulus battery and the gray-code state index, including chain degradation."""

import numpy as np
import pytest
from scipy.ndimage import zoom

from syncsummoner.probe import patterns

WIDTH, HEIGHT = 96, 64
GAIN, PEDESTAL = 0.96, 18.0 / 255.0


def frame(name="zoneplate", *, width=WIDTH, height=HEIGHT, index=0, seed=7):
    """Frame."""
    return patterns.generate(
        name, width=width, height=height, frame_index=index, rng=np.random.default_rng(seed)
    )


def degrade(image, *, noise=0.01, scale=1.0, seed=3):
    """Model the measured analog chain: luma gain, black pedestal, noise, scaling."""
    rng = np.random.default_rng(seed)
    out = np.clip(GAIN * image + PEDESTAL + rng.normal(0.0, noise, image.shape), 0.0, 1.0)
    if scale != 1.0:
        out = zoom(out, (1.0, scale, 1.0), order=1)
    return out.astype(np.float32)


@pytest.mark.parametrize("name", patterns.PATTERNS)
def test_generate_contract(name):
    """Generate contract."""
    out = frame(name, index=5)
    assert out.shape == (HEIGHT, WIDTH, 3)
    assert out.dtype == np.float32
    assert 0.0 <= out.min() and out.max() <= 1.0


def test_unknown_pattern():
    """Unknown pattern."""
    with pytest.raises(ValueError, match="unknown pattern"):
        frame("does_not_exist")


@pytest.mark.parametrize("name", ["grating_sweep", "motion_ball"])
def test_time_varying(name):
    """Time varying."""
    assert not np.allclose(frame(name, index=0), frame(name, index=9))


@pytest.mark.parametrize("name", ["zoneplate", "smpte_bars", "siemens_star", "noise_1f"])
def test_time_invariant(name):
    """Time invariant."""
    assert np.array_equal(frame(name, index=0), frame(name, index=9))


def test_noise_1f_seeded():
    """Noise 1/f seeded."""
    assert np.array_equal(frame("noise_1f", seed=1), frame("noise_1f", seed=1))
    assert not np.allclose(frame("noise_1f", seed=1), frame("noise_1f", seed=2))


def test_luma_ramp_quantization_is_countable():
    """Luma ramp quantization is countable."""
    ramp = patterns.generate("luma_ramp", width=WIDTH, height=8, rng=np.random.default_rng(0), steps=5)
    assert len(np.unique(ramp)) == 5
    assert len(np.unique(frame("luma_ramp"))) > 5


def test_smpte_bars_are_chromatic():
    """SMPTE bars are chromatic."""
    bars = frame("smpte_bars")
    assert np.ptp(bars, axis=2).max() > 0.5


def test_motion_ball_has_impact():
    """Motion ball has impact."""
    centroids = []
    for index in range(30):
        ball = frame("motion_ball", index=index)[:, :, 0]
        rows = np.arange(ball.shape[0])
        centroids.append(float((rows * ball.sum(axis=1)).sum() / ball.sum()))
    velocity = np.diff(centroids)
    assert np.abs(np.diff(velocity)).max() > 3 * np.abs(np.diff(velocity)).mean()


@pytest.mark.parametrize("index", [0, 1, 2, 3, 85, 128, 200, 254, 255])
def test_state_index_roundtrip_clean(index):
    """State index round-trips clean."""
    tagged = patterns.with_state_index(frame(), index)
    assert patterns.read_state_index(tagged) == index


def test_state_index_roundtrip_degraded_all_codes():
    """State index round-trips degraded all codes."""
    base = frame("noise_1f")
    for index in range(patterns.state_index_capacity(8)):
        tagged = degrade(patterns.with_state_index(base, index))
        assert patterns.read_state_index(tagged) == index


@pytest.mark.parametrize("scale", [0.5, 0.83, 1.37])
def test_state_index_survives_horizontal_scaling(scale):
    """State index survives horizontal scaling."""
    base = frame("smpte_bars")
    for index in (0, 37, 200, 255):
        tagged = degrade(patterns.with_state_index(base, index), scale=scale)
        assert patterns.read_state_index(tagged) == index


@pytest.mark.parametrize("bits,strip_px", [(4, 4), (6, 12), (10, 8)])
def test_state_index_widths(bits, strip_px):
    """State index widths."""
    base = frame()
    for index in (0, 1, (1 << bits) - 1):
        tagged = patterns.with_state_index(base, index, bits=bits, strip_px=strip_px)
        assert patterns.read_state_index(degrade(tagged), bits=bits, strip_px=strip_px) == index


def test_state_index_wraps_modulo_capacity():
    """State index wraps modulo capacity."""
    base = frame()
    tagged = patterns.with_state_index(base, 256 + 9, bits=8)
    assert patterns.read_state_index(tagged) == 9


def test_gray_code_single_bit_transitions():
    """Gray code single bit transitions."""
    base = np.zeros((16, WIDTH, 3), dtype=np.float32)
    centres = np.round((np.arange(10) + 0.5) * WIDTH / 10).astype(int)
    cells = np.array(
        [
            patterns.with_state_index(base, i)[0, centres, 0] > 0.5
            for i in range(patterns.state_index_capacity(8))
        ]
    )
    assert np.abs(np.diff(cells.astype(int), axis=0)).sum(axis=1).max() == 1


def test_no_strip_reads_none():
    """No strip reads none."""
    assert patterns.read_state_index(np.full((32, WIDTH, 3), 0.5, dtype=np.float32)) is None


def test_strip_leaves_body_untouched():
    """Strip leaves body untouched."""
    base = frame()
    tagged = patterns.with_state_index(base, 42, strip_px=8)
    assert np.array_equal(patterns.crop_strip(tagged, strip_px=8), base[8:])
    assert patterns.crop_strip(tagged, strip_px=8).shape == (HEIGHT - 8, WIDTH, 3)


def test_with_state_index_does_not_mutate():
    """With state index does not mutate."""
    base = frame()
    copy = base.copy()
    patterns.with_state_index(base, 11)
    assert np.array_equal(base, copy)

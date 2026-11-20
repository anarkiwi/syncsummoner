"""Channel order judged from planar YUV: each exchange is told apart, whatever the gain and pedestal."""

# pylint: disable=missing-function-docstring

import numpy as np
import pytest

from syncsummoner.probe.chroma import (
    AS_PLAYED,
    ORDERS,
    TO_YUV,
    ChromaOrder,
    order_of,
    to_rgb,
    to_yuv,
    variants,
)

H, W = 18, 32


def field(n=3, seed=0):
    """RGB frames with luma, U and V varying independently, as 8-bit planar YUV."""
    rng = np.random.default_rng(seed)
    x = np.linspace(-1, 1, W)[None, :].repeat(H, 0)
    y = np.linspace(-1, 1, H)[:, None].repeat(W, 1)
    frames = []
    for _ in range(n):
        a, b = rng.uniform(0.5, 1.0, 2)
        yuv = np.stack([0.45 + 0.1 * y, 0.25 * a * x, 0.25 * b * y], -1)
        frames.append(np.clip(yuv @ np.linalg.inv(TO_YUV).T, 0, 1))
    return to_yuv(np.stack(frames))


def test_to_yuv_puts_grey_on_the_chroma_zero_and_planes_first():
    grey = to_yuv(np.full((2, 4, 6, 3), 0.5))
    assert grey.shape == (2, 3, 4, 6)
    assert np.allclose(grey[:, 0], 127.5) and np.allclose(grey[:, 1:], 128.0)


def test_variants_are_the_picture_and_its_three_exchanges():
    shown = field(1)
    found = variants(shown)
    assert found.shape == (4, 1, 3, H, W)
    assert np.allclose(found[0], shown - np.array([0, 128, 128])[:, None, None])
    assert np.allclose(found[1], found[0][:, [0, 2, 1]])
    assert np.allclose(found[3], found[2][:, [0, 2, 1]])
    twice = variants(variants(shown)[2] + np.array([0, 128, 128])[:, None, None])[2]
    assert np.allclose(twice, found[0])


@pytest.mark.parametrize("which", range(4))
def test_each_order_is_found_at_any_gain_and_pedestal(which):
    shown = field()
    captured = 0.6 * variants(shown)[which] + np.array([20.0, 128.0, 128.0])[:, None, None]
    got = order_of(captured, shown)
    assert got.order == ORDERS[which] and got.frames == 3
    assert got.scores[ORDERS[which]] == pytest.approx([1.0, 1.0, 1.0], abs=1e-6)
    assert got.ok == (which in (0, 3)) and got.judged


def test_ok_judges_the_chroma_pair_by_its_margin():
    """Red and blue exchanged reads near enough a chroma exchange that only the pair is gated on."""
    scores = {o: [0.9, 0.1, 0.1] for o in ORDERS}
    found = ChromaOrder(AS_PLAYED, scores, 4, -2.0)
    assert not found.ok and not found.judged
    assert found.as_dict() == {
        "order": AS_PLAYED,
        "ok": False,
        "judged": False,
        "margin": -2.0,
        "frames": 4,
        "scores": scores,
    }
    assert ChromaOrder(AS_PLAYED, scores, 4, 3.0).judged


def test_noise_alone_is_not_judged_and_one_clear_frame_is():
    rng = np.random.default_rng(5)
    shown = field(8)
    noise = rng.normal(128, 20, shown.shape)
    assert not order_of(noise, shown).judged
    one = order_of(variants(shown[:1])[1] + 128, shown[:1])
    assert one.judged and not one.ok and one.margin == -np.inf


def test_a_flat_plane_scores_zero_rather_than_nan():
    shown = field(1)
    flat = shown.copy()
    flat[:, 0] = 100.0
    got = order_of(flat, shown)
    assert all(np.isfinite(v).all() for v in got.scores.values())
    assert got.scores[AS_PLAYED][0] == 0.0


def test_no_captures_cannot_be_judged():
    with pytest.raises(ValueError, match="no captures"):
        order_of(np.zeros((0, 3, H, W)), np.zeros((0, 3, H, W)))


def test_to_rgb_undoes_to_yuv_and_reads_studio_swing():
    rgb = np.random.default_rng(2).uniform(0.0, 1.0, (3, 4, 6, 3))
    assert np.allclose(to_rgb(to_yuv(rgb)), rgb, atol=1e-5)
    studio = np.stack([np.full((4, 6), v) for v in (235.0, 128.0, 128.0)])[None]
    black = np.stack([np.full((4, 6), v) for v in (16.0, 128.0, 128.0)])[None]
    assert np.allclose(to_rgb(studio, full=False), 1.0) and np.allclose(to_rgb(black, full=False), 0.0)
    red = np.array([[[1.0, 0.0, 0.0]]])
    bt709 = np.array([0.2126, 0.5 / (1 - 0.0722) * -0.2126, 0.5]) * 255.0 + [0.0, 128.0, 128.0]
    assert np.allclose(to_rgb(bt709[:, None, None], space="bt709"), red, atol=1e-5)

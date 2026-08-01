"""Capture: V4L2 configuration, RGB conversion, no-signal splash detection."""

# pylint: disable=missing-function-docstring
# pylint: disable=no-member  ; cv2 is a compiled extension pylint cannot introspect

import cv2
import numpy as np
import pytest

from syncsummoner.device import capture as cap
from syncsummoner.device.capture import Capture, CaptureError

from .conftest import FakeClock, FakeVideoCapture, bgr_frame

SIZE = (48, 64)


@pytest.fixture(name="opened")
def opened_fixture(monkeypatch):
    """A Capture bound to a fake cv2.VideoCapture, plus that fake."""
    fake = FakeVideoCapture("/dev/video0")
    monkeypatch.setattr(cv2, "VideoCapture", lambda *args, **kwargs: fake)
    clock = FakeClock()
    return Capture(sleep=clock.sleep, clock=clock), fake, clock


def splash(size=SIZE, text_frac=0.06):
    """Achromatic bilevel slide: white field with black glyph pixels."""
    frame = np.ones(size + (3,), dtype=np.float32)
    rng = np.random.default_rng(1)
    mask = rng.random(size) < text_frac
    frame[mask] = 0.0
    return frame


def colorbars(size=SIZE):
    """Saturated vertical bars."""
    hues = np.array(
        [[1, 1, 1], [1, 1, 0], [0, 1, 1], [0, 1, 0], [1, 0, 1], [1, 0, 0], [0, 0, 1], [0, 0, 0]],
        dtype=np.float32,
    )
    row = np.repeat(hues, size[1] // len(hues), axis=0)
    return np.broadcast_to(row, size + (3,)).copy()


def mono_content(size=SIZE):
    """Achromatic continuous-tone content, e.g. a mono zone plate."""
    y, x = np.indices(size).astype(np.float32)
    ramp = 0.5 + 0.45 * np.sin((x**2 + y**2) / 60.0)
    return np.repeat(ramp[:, :, None], 3, axis=2)


def test_open_configures_v4l2(opened):
    capture, fake, _ = opened
    capture.open()
    assert fake.props[cv2.CAP_PROP_FRAME_WIDTH] == 720
    assert fake.props[cv2.CAP_PROP_FRAME_HEIGHT] == 576
    assert fake.props[cv2.CAP_PROP_FPS] == 50
    assert fake.props[cv2.CAP_PROP_FOURCC] == cv2.VideoWriter_fourcc(*"YUYV")
    assert fake.props[cv2.CAP_PROP_CONVERT_RGB] == 1


def test_open_is_idempotent(opened):
    capture, fake, _ = opened
    assert capture.open().open()._cap is fake  # pylint: disable=protected-access


def test_open_raises_when_card_absent(opened):
    capture, fake, _ = opened
    fake.opened = False
    with pytest.raises(CaptureError, match="V4L2"):
        capture.open()


def test_read_before_open_raises(opened):
    capture, _, _ = opened
    with pytest.raises(CaptureError, match="not open"):
        capture.read()


def test_read_converts_bgr_to_rgb_float(opened):
    capture, fake, _ = opened
    fake.frames = [bgr_frame((255, 128, 0))]
    frame = capture.open().read()
    assert frame.dtype == np.float32
    assert frame.shape == (4, 4, 3)
    assert frame[0, 0] == pytest.approx([1.0, 128 / 255, 0.0], abs=1e-6)


def test_read_returns_none_on_failed_grab(opened):
    capture, _, _ = opened
    assert capture.open().read() is None


def test_context_manager_releases(opened):
    capture, fake, _ = opened
    with capture:
        pass
    assert fake.released


def test_close_is_idempotent(opened):
    capture, _, _ = opened
    capture.open().close()
    capture.close()


def test_chroma_fraction_separates_bars_from_splash(opened):
    capture, _, _ = opened
    assert capture.chroma_fraction(colorbars()) > 0.5
    assert capture.chroma_fraction(splash()) == 0.0


def test_splash_is_no_signal(opened):
    capture, _, _ = opened
    assert capture.is_no_signal(splash())


def test_colorbars_are_content(opened):
    capture, _, _ = opened
    assert not capture.is_no_signal(colorbars())


def test_monochrome_content_is_not_no_signal(opened):
    capture, _, _ = opened
    frame = mono_content()
    assert capture.chroma_fraction(frame) == 0.0
    assert not capture.is_no_signal(frame)


def test_black_is_not_no_signal(opened):
    capture, _, _ = opened
    assert not capture.is_no_signal(np.zeros(SIZE + (3,), dtype=np.float32))


def test_high_variance_splash_would_fool_a_variance_test(opened):
    capture, _, _ = opened
    frame = splash()
    assert frame.std() > 0.1
    assert capture.is_no_signal(frame)


def test_missing_frame_is_no_signal(opened):
    capture, _, _ = opened
    assert capture.is_no_signal(None)


def test_wait_for_lock_succeeds_after_splash(opened):
    capture, fake, clock = opened
    fake.frames = [
        (splash() * 255).astype(np.uint8),
        (colorbars() * 255).astype(np.uint8)[:, :, ::-1],
    ]
    assert capture.wait_for_lock(1.0)
    assert clock.slept == [0.05]


def test_wait_for_lock_times_out(opened):
    capture, fake, clock = opened
    fake.frames = [(splash() * 255).astype(np.uint8)] * 100
    assert not capture.wait_for_lock(0.2)
    assert clock.now >= 0.2


def test_luma_weights_sum_to_one():
    assert cap.LUMA.sum() == pytest.approx(1.0, abs=1e-6)

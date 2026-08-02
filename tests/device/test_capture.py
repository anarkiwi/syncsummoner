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


def _stream(target, frames):
    """Drive a fake capture from a fixed frame sequence."""
    it = iter(frames)
    target.read = lambda: next(it, None)
    return target


def test_wait_for_content_requires_motion_not_just_a_signal():
    """The rig's drop freezes the output; a splash test alone would pass it."""
    rng = np.random.default_rng(0)
    frozen = np.full((8, 8, 3), 0.4, np.float32)
    port = Capture()
    port.open = lambda: None
    port.is_no_signal = lambda _f: False
    _stream(port, [frozen] * 40)
    assert not port.wait_for_content(timeout_s=0.0, run=4)

    moving = [rng.random((8, 8, 3)).astype(np.float32) for _ in range(40)]
    _stream(port, moving)
    assert port.wait_for_content(timeout_s=5.0, run=4)


def test_wait_for_content_resets_the_run_on_a_splash():
    """A splash mid-run must not count toward the required streak."""
    rng = np.random.default_rng(1)
    good = [rng.random((8, 8, 3)).astype(np.float32) for _ in range(20)]
    blank = np.zeros((8, 8, 3), np.float32)
    port = Capture()
    port.open = lambda: None
    port.is_no_signal = lambda frame: frame is blank
    _stream(port, good[:3] + [blank] + good)
    assert port.wait_for_content(timeout_s=5.0, run=5)


def test_splash_test_reads_a_bounded_sample_not_every_pixel():
    """Reading every pixel of a 1080p frame cost 3x a 30fps budget and pegged a core."""
    frame = np.zeros((1080, 1920, 3), np.float32)
    view = Capture._decimate(frame)  # pylint: disable=protected-access
    assert view.size // 3 <= 4 * cap.SPLASH_SAMPLES
    assert view.shape[0] < frame.shape[0] and view.shape[1] < frame.shape[1]


def test_small_frames_are_not_decimated():
    frame = np.zeros((45, 80, 3), np.float32)
    assert Capture._decimate(frame).shape == frame.shape  # pylint: disable=protected-access


def test_strided_chroma_fraction_tracks_the_full_frame_answer():
    """The statistic is an area fraction, so a sample must agree with the whole."""
    rng = np.random.default_rng(3)
    frame = rng.random((540, 960, 3)).astype(np.float32)
    frame[:, :480] = 0.5
    port = Capture()
    coarse = port.chroma_fraction(frame)
    exact = port.chroma_fraction(frame, samples=frame.shape[0] * frame.shape[1])
    assert abs(coarse - exact) < 0.02


def test_frames_stops_at_the_budget_when_nothing_is_usable():
    """An unbounded read-until-good loop stalls forever on a rejected stimulus."""
    clock = FakeClock()
    port = Capture(clock=clock, sleep=clock.sleep)
    port.open = lambda: None
    port.read = lambda: (clock.sleep(0.01), np.zeros((8, 8, 3), np.float32))[1]
    port.is_no_signal = lambda _f: True
    assert not port.frames(5, timeout_s=0.2)


def test_frames_returns_what_it_collected():
    clock = FakeClock()
    rng = np.random.default_rng(2)
    port = Capture(clock=clock, sleep=clock.sleep)
    port.open = lambda: None
    port.read = lambda: (clock.sleep(0.01), rng.random((8, 8, 3)).astype(np.float32))[1]
    port.is_no_signal = lambda _f: False
    assert len(port.frames(4, timeout_s=5.0, settle=2)) == 4

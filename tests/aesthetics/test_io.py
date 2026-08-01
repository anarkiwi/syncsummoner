"""Decode shim, exercised against a stub VideoCapture; no files are touched."""

# cv2 is a C extension; pylint cannot introspect its members.
# pylint: disable=no-member

import cv2
import numpy as np
import pytest

from syncsummoner.aesthetics.io import DEFAULT_FPS, read_clip


class StubCapture:
    """Minimal cv2.VideoCapture stand-in yielding synthetic BGR frames."""

    def __init__(self, frames, fps=25.0, opened=True):
        self.frames = list(frames)
        self.fps = fps
        self.opened = opened
        self.released = False

    def isOpened(self):  # pylint: disable=invalid-name
        """Match the OpenCV method name."""
        return self.opened

    def get(self, prop):
        """Report frames per second, ignoring other properties."""
        return self.fps if prop == cv2.CAP_PROP_FPS else 0.0

    def read(self):
        """Pop the next frame, or signal end of stream."""
        return (True, self.frames.pop(0)) if self.frames else (False, None)

    def release(self):
        """Record that the capture was closed."""
        self.released = True


def stub(monkeypatch, **kwargs):
    """Install a StubCapture factory in place of cv2.VideoCapture."""
    capture = StubCapture(**kwargs)
    monkeypatch.setattr(cv2, "VideoCapture", lambda path: capture)
    return capture


def bgr_frames(n=6, size=8):
    """Sequence of uint8 BGR frames with a distinctive blue channel."""
    frames = np.zeros((n, size, size, 3), dtype=np.uint8)
    frames[..., 0] = 255
    frames[..., 2] = np.arange(n, dtype=np.uint8)[:, None, None]
    return list(frames)


def test_decodes_to_rgb_float_frames(monkeypatch):
    """Frames arrive as RGB float32 in [0, 1] with the container fps."""
    capture = stub(monkeypatch, frames=bgr_frames())
    frames, fps = read_clip("clip.mp4")
    assert frames.shape == (6, 8, 8, 3) and frames.dtype == np.float32
    assert fps == 25.0
    assert frames[0, 0, 0, 2] == pytest.approx(1.0)
    assert frames[0, 0, 0, 0] == pytest.approx(0.0)
    assert capture.released


def test_stride_and_max_frames(monkeypatch):
    """Striding subsamples the stream and scales the reported rate."""
    stub(monkeypatch, frames=bgr_frames(n=9))
    frames, fps = read_clip("clip.mp4", stride=3, max_frames=2)
    assert frames.shape[0] == 2 and fps == pytest.approx(25.0 / 3.0)


def test_max_width_downscales(monkeypatch):
    """Wide frames are resized, preserving aspect ratio."""
    stub(monkeypatch, frames=[np.zeros((16, 32, 3), np.uint8)] * 2)
    frames, _ = read_clip("clip.mp4", max_width=8)
    assert frames.shape[1:3] == (4, 8)


def test_missing_fps_falls_back(monkeypatch):
    """A container without a usable frame rate falls back to the default."""
    stub(monkeypatch, frames=bgr_frames(n=2), fps=0.0)
    assert read_clip("clip.mp4")[1] == DEFAULT_FPS


def test_unopenable_source(monkeypatch):
    """An unopenable path raises FileNotFoundError."""
    stub(monkeypatch, frames=[], opened=False)
    with pytest.raises(FileNotFoundError):
        read_clip("missing.mp4")


def test_empty_stream(monkeypatch):
    """A stream that decodes no frames raises."""
    stub(monkeypatch, frames=[])
    with pytest.raises(ValueError):
        read_clip("empty.mp4")


def test_invalid_stride():
    """Stride must be at least one."""
    with pytest.raises(ValueError):
        read_clip("clip.mp4", stride=0)

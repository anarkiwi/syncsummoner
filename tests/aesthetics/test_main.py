"""The standalone entry point that doubles as the extraction-seam test."""

# cv2 is a C extension; pylint cannot introspect its members.
# pylint: disable=no-member

import json

import cv2
import numpy as np
import pytest

from syncsummoner.aesthetics.__main__ import _jsonable, main
from tests.aesthetics import drifting, texture


def stub_capture(monkeypatch, n_frames=24):
    """Serve a drifting synthetic clip in place of a decoded file."""
    frames = drifting(texture(np.random.default_rng(0)), n_frames, shift=8)
    bgr = [np.ascontiguousarray((f[..., ::-1] * 255).astype(np.uint8)) for f in frames]

    class Capture:
        """Stub capture over the synthetic frames."""

        def __init__(self):
            self.pending = list(bgr)

        def isOpened(self):  # pylint: disable=invalid-name
            """Always open."""
            return True

        def get(self, prop):
            """Report 30 fps."""
            return 30.0 if prop == cv2.CAP_PROP_FPS else 0.0

        def read(self):
            """Pop the next frame."""
            return (True, self.pending.pop(0)) if self.pending else (False, None)

        def release(self):
            """No resources to free."""

    monkeypatch.setattr(cv2, "VideoCapture", lambda path: Capture())


def test_score_command_prints_json(monkeypatch, capsys):
    """`score` emits the descriptor plus the aggregate as JSON."""
    stub_capture(monkeypatch)
    assert main(["score", "clip.mp4", "--seed", "3"]) == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["n_frames"] == 24 and payload["fps"] == 30.0
    assert payload["dynamics"]["stability"] == "periodic"
    assert len(payload["channel_energy"]) == 5
    assert -1.0 <= payload["score"] <= 1.0
    assert isinstance(payload["analyzer_version"], str)


def test_argv_is_used_when_no_arguments_are_passed(monkeypatch, capsys):
    """Omitting argv reads sys.argv, as the module entry point does."""
    stub_capture(monkeypatch, n_frames=4)
    monkeypatch.setattr("sys.argv", ["prog", "score", "clip.mp4", "--stride", "1"])
    assert main() == 0
    assert json.loads(capsys.readouterr().out)["n_frames"] == 4


def test_unknown_command_exits():
    """An unrecognized subcommand fails the argument parser."""
    with pytest.raises(SystemExit):
        main(["bogus"])


def test_json_encoder_rejects_unknown_types():
    """The encoder converts numpy and enums only."""
    assert _jsonable(np.float32(0.5)) == pytest.approx(0.5)
    assert _jsonable(np.arange(2)) == [0, 1]
    with pytest.raises(TypeError):
        _jsonable(object())

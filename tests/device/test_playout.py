"""Playout: BGR565 packing and framebuffer push, with no Pi and no network."""

# pylint: disable=missing-function-docstring

import subprocess

import numpy as np
import pytest

from syncsummoner.device import host, playout as po
from syncsummoner.device.playout import Playout, to_fb565

from .conftest import FakeClock


class FakeRunner:
    """Records every command and payload handed to it."""

    def __init__(self):
        self.calls = []

    def __call__(self, command, data):
        self.calls.append((command, data))


def frame(rgb, height=1080, width=1920):
    """Constant-colour RGB float frame."""
    out = np.empty((height, width, 3), dtype=np.float32)
    out[:, :] = rgb
    return out


@pytest.mark.parametrize(
    "rgb,word",
    [
        ((0.0, 0.0, 0.0), 0x0000),
        ((1.0, 1.0, 1.0), 0xFFFF),
        ((1.0, 0.0, 0.0), 0x001F),
        ((0.0, 1.0, 0.0), 0x07E0),
        ((0.0, 0.0, 1.0), 0xF800),
    ],
)
def test_fb565_primaries_use_measured_bgr_word_order(rgb, word):
    """Blue occupies the high bits; red the low. Measured against the real chain."""
    data = to_fb565(frame(rgb, 2, 2), limited_range=False)
    assert np.all(np.frombuffer(data, dtype="<u2") == word)


def test_fb565_is_little_endian():
    data = to_fb565(frame((0.0, 0.0, 1.0), 1, 1), limited_range=False)
    assert data == b"\x00\xf8"


def test_fb565_quantises_per_channel():
    words = np.frombuffer(to_fb565(frame((0.5, 0.5, 0.5), 1, 1), limited_range=False), dtype="<u2")
    word = int(words[0])
    assert (word >> 11, (word >> 5) & 0x3F, word & 0x1F) == (16, 32, 16)


def test_fb565_clips_out_of_range():
    odd = np.array([[[-1.0, 2.0, 0.5]]], dtype=np.float32)
    word = int(np.frombuffer(to_fb565(odd, limited_range=False), dtype="<u2")[0])
    assert (word & 0x1F, (word >> 5) & 0x3F) == (0, 63)


def test_limited_range_is_the_default_and_maps_to_studio_swing():
    """Full range clips above ~92%; studio swing measures unity gain end to end."""
    black = int(np.frombuffer(to_fb565(frame((0.0, 0.0, 0.0), 1, 1)), dtype="<u2")[0])
    white = int(np.frombuffer(to_fb565(frame((1.0, 1.0, 1.0), 1, 1)), dtype="<u2")[0])
    assert (black >> 5) & 0x3F == round(16 / 255 * 63)
    assert (white >> 5) & 0x3F == round(235 / 255 * 63)
    assert black != 0x0000 and white != 0xFFFF


def test_rgb565_rejects_bad_shape():
    with pytest.raises(ValueError, match=r"\(H, W, 3\)"):
        to_fb565(np.zeros((4, 4), dtype=np.float32))


def test_frame_bytes_matches_measured_framebuffer():
    assert Playout(runner=FakeRunner()).frame_bytes == 4147200


def test_show_pushes_one_framebuffer_write():
    runner = FakeRunner()
    Playout(runner=runner).show(frame((0.0, 0.0, 1.0)))
    command, data = runner.calls[0]
    assert command == "cat > /dev/fb0"
    assert len(data) == 4147200
    words = np.frombuffer(data, dtype="<u2")
    assert np.all(words == words[0])
    assert int(words[0]) >> 11 == round(235 / 255 * 31)


def test_show_rejects_wrong_geometry():
    with pytest.raises(ValueError, match=r"\(1080, 1920\)"):
        Playout(runner=FakeRunner()).show(frame((0, 0, 0), 240, 320))


def test_encode_rejects_geometry_mismatch_with_stride():
    play = Playout(runner=FakeRunner(), width=8, height=4)
    assert len(play.encode(frame((1, 1, 1), 4, 8))) == 64


def test_play_paces_a_sequence():
    runner = FakeRunner()
    clock = FakeClock()
    play = Playout(runner=runner, width=2, height=2, sleep=clock.sleep, clock=clock)
    frames = [frame((v / 4, 0, 0), 2, 2) for v in range(4)]
    assert play.play(frames, fps=50.0) == 4
    assert len(runner.calls) == 4
    assert clock.now == pytest.approx(4 / 50.0)


def test_play_does_not_sleep_when_already_behind():
    clock = FakeClock()
    runner = FakeRunner()

    def slow(command, data):
        runner(command, data)
        clock.now += 1.0

    play = Playout(runner=slow, width=2, height=2, sleep=clock.sleep, clock=clock)
    assert play.play([frame((0, 0, 0), 2, 2)] * 3, fps=50.0) == 3
    assert not clock.slept


def test_play_empty_sequence():
    play = Playout(runner=FakeRunner(), width=2, height=2)
    assert play.play([]) == 0


def test_play_rejects_bad_fps():
    with pytest.raises(ValueError, match="positive"):
        Playout(runner=FakeRunner()).play([], fps=0)


def test_default_runner_is_ssh(monkeypatch):
    seen = {}

    def fake_run(argv, **kwargs):
        seen.update(argv=argv, input=kwargs["input"])
        return subprocess.CompletedProcess(argv, 0, stdout=b"", stderr=b"")

    monkeypatch.setattr(host.subprocess, "run", fake_run)
    Playout(width=2, height=2).show(frame((0, 0, 0), 2, 2))
    assert seen["argv"] == ["ssh", "-o", "BatchMode=yes", po.DEFAULT_HOST, f"cat > {po.DEFAULT_FB}"]
    assert len(seen["input"]) == 8


class Recorder:
    """Runner recording every command and payload it is handed."""

    def __init__(self):
        self.calls = []

    def __call__(self, command, data=None):
        self.calls.append((command, None if data is None else len(data)))
        return ""


def test_a_clip_player_uploads_once_and_plays_from_the_pi(tmp_path):
    """The Pi decrypts every pushed byte, which caps a pushed pass at a few frames a second."""
    from syncsummoner.device.playout import ClipPlayer

    clip = tmp_path / "clip.mkv"
    clip.write_bytes(b"x" * 2048)
    runner = Recorder()
    player = ClipPlayer("pi@rig", width=1920, height=1080, runner=runner)
    assert player.upload(str(clip)) == 2048
    with player.playing(fps=30.0):
        pass
    commands = [c for c, _ in runner.calls]
    assert any(player.CLIP_PATH in c and c.startswith("cat >") for c in commands), "clip goes up once"
    assert any("ffmpeg" in c and "-re" in c and str(player.frame_bytes) in c for c in commands)
    assert any(c.startswith("kill -TERM") for c in commands), "and is stopped by pid, not by pattern"


def test_a_clip_player_decodes_at_the_framebuffer_geometry():
    from syncsummoner.device.playout import ClipPlayer

    player = ClipPlayer("pi@rig", width=1280, height=720, runner=Recorder())
    assert "scale=1280:720" in player.command().format(fps=25)


def test_a_clip_player_allows_a_whole_clip_to_go_up(monkeypatch):
    """A 449MB clip at ssh speed needs minutes, and the default runner gives thirty seconds."""
    seen = []
    monkeypatch.setattr(po, "ssh_runner", lambda host, timeout: seen.append(timeout))
    po.ClipPlayer("pi@rig")
    po.Playout("pi@rig")
    assert seen[0] == po.UPLOAD_TIMEOUT >= 300.0
    assert seen[1] < seen[0], "a still frame does not need a clip's budget"

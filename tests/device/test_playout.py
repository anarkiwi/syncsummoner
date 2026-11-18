"""Playout: BGR565 packing, exact rates and flip-log judging, with no Pi and no network."""

# pylint: disable=missing-function-docstring

import hashlib
from fractions import Fraction

import numpy as np
import pytest

from syncsummoner.device import playout as po
from syncsummoner.device.playout import Playout, to_fb565


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


def test_encode_rejects_geometry_mismatch():
    with pytest.raises(ValueError, match=r"\(1080, 1920\)"):
        Playout(runner=FakeRunner()).encode(frame((0, 0, 0), 240, 320))


def test_encode_packs_at_the_display_geometry():
    play = Playout(runner=FakeRunner(), width=8, height=4)
    assert len(play.encode(frame((1, 1, 1), 4, 8))) == 64 == play.frame_bytes


@pytest.mark.parametrize(
    "fps, exact",
    [
        (29.97, Fraction(30000, 1001)),
        (59.94, Fraction(60000, 1001)),
        (23.976, Fraction(24000, 1001)),
        (30.0, Fraction(30)),
        (10.0, Fraction(10)),
        (12.5, Fraction(25, 2)),
        (7.5, Fraction(15, 2)),
        (0.3, Fraction(3, 10)),
        ("30000/1001", Fraction(30000, 1001)),
        (Fraction(24), Fraction(24)),
    ],
)
def test_rate_is_exact_and_recognises_ntsc(fps, exact):
    assert po.rate(fps) == exact


def flips(rows):
    """A flip log from ``(v, seq, frame, flags)`` rows."""
    return np.array([(v, seq, f, 0, flags) for v, seq, f, flags in rows], dtype=po.FLIP)


@pytest.mark.parametrize(
    "rows, hold, want",
    [
        ([], 1, (0, 0, 0, 0)),
        ([(0, 100, 0, po.SEGMENT), (1, 101, 0, 0), (2, 102, 1, 0), (3, 103, 1, 0)], 2, (4, 1, 0, 0)),
        ([(0, 100, 0, po.SEGMENT), (1, 102, 1, 0), (2, 103, 2, 0)], 1, (3, 1, 1, 0)),
        ([(0, 100, 0, po.SEGMENT), (1, 101, 1, 0), (2, 102, 1, 0)], 1, (3, 1, 0, 1)),
        (
            [(0, 100, 0, po.SEGMENT), (1, 101, 1, 0), (9, 500, 1, po.SEGMENT), (10, 501, 10, 0)],
            1,
            (4, 2, 0, 0),
        ),
        ([(0, 100, 0, po.SEGMENT), (1, 101, 1, 0), (3, 103, 2, po.MISSED), (4, 104, 4, 0)], 1, (4, 1, 1, 0)),
    ],
    ids=["empty", "held", "missed", "stale", "resumed", "flagged"],
)
def test_flip_summary_counts_misses_and_off_cadence_frames_within_segments(rows, hold, want):
    got = po.flip_summary(flips(rows), hold)
    assert (got["flips"], got["segments"], got["missed"], got["out_of_cadence"]) == want


def test_a_late_flip_may_mis_show_every_frame_due_from_the_vblank_after_the_last_to_its_own():
    log = flips(
        [(0, 100, 0, po.SEGMENT), (1, 101, 0, 0), (2, 102, 1, 0), (5, 105, 1, po.MISSED), (6, 106, 3, 0)]
    )
    assert po.misses(log, 2).tolist() == [[1, 2]]
    assert po.misses(log[:3], 2).shape == (0, 2)


@pytest.mark.parametrize(
    "spans, want",
    [(None, (1, 1)), ([(0, 2)], (0, 0)), ([(2, 3)], (1, 0)), ([(3, 4)], (1, 0)), ([(5, 9)], (0, 1))],
    ids=["all", "before", "first-mis-shown", "skipped", "after"],
)
def test_flip_summary_judges_only_the_frames_its_spans_hold(spans, want):
    log = flips(
        [(0, 100, 0, po.SEGMENT), (1, 101, 1, 0), (3, 103, 2, po.MISSED), (4, 104, 4, 0), (5, 105, 4, 0)]
    )
    got = po.flip_summary(log, 1, spans)
    assert (got["missed"], got["out_of_cadence"]) == want


def test_flip_records_match_the_players_own_layout():
    fields = (3, 1003, 1, 123456789, po.SEGMENT)
    assert np.frombuffer(po.RECORD.pack(*fields), dtype=po.FLIP)[0].tolist() == fields


def test_the_player_program_shipped_is_the_module_run_on_the_pi():
    source = po.player_source().decode()
    assert source.startswith('"""Vsync-locked clip player') and "def main(" in source
    assert "import numpy" not in source and "from syncsummoner" not in source


def test_a_player_allows_a_whole_clip_to_go_up(monkeypatch):
    seen = []
    monkeypatch.setattr(po, "ssh_runner", lambda host, timeout: seen.append(timeout))
    po.ClipPlayer("pi@rig")
    po.Playout("pi@rig")
    assert seen[0] == po.UPLOAD_TIMEOUT >= 300.0
    assert seen[1] < seen[0], "a still frame does not need a clip's budget"


class ClipHost:
    """A Pi holding at most one clip, answering ``sha256sum`` of it; ``corrupt`` mangles what it stores."""

    def __init__(self, corrupt=False):
        self.held, self.corrupt, self.sent = None, corrupt, 0

    def __call__(self, command, data=None):
        if command.startswith("sha256sum"):
            return "" if self.held is None else f"{hashlib.sha256(self.held).hexdigest()}  clip.mkv"
        if "cat >" in command:
            self.sent += len(data)
            self.held = data[::-1] if self.corrupt else data
        return ""


def test_a_clip_the_pi_already_holds_is_not_sent_again(tmp_path):
    clip = tmp_path / "clip.mkv"
    clip.write_bytes(b"frames" * 1000)
    host = ClipHost()
    player = po.ClipPlayer(runner=host)
    assert player.upload(str(clip)) == 6000 and host.sent == 6000
    assert player.upload(str(clip)) == 0 and host.sent == 6000
    clip.write_bytes(b"other" * 10)
    assert player.upload(str(clip)) == 50 and host.sent == 6050


def test_a_clip_that_arrives_different_is_a_playout_error(tmp_path):
    clip = tmp_path / "clip.mkv"
    clip.write_bytes(b"abc")
    with pytest.raises(po.PlayoutError, match="hashes"):
        po.ClipPlayer(runner=ClipHost(corrupt=True)).upload(str(clip))
    assert po.file_digest(clip, chunk=1) == hashlib.sha256(b"abc").hexdigest()


def test_a_clip_player_plays_only_the_spans_asked_for():
    player = po.ClipPlayer(runner=FakeRunner())
    assert "--spans" not in player.command(Fraction(30))
    command = player.command(Fraction(30), [(10, 40), (20, 50)])
    assert command.split("--spans ")[1].split()[0] == "10:40,20:50"


def test_a_loop_player_refuses_spans():
    player = po.LoopPlayer(runner=FakeRunner())
    assert "--spans" not in player.command(Fraction(10), None)
    with pytest.raises(ValueError, match="plays no spans"):
        player.command(Fraction(10), [(0, 3)])

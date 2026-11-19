"""Playout: BGR565 packing, exact rates and flip-log judging, with no Pi and no network."""

# pylint: disable=missing-function-docstring

import hashlib
import io
import itertools
import os
import queue
import threading
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
    ],
    ids=["empty", "held", "missed", "stale", "resumed"],
)
def test_flip_summary_counts_misses_and_off_cadence_frames_within_segments(rows, hold, want):
    got = po.flip_summary(flips(rows), hold)
    assert (got["flips"], got["segments"], got["missed"], got["out_of_cadence"]) == want


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


def record(index):
    return po.RECORD.pack(index, index, index, 0, 0)


class Probe:
    """A raw pipe reader that reports how many bytes each read handed back."""

    def __init__(self, fd):
        self.raw, self.reads = os.fdopen(fd, "rb", buffering=0), queue.Queue()

    def readinto(self, buf):
        got = self.raw.readinto(buf)
        self.reads.put(got)
        return got

    def close(self):
        self.raw.close()


@pytest.fixture(name="feed")
def fixture_feed():
    """A Shown over a pipe, its write end, its reads and a clock counting from 100."""
    r, w = os.pipe()
    probe = Probe(r)
    shown = po.Shown(probe, clock=itertools.count(100).__next__)
    yield shown, w, probe.reads
    try:
        os.close(w)
    except OSError:
        pass
    shown.close()


def test_shown_reports_a_frame_only_once_its_whole_record_is_in(feed):
    shown, w, reads = feed
    assert shown.latest() == po.NONE
    os.write(w, record(7)[:13])
    assert reads.get() == 13 and shown.latest() == po.NONE
    os.write(w, record(7)[13:])
    assert reads.get() == po.RECORD.size - 13
    assert shown.wait(7) == 100 and shown.latest() == 7


def test_shown_logs_each_distinct_frame_when_first_heard_and_waits_for_one(feed):
    shown, w, _ = feed
    got = []
    waiter = threading.Thread(target=lambda: got.append(shown.wait(2)))
    waiter.start()
    for index in (0, 0, 1, 3):
        os.write(w, record(index))
    waiter.join()
    assert got == [103] and shown.wait(1) == 102 and shown.wait(0) == 100 and shown.latest() == 3
    assert shown.log().tolist() == [(0, 100, 0), (1, 102, 0), (3, 103, 0)] and shown.log().dtype == po.SHOWN


def test_shown_stops_waiting_when_asked_or_when_the_stream_ends(feed):
    shown, w, _ = feed
    stop, got = threading.Event(), []
    waiter = threading.Thread(target=lambda: got.append(shown.wait(5, stop)))
    waiter.start()
    os.write(w, record(4))
    stop.set()
    waiter.join()
    assert got == [None] and shown.wait(5, stop) is None
    os.close(w)
    assert shown.wait(5) is None and shown.latest() == 4


class FakeProc:
    """A streamed command whose stdout holds the given bytes."""

    def __init__(self, data=b""):
        self.stdout, self.stdin, self.done = io.BytesIO(data), io.BytesIO(), []

    def terminate(self):
        self.done.append("terminate")

    def wait(self):
        self.done.append("wait")


def test_a_player_follows_its_live_flips_over_the_stream_it_was_given():
    started = []
    procs = iter([FakeProc(record(0) + record(1)), FakeProc()])
    runner = FakeRunner()
    player = po.ClipPlayer(runner=runner, stream=lambda command: started.append(command) or next(procs))
    command = player.command(Fraction(30))
    assert command.split("--live ")[1].split()[0] == player.path("live.fifo")
    assert command.split("--ask ")[1].split()[0] == player.path("ask.fifo")
    with player.following(clock=itertools.count().__next__) as shown:
        assert shown.wait(1) == 1 and shown.wait(2) is None
    assert started == [f"exec cat {player.path('live.fifo')}", f"exec cat > {player.path('ask.fifo')}"]
    assert "mkfifo" in runner.calls[0][0], "its pipes are made before it is followed"


def test_a_reply_pairs_with_the_request_before_it_and_never_counts_as_a_flip():
    read, write = os.pipe()
    asks = io.BytesIO()
    with os.fdopen(read, "rb", buffering=0) as stream, os.fdopen(write, "wb", buffering=0) as pi:
        shown = po.Shown(stream, asks=asks, clock=itertools.count(10).__next__)
        pi.write(record(0) + record(1))
        assert shown.wait(1) is not None
        shown.ask()
        pi.write(po.RECORD.pack(1, 1, 1, 0, po.REPLY) + po.RECORD.pack(30, 12, 2, 0, po.SEGMENT))
        assert shown.wait(2) is not None
        pi.close()
        shown.close()
    assert asks.getvalue() == b"?"
    assert shown.log()["frame"].tolist() == [0, 1, 2]
    assert shown.log()["base"].tolist() == [0, 0, -18], "each record keeps its segment's base"
    asked = shown.asked()
    assert asked.size == 1 and asked["t0_ns"][0] < asked["t1_ns"][0] and asked["frame"][0] == 1
    assert asked["base"][0] == 0


def test_a_player_heard_flipping_returns_before_its_status_says_so():
    player = po.ClipPlayer(runner=lambda command, data=None: None, stream=lambda command: FakeProc())
    player.status = lambda: {"state": "starting"}
    player.pid = lambda: 42
    player.stop = lambda: {}
    heard = iter([False, True])
    assert player.start(fps=Fraction(30), heard=lambda: next(heard)) is None


def test_asking_a_player_that_stopped_reading_goes_quiet():
    class Gone:
        """A pipe whose reader has gone."""

        def write(self, data):
            raise BrokenPipeError(32, "Broken pipe")

        def flush(self):
            pass

    shown = po.Shown(io.BytesIO(), asks=Gone(), clock=itertools.count().__next__)
    shown.ask()
    shown.ask()
    shown.close()
    assert shown.asked().size == 0

"""The Pi-side KMS player's pacing and miss detection, against a simulated display."""

# pylint: disable=missing-function-docstring

import io
import json
import os
import shutil
import struct
import subprocess
import sys
import threading
import time
from fractions import Fraction
from types import SimpleNamespace

import pytest

from syncsummoner.device import kmsplay as K

PERIOD = Fraction(10**9, 30)


class FakeDisplay:
    """Vblanks every ``period_ns`` on a virtual clock; a flip lands on the vblank after its commit.

    ``late`` names clip vblanks whose commit misses its vblank and lands one later; ``lost``
    names ones whose event never arrives. While inactive no vblanks occur, and reactivating
    starts a grid ``phase_ns`` off the old one.
    """

    def __init__(self, *, period_ns=PERIOD, late=(), lost=(), phase_ns=0, seq0=1000):
        self.period = period_ns
        self.now = Fraction(0)
        self.grid = Fraction(0)
        self.seq_at_grid = seq0
        self.late, self.lost = set(late), set(lost)
        self.phase = phase_ns
        self.active = True
        self.pending = None
        self.shown = {}
        self.screen = None
        self.filled = {}
        self.started = None
        self.actives = []

    def clock(self):
        return int(self.now)

    def sleep(self, seconds):
        self.now += Fraction(seconds) * 10**9

    def fill(self, buf, data):
        self.filled[buf] = data

    def start(self, buf):
        self.started = buf
        self.screen = self.filled[buf]

    def set_active(self, active, buf):
        self.actives.append(active)
        if active and not self.active:
            self.grid = self.now + self.phase
        self.active = active
        self.screen = self.filled[buf]

    def flip(self, buf, v):
        assert self.pending is None, "a commit while one is pending is refused by KMS"
        self.pending = (buf, v, self.filled[buf])

    def wait(self, timeout_s):
        assert self.active and self.pending is not None
        buf, v, data = self.pending
        self.pending = None
        ticks = (self.now - self.grid) // self.period + 1
        if v in self.late:
            ticks += 1
        if v in self.lost:
            self.now += Fraction(timeout_s) * 10**9
            return None
        self.now = self.grid + ticks * self.period
        seq = self.seq_at_grid + ticks
        self.grid, self.seq_at_grid = self.now, seq
        self.screen = data
        self.shown[v] = (buf, data)
        return seq, int(self.now), v


def frames(count):
    return iter([struct.pack("<q", k) for k in range(count)])


class Prerolled:
    """A read-ahead with everything already decoded."""

    def __init__(self, source):
        self._inner = K.Prerolled(source, depth=10_000)
        self._inner.wait_full(5.0)

    def get(self, index):
        return self._inner.get(index)


def play(display, count, *, hold=1, stop_after=None, signals=None, at=None, period_ns=None):
    """Run a player over ``count`` frames, stopping after ``stop_after`` flips; returns it and its records."""
    records, signals = [], {} if signals is None else signals
    limit = count * hold if stop_after is None else stop_after

    def record(*fields):
        records.append(fields)
        for flips, (key, value) in (at or {}).items():
            if len(records) == flips:
                signals[key] = value
        if len(records) >= limit:
            signals["stop"] = True

    player = K.Player(
        display,
        Prerolled(frames(count)),
        hold=hold,
        period_ns=display.period if period_ns is None else period_ns,
        record=record,
        signals=signals,
        clock=display.clock,
        sleep=display.sleep,
    )
    player.run()
    return player, records


def shown(records):
    return [r[2] for r in records]


def test_each_frame_is_flipped_on_consecutive_vblanks():
    display = FakeDisplay()
    player, records = play(display, 20)
    assert shown(records) == list(range(20))
    assert [r[1] for r in records] == [1001 + k for k in range(20)]
    assert [r[0] for r in records] == list(range(20))
    assert player.flips == 20 and player.segments == 1
    assert records[0][4] == K.SEGMENT and not any(r[4] for r in records[1:])


def test_a_frame_is_held_for_exactly_hold_vblanks():
    display = FakeDisplay(period_ns=PERIOD / 2)
    _, records = play(display, 5, hold=2)
    assert shown(records) == [0, 0, 1, 1, 2, 2, 3, 3, 4, 4]
    assert all(b[1] - a[1] == 1 for a, b in zip(records, records[1:]))


def test_the_screen_shows_the_frame_that_was_due():
    display = FakeDisplay()
    play(display, 6)
    assert [struct.unpack("<q", display.shown[v][1])[0] for v in range(6)] == list(range(6))
    assert len({display.shown[v][0] for v in range(3)}) == 3, "three buffers in rotation"


def test_a_late_flip_is_a_fault_not_a_slip():
    with pytest.raises(K.PlayoutFault, match="vblank 7.*1 vblank.*missed"):
        play(FakeDisplay(late={7}), 20)


def test_a_flip_that_never_completes_is_a_fault():
    with pytest.raises(K.PlayoutFault, match="no flip completed"):
        play(FakeDisplay(lost={4}), 20)


def test_a_drifting_event_clock_is_a_fault():
    """A display clock off its nominal period walks off the grid and is caught."""
    display = FakeDisplay(period_ns=PERIOD * Fraction(101, 100))
    with pytest.raises(K.PlayoutFault, match="off a grid"):
        play(display, 200, period_ns=PERIOD)
    assert display.clock() < 100 * display.period


def test_a_clip_that_ends_holds_its_last_frame():
    display = FakeDisplay()
    signals = {}
    records = []
    player = K.Player(
        display,
        Prerolled(frames(4)),
        hold=1,
        period_ns=PERIOD,
        record=lambda *f: records.append(f),
        signals=signals,
        clock=display.clock,
        sleep=lambda s: (display.sleep(s), signals.__setitem__("stop", display.now > 10**9)),
    )
    player.run()
    assert player.ended and shown(records) == [0, 1, 2, 3]
    assert struct.unpack("<q", display.screen)[0] == 3


def test_an_empty_clip_is_a_fault():
    display = FakeDisplay()
    with pytest.raises(K.PlayoutFault, match="no frames"):
        K.Player(
            display,
            Prerolled(frames(0)),
            hold=1,
            period_ns=PERIOD,
            record=print,
            signals={},
            clock=display.clock,
        ).run()


def test_the_link_drop_keeps_the_timeline_running():
    """Frames due while the link is down are skipped; the next segment shows the frame due then."""
    display = FakeDisplay(phase_ns=PERIOD / 3)
    signals = {}
    downs = []

    def sleep(seconds):
        display.sleep(seconds)
        downs.append(display.clock())
        if len(downs) == 30:
            signals["up"] = True

    player = K.Player(
        display,
        Prerolled(frames(200)),
        hold=1,
        period_ns=PERIOD,
        record=lambda *f: (records.append(f), signals.update(stop=len(records) >= 60)),
        signals=signals,
        clock=display.clock,
        sleep=sleep,
    )
    records = []
    signals_at = {10: "down"}
    original = player.record

    def record(*fields):
        original(*fields)
        if len(records) in signals_at:
            signals[signals_at[len(records)]] = True

    player.record = record
    player.run()
    assert display.actives == [False, True]
    resumed = next(k for k, r in enumerate(records) if k and r[4] & K.SEGMENT)
    v, _, frame, t_ns, _ = records[resumed]
    assert (
        v == round((t_ns - records[0][3]) / PERIOD) and v > 30
    ), "the segment sits on the timeline, not where it paused"
    assert frame == 9, "the first flip after the link returns shows the frame left up"
    after = records[resumed + 1 :]
    assert [r[2] for r in after] == [r[0] for r in after], "and every later flip the frame due on it"
    assert player.segments == 2
    assert player.timeline.phases_ns[1] == pytest.approx(float(PERIOD / 3), abs=1)


def test_stop_during_the_link_drop_ends_the_player():
    display = FakeDisplay()
    signals = {}

    def sleep(seconds):
        display.sleep(seconds)
        signals["stop"] = True

    player = K.Player(
        display,
        Prerolled(frames(50)),
        hold=1,
        period_ns=PERIOD,
        record=lambda *f: signals.update(down=f[0] == 3),
        signals=signals,
        clock=display.clock,
        sleep=sleep,
    )
    player.run()
    assert display.actives == [False]


def test_link_requests_after_the_clip_ends_still_drive_the_link():
    display = FakeDisplay()
    signals = {}
    ticks = []

    def sleep(seconds):
        display.sleep(seconds)
        ticks.append(1)
        signals.update({1: {"down": True}, 2: {"up": True}, 3: {"stop": True}}.get(len(ticks), {}))

    K.Player(
        display,
        Prerolled(frames(3)),
        hold=1,
        period_ns=PERIOD,
        record=lambda *f: None,
        signals=signals,
        clock=display.clock,
        sleep=sleep,
    ).run()
    assert display.actives == [False, True]


def mode(w=1920, h=1080, clock=74250, htotal=2200, vtotal=1125, flags=0):
    return SimpleNamespace(hdisplay=w, vdisplay=h, clock=clock, htotal=htotal, vtotal=vtotal, flags=flags)


def test_refresh_is_exact():
    assert K.refresh(mode()) == 30
    assert K.refresh(mode(clock=74176, htotal=2200, vtotal=1125)) == Fraction(74176000, 2475000)


def test_the_current_mode_is_kept_when_it_divides():
    current = mode()
    picked, hold = K.pick_mode([current, mode(clock=148500)], 1920, 1080, Fraction(30))
    assert picked is current and hold == 1


def test_a_whole_multiple_is_held_for():
    picked, hold = K.pick_mode([mode(clock=148500)], 1920, 1080, Fraction(30))
    assert hold == 2 and K.refresh(picked) == 60


def test_no_dividing_mode_is_a_fault_naming_what_is_offered():
    with pytest.raises(K.PlayoutFault, match="12 fps; offered 30.0000"):
        K.pick_mode([None, mode(), mode(flags=K.INTERLACE), mode(w=1280, h=720)], 1920, 1080, Fraction(12))


def test_rates_parse_exactly():
    assert K.parse_rate("30000/1001") == Fraction(30000, 1001)
    assert K.parse_rate("30") == 30
    with pytest.raises(ValueError):
        K.parse_rate("0")


def test_prerolled_frames_come_in_order_and_skip_forward():
    pre = K.Prerolled(frames(5), depth=8)
    pre.wait_full(5.0)
    assert struct.unpack("<q", pre.get(0))[0] == 0
    assert struct.unpack("<q", pre.get(3))[0] == 3
    with pytest.raises(K.PlayoutFault, match="asked for after"):
        pre.get(2)


def test_prerolled_end_is_none():
    pre = K.Prerolled(frames(1), depth=4)
    pre.wait_full(5.0)
    assert pre.get(0) is not None and pre.get(1) is None and pre.ended and pre.get(2) is None


def test_an_underrun_is_a_fault():
    with pytest.raises(K.PlayoutFault, match="frame 0 was not decoded"):
        K.Prerolled(_Stall(), depth=2).get(0)


class _Stall:
    """A source that never yields."""

    def __iter__(self):
        return self

    def __next__(self):
        time.sleep(60)
        raise StopIteration


def test_a_failing_source_is_a_fault():
    def broken():
        yield b"x"
        raise OSError("disk gone")

    pre = K.Prerolled(broken(), depth=4)
    pre.wait_full(5.0)
    pre.get(0)
    with pytest.raises(K.PlayoutFault, match="disk gone"):
        pre.get(1)


def test_preroll_that_never_fills_is_a_fault():
    pre = K.Prerolled(_Stall(), depth=2)
    ticks = iter(range(100))
    with pytest.raises(K.PlayoutFault, match="preroll held 0 of 2"):
        pre.wait_full(3.0, clock=lambda: next(ticks))


def test_preroll_depth_must_be_positive():
    with pytest.raises(ValueError):
        K.Prerolled(iter(()), depth=0)


def test_read_exactly_reassembles_short_reads():
    class Trickle(io.RawIOBase):
        """A pipe handing back at most three bytes per read."""

        def __init__(self, data):
            self.data = data

        def readinto(self, b):
            n = min(3, len(b), len(self.data))
            b[:n], self.data = self.data[:n], self.data[n:]
            return n

    stream = Trickle(b"0123456789")
    assert K.read_exactly(stream, 4) == b"0123"
    assert K.read_exactly(stream, 4) == b"4567"
    assert K.read_exactly(stream, 4) is None


def test_decode_argv_keeps_every_frame_at_display_geometry():
    argv = K.decode_argv("c.mkv", 1280, 720, loop=True)
    assert "-re" not in argv and "-r" not in argv
    assert argv[argv.index("-stream_loop") + 1] == "-1"
    assert "scale=1280:720" in argv and "bgr565le" in argv
    assert "-stream_loop" not in K.decode_argv("c.mkv", 8, 8, loop=False)


def test_clip_frames_reads_whole_frames_and_reports_a_decoder_error():
    good = [sys.executable, "-c", "import sys; sys.stdout.buffer.write(b'abcdefgh')"]
    assert list(K.clip_frames(good, 4)) == [b"abcd", b"efgh"]
    bad = [sys.executable, "-c", "import sys; sys.stderr.write('bad input\\n'); sys.exit(1)"]
    with pytest.raises(K.PlayoutFault, match="bad input"):
        list(K.clip_frames(bad, 4))


def test_clip_rate_reads_the_container(monkeypatch):
    monkeypatch.setattr(
        K.subprocess, "run", lambda argv, **kw: subprocess.CompletedProcess(argv, 0, stdout="30000/1001\n")
    )
    assert K.clip_rate("x.mkv") == Fraction(30000, 1001)


def test_dir_frames_cycle_when_looping(tmp_path):
    for k in range(3):
        (tmp_path / f"frame.{k:03d}").write_bytes(bytes([k]))
    looped = K.dir_frames(str(tmp_path), loop=True)
    assert [next(looped) for _ in range(7)] == [bytes([k % 3]) for k in range(7)]
    assert list(K.dir_frames(str(tmp_path), loop=False)) == [b"\x00", b"\x01", b"\x02"]
    assert not list(K.dir_frames(str(tmp_path / "none"), loop=True))


def test_status_is_replaced_whole(tmp_path):
    path = tmp_path / "s.json"
    K.write_status(str(path), {"state": "playing"})
    assert json.loads(path.read_text()) == {"state": "playing"}
    assert not (tmp_path / "s.json.tmp").exists()


def test_arguments_name_one_source():
    args = K.parse_args(["--clip", "c.mkv", "--fps", "30", "--flips", "f", "--status", "s", "--loop"])
    assert args.fps == 30 and args.loop and args.frames is None
    with pytest.raises(SystemExit):
        K.parse_args(["--clip", "c", "--frames", "d", "--fps", "30", "--flips", "f", "--status", "s"])
    assert args.live is None
    assert (
        K.parse_args(["--frames", "d", "--fps", "30", "--flips", "f", "--status", "s", "--live", "p"]).live
        == "p"
    )


def packed(*indices):
    return [K.RECORD.pack(f, f, f, f, 0) for f in indices]


def drain(fd):
    """Everything a non-blocking reader can take now."""
    out = b""
    while True:
        try:
            more = os.read(fd, 1 << 16)
        except BlockingIOError:
            return out
        if not more:
            return out
        out += more


def reader(path):
    return os.open(path, os.O_RDONLY | os.O_NONBLOCK)


def test_the_live_feed_drops_records_until_a_reader_comes_then_delivers_them_whole(tmp_path):
    path = str(tmp_path / "live")
    feed = K.LiveFeed(path)
    for rec in packed(0, 1):
        feed.write(rec)
    fd = reader(path)
    sent = packed(2, 3, 4)
    for rec in sent:
        feed.write(rec)
    assert drain(fd) == b"".join(sent)
    feed.close()
    assert drain(fd) == b""
    os.close(fd)


def test_a_full_live_pipe_drops_whole_records_instead_of_blocking(tmp_path):
    path = str(tmp_path / "live")
    feed = K.LiveFeed(path)
    fd = reader(path)
    sent = packed(*range(4 * (1 << 16) // K.RECORD.size))
    for rec in sent:
        feed.write(rec)
    got = drain(fd)
    assert 0 < len(got) < len(b"".join(sent)) and len(got) % K.RECORD.size == 0
    assert got == b"".join(sent)[: len(got)]
    feed.write(sent[0])
    assert drain(fd) == sent[0]
    feed.close()
    os.close(fd)


def test_the_live_feed_waits_for_a_new_reader_once_one_leaves(tmp_path):
    path = str(tmp_path / "live")
    feed = K.LiveFeed(path)
    fd = reader(path)
    feed.write(packed(0)[0])
    os.close(fd)
    feed.write(packed(1)[0])
    fd = reader(path)
    feed.write(packed(2)[0])
    assert drain(fd) == packed(2)[0]
    feed.close()
    os.close(fd)


def test_a_fresh_live_feed_leaves_a_reader_of_the_old_one_with_nothing(tmp_path):
    path = str(tmp_path / "live")
    old = K.LiveFeed(path)
    stale = reader(path)
    old.write(packed(0)[0])
    assert drain(stale) == packed(0)[0]
    new = K.LiveFeed(path)
    assert os.stat(path).st_ino != os.fstat(stale).st_ino
    fresh = reader(path)
    new.write(packed(1)[0])
    old.close()
    assert drain(stale) == b"" and drain(fresh) == packed(1)[0]
    new.close()
    os.close(stale)
    os.close(fresh)


@pytest.mark.parametrize("refused", [None, PermissionError, OSError])
def test_realtime_asks_for_fifo_at_the_pacing_priority_and_reports_a_refusal(monkeypatch, refused):
    asked = []

    def setscheduler(pid, policy, param):
        asked.append((pid, policy, param.sched_priority))
        if refused:
            raise refused

    monkeypatch.setattr(K.os, "sched_setscheduler", setscheduler)
    assert K.realtime() is (refused is None)
    assert asked == [(0, K.os.SCHED_FIFO, K.REALTIME_PRIORITY)]


def test_a_span_is_sought_half_a_frame_early_and_trimmed_to_its_length_before_it_is_scaled():
    argv = K.decode_argv("c.mkv", 8, 6, loop=False, span=(3, 9), rate=Fraction(30))
    assert argv[argv.index("-vf") + 1] == "trim=end_frame=6,scale=8:6"
    assert argv[argv.index("-ss") + 1] == "0.083333" and argv.index("-ss") < argv.index("-i")
    assert "-ss" not in K.decode_argv("c.mkv", 8, 6, loop=False, span=(0, 9), rate=Fraction(30))


@pytest.mark.skipif(shutil.which("ffmpeg") is None, reason="needs ffmpeg")
def test_a_sought_span_decodes_the_frames_a_full_decode_trims_to(tmp_path):
    clip = str(tmp_path / "c.mkv")
    src = "testsrc2=size=64x48:rate=30"
    encode = ["ffmpeg", "-loglevel", "error", "-f", "lavfi", "-i", src, "-frames:v", "90"]
    subprocess.run(encode + ["-c:v", "libx264", "-g", "25", "-pix_fmt", "yuv420p", clip], check=True)
    whole = b"".join(K.clip_frames(K.decode_argv(clip, 64, 48, loop=False), 64 * 48 * 2))
    size = 64 * 48 * 2
    for a, b in ((0, 7), (1, 4), (26, 40), (52, 90)):
        got = b"".join(K.span_frames(clip, 64, 48, [(a, b)], size, Fraction(30)))
        assert got == whole[a * size : b * size], (a, b)


def test_spans_parse_as_ranges_and_refuse_empty_or_backward_ones():
    assert K.parse_spans("0:5,2:9") == [(0, 5), (2, 9)]
    for bad in ("", "5:5", "3:1", "-1:4", "1:2:3"):
        with pytest.raises(ValueError):
            K.parse_spans(bad)


def test_spans_play_as_one_stream_each_next_decoder_started_before_it_is_due(monkeypatch):
    started, procs = [], []
    real = K._decoder  # pylint: disable=protected-access

    def decoder(argv):
        started.append(argv[-1])
        procs.append(real(argv))
        return procs[-1]

    def argv_for(path, width, height, *, loop, span, rate):
        del path, width, height, loop, rate
        data = bytes(range(span[0], span[1]))
        return [sys.executable, "-c", f"import sys; sys.stdout.buffer.write({data!r})", f"{span[0]}"]

    monkeypatch.setattr(K, "_decoder", decoder)
    monkeypatch.setattr(K, "decode_argv", argv_for)
    stream = K.span_frames("c.mkv", 1, 1, [(0, 2), (5, 7), (1, 2)], 1, Fraction(30))
    assert next(stream) == b"\x00" and started == ["0", "5"], "the next span decodes while this one plays"
    assert list(stream) == [b"\x01", b"\x05", b"\x06", b"\x01"] and started == ["0", "5", "1"]
    procs.clear()
    closed = K.span_frames("c.mkv", 1, 1, [(0, 2), (5, 7)], 1, Fraction(30))
    next(closed)
    closed.close()
    assert len(procs) == 2 and all(p.poll() is not None for p in procs), "closing early reaps both decoders"


def test_a_decoder_chattering_on_stderr_never_stalls_its_frames():
    chatter = (
        "import sys; [sys.stderr.write('warning %d\\n' % i) for i in range(20000)]; "
        "sys.stdout.buffer.write(b'abcdef')"
    )
    got, done = [], threading.Event()

    def read():
        got.extend(K.clip_frames([sys.executable, "-c", chatter], 2))
        done.set()

    threading.Thread(target=read, daemon=True).start()
    assert done.wait(30), "the decoder blocked on its own stderr"
    assert got == [b"ab", b"cd", b"ef"]


def test_a_failing_decoder_faults_with_its_last_line():
    argv = [sys.executable, "-c", "import sys; sys.stderr.write('first\\nlast words\\n\\n'); sys.exit(3)"]
    with pytest.raises(K.PlayoutFault, match="decoding failed: last words$"):
        list(K.clip_frames(argv, 2))


def test_each_request_is_answered_with_the_last_flip_flagged_as_a_reply():
    sent = []
    live = SimpleNamespace(write=sent.append)
    last = [None, K.RECORD.pack(4, 4, 4, 99, K.SEGMENT)]
    K.answer(io.BytesIO(b"??"), lambda: last.pop(0), live)
    assert len(sent) == 1, "nothing to answer before the first flip"
    assert K.RECORD.unpack(sent[0]) == (4, 4, 4, 99, K.SEGMENT | K.REPLY)


def test_each_flip_is_logged_offered_live_and_kept_for_replies():
    log, offered = io.BytesIO(), []
    record = K.recorder(log, SimpleNamespace(write=offered.append), None)
    record(1, 2, 3, 4, 0)
    record(2, 3, 4, 5, 0)
    assert log.getvalue() == K.RECORD.pack(1, 2, 3, 4, 0) + K.RECORD.pack(2, 3, 4, 5, 0) == b"".join(offered)
    assert K.recorder(io.BytesIO(), None, None) is not None

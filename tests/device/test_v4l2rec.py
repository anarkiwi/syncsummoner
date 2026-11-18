"""The V4L2 recorder's frame log, encoder pipe and command line, with no capture card."""

# pylint: disable=missing-function-docstring

import hashlib
import io
import shutil
import subprocess
import threading
from fractions import Fraction
from types import SimpleNamespace

import numpy as np
import pytest

from syncsummoner.device import v4l2rec as vr


def capture(sequence, secs, usecs, data, flags=0):
    """A dequeued frame as linuxpy hands it over."""
    buff = SimpleNamespace(
        sequence=sequence,
        timestamp=SimpleNamespace(secs=secs, usecs=usecs),
        bytesused=len(data),
        flags=flags,
    )
    return SimpleNamespace(buff=buff, data=data)


def digest(data):
    return int.from_bytes(hashlib.blake2b(data, digest_size=8).digest(), "little")


def ivf(data):
    """The ``(pts, bytes)`` of every frame in an IVF stream, after checking its header."""
    assert data[:4] == b"DKIF"
    at, found = vr.IVF.size, []
    while at < len(data):
        size, pts = vr.IVF_FRAME.unpack_from(data, at)
        at += vr.IVF_FRAME.size
        found.append((pts, data[at : at + size]))
        at += size
    return found


def rows(data):
    """A deduplicated sidecar's rows from its bytes."""
    assert data[: len(vr.MAGIC)] == vr.MAGIC
    return np.frombuffer(data[vr.HEADER.size :], dtype=vr.STORED)


def test_record_stores_frames_untouched_and_logs_each_ones_sequence_and_time(tmp_path):
    frames = [
        capture(5, 100, 250, b"ab"),
        capture(6, 100, 33_583, b"cde"),
        capture(9, 101, 0, b"f", vr.FLAG_ERROR),
    ]
    sink = io.BytesIO()
    with open(vr.sidecar(tmp_path / "take.mkv"), "wb") as log:
        assert vr.record(frames, sink, log, stop=lambda: False) == 3
    assert ivf(sink.getvalue()) == [(0, b"ab"), (33, b"cde"), (67, b"f")]
    got = vr.read_log(tmp_path / "take.mkv")
    assert got.tolist() == [
        (5, 100_000_250_000, 2, 0, digest(b"ab")),
        (6, 100_033_583_000, 3, 0, digest(b"cde")),
        (9, 101_000_000_000, 1, vr.FLAG_ERROR, digest(b"f")),
    ]
    assert np.count_nonzero(np.diff(got["sequence"]) != 1) == 1, "a dropped frame is a gap in the sequence"


@pytest.mark.parametrize(
    ("limit", "stop_at", "kept"), [(2, None, 2), (None, 3, 3), (0, None, 0), (None, None, 5)]
)
def test_record_ends_at_its_limit_or_when_asked(limit, stop_at, kept):
    pulled = []

    def frames():
        for k in range(5):
            pulled.append(k)
            yield capture(k, 0, k, bytes([k]))

    sink, log = io.BytesIO(), io.BytesIO()
    assert vr.record(frames(), sink, log, stop=lambda: len(pulled) > (stop_at or 99), limit=limit) == kept
    assert (
        [data for _, data in ivf(sink.getvalue())] == [bytes([k]) for k in range(kept)]
        if kept
        else not sink.getvalue()
    )
    assert len(log.getvalue()) == (vr.HEADER.size + kept * vr.STORED.itemsize if kept else 0)


def test_the_sidecar_sits_beside_the_take_and_a_torn_tail_is_dropped(tmp_path):
    take = tmp_path / "take.mkv"
    assert vr.sidecar(take) == tmp_path / "take.mkv.v4l2" == vr.sidecar(str(take))
    logged = np.array([(1, 2, 3, 4, 5), (6, 7, 8, 9, 10)], dtype=vr.FRAME)
    vr.sidecar(take).write_bytes(logged.tobytes() + b"\x01\x02\x03")
    assert vr.read_log(take).tolist() == logged.tolist()


def test_a_deduplicated_sidecar_reads_as_the_frame_log_with_its_layout_and_a_torn_tail_dropped(tmp_path):
    take = tmp_path / "take.mkv"
    layout = vr.Layout(vr.FOURCC["yuyv422"], 64, 32, Fraction(1, 90_000))
    stored = np.array([(1, 2, 3, 4, 5, 0, 0), (6, 7, 8, 9, 5, 0, 3000)], dtype=vr.STORED)
    vr.sidecar(take).write_bytes(layout.sidecar_header() + stored.tobytes() + b"\x01")
    got, found = vr.read_sidecar(take)
    assert found == layout and got.tolist() == stored.tolist()
    assert vr.read_log(take).dtype == vr.FRAME and vr.read_log(take).tolist() == [
        r[:5] for r in stored.tolist()
    ]
    vr.sidecar(take).write_bytes(layout.sidecar_header()[:7] + b"\x09" + layout.sidecar_header()[8:])
    with pytest.raises(ValueError, match="version 9"):
        vr.read_sidecar(take)


@pytest.mark.parametrize(
    ("rate", "first"), [(Fraction(60), [0, 17, 33, 50, 67]), (vr.card_rate(29.97), [0, 33, 67, 100, 133])]
)
def test_timestamps_are_the_captures_at_the_cards_rate_to_the_millisecond(rate, first):
    assert vr.timestamp(np.arange(5), rate).tolist() == first
    assert vr.card_rate(59.94) == Fraction(2997, 50) and vr.timestamp(3, Fraction(60)) == 50


def test_encoder_argv_reads_timestamped_frames_from_the_pipe():
    argv = vr.encoder_argv(["-c:v", "copy"], "out.mkv", ffmpeg="ff")
    assert argv[0] == "ff" and argv[argv.index("-i") - 2 : argv.index("-i") + 2] == [
        "-f",
        "ivf",
        "-i",
        "pipe:0",
    ]
    assert argv[-3:] == ["-c:v", "copy", "out.mkv"]


def test_a_repeated_capture_is_logged_against_the_frame_already_stored():
    """Equal digests alone store nothing new only when the bytes match too."""
    frames = [capture(k, 0, k, data) for k, data in enumerate([b"a", b"a", b"b", b"b", b"b", b"a"])]
    sink, log = io.BytesIO(), io.BytesIO()
    vr.record(frames, sink, log, stop=lambda: False, rate=Fraction(60))
    assert ivf(sink.getvalue()) == [(0, b"a"), (33, b"b"), (83, b"a")]
    got = rows(log.getvalue())
    assert got["stored"].tolist() == [0, 0, 1, 1, 1, 2] and got["pts"].tolist() == [0, 17, 33, 50, 67, 83]
    out = vr.Output(io.BytesIO(), io.BytesIO())
    out.write(b"x", (0, 0, 1, 0), 7, vr.Layout(), Fraction(30))
    out.write(b"y", (1, 0, 1, 0), 7, vr.Layout(), Fraction(30))
    assert [d for _, d in ivf(out.sink.getvalue())] == [b"x", b"y"], "a digest collision is not a repeat"


@pytest.mark.skipif(shutil.which("ffmpeg") is None, reason="needs ffmpeg")
def test_recorded_frames_reach_the_file_once_per_run(tmp_path):
    width, height = 16, 8
    shades = [0, 1, 1, 2, 3, 3]
    frames = [
        capture(k, 10, k * 33_333, bytes([16 * s]) * (width * height * 2)) for k, s in enumerate(shades)
    ]
    out = tmp_path / "raw.mkv"
    argv = vr.encoder_argv(["-c:v", "rawvideo"], str(out), ffmpeg="ffmpeg")
    layout = vr.Layout(vr.FOURCC["yuyv422"], width, height)
    with subprocess.Popen(argv, stdin=subprocess.PIPE) as enc, open(vr.sidecar(out), "wb") as log:
        vr.record(frames, enc.stdin, log, stop=lambda: False, layout=layout)
        enc.stdin.close()
        assert enc.wait() == 0
    probe = [
        "ffprobe",
        "-v",
        "error",
        "-count_frames",
        "-show_entries",
        "stream=nb_read_frames",
        "-of",
        "csv=p=0",
    ]
    assert int(subprocess.run([*probe, str(out)], capture_output=True, text=True, check=True).stdout) == 4
    assert len(vr.read_log(out)) == len(shades) and vr.read_sidecar(out)[0]["stored"].tolist() == [
        0,
        1,
        1,
        2,
        3,
        3,
    ]


@pytest.mark.parametrize(
    ("seconds", "rate", "frames"), [(None, 30.0, None), (2.0, 30.0, 60), (1.0, 29.97, 30), (0.5, 60.0, 30)]
)
def test_frame_limit_counts_at_the_cards_actual_rate(seconds, rate, frames):
    assert vr.frame_limit(seconds, rate) == frames


@pytest.mark.parametrize(
    ("typed", "asked"), [(b"xy\nq", True), ("q", True), (b"xyq", False), (b"xyz", False), (b"", False)]
)
def test_q_starting_a_line_asks_for_a_stop_and_a_closed_stdin_does_not(typed, asked):
    flag = threading.Event()
    vr.watch_stdin(io.BytesIO(typed) if isinstance(typed, bytes) else io.StringIO(typed), flag)
    assert flag.is_set() is asked


def test_arguments_default_to_the_session_card_and_pass_ffmpegs_through():
    args = vr.parse_args(["take.mkv"])
    assert (args.device, args.width, args.height, args.fps, args.input_format) == (
        "/dev/video0",
        1920,
        1080,
        30.0,
        "mjpeg",
    )
    assert args.seconds is None and args.out_args == []
    args = vr.parse_args(["--seconds", "3", "take.mkv", "--", "-c:v", "ffv1", "-level", "3"])
    assert args.seconds == 3.0 and args.out_args == ["-c:v", "ffv1", "-level", "3"]


def test_a_stalled_sink_does_not_stop_frames_being_dequeued():
    gate, pulled = threading.Event(), []

    class Stalled:
        """A sink that blocks until every frame has been dequeued."""

        def __init__(self):
            self.got = []

        def write(self, data):
            gate.wait()
            self.got.append(data)

    def frames():
        for k in range(5):
            pulled.append(k)
            yield capture(k, 10, k * 33_333, bytes([k]) * 4)
        gate.set()

    sink, log = Stalled(), io.BytesIO()
    assert vr.record(frames(), sink, log, stop=lambda: False, backlog=8) == 5
    assert pulled == list(range(5))
    assert [d for _, d in ivf(b"".join(sink.got))] == [bytes([k]) * 4 for k in range(5)]
    assert rows(log.getvalue())["sequence"].tolist() == list(range(5))


def test_a_failed_write_is_raised_and_stops_the_recording():
    class Broken:
        """A sink whose encoder has gone."""

        def write(self, data):
            raise BrokenPipeError("encoder gone")

    frames = (capture(k, 10, k, b"x") for k in range(1000))
    with pytest.raises(BrokenPipeError):
        vr.record(frames, Broken(), io.BytesIO(), stop=lambda: False, backlog=2)


def test_a_roll_splits_the_captures_between_files_sharing_those_either_side_of_it():
    """Captures from the roll's opening go to the new file, those before its closing to the old one too."""
    logs, closed = {}, []

    def opener(path):
        logs[path] = (io.BytesIO(), io.BytesIO())
        return vr.Output(*logs[path], close=lambda: closed.append(path))

    outputs = vr.Outputs(opener("a.mkv"), opener)
    rolled = threading.Event()

    def frames():
        for k in range(8):
            if k == 2:
                outputs.roll(4_000, 6_000, "b.mkv")
                rolled.set()
            yield capture(k, 0, k, bytes([k]))

    assert vr.record(frames(), None, None, stop=lambda: False, outputs=outputs) == 8 and rolled.is_set()
    seq = {p: rows(log.getvalue())["sequence"].tolist() for p, (_, log) in logs.items()}
    assert seq == {"a.mkv": [0, 1, 2, 3, 4, 5], "b.mkv": [4, 5, 6, 7]}
    assert ivf(logs["b.mkv"][0].getvalue()) == [
        (0, bytes([4])),
        (33, bytes([5])),
        (67, bytes([6])),
        (100, bytes([7])),
    ]
    assert rows(logs["b.mkv"][1].getvalue())["stored"].tolist() == [0, 1, 2, 3], "each file numbers its own"
    assert closed == ["a.mkv"], "the old file is closed once the captures move past it"


def test_a_recording_with_one_file_cannot_roll():
    with pytest.raises(ValueError, match="cannot roll"):
        vr.Outputs(vr.Output(io.BytesIO(), io.BytesIO())).roll(0, 1, "b.mkv")


def test_a_roll_line_on_stdin_reaches_the_outputs_and_a_q_inside_a_path_stops_nothing():
    flag, rolls = threading.Event(), []
    vr.watch_stdin(io.BytesIO(b"roll 5 9 /q/take.mkv\nbogus\nq"), flag, lambda *a: rolls.append(a))
    assert rolls == [(5, 9, "/q/take.mkv")] and flag.is_set()

"""Native frame archive: streaming FFV1 encode, keyed commit, bit-exact random access."""

import json
import shutil
import subprocess
from types import SimpleNamespace

import numpy as np
import pytest

from syncsummoner.probe import fit
from syncsummoner.probe.archive import CHANNELS, ArchiveError, FrameArchive, FrameRow
from syncsummoner.probe.store import KeyKind, ProgramKey
from tests import MEASURED_YVYU, native_frame

CLOCK = 1_700_000_000.0
SIZE = (24, 32)
KEY = ProgramKey("bitcrush", "deadbeef", KeyKind.BINARY)
OTHER = ProgramKey("bitcrush", "feedface", KeyKind.BINARY)


def params(seed):
    """A raw 12-value parameter vector."""
    return tuple(int((seed * 37 + i * 11) % 1024) for i in range(12))


def frames(count, size=SIZE, seed=0):
    """Synthetic packed 4:2:2 frames with both structure and entropy."""
    rng = np.random.default_rng(seed)
    ramp = np.linspace(0, 255, size[1], dtype=np.float64)[None, :, None]
    return [
        np.clip(ramp + rng.integers(0, 64, size + (CHANNELS,)) + 3 * i, 0, 255).astype(np.uint8)
        for i in range(count)
    ]


class FakePipe:
    """Encoder stdin recording one chunk per frame, optionally breaking early."""

    def __init__(self, proc):
        self.proc = proc
        self.closed = False

    def write(self, data):
        """Record the frame, or fail as a dead encoder's pipe does."""
        if self.proc.fail_at is not None and len(self.proc.chunks) >= self.proc.fail_at:
            raise BrokenPipeError("encoder gone")
        self.proc.chunks.append(bytes(data))

    def close(self):
        """Signal end of stream."""
        self.closed = True


class FakeEncoder:
    """``Popen`` stand-in that writes the piped bytes out as the archive's video."""

    def __init__(self, argv, *, stderr=None, returncode=0, fail_at=None):
        self.argv = list(argv)
        self.stderr = stderr
        self.fail_at = fail_at
        self.chunks = []
        self.stdin = FakePipe(self)
        self.returncode = None
        self.waits = 0
        self._exit = returncode

    def wait(self):
        """Reap, publishing the concatenated frames when the encode succeeded."""
        self.waits += 1
        self.returncode = self._exit
        if not self._exit:
            with open(self.argv[-1], "wb") as handle:
                handle.write(b"".join(self.chunks))
        return self._exit


class FakeFfmpeg:
    """Popen factory recording every encoder launched."""

    def __init__(self, *, returncode=0, fail_at=None, message=b""):
        self.returncode = returncode
        self.fail_at = fail_at
        self.message = message
        self.procs = []

    def __call__(self, argv, *, stdin=None, stdout=None, stderr=None):
        """Launch a fake encoder, leaving any diagnostic in the caller's log file."""
        del stdin, stdout
        if self.message and stderr is not None:
            stderr.write(self.message)
        proc = FakeEncoder(argv, stderr=stderr, returncode=self.returncode, fail_at=self.fail_at)
        self.procs.append(proc)
        return proc


class FakeDecoder:
    """``subprocess.run`` stand-in slicing the fake archive's concatenated frames."""

    def __init__(self, fps, size, *, returncode=0, truncate=False):
        self.fps = fps
        self.nbytes = size[0] * size[1] * CHANNELS
        self.returncode = returncode
        self.truncate = truncate
        self.calls = []

    def __call__(self, argv, capture_output=False, check=False):
        """Return the frame the seek offset names."""
        del capture_output, check
        offset = float(argv[argv.index("-ss") + 1])
        index = round(offset * self.fps + 0.5)
        self.calls.append(index)
        with open(argv[argv.index("-i") + 1], "rb") as handle:
            data = handle.read()[index * self.nbytes : (index + 1) * self.nbytes]
        return subprocess.CompletedProcess(
            argv, self.returncode, stdout=data[:-1] if self.truncate else data, stderr=b""
        )


def archive_at(tmp_path, **kwargs):
    """Archive rooted in a temp directory with a frozen clock."""
    return FrameArchive(tmp_path / "raw", clock=lambda: CLOCK, **kwargs)


def write_archive(tmp_path, count=4, *, program="bitcrush", key=KEY, **kwargs):
    """Archive ``count`` synthetic frames through a fake encoder."""
    ffmpeg = FakeFfmpeg(**kwargs)
    archive = archive_at(tmp_path, popen=ffmpeg, run=FakeDecoder(25, SIZE))
    made = frames(count)
    with archive.writer(program, key, width=SIZE[1], height=SIZE[0]) as writer:
        for i, frame in enumerate(made):
            index = writer.write(frame, params=params(i), setpoint=i // 2, loop_index=i % 3)
            assert index == i
            assert len(ffmpeg.procs[0].chunks) == i + 1
    return archive, ffmpeg, made


def test_frames_stream_out_as_they_arrive(tmp_path):
    """Nothing is buffered: each frame reaches the encoder before the next call returns."""
    _, ffmpeg, made = write_archive(tmp_path, 5)
    assert [bytes(f.data) for f in made] == ffmpeg.procs[0].chunks
    assert ffmpeg.procs[0].stdin.closed and ffmpeg.procs[0].waits == 1


def test_encoder_is_asked_for_lossless_intra_ffv1(tmp_path):
    """The encode is FFV1 in Matroska, fed and stored as the card's own 4:2:2 samples."""
    _, ffmpeg, _ = write_archive(tmp_path, 2)
    argv = ffmpeg.procs[0].argv
    assert argv[argv.index("-c:v") + 1] == "ffv1"
    assert argv[argv.index("-f") + 1] == "rawvideo"
    assert argv[-2] == "matroska"
    assert argv.count("-pix_fmt") == 2 and argv[-4] == "yuv422p"
    assert "yvyu422" in argv and f"{SIZE[1]}x{SIZE[0]}" in argv
    assert argv[argv.index("-i") + 1] == "pipe:0"


def test_commit_publishes_video_sidecar_and_metadata(tmp_path):
    """A clean exit leaves all three artefacts and no temp file."""
    archive, _, made = write_archive(tmp_path, 4)
    video, data, meta_path = archive.paths("bitcrush")
    assert video.exists() and data.exists() and meta_path.exists()
    assert video.stat().st_size == sum(f.nbytes for f in made)
    assert sorted(p.name for p in archive.directory.iterdir()) == sorted(
        (data.name, meta_path.name, video.name)
    )


def test_metadata_is_self_describing(tmp_path):
    """Metadata names the key, the geometry, the codec and the yield."""
    archive, _, made = write_archive(tmp_path, 4)
    meta = json.loads(archive.paths("bitcrush")[2].read_text(encoding="utf-8"))
    assert meta["program"] == "bitcrush"
    assert (meta["key_kind"], meta["key_material"], meta["key_digest"]) == (
        "binary",
        "deadbeef",
        KEY.digest,
    )
    assert (meta["codec"], meta["container"]) == ("ffv1", "matroska")
    assert (meta["pix_fmt"], meta["stored_pix_fmt"], meta["channels"]) == ("yvyu422", "yuv422p", 2)
    assert (meta["width"], meta["height"], meta["fps"]) == (SIZE[1], SIZE[0], 25)
    assert meta["frames"] == 4
    assert meta["bytes_per_frame"] == pytest.approx(made[0].nbytes)
    assert meta["video"].endswith(".mkv") and meta["data"].endswith(".parquet")
    assert meta["created"].startswith("2023-11-14T")


def test_sidecar_rows_carry_one_row_per_frame(tmp_path):
    """Every archived frame gets its index, program, vector, setpoint, loop and time."""
    archive, _, _ = write_archive(tmp_path, 4)
    rows = archive.rows("bitcrush")
    assert [r.frame for r in rows] == [0, 1, 2, 3]
    assert [r.setpoint for r in rows] == [0, 0, 1, 1]
    assert [r.loop_index for r in rows] == [0, 1, 2, 0]
    assert [r.params for r in rows] == [params(i) for i in range(4)]
    assert {r.program for r in rows} == {"bitcrush"}
    assert {r.captured for r in rows} == {CLOCK}


def test_sidecar_is_exactly_what_the_parquet_helpers_read(tmp_path):
    """The sidecar reuses the project's measurement helpers, not a new format."""
    archive, _, _ = write_archive(tmp_path, 3)
    flat = fit.load_measurements(archive.paths("bitcrush")[1])
    assert len(flat) == 3
    assert flat[0]["p1"] == params(0)[0] and flat[0]["frame"] == 0
    assert FrameRow.from_row(flat[2]).params == params(2)


def test_an_undecoded_loop_index_stays_null(tmp_path):
    """A codeframe that did not decode is recorded as unknown, not as zero."""
    ffmpeg = FakeFfmpeg()
    archive = archive_at(tmp_path, popen=ffmpeg)
    with archive.writer("bitcrush", KEY, width=SIZE[1], height=SIZE[0]) as writer:
        writer.write(frames(1)[0], params=params(0), setpoint=0)
        writer.write(frames(1)[0], params=params(1), setpoint=0, loop_index=5, captured=CLOCK + 1)
    rows = archive.rows("bitcrush")
    assert [r.loop_index for r in rows] == [None, 5]
    assert [r.captured for r in rows] == [CLOCK, CLOCK + 1]


def test_a_failed_encode_raises_and_commits_nothing(tmp_path):
    """A nonzero ffmpeg exit is an error, and leaves no archive behind."""
    archive = archive_at(tmp_path, popen=FakeFfmpeg(returncode=1, message=b"ffv1 broke"))
    with pytest.raises(ArchiveError, match="exited 1: ffv1 broke"):
        with archive.writer("bitcrush", KEY, width=SIZE[1], height=SIZE[0]) as writer:
            writer.write(frames(1)[0], params=params(0), setpoint=0)
    assert archive.meta("bitcrush") is None
    assert not list(archive.directory.iterdir())


def test_a_dead_encoder_is_reported_not_silently_dropped(tmp_path):
    """Frames refused by a broken pipe raise rather than vanishing."""
    archive = archive_at(tmp_path, popen=FakeFfmpeg(returncode=1, fail_at=2))
    with pytest.raises(ArchiveError, match="after 2"):
        with archive.writer("bitcrush", KEY, width=SIZE[1], height=SIZE[0]) as writer:
            for i, frame in enumerate(frames(4)):
                writer.write(frame, params=params(i), setpoint=0)
    assert archive.meta("bitcrush") is None


def test_an_exception_in_the_body_reaps_the_encoder_without_committing(tmp_path):
    """A caller that dies mid-sweep leaves no half-archive and no orphan process."""
    ffmpeg = FakeFfmpeg()
    archive = archive_at(tmp_path, popen=ffmpeg)
    with pytest.raises(KeyboardInterrupt):
        with archive.writer("bitcrush", KEY, width=SIZE[1], height=SIZE[0]) as writer:
            writer.write(frames(1)[0], params=params(0), setpoint=0)
            raise KeyboardInterrupt
    assert ffmpeg.procs[0].waits == 1 and ffmpeg.procs[0].stdin.closed
    assert archive.meta("bitcrush") is None
    assert not list(archive.directory.iterdir())


def test_an_empty_archive_is_not_a_commit(tmp_path):
    """A sweep that captured nothing stays unarchived."""
    archive = archive_at(tmp_path, popen=FakeFfmpeg())
    with pytest.raises(ArchiveError, match="no frames"):
        with archive.writer("bitcrush", KEY, width=SIZE[1], height=SIZE[0]):
            pass
    assert archive.meta("bitcrush") is None


@pytest.mark.parametrize(
    "frame,match",
    [
        (np.zeros(SIZE + (CHANNELS,), np.float32), "uint8"),
        (np.zeros((8, 8, CHANNELS), np.uint8), "does not match"),
        (np.zeros(SIZE + (3,), np.uint8), "does not match"),
        ([[0]], "uint8"),
    ],
    ids=["float", "wrong-shape", "upsampled-bgr", "not-an-array"],
)
def test_only_native_uint8_packed_frames_are_archived(tmp_path, frame, match):
    """Anything already converted or reshaped is rejected: the archive is the native path."""
    archive = archive_at(tmp_path, popen=FakeFfmpeg())
    with pytest.raises(ArchiveError, match=match):
        with archive.writer("bitcrush", KEY, width=SIZE[1], height=SIZE[0]) as writer:
            writer.write(frame, params=params(0), setpoint=0)


def test_a_short_parameter_vector_is_rejected(tmp_path):
    """A row must carry the whole device state, not part of it."""
    archive = archive_at(tmp_path, popen=FakeFfmpeg())
    with pytest.raises(ArchiveError, match="expected 12"):
        with archive.writer("bitcrush", KEY, width=SIZE[1], height=SIZE[0]) as writer:
            writer.write(frames(1)[0], params=(0, 1, 2), setpoint=0)


def test_writing_before_the_context_opens_raises(tmp_path):
    """The encoder only exists inside the context manager."""
    writer = archive_at(tmp_path).writer("bitcrush", KEY, width=SIZE[1], height=SIZE[0])
    assert writer.count == 0
    with pytest.raises(ArchiveError, match="not open"):
        writer.write(frames(1)[0], params=params(0), setpoint=0)


def test_a_reflashed_program_invalidates_the_archive(tmp_path):
    """Raw frames are keyed exactly as derived results are."""
    archive, _, _ = write_archive(tmp_path, 3)
    assert archive.has("bitcrush", KEY) is True
    assert archive.has("bitcrush", OTHER) is False
    assert archive.reader("bitcrush", OTHER) is None
    assert archive.reader("bitcrush", KEY) is not None


def test_committed_lists_only_complete_archives(tmp_path):
    """The listing skips unparsable sidecars and archives missing their video."""
    archive, _, _ = write_archive(tmp_path, 3)
    (archive.directory / "junk.json").write_text("{not json", encoding="utf-8")
    (archive.directory / "orphan.json").write_text('{"program": "orphan"}', encoding="utf-8")
    assert sorted(archive.committed()) == ["bitcrush"]
    assert archive.committed()["bitcrush"]["frames"] == 3
    archive.paths("bitcrush")[0].unlink()
    assert not archive.committed()
    assert archive.rows("bitcrush") == []


def test_a_newer_schema_is_not_trusted(tmp_path):
    """An archive from a future writer reads as absent rather than being misparsed."""
    archive, _, _ = write_archive(tmp_path, 2)
    meta_path = archive.paths("bitcrush")[2]
    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    meta_path.write_text(json.dumps(meta | {"schema_version": 99}), encoding="utf-8")
    assert archive.meta("bitcrush") is None
    assert archive.reader("bitcrush") is None


def test_awkward_program_names_get_distinct_archives(tmp_path):
    """Names needing escaping stay distinct on disk, as in the result store."""
    for name in ("a/b", "a b", "a_b"):
        write_archive(tmp_path, 2, program=name, key=ProgramKey(name, "x"))
    archive = archive_at(tmp_path, run=FakeDecoder(25, SIZE))
    assert sorted(archive.committed()) == ["a b", "a/b", "a_b"]
    assert len({archive.paths(n)[0] for n in ("a/b", "a b", "a_b")}) == 3


def test_reader_random_access_selects_the_named_frame(tmp_path):
    """Frame N decodes to frame N, in any order, without reading its neighbours."""
    archive, _, made = write_archive(tmp_path, 5)
    reader = archive.reader("bitcrush", KEY)
    assert reader.count == 5 and reader.shape == SIZE + (CHANNELS,)
    for index in (4, 0, 2, 1, 3):
        assert np.array_equal(reader.frame(index), made[index])
    assert archive.run.calls == [4, 0, 2, 1, 3]


def test_reader_selects_rows_by_vector_setpoint_and_loop(tmp_path):
    """Rows are the index: a caller finds frames by state without decoding any."""
    archive, _, _ = write_archive(tmp_path, 6)
    reader = archive.reader("bitcrush")
    assert [r.frame for r in reader.select(setpoint=1)] == [2, 3]
    assert [r.frame for r in reader.select(loop_index=0)] == [0, 3]
    assert [r.frame for r in reader.select(params=params(4))] == [4]
    assert [r.frame for r in reader.select(setpoint=2, loop_index=2)] == [5]
    assert reader.select(params=params(4), setpoint=0) == []
    assert len(reader.select()) == 6


def test_reader_rejects_an_out_of_range_frame(tmp_path):
    """Asking past the end is an error, not a silently wrong frame."""
    archive, _, _ = write_archive(tmp_path, 2)
    reader = archive.reader("bitcrush")
    for index in (-1, 2):
        with pytest.raises(IndexError):
            reader.frame(index)


@pytest.mark.parametrize("kwargs", [{"returncode": 1}, {"truncate": True}], ids=["exit", "short"])
def test_a_failed_or_short_decode_raises(tmp_path, kwargs):
    """A frame that did not decode whole is an error, never a partial array."""
    archive, _, _ = write_archive(tmp_path, 2)
    archive.run = FakeDecoder(25, SIZE, **kwargs)
    with pytest.raises(ArchiveError, match="bytes"):
        archive.reader("bitcrush").frame(1)


def test_absent_archive_reads_as_nothing(tmp_path):
    """An unarchived program yields no metadata, no rows and no reader."""
    archive = archive_at(tmp_path)
    assert archive.meta("bitcrush") is None
    assert archive.rows("bitcrush") == []
    assert archive.reader("bitcrush") is None
    assert not archive.committed()


def test_an_odd_width_cannot_be_packed_422(tmp_path):
    """4:2:2 shares chroma across pixel pairs, so an odd width fails before any encode."""
    with pytest.raises(ArchiveError, match="odd"):
        archive_at(tmp_path).writer("bitcrush", KEY, width=31, height=16)


@pytest.mark.skipif(shutil.which("ffmpeg") is None, reason="ffmpeg is not installed")
@pytest.mark.parametrize("seed", [7, 11], ids=["structured", "noisy"])
def test_real_ffv1_round_trip_is_bit_exact(tmp_path, seed):
    """The claim the module rests on: packed YVYU in, the same bytes back out.

    Packed YVYU to planar yuv422p exchanges the two chroma planes and nothing
    else, so the samples survive and the stored order becomes the canonical one.
    """
    archive = archive_at(tmp_path)
    made = frames(6, size=(48, 64), seed=seed)
    with archive.writer("bitcrush", KEY, width=64, height=48) as writer:
        for i, frame in enumerate(made):
            writer.write(frame, params=params(i), setpoint=i // 3, loop_index=i % 2)
    reader = archive.reader("bitcrush", KEY)
    assert reader.count == 6
    for index in (5, 0, 3, 1, 4, 2):
        decoded = reader.frame(index)
        assert decoded.dtype == np.uint8 and decoded.shape == (48, 64, CHANNELS)
        assert np.array_equal(decoded, made[index])
    assert reader.meta["bytes_per_frame"] > 0


def decode_rgb(video, height, width):
    """Decode one frame with ffmpeg's own conversion, as any later reader would."""
    argv = ["ffmpeg", "-loglevel", "error", "-i", str(video), "-frames:v", "1"]
    argv += ["-f", "rawvideo", "-pix_fmt", "rgb24", "pipe:1"]
    out = subprocess.run(argv, capture_output=True, check=True).stdout
    return np.frombuffer(out, np.uint8).reshape(height, width, 3)


@pytest.mark.skipif(shutil.which("ffmpeg") is None, reason="ffmpeg is not installed")
@pytest.mark.parametrize("colour", sorted(MEASURED_YVYU))
def test_real_round_trip_preserves_the_colour_not_just_the_bytes(tmp_path, colour):
    """Tagging the pipe YVYU is what makes the stored yuv422p mean what it says.

    Under the advertised tag the planes would be stored exchanged, so the bytes
    would still round-trip while every later reader saw red and blue swapped.
    """
    planes, expected = MEASURED_YVYU[colour]
    native = native_frame(planes, height=16, width=16)
    archive = archive_at(tmp_path)
    with archive.writer("bitcrush", KEY, width=16, height=16) as writer:
        writer.write(native, params=params(0), setpoint=0)
    reader = archive.reader("bitcrush", KEY)
    assert np.array_equal(reader.frame(0), native)
    assert decode_rgb(reader.video, 16, 16)[8, 8] == pytest.approx(expected, abs=3.0)


def test_an_archive_tagged_with_the_advertised_order_is_refused(tmp_path):
    """A schema-1 archive stored its chroma planes exchanged; decoding it now would lie."""
    archive, _, _ = write_archive(tmp_path, 2)
    meta_path = archive.paths("bitcrush")[2]
    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    meta_path.write_text(json.dumps(meta | {"schema_version": 1, "pix_fmt": "yuyv422"}), encoding="utf-8")
    assert archive.meta("bitcrush") is None
    assert archive.reader("bitcrush") is None


class FakeStreamer:
    """``Popen`` stand-in serving the fake archive's frames in order on stdout."""

    def __init__(self, size=SIZE, *, truncate=False):
        self.nbytes = size[0] * size[1] * CHANNELS
        self.truncate = truncate
        self.calls = []

    def __call__(self, argv, stdout=None, stderr=None):
        del stdout, stderr
        import io

        start = round(float(argv[argv.index("-ss") + 1]) * 25 + 0.5)
        want = int(argv[argv.index("-frames:v") + 1])
        self.calls.append((start, want))
        with open(argv[argv.index("-i") + 1], "rb") as handle:
            data = handle.read()[start * self.nbytes :][: want * self.nbytes]
        return SimpleNamespace(stdout=io.BytesIO(data[:-1] if self.truncate else data), wait=lambda: 0)


def test_stream_reads_the_archive_in_order_from_one_decoder(tmp_path):
    """A per-frame seek costs a process each time; recomputing metrics reads in order."""
    archive, _, made = write_archive(tmp_path, count=6)
    streamer = FakeStreamer()
    reader = archive.reader("bitcrush")
    reader._popen = streamer  # pylint: disable=protected-access
    got = list(reader.stream())
    assert len(got) == 6 and streamer.calls == [(0, 6)]
    assert all(np.array_equal(a, b) for a, b in zip(got, made)), "bit exact, in order"


def test_stream_takes_a_slice_of_the_archive(tmp_path):
    """Only the setpoint being measured need be decoded."""
    archive, _, made = write_archive(tmp_path, count=6)
    reader = archive.reader("bitcrush")
    reader._popen = FakeStreamer()  # pylint: disable=protected-access
    got = list(reader.stream(start=2, count=3))
    assert [f.mean() for f in got] == [f.mean() for f in made[2:5]]


def test_stream_of_an_empty_range_runs_no_decoder(tmp_path):
    """A range past the end costs no process at all."""
    archive, _, _ = write_archive(tmp_path, count=2)
    streamer = FakeStreamer()
    reader = archive.reader("bitcrush")
    reader._popen = streamer  # pylint: disable=protected-access
    assert not list(reader.stream(start=5)) and not streamer.calls


def test_stream_stops_when_the_decoder_ends_early(tmp_path):
    """A short read ends the stream rather than reshaping a partial frame."""
    archive, _, _ = write_archive(tmp_path, count=4)
    reader = archive.reader("bitcrush")
    reader._popen = FakeStreamer(truncate=True)  # pylint: disable=protected-access
    assert len(list(reader.stream())) == 3, "a short read ends the stream rather than reshaping it"

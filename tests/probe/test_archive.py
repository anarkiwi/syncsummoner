"""Native frame archive: a recording published in one rename, with a per-frame sidecar."""

import io
import json
import shutil
import subprocess
from types import SimpleNamespace

import numpy as np
import pytest

from syncsummoner.probe import fit
from syncsummoner.probe.archive import (
    ARCHIVE_SCHEMA_VERSION,
    CHANNELS,
    GAP,
    ArchiveError,
    FrameArchive,
    FrameRow,
)
from syncsummoner.probe.store import KeyKind, ProgramKey

CLOCK = 1_700_000_000.0
#: Height, width; the recorder's geometry, not the reader's.
SIZE = (24, 32)
FPS = 30
KEY = ProgramKey("bitcrush", "deadbeef", KeyKind.BINARY)
OTHER = ProgramKey("bitcrush", "feedface", KeyKind.BINARY)


def params(seed):
    """A raw 12-value parameter vector."""
    return tuple(int((seed * 37 + i * 11) % 1024) for i in range(12))


def frames(count, size=SIZE, seed=0):
    """Synthetic RGB frames, as the reader decodes them back."""
    rng = np.random.default_rng(seed)
    ramp = np.linspace(0, 255, size[1], dtype=np.float64)[None, :, None]
    return [
        np.clip(ramp + rng.integers(0, 64, size + (CHANNELS,)) + 3 * i, 0, 255).astype(np.uint8)
        for i in range(count)
    ]


def rows(count, *, program="bitcrush", gaps=()):
    """One sidecar row per recorded frame, held two frames per setpoint."""
    return [
        FrameRow(
            frame=i,
            program=program,
            params=params(i),
            setpoint=GAP if i in gaps else i // 2,
            captured=CLOCK + i / FPS,
        )
        for i in range(count)
    ]


class FakeDecoder:
    """``subprocess.run`` stand-in slicing the recording's concatenated frames."""

    def __init__(self, fps, size, *, returncode=0, truncate=False):
        self.fps = fps
        self.nbytes = size[0] * size[1] * CHANNELS
        self.returncode = returncode
        self.truncate = truncate
        self.calls = []

    def __call__(self, argv, capture_output=False, check=False):
        """Return the frame the seek offset names."""
        del capture_output, check
        index = round(float(argv[argv.index("-ss") + 1]) * self.fps + 0.5)
        self.calls.append(index)
        with open(argv[argv.index("-i") + 1], "rb") as handle:
            data = handle.read()[index * self.nbytes : (index + 1) * self.nbytes]
        return subprocess.CompletedProcess(
            argv, self.returncode, stdout=data[:-1] if self.truncate else data, stderr=b""
        )


class FakeStreamer:
    """``Popen`` stand-in serving the recording's frames in order on stdout."""

    def __init__(self, size=SIZE, *, truncate=False):
        self.size = size
        self.nbytes = size[0] * size[1] * CHANNELS
        self.truncate = truncate
        self.calls = []

    def __call__(self, argv, stdout=None, stderr=None):
        """Serve the slice asked for, downscaling as the decoder would when told to."""
        del stdout, stderr
        start = round(float(argv[argv.index("-ss") + 1]) * FPS + 0.5)
        want = int(argv[argv.index("-frames:v") + 1])
        scale = argv[argv.index("-vf") + 1] if "-vf" in argv else None
        self.calls.append((start, want, scale))
        with open(argv[argv.index("-i") + 1], "rb") as handle:
            data = handle.read()[start * self.nbytes :][: want * self.nbytes]
        if scale is not None:
            width = int(scale.removeprefix("scale=").split(":")[0])
            height = 2 * round(self.size[0] * width / self.size[1] / 2)
            data = bytes(want * height * width * CHANNELS)
        return SimpleNamespace(stdout=io.BytesIO(data[:-1] if self.truncate else data), wait=lambda: 0)


def archive_at(tmp_path, **kwargs):
    """Archive rooted in a temp directory with a frozen clock."""
    return FrameArchive(tmp_path / "raw", clock=lambda: CLOCK, **kwargs)


def write_archive(tmp_path, count=4, *, program="bitcrush", key=KEY, gaps=(), **kwargs):
    """Commit a recording of ``count`` synthetic frames, as the recorder would leave it."""
    kwargs.setdefault("run", FakeDecoder(FPS, SIZE))
    archive = archive_at(tmp_path, **kwargs)
    made = frames(count)
    scratch = archive.scratch(program)
    scratch.write_bytes(b"".join(f.tobytes() for f in made))
    meta = archive.commit(
        program,
        key,
        scratch,
        rows(count, program=program, gaps=gaps),
        width=SIZE[1],
        height=SIZE[0],
        fps=FPS,
    )
    return archive, made, meta


def test_a_recording_is_published_by_renaming_it_into_place(tmp_path):
    """The recorder writes beside the archive, so publishing costs no copy of the video."""
    archive = archive_at(tmp_path)
    scratch = archive.scratch("bitcrush")
    assert scratch.parent == archive.directory and not scratch.exists()
    scratch.write_bytes(b"a recording")
    archive.commit("bitcrush", KEY, scratch, rows(2), width=SIZE[1], height=SIZE[0], fps=FPS)
    video = archive.paths("bitcrush")[0]
    assert not scratch.exists() and video.read_bytes() == b"a recording"


def test_commit_publishes_video_sidecar_and_metadata(tmp_path):
    """A clean commit leaves all three artefacts and no temp file."""
    archive, made, _ = write_archive(tmp_path, 4)
    video, data, meta_path = archive.paths("bitcrush")
    assert video.exists() and data.exists() and meta_path.exists()
    assert video.stat().st_size == sum(f.nbytes for f in made)
    assert sorted(p.name for p in archive.directory.iterdir()) == sorted(
        (data.name, meta_path.name, video.name)
    )


def test_metadata_is_self_describing(tmp_path):
    """Metadata names the key, the geometry, the codec and the yield."""
    archive, made, _ = write_archive(tmp_path, 4, gaps=(3,))
    meta = json.loads(archive.paths("bitcrush")[2].read_text(encoding="utf-8"))
    assert meta["schema_version"] == ARCHIVE_SCHEMA_VERSION == 3
    assert meta["program"] == "bitcrush"
    assert (meta["key_kind"], meta["key_material"], meta["key_digest"]) == (
        "binary",
        "deadbeef",
        KEY.digest,
    )
    assert (meta["codec"], meta["container"]) == ("ffv1", "matroska")
    assert (meta["pix_fmt"], meta["stored_pix_fmt"]) == ("yuyv422", "yuv422p")
    assert (meta["width"], meta["height"], meta["fps"]) == (SIZE[1], SIZE[0], FPS)
    assert (meta["frames"], meta["measured"]) == (4, 3), "a gap frame is stored but not measured"
    assert meta["bytes_per_frame"] == pytest.approx(made[0].nbytes)
    assert meta["video"].endswith(".mkv") and meta["data"].endswith(".parquet")
    assert meta["created"].startswith("2023-11-14T")


def test_sidecar_rows_carry_one_row_per_recorded_frame(tmp_path):
    """Every frame the card delivered gets its index, vector, setpoint and capture time."""
    archive, _, _ = write_archive(tmp_path, 4, gaps=(1,))
    stored = archive.rows("bitcrush")
    assert [r.frame for r in stored] == [0, 1, 2, 3]
    assert [r.setpoint for r in stored] == [0, GAP, 1, 1]
    assert [r.params for r in stored] == [params(i) for i in range(4)]
    assert {r.program for r in stored} == {"bitcrush"}
    assert [r.captured for r in stored] == [CLOCK + i / FPS for i in range(4)]


def test_sidecar_is_exactly_what_the_parquet_helpers_read(tmp_path):
    """The sidecar reuses the project's measurement helpers, not a new format."""
    archive, _, _ = write_archive(tmp_path, 3)
    flat = fit.load_measurements(archive.paths("bitcrush")[1])
    assert len(flat) == 3
    assert flat[0]["p1"] == params(0)[0] and flat[0]["frame"] == 0
    assert FrameRow.from_row(flat[2]).params == params(2)


def test_an_empty_recording_is_not_a_commit(tmp_path):
    """A sweep that captured nothing stays unarchived."""
    archive = archive_at(tmp_path)
    scratch = archive.scratch("bitcrush")
    scratch.write_bytes(b"nothing in it")
    with pytest.raises(ArchiveError, match="no frames"):
        archive.commit("bitcrush", KEY, scratch, [], width=SIZE[1], height=SIZE[0])
    assert archive.meta("bitcrush") is None and scratch.exists(), "nothing was published"


def test_a_repeated_frame_index_is_refused(tmp_path):
    """Two rows for one frame would make every later lookup ambiguous."""
    archive = archive_at(tmp_path)
    scratch = archive.scratch("bitcrush")
    scratch.write_bytes(b"a recording")
    doubled = rows(2) + rows(1)
    with pytest.raises(ArchiveError, match="repeated frame indices"):
        archive.commit("bitcrush", KEY, scratch, doubled, width=SIZE[1], height=SIZE[0])
    assert archive.meta("bitcrush") is None


def test_a_short_parameter_vector_is_rejected(tmp_path):
    """A selection must name the whole device state, not part of it."""
    archive, _, _ = write_archive(tmp_path, 2)
    with pytest.raises(ArchiveError, match="expected 12"):
        archive.reader("bitcrush").select(params=(0, 1, 2))


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


def test_an_archive_written_with_the_chroma_pair_reversed_is_refused(tmp_path):
    """Schema 2 stored its chroma planes exchanged; decoding it now would swap red and blue."""
    archive, _, _ = write_archive(tmp_path, 2)
    meta_path = archive.paths("bitcrush")[2]
    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    meta_path.write_text(json.dumps(meta | {"schema_version": 2, "pix_fmt": "yvyu422"}), encoding="utf-8")
    assert archive.meta("bitcrush") is None
    assert archive.reader("bitcrush") is None and not archive.committed()


def test_awkward_program_names_get_distinct_archives(tmp_path):
    """Names needing escaping stay distinct on disk, as in the result store."""
    for name in ("a/b", "a b", "a_b"):
        write_archive(tmp_path, 2, program=name, key=ProgramKey(name, "x"))
    archive = archive_at(tmp_path, run=FakeDecoder(FPS, SIZE))
    assert sorted(archive.committed()) == ["a b", "a/b", "a_b"]
    assert len({archive.paths(n)[0] for n in ("a/b", "a b", "a_b")}) == 3


def test_reader_random_access_selects_the_named_frame(tmp_path):
    """Frame N decodes to frame N, in any order, without reading its neighbours."""
    archive, made, _ = write_archive(tmp_path, 5)
    reader = archive.reader("bitcrush", KEY)
    assert reader.count == 5 and reader.shape() == SIZE + (CHANNELS,)
    for index in (4, 0, 2, 1, 3):
        assert np.array_equal(reader.frame(index), made[index])
    assert archive.run.calls == [4, 0, 2, 1, 3]


def test_reader_shape_follows_the_width_the_decoder_was_asked_for(tmp_path):
    """Scaling happens in ffmpeg, so the caller must know the shape it will get back."""
    archive, _, _ = write_archive(tmp_path, 2)
    reader = archive.reader("bitcrush")
    assert reader.shape(16) == (12, 16, CHANNELS), "the aspect is kept, on an even height"
    assert reader.shape(SIZE[1] * 2) == reader.shape(None) == SIZE + (CHANNELS,)
    assert "-vf" not in reader.command(width=SIZE[1] * 2), "no upscale is ever asked for"
    argv = reader.command(start=1, count=3, width=16)
    assert argv[argv.index("-vf") + 1] == "scale=16:-2"
    assert argv[argv.index("-frames:v") + 1] == "3" and argv[-2] == "rgb24"


def test_reader_selects_rows_by_vector_and_setpoint(tmp_path):
    """Rows are the index: a caller finds frames by state without decoding any."""
    archive, _, _ = write_archive(tmp_path, 6, gaps=(0,))
    reader = archive.reader("bitcrush")
    assert [r.frame for r in reader.select(setpoint=1)] == [2, 3]
    assert [r.frame for r in reader.select(setpoint=GAP)] == [0]
    assert [r.frame for r in reader.select(params=params(4))] == [4]
    assert [r.frame for r in reader.select(params=params(5), setpoint=2)] == [5]
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
    archive.run = FakeDecoder(FPS, SIZE, **kwargs)
    with pytest.raises(ArchiveError, match="bytes"):
        archive.reader("bitcrush").frame(1)


def test_absent_archive_reads_as_nothing(tmp_path):
    """An unarchived program yields no metadata, no rows and no reader."""
    archive = archive_at(tmp_path)
    assert archive.meta("bitcrush") is None
    assert archive.rows("bitcrush") == []
    assert archive.reader("bitcrush") is None
    assert not archive.committed()


def streaming_reader(archive, streamer):
    """Reader whose decoder is the given stand-in."""
    reader = archive.reader("bitcrush")
    reader._popen = streamer  # pylint: disable=protected-access
    return reader


def test_stream_reads_the_archive_in_order_from_one_decoder(tmp_path):
    """A per-frame seek costs a process each time; recomputing metrics reads in order."""
    archive, made, _ = write_archive(tmp_path, count=6)
    streamer = FakeStreamer()
    got = list(streaming_reader(archive, streamer).stream())
    assert len(got) == 6 and streamer.calls == [(0, 6, None)]
    assert all(np.array_equal(a, b) for a, b in zip(got, made)), "bit exact, in order"


def test_stream_takes_a_slice_of_the_archive(tmp_path):
    """Only the setpoint being measured need be decoded."""
    archive, made, _ = write_archive(tmp_path, count=6)
    got = list(streaming_reader(archive, FakeStreamer()).stream(start=2, count=3))
    assert [f.mean() for f in got] == [f.mean() for f in made[2:5]]


def test_stream_has_the_decoder_do_the_downscale(tmp_path):
    """Every metric runs at the analysis width, and ffmpeg scales for free on the way out."""
    archive, _, _ = write_archive(tmp_path, count=3)
    streamer = FakeStreamer()
    reader = streaming_reader(archive, streamer)
    got = list(reader.stream(width=16))
    assert streamer.calls == [(0, 3, "scale=16:-2")]
    assert [f.shape for f in got] == [reader.shape(16)] * 3


def test_stream_of_an_empty_range_runs_no_decoder(tmp_path):
    """A range past the end costs no process at all."""
    archive, _, _ = write_archive(tmp_path, count=2)
    streamer = FakeStreamer()
    assert not list(streaming_reader(archive, streamer).stream(start=5)) and not streamer.calls


def test_stream_stops_when_the_decoder_ends_early(tmp_path):
    """A short read ends the stream rather than reshaping a partial frame."""
    archive, _, _ = write_archive(tmp_path, count=4)
    reader = streaming_reader(archive, FakeStreamer(truncate=True))
    assert len(list(reader.stream())) == 3, "a short read ends the stream rather than reshaping it"


def test_a_dark_verdict_survives_a_restart_and_is_keyed(tmp_path):
    """A dark program has no archive, so only the verdict stops a resume re-probing it."""
    archive = archive_at(tmp_path)
    assert not archive.dark("bitcrush", KEY)
    archive.mark_dark("bitcrush", KEY, 11.2)
    assert archive.dark("bitcrush", KEY) and not archive.dark("bitcrush", OTHER)
    stored = json.loads(archive.dark_path("bitcrush").read_text(encoding="utf-8"))
    assert stored["luma"] == 11.2 and stored["key_digest"] == KEY.digest


def test_a_dark_verdict_does_not_make_the_program_look_archived(tmp_path):
    """Nothing downstream may mistake a black program for measured frames."""
    archive = archive_at(tmp_path)
    archive.mark_dark("bitcrush", KEY, 0.0)
    assert not archive.has("bitcrush", KEY) and archive.reader("bitcrush") is None
    assert "bitcrush" not in archive.committed(), "a verdict is not an archive"


BLOCKS = ((220, 30, 30), (30, 220, 30), (30, 30, 220), (200, 200, 200))


def block_frame(shift, *, height=32, width=64):
    """Flat colour bars, rotated by ``shift``, so a seek lands provably on one frame."""
    frame = np.empty((height, width, CHANNELS), dtype=np.uint8)
    for i in range(len(BLOCKS)):
        frame[:, i * width // len(BLOCKS) : (i + 1) * width // len(BLOCKS)] = BLOCKS[
            (i + shift) % len(BLOCKS)
        ]
    return frame


def encode_ffv1(source, target, *, height, width, fps=FPS):
    """Encode raw RGB frames as the recorder does: lossless intra FFV1 in Matroska."""
    subprocess.run(
        [
            "ffmpeg",
            "-loglevel",
            "error",
            "-y",
            "-f",
            "rawvideo",
            "-pix_fmt",
            "rgb24",
            "-video_size",
            f"{width}x{height}",
            "-framerate",
            str(fps),
            "-i",
            str(source),
            "-c:v",
            "ffv1",
            "-level",
            "3",
            "-pix_fmt",
            "yuv422p",
            str(target),
        ],
        check=True,
        capture_output=True,
    )


@pytest.mark.skipif(shutil.which("ffmpeg") is None, reason="ffmpeg is not installed")
def test_a_real_recording_reads_back_frame_for_frame_in_colour(tmp_path):
    """The claim the module rests on: what the card sent is what a later reader sees.

    Random access is by input seek, which only lands on the named frame because
    every FFV1 frame is a keyframe.
    """
    made = [block_frame(shift) for shift in range(4)]
    source = tmp_path / "source.rgb"
    source.write_bytes(b"".join(f.tobytes() for f in made))
    archive = archive_at(tmp_path)
    scratch = archive.scratch("bitcrush")
    encode_ffv1(source, scratch, height=32, width=64)
    archive.commit("bitcrush", KEY, scratch, rows(4), width=64, height=32, fps=FPS)
    reader = archive.reader("bitcrush", KEY)
    assert reader.count == 4 and reader.shape() == (32, 64, CHANNELS)
    for index in (3, 0, 2, 1):
        decoded = reader.frame(index)
        assert decoded.dtype == np.uint8 and decoded.shape == (32, 64, CHANNELS)
        for i, colour in enumerate(BLOCKS[index:] + BLOCKS[:index]):
            assert decoded[16, i * 16 + 8] == pytest.approx(colour, abs=4)
    assert [f.shape for f in reader.stream()] == [(32, 64, CHANNELS)] * 4
    assert [f.shape for f in reader.stream(width=32)] == [reader.shape(32)] * 4

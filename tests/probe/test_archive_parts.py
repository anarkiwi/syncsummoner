"""Archives of several parts: takes linked in beside an ffconcat listing the decoder reads as one."""

# pylint: disable=missing-function-docstring

import errno
import os
import shutil
import subprocess

import numpy as np
import pytest

from syncsummoner.probe import archive as A
from syncsummoner.probe.archive import ArchiveError, FrameArchive, FrameReader, FrameRow
from syncsummoner.probe.store import KeyKind, ProgramKey

HAS_FFMPEG = shutil.which("ffmpeg") is not None and shutil.which("ffprobe") is not None


def rows(count, program="p"):
    """One sidecar row per frame, each its own setpoint."""
    return [FrameRow(frame=i, program=program, params=(i,) * 12, setpoint=i) for i in range(count)]


def durations(spans):
    """``subprocess.run`` stand-in answering a duration probe with each part's own span."""

    def run(argv, **_kw):
        return subprocess.CompletedProcess(argv, 0, stdout=f"{spans[argv[-1]]}\n".encode(), stderr=b"")

    return run


def takes(directory, sizes):
    """Fake take files of ``sizes`` bytes, each a distinct fill."""
    directory.mkdir(parents=True, exist_ok=True)
    made = []
    for k, size in enumerate(sizes):
        take = directory / f"take-{k}.mkv"
        take.write_bytes(bytes([k + 1]) * size)
        made.append(take)
    return made


def commit(archive, videos, count, program="p", **geometry):
    """Commit ``videos`` as ``program``'s archive of ``count`` frames."""
    key = ProgramKey(program, "rc4", KeyKind.NAME_FIRMWARE)
    return archive.commit(
        program, key, videos, rows(count, program), **({"width": 4, "height": 2} | geometry)
    )


def test_a_part_is_a_hard_link_to_its_take(tmp_path):
    source = takes(tmp_path, [5])[0]
    A.link(source, tmp_path / "part.mkv")
    assert (tmp_path / "part.mkv").stat().st_ino == source.stat().st_ino and source.stat().st_nlink == 2


def test_across_filesystems_a_part_is_copied(tmp_path, monkeypatch):
    source = takes(tmp_path, [5])[0]
    os.utime(source, (1_000_000, 1_000_000))

    def cross(*_a):
        raise OSError(errno.EXDEV, "cross-device link")

    monkeypatch.setattr(A.os, "link", cross)
    A.link(source, tmp_path / "part.mkv")
    part = tmp_path / "part.mkv"
    assert part.read_bytes() == source.read_bytes() and part.stat().st_ino != source.stat().st_ino
    assert part.stat().st_mtime == 1_000_000 and sorted(p.name for p in tmp_path.iterdir()) == [
        "part.mkv",
        "take-0.mkv",
    ]


def test_any_other_link_failure_is_raised_and_leaves_nothing(tmp_path, monkeypatch):
    source = takes(tmp_path, [5])[0]

    def denied(*_a):
        raise OSError(errno.EACCES, "denied")

    monkeypatch.setattr(A.os, "link", denied)
    with pytest.raises(PermissionError):
        A.link(source, tmp_path / "part.mkv")
    assert [p.name for p in tmp_path.iterdir()] == ["take-0.mkv"]


def test_relinking_the_same_take_leaves_no_temp_link(tmp_path):
    source = takes(tmp_path, [5])[0]
    for _ in range(2):
        A.link(source, tmp_path / "part.mkv")
    assert sorted(p.name for p in tmp_path.iterdir()) == ["part.mkv", "take-0.mkv"]


@pytest.mark.parametrize("stdout, returncode", [(b"N/A\n", 0), (b"0.000000\n", 0), (b"", 1)])
def test_a_part_without_a_duration_is_refused(stdout, returncode):
    def run(argv, **_kw):
        return subprocess.CompletedProcess(argv, returncode, stdout=stdout, stderr=b"bad")

    with pytest.raises(ArchiveError, match="no container duration"):
        A.container_duration("x.mkv", run=run)


def test_takes_are_linked_in_behind_a_listing_of_them(tmp_path):
    sources = takes(tmp_path / "takes", [3, 5, 7])
    archive = FrameArchive(tmp_path / "a", run=durations(dict(zip(map(str, sources), ["0.2", "0.3", "0.4"]))))
    meta = commit(archive, sources, 6)
    assert meta["video"] == "p.ffconcat" and meta["parts"] == [f"p.part0{k}.mkv" for k in range(3)]
    assert (archive.directory / "p.ffconcat").read_text(encoding="utf-8") == (
        "ffconcat version 1.0\n"
        "file 'p.part00.mkv'\nduration 0.2\n"
        "file 'p.part01.mkv'\nduration 0.3\n"
        "file 'p.part02.mkv'\nduration 0.4\n"
    )
    assert all((archive.directory / n).samefile(s) for n, s in zip(meta["parts"], sources))
    assert meta["bytes_per_frame"] == pytest.approx(15 / 6)
    assert sorted(p.name for p in archive.directory.iterdir()) == sorted(
        ["p.ffconcat", "p.json", "p.parquet", *meta["parts"]]
    )
    assert archive.reader("p").video == archive.directory / "p.ffconcat"


def test_a_lost_part_uncommits_the_archive(tmp_path):
    sources = takes(tmp_path / "takes", [3, 5])
    archive = FrameArchive(tmp_path / "a", run=durations(dict.fromkeys(map(str, sources), "0.1")))
    commit(archive, sources, 2)
    (archive.directory / "p.part01.mkv").unlink()
    assert archive.meta("p") is None and archive.reader("p") is None and not archive.committed()


def test_a_recommit_replaces_what_the_last_commit_named_and_nothing_else(tmp_path):
    sources = takes(tmp_path / "takes", [3, 5, 7])
    archive = FrameArchive(tmp_path / "a", run=durations(dict.fromkeys(map(str, sources), "0.1")))
    scratch = archive.scratch("p")
    scratch.write_bytes(b"joined")
    commit(archive, scratch, 3)
    (archive.directory / "c.mkv").write_bytes(b"clip")
    commit(archive, sources, 3)
    assert not (archive.directory / "p.mkv").exists(), "the joined video is gone once the listing commits"
    meta = commit(archive, sources[1:], 2)
    assert (
        meta["parts"] == ["p.part00.mkv", "p.part01.mkv"]
        and not (archive.directory / "p.part02.mkv").exists()
    )
    assert (archive.directory / "p.part00.mkv").read_bytes() == sources[1].read_bytes()
    assert (archive.directory / "c.mkv").exists() and archive.meta("p")["frames"] == 2
    archive.scratch("p").write_bytes(b"v")
    meta = commit(archive, archive.scratch("p"), 1)
    assert meta["video"] == "p.mkv" and "parts" not in meta
    assert sorted(p.name for p in archive.directory.iterdir()) == ["c.mkv", "p.json", "p.mkv", "p.parquet"]


def test_a_refused_part_leaves_the_earlier_archive_committed(tmp_path):
    sources = takes(tmp_path / "takes", [3, 5])
    archive = FrameArchive(tmp_path / "a", run=durations({str(sources[0]): "0.1", str(sources[1]): "N/A"}))
    commit(archive, sources[:1], 1)
    with pytest.raises(ArchiveError):
        commit(archive, sources, 2)
    assert archive.meta("p")["parts"] == ["p.part00.mkv"]


def test_joined_and_listed_archives_share_a_directory(tmp_path):
    sources = takes(tmp_path / "takes", [3, 5])
    archive = FrameArchive(tmp_path / "a", run=durations(dict.fromkeys(map(str, sources), "0.1")))
    scratch = archive.scratch("old")
    scratch.write_bytes(b"joined")
    commit(archive, scratch, 2, program="old")
    commit(archive, sources, 4, program="new")
    assert sorted(archive.committed()) == ["new", "old"]
    assert (
        archive.reader("old").video.name == "old.mkv" and archive.reader("new").video.name == "new.ffconcat"
    )
    assert [r.frame for r in archive.rows("new")] == [0, 1, 2, 3] and archive.meta("old")["frames"] == 2


def clips(directory, frames):
    """Real MJPEG takes of ``frames`` frames each, every one a different picture."""
    sources = ("testsrc", "testsrc2", "mandelbrot")
    made = []
    for k, (source, count) in enumerate(zip(sources, frames)):
        take = directory / f"take-{k}.mkv"
        lavfi = f"{source}=size=64x48:rate=30"
        argv = ["ffmpeg", "-v", "error", "-f", "lavfi", "-i", lavfi, "-frames:v", str(count), "-c:v", "mjpeg"]
        subprocess.run([*argv, str(take)], check=True)
        made.append(take)
    return made


@pytest.mark.skipif(not HAS_FFMPEG, reason="ffmpeg is not installed")
def test_a_listing_decodes_as_the_takes_joined_would(tmp_path):
    counts = [4, 5, 3]
    sources = clips(tmp_path, counts)
    joined = tmp_path / "joined.mkv"
    (tmp_path / "list.txt").write_text("".join(f"file '{s}'\n" for s in sources), encoding="utf-8")
    argv = ["ffmpeg", "-v", "error", "-f", "concat", "-safe", "0", "-i", str(tmp_path / "list.txt")]
    subprocess.run([*argv, "-map", "0:v", "-c", "copy", str(joined)], check=True)
    archive = FrameArchive(tmp_path / "a")
    total = sum(counts)
    meta = commit(archive, sources, total, width=64, height=48, fps=30.0)
    listed = archive.reader("p")
    whole = FrameReader(joined, meta, rows(total), ffmpeg="ffmpeg", run=subprocess.run)
    expect = list(whole.stream())
    assert len(expect) == total and len({f.tobytes() for f in expect}) > 1
    assert [f.tobytes() for f in listed.stream()] == [f.tobytes() for f in expect]
    assert [f.tobytes() for f in listed.passes()] == [f.tobytes() for f in expect]
    for index in (0, counts[0] - 1, counts[0], counts[0] + 2, total - 1):
        assert np.array_equal(listed.frame(index), expect[index]), f"frame {index}"
    assert [f.tobytes() for f in listed.stream(start=counts[0] + 1, count=3)] == [
        f.tobytes() for f in expect[counts[0] + 1 : counts[0] + 4]
    ]
    assert all((archive.directory / n).samefile(s) for n, s in zip(meta["parts"], sources))

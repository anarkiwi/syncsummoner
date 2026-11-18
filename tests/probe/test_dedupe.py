"""Old takes rewritten deduplicated: exact or untouched, and every hard-linked name moved together."""

# pylint: disable=missing-function-docstring

import os
import shutil

import numpy as np
import pytest

from syncsummoner.device import stored as st
from syncsummoner.device import v4l2rec as vr
from syncsummoner.probe import dedupe as dd
from tests import takes

pytestmark = pytest.mark.skipif(shutil.which("ffmpeg") is None, reason="needs ffmpeg")

SHADES = [0, 0, 0, 1, 2, 2, 3, 3, 3, 3, 1, 4, 5, 5]


@pytest.fixture(name="old")
def _old(tmp_path):
    datas = takes.jpegs(SHADES)
    (tmp_path / "takes").mkdir()
    return takes.old(tmp_path / "takes" / "take.mkv", datas), datas


def snapshot(take):
    return take.read_bytes(), vr.sidecar(take).read_bytes()


def test_an_old_take_round_trips_byte_for_byte_and_reads_as_before(old):
    take, datas = old
    with st.captures(take) as source:
        before, log = takes.framemd5(source), vr.read_log(take)
    done = dd.dedupe(take)
    assert (done.status, done.captures, done.stored) == ("deduplicated", len(SHADES), 7), done.why
    assert done.after < done.before
    assert [d for _, d in st.Stored.of(take).frames()] == datas
    with st.captures(take) as source:
        assert takes.framemd5(source) == before
    assert vr.read_log(take).tolist() == log.tolist()
    assert not [p for p in take.parent.iterdir() if p.name.startswith(".")], "nothing is left beside it"
    assert dd.dedupe(take).status == "already"


def test_a_dry_run_writes_nothing_and_counts_the_repeats(old):
    take, datas = old
    was = snapshot(take)
    done = dd.dedupe(take, dry_run=True)
    assert (done.status, done.stored) == ("dry-run", 7)
    assert done.before - done.after == sum(len(d) for d, p in zip(datas, [None, *datas]) if d == p)
    assert snapshot(take) == was


def test_a_corrupted_copy_is_refused_and_kept(old):
    take, datas = old
    blob = bytearray(take.read_bytes())
    at = blob.find(datas[4][-24:-2])
    blob[at] ^= 0xFF
    take.write_bytes(bytes(blob))
    was = snapshot(take)
    done = dd.dedupe(take)
    assert done.status == "refused" and "capture 4" in done.why
    assert snapshot(take) == was and vr.read_sidecar(take)[1] is None


def test_a_take_whose_packets_are_not_its_captures_is_refused(old):
    take, _ = old
    rows = vr.read_log(take).copy()
    rows["bytes"][2] += 1
    vr.sidecar(take).write_bytes(rows.tobytes())
    assert "not the captures" in dd.dedupe(take).why
    vr.sidecar(take).unlink()
    assert dd.dedupe(take).why == "no sidecar"


def test_captures_logged_past_the_file_keep_their_rows(old):
    take, datas = old
    rows = vr.read_log(take)
    vr.sidecar(take).write_bytes(np.concatenate([rows, rows[-2:]]).tobytes())
    assert dd.dedupe(take).status == "deduplicated"
    got = st.Stored.of(take)
    assert got.index[-3:].tolist() == [6, 7, 8] and vr.read_log(take).size == rows.size + 2
    assert [d for _, d in got.frames()] == datas


def test_a_hard_linked_take_moves_every_name_to_one_new_inode(old, tmp_path):
    take, datas = old
    (tmp_path / "archive").mkdir()
    other = tmp_path / "archive" / "part.mkv"
    os.link(take, other)
    was = snapshot(take)
    done = dd.dedupe(take)
    assert done.status == "skipped" and "pass --links" in done.why and done.after < done.before
    assert snapshot(take) == was and os.stat(take).st_nlink == 2
    done = dd.dedupe(take, links=[tmp_path])
    assert done.status == "deduplicated", done.why
    assert os.path.samestat(take.stat(), other.stat()) and take.stat().st_nlink == 2
    assert os.path.samestat(vr.sidecar(take).stat(), vr.sidecar(other).stat())
    assert [d for _, d in st.Stored.of(other).frames()] == datas


def test_a_name_in_a_read_only_folder_is_left_and_its_saving_reported(old, tmp_path):
    take, _ = old
    (tmp_path / "reference").mkdir()
    keep = tmp_path / "reference" / "take.mkv"
    os.link(take, keep)
    os.chmod(keep.parent, 0o555)
    try:
        was = snapshot(take)
        done = dd.dedupe(take, links=[tmp_path])
        assert done.status == "skipped" and "read-only" in done.why and done.after < done.before
        assert snapshot(take) == was and os.path.samestat(take.stat(), keep.stat())
    finally:
        os.chmod(keep.parent, 0o755)


def test_a_name_beside_another_sidecar_is_refused(old, tmp_path):
    take, _ = old
    other = tmp_path / "other.mkv"
    os.link(take, other)
    vr.sidecar(other).write_bytes(vr.sidecar(take).read_bytes())
    assert "not the take's sidecar" in dd.dedupe(take, links=[tmp_path]).why


def test_an_interrupted_rewrite_is_reported_not_overwritten(old):
    take, _ = old
    left = take.with_name(f".{take.name}.dedupe.mkv")
    left.write_bytes(b"x")
    assert "interrupted" in dd.dedupe(take).why and left.read_bytes() == b"x"


def test_dedupe_all_prints_a_row_per_take_and_the_totals(old, capsys):
    take, _ = old
    done = dd.dedupe_all([take, take.with_name("absent.mkv")], dry_run=True)
    assert [o.status for o in done] == ["dry-run", "refused"]
    out = capsys.readouterr().out.splitlines()
    assert out[0].startswith("status") and out[-1].split()[0] == str(len(SHADES))

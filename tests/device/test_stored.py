"""Deduplicated takes read back capture for capture, as takes storing every capture read."""

# pylint: disable=missing-function-docstring

import shutil

import numpy as np
import pytest

from syncsummoner.device import stored as st
from syncsummoner.device import v4l2rec as vr
from tests import takes

pytestmark = pytest.mark.skipif(shutil.which("ffmpeg") is None, reason="needs ffmpeg")

SHADES = [0, 0, 1, 1, 1, 2, 3, 3, 0, 4, 4, 4, 4, 5, 5, 6]


@pytest.fixture(name="pair")
def _pair(tmp_path):
    datas = takes.jpegs(SHADES)
    return takes.recorded(tmp_path / "new.mkv", datas), takes.old(tmp_path / "old.mkv", datas), datas


def test_a_deduplicated_take_stores_each_run_once_and_yields_every_capture_byte_for_byte(pair):
    new, old, datas = pair
    found = st.Stored.of(new)
    assert st.Stored.of(old) is None, "an old take stores every capture"
    assert found.index.tolist() == [0, 0, 1, 1, 1, 2, 3, 3, 4, 5, 5, 5, 5, 6, 6, 7]
    assert len(takes.framemd5(["-i", str(new)])) == 8
    got = list(found.frames())
    assert [d for _, d in got] == datas
    assert [p for p, _ in got] == vr.timestamp(np.arange(len(datas)), vr.Fraction(60)).tolist()
    assert [d for _, d in found.frames(6)] == datas[6:], "a seek lands on the exact stored frame"


def test_every_reader_sees_the_old_takes_stream(pair):
    new, old, _ = pair
    with st.captures(old) as source:
        assert source == ["-i", str(old)]
        expected = takes.framemd5(source)
    with st.captures(new) as source:
        assert takes.framemd5(source) == expected
    for start in (0.1, 0.11, 0.2):
        with st.captures(old, start_s=start) as a, st.captures(new, start_s=start) as b:
            assert takes.framemd5(b, "-copyts") == takes.framemd5(a, "-copyts")
    with st.captures(new, from_end_s=0.05) as source:
        assert [line.split(",")[-1] for line in takes.framemd5(source)] == [
            line.split(",")[-1] for line in expected[-3:]
        ]


def test_a_reader_stopping_early_or_never_reading_ends_the_feed(pair):
    new, _, _ = pair
    with st.captures(new) as source:
        assert len(takes.framemd5(source, out=("-frames:v", "2"))) <= 3
    with st.captures(new):
        pass


def test_a_corrupted_stored_frame_is_refused(pair):
    new, _, _ = pair
    rows, layout = vr.read_sidecar(new)
    bad = rows.copy()
    bad["digest"][5] ^= 1
    vr.sidecar(new).write_bytes(layout.sidecar_header() + bad.tobytes())
    with pytest.raises(st.StoredError, match="stored frame 2"):
        list(st.Stored.of(new).frames())
    with pytest.raises(st.StoredError):
        with st.captures(new) as source:
            takes.framemd5(source)
    bad = rows.copy()
    bad["stored"][3] = 3
    vr.sidecar(new).write_bytes(layout.sidecar_header() + bad.tobytes())
    with pytest.raises(st.StoredError, match="in order"):
        st.Stored.of(new)


def test_a_take_cut_short_yields_the_captures_it_holds(pair, tmp_path):
    new, _, datas = pair
    rows, layout = vr.read_sidecar(new)
    more = np.concatenate([rows, rows[-1:]])
    more["stored"][-1] += 1
    vr.sidecar(new).write_bytes(layout.sidecar_header() + more.tobytes())
    assert [d for _, d in st.Stored.of(new).frames()] == datas
    assert not list(st.Stored.of(new).frames(len(more)))
    assert st.Stored.of(tmp_path / "none.mkv") is None


def test_a_listing_of_parts_reads_as_the_old_parts_listed(pair, tmp_path):
    new, old, datas = pair
    takes.old(tmp_path / "other.mkv", datas[:5])
    text = "ffconcat version 1.0\nfile '{}'\nduration 0.267\nfile 'other.mkv'\nduration 0.083\n"
    (tmp_path / "old.ffconcat").write_text(text.format(old.name), encoding="utf-8")
    (tmp_path / "new.ffconcat").write_text(text.format(new.name), encoding="utf-8")
    with st.captures(tmp_path / "old.ffconcat") as source:
        expected = takes.framemd5(["-safe", "0", *source])
    with st.captures(tmp_path / "new.ffconcat") as source:
        assert takes.framemd5(source) == expected and len(expected) == len(datas) + 5
    with pytest.raises(ValueError, match="from its start"):
        with st.captures(tmp_path / "new.ffconcat", start_s=1.0):
            pass

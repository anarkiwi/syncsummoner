"""Replay: measurements recomputed from an archived run, with no device."""

# pylint: disable=missing-function-docstring

import numpy as np

from syncsummoner.device.profile import Source
from syncsummoner.probe import replay as R
from syncsummoner.probe.archive import CHANNELS, FrameRow
from tests.probe.test_runner import FakeAnalyzer

SIZE = (16, 24)


def native(value):
    """A packed 4:2:2 frame of one constant sample value."""
    return np.full(SIZE + (CHANNELS,), value, dtype=np.uint8)


class Reader:
    """Frame reader stub: rows and their frames, as the real one streams them."""

    def __init__(self, frames, *, setpoints, firmware="1.0.0"):
        self.frames = list(frames)
        self.meta = {"key_material": firmware, "width": SIZE[1], "height": SIZE[0], "frames": len(frames)}
        self.rows = [
            FrameRow(frame=i, program="p", params=tuple([s] * 12), setpoint=s)
            for i, s in enumerate(setpoints)
        ]

    def stream(self, *, start=0, count=None):
        """Frames in order, honouring the slice the caller asked for."""
        end = len(self.frames) if count is None else start + count
        return iter(self.frames[start:end])


class Archive:
    """Frame archive stub holding one reader per program."""

    def __init__(self, readers):
        self.readers = dict(readers)

    def committed(self):
        return {name: {} for name in self.readers}

    def reader(self, program, key=None):
        del key
        return self.readers.get(program)


def test_setpoints_group_the_stream_by_the_vector_that_produced_it():
    reader = Reader([native(v) for v in (10, 20, 30, 40)], setpoints=[0, 0, 1, 1])
    got = [(sp, params[0], len(frames)) for sp, params, frames in R.setpoints(reader)]
    assert got == [(0, 0, 2), (1, 1, 2)]


def test_setpoints_of_an_empty_archive():
    assert not list(R.setpoints(Reader([], setpoints=[])))


def test_blank_digest_is_the_frame_two_programs_share():
    """Two programs cannot both produce one frame from different vectors."""
    blank = native(7)
    archive = Archive(
        {
            "a": Reader([blank, native(90)], setpoints=[0, 1]),
            "b": Reader([blank, native(120)], setpoints=[0, 1]),
        }
    )
    assert R.blank_digest(archive, ["a", "b"]) == R._digest(blank)  # pylint: disable=protected-access


def test_blank_digest_is_none_when_no_two_programs_open_alike():
    archive = Archive({"a": Reader([native(20)], setpoints=[0]), "b": Reader([native(90)], setpoints=[0])})
    assert R.blank_digest(archive, ["a", "b", "missing"]) is None


def test_a_setpoint_that_is_only_the_blanking_frame_is_dropped():
    """Those rows carry a real vector against a picture the device never drew."""
    blank = native(7)
    reader = Reader([blank, blank, native(90), native(95)], setpoints=[0, 0, 1, 1])
    records = R.program_records(
        Archive({"p": reader}), "p", analyzer=FakeAnalyzer(), blank=R._digest(blank)
    )  # pylint: disable=protected-access
    assert len(records) == 1 and records[0].state_index == 1


def test_records_carry_the_archive_provenance():
    reader = Reader([native(90), native(95)], setpoints=[0, 0], firmware="1.0.0-rc.40")
    record = R.program_records(Archive({"p": reader}), "p", analyzer=FakeAnalyzer())[0]
    assert record.program == "p" and record.firmware == "1.0.0-rc.40"
    assert record.source is Source.HW and record.stimulus == R.STIMULUS
    assert record.params == tuple([0] * 12) and record.metrics


def test_records_of_a_program_the_archive_does_not_hold():
    assert R.program_records(Archive({}), "absent", analyzer=FakeAnalyzer()) == []


def test_replay_measures_every_committed_program():
    blank = native(7)
    archive = Archive(
        {
            "a": Reader([blank, native(90), native(95)], setpoints=[0, 1, 1]),
            "b": Reader([blank, native(120), native(125)], setpoints=[0, 1, 1]),
        }
    )
    notes = []
    out = R.replay(archive, analyzer=FakeAnalyzer(), log=notes.append)
    assert sorted(out) == ["a", "b"] and all(len(v) == 1 for v in out.values())
    assert any("blanking frame" in n for n in notes)


def test_replay_notes_when_nothing_is_shared_between_programs():
    archive = Archive({"a": Reader([native(90), native(95)], setpoints=[0, 0])})
    notes = []
    out = R.replay(archive, analyzer=FakeAnalyzer(), programs=["a"], log=notes.append)
    assert len(out["a"]) == 1 and any("no blanking frame" in n for n in notes)

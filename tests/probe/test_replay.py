"""Replay: measurements recomputed from an archived run, with no device."""

# pylint: disable=missing-function-docstring

import numpy as np

from syncsummoner.device.profile import Source
from syncsummoner.probe import replay as R
from syncsummoner.probe.archive import GAP, CHANNELS, FrameRow
from syncsummoner.probe.runner import ANALYSIS_WIDTH
from tests.probe.test_runner import FakeAnalyzer

SIZE = (16, 24)


def rgb(value, size=SIZE):
    """One RGB frame of a constant level, as the archive decodes them."""
    return np.full(size + (CHANNELS,), value, dtype=np.uint8)


class Reader:
    """Frame reader stub: rows and their frames, as the real one streams them."""

    def __init__(self, frames, *, setpoints, firmware="1.0.0"):
        self.frames = list(frames)
        self.meta = {"key_material": firmware, "width": SIZE[1], "height": SIZE[0], "frames": len(frames)}
        self.widths = []
        self.rows = [
            FrameRow(frame=i, program="p", params=tuple([max(s, 0)] * 12), setpoint=s)
            for i, s in enumerate(setpoints)
        ]

    def stream(self, *, start=0, count=None, width=None):
        """Frames in order, recording the width the decoder was asked to scale to."""
        self.widths.append(width)
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
    reader = Reader([rgb(v) for v in (10, 20, 30, 40)], setpoints=[0, 0, 1, 1])
    got = [(sp, params[0], len(frames)) for sp, params, frames in R.setpoints(reader)]
    assert got == [(0, 0, 2), (1, 1, 2)]


def test_setpoints_hands_on_float_frames_the_analyzer_can_take():
    reader = Reader([rgb(255), rgb(0)], setpoints=[0, 0])
    _, _, frames = next(R.setpoints(reader))
    assert [f.dtype for f in frames] == [np.dtype(np.float32)] * 2
    assert [float(f.max()) for f in frames] == [1.0, 0.0]


def test_setpoints_drops_the_frames_captured_between_setpoints():
    """A gap frame was captured while the parameters were still moving."""
    reader = Reader([rgb(v) for v in (10, 20, 30, 40)], setpoints=[GAP, 0, GAP, 1])
    got = [(sp, len(frames), round(255 * float(frames[0].max()))) for sp, _, frames in R.setpoints(reader)]
    assert got == [(0, 1, 20), (1, 1, 40)]


def test_a_gap_inside_one_setpoint_does_not_split_it():
    """A dropped frame mid-dwell is a gap in the stream, not a second setpoint."""
    reader = Reader([rgb(v) for v in (10, 20, 30)], setpoints=[0, GAP, 0])
    assert [(sp, len(frames)) for sp, _, frames in R.setpoints(reader)] == [(0, 2)]


def test_an_archive_of_nothing_but_gaps_measures_nothing():
    reader = Reader([rgb(10), rgb(20)], setpoints=[GAP, GAP])
    assert not list(R.setpoints(reader))
    assert not R.program_records(Archive({"p": reader}), "p", analyzer=FakeAnalyzer())


def test_setpoints_of_an_empty_archive():
    assert not list(R.setpoints(Reader([], setpoints=[])))


def test_the_decoder_is_asked_for_the_analysis_width():
    """Scaling in ffmpeg costs nothing here; scaling every frame in Python costs the read."""
    reader = Reader([rgb(90)], setpoints=[0])
    list(R.setpoints(reader))
    list(R.setpoints(reader, width=64))
    assert reader.widths == [ANALYSIS_WIDTH, 64]


def test_records_carry_the_archive_provenance():
    reader = Reader([rgb(90), rgb(95)], setpoints=[0, 0], firmware="1.0.0-rc.40")
    record = R.program_records(Archive({"p": reader}), "p", analyzer=FakeAnalyzer())[0]
    assert record.program == "p" and record.firmware == "1.0.0-rc.40"
    assert record.source is Source.HW and record.stimulus == R.STIMULUS
    assert record.params == tuple([0] * 12) and record.metrics


def test_records_of_a_program_the_archive_does_not_hold():
    assert not R.program_records(Archive({}), "absent", analyzer=FakeAnalyzer())


def test_replay_measures_every_committed_program():
    archive = Archive(
        {
            "a": Reader([rgb(90), rgb(95), rgb(90)], setpoints=[GAP, 1, 1]),
            "b": Reader([rgb(120), rgb(125), rgb(120)], setpoints=[GAP, 1, 1]),
        }
    )
    notes = []
    out = R.replay(archive, analyzer=FakeAnalyzer(), log=notes.append)
    assert sorted(out) == ["a", "b"] and all(len(v) == 1 for v in out.values())
    assert sorted(notes) == ["a: 1 setpoints measured", "b: 1 setpoints measured"]


def test_replay_measures_only_the_programs_it_was_given():
    archive = Archive({name: Reader([rgb(90), rgb(95)], setpoints=[0, 0]) for name in ("a", "b")})
    assert sorted(R.replay(archive, analyzer=FakeAnalyzer(), programs=["a"])) == ["a"]


def test_replay_notes_a_program_it_could_not_measure_but_reports_nothing_for_it():
    notes = []
    assert not R.replay(Archive({}), analyzer=FakeAnalyzer(), programs=["absent"], log=notes.append)
    assert notes == ["absent: 0 setpoints measured"]


def test_replay_reads_several_programs_at_once():
    """Decoding dominates and releases the lock, so the work overlaps."""
    archive = Archive({name: Reader([rgb(90), rgb(95)], setpoints=[0, 0]) for name in ("a", "b", "c", "d")})
    out = R.replay(archive, analyzer=FakeAnalyzer(), jobs=4)
    assert sorted(out) == ["a", "b", "c", "d"] and all(len(v) == 1 for v in out.values())

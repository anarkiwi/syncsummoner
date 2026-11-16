"""Replay: measurements recomputed from an archived run, with no device."""

# pylint: disable=missing-function-docstring

import os
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor

import numpy as np
import pytest

from syncsummoner.device.profile import Source
from syncsummoner.probe import codeframes as cf
from syncsummoner.probe import replay as R
from syncsummoner.probe.archive import GAP, CHANNELS, ArchiveError, FrameRow
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

    def passes(self, *, width=None, native=(), into=None):
        """Every frame once, the native ones asked for held in ``into`` as the decoder reaches them."""
        self.widths.append(width)
        keep = set(native)
        for i, frame in enumerate(self.frames):
            if i in keep:
                into[i] = frame
            yield frame


class Broken(Reader):
    """A reader whose decoder fails part way through the archive."""

    def passes(self, *, width=None, native=(), into=None):
        del width, native, into
        yield self.frames[0]
        raise ArchiveError("decoding gave 0 bytes")


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


@pytest.fixture(name="loop", scope="module")
def loop_fixture():
    """A loop at the stub's own geometry, so behaviour has something to register against."""
    return cf.build_loop(width=SIZE[1], height=SIZE[0], rng=np.random.default_rng(3), count=4, strip_px=2)


def sweep(levels=(90, 95, 100, 105), *, dwell=4):
    """A reader over a setpoint per level, a gap before each, every vector distinct."""
    frames, setpoints = [], []
    for setpoint, level in enumerate(levels):
        frames += [rgb(0)] + [rgb(level + k) for k in range(dwell)]
        setpoints += [GAP] + [setpoint] * dwell
    reader = Reader(frames, setpoints=setpoints)
    reader.rows = [
        FrameRow(
            frame=r.frame, program="p", params=(max(r.setpoint, 0) * 10,) + (0,) * 11, setpoint=r.setpoint
        )
        for r in reader.rows
    ]
    return reader


def test_one_program_is_measured_fitted_and_read_from_one_decode(loop):
    reader = sweep()
    result = R.replay_program(Archive({"p": reader}), "p", analyzer=FakeAnalyzer(), loop=loop)
    assert reader.widths == [ANALYSIS_WIDTH], "one decoder, at the analysis width"
    assert [r.state_index for r in result.records] == [0, 1, 2, 3]
    assert result.profile is not None and result.profile.program == "p"
    assert 0.0 <= result.behaviour.pointwise <= 1.0
    assert any("no program info archived" in note for note in result.notes)


def test_a_program_the_archive_does_not_hold_is_a_note(loop):
    result = R.replay_program(Archive({}), "absent", analyzer=FakeAnalyzer(), loop=loop)
    assert not result.records and result.profile is None and result.notes == ["absent: no committed archive"]


def test_a_program_that_will_not_decode_does_not_end_the_refit(loop):
    reader = Broken([rgb(90)] * 4, setpoints=[0] * 4)
    result = R.replay_program(Archive({"p": reader}), "p", analyzer=FakeAnalyzer(), loop=loop)
    assert not result.records and result.profile is None and result.behaviour == R.B.Behaviour()
    assert "ArchiveError" in result.notes[0]


def test_replay_keys_what_measured_and_notes_what_did_not(loop):
    archive = Archive({"a": sweep(), "b": sweep((120, 125))})
    notes = []
    out = R.replay(archive, analyzer=FakeAnalyzer(), loop=loop, log=notes.append, executor=ThreadPoolExecutor)
    assert sorted(out) == ["a", "b"] and [len(out[n].records) for n in "ab"] == [4, 2]
    out = R.replay(archive, analyzer=FakeAnalyzer(), programs=["a", "gone"], loop=loop, log=notes.append)
    assert sorted(out) == ["a"] and "gone: no committed archive" in notes


def test_worker_processes_return_what_threads_do(loop):
    """The pool sends the archive and the analyzer, a module by its name, to each worker."""
    from syncsummoner import aesthetics  # pylint: disable=import-outside-toplevel

    archive = Archive({name: sweep() for name in ("a", "b", "c")})
    threaded = R.replay(archive, analyzer=aesthetics, loop=loop, jobs=2, executor=ThreadPoolExecutor)
    pooled = R.replay(archive, analyzer=aesthetics, loop=loop, jobs=2, executor=ProcessPoolExecutor)
    assert sorted(pooled) == ["a", "b", "c"]
    for name, result in pooled.items():
        assert [r.metrics for r in result.records] == [r.metrics for r in threaded[name].records]
        assert result.behaviour == threaded[name].behaviour


def test_workers_start_under_single_threaded_libraries_and_the_caller_keeps_its_own(monkeypatch):
    """A worker sizes its BLAS pool from the environment it starts in, not the parent's."""
    monkeypatch.setenv("OPENBLAS_NUM_THREADS", "7")
    monkeypatch.delenv("OMP_NUM_THREADS", raising=False)
    seen = {}

    class Recording(ThreadPoolExecutor):
        """A pool that records the environment it was started in."""

        def __init__(self, **kwargs):
            seen.update({name: os.environ.get(name) for name in R.SINGLE_THREADED})
            super().__init__(**kwargs)

    R.replay(Archive({}), analyzer=FakeAnalyzer(), programs=[], executor=Recording)
    assert seen == R.SINGLE_THREADED
    assert os.environ["OPENBLAS_NUM_THREADS"] == "7" and "OMP_NUM_THREADS" not in os.environ

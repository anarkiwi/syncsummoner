"""Replay: measurements recomputed from an archived run, with no device."""

# pylint: disable=missing-function-docstring

import json
import os
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
from pathlib import Path

import numpy as np
import pytest

from syncsummoner import aesthetics
from syncsummoner.device.profile import Source
from syncsummoner.device.recorder import INSPECT_SIZE
from syncsummoner.probe import codeframes as cf
from syncsummoner.probe import replay as R
from syncsummoner.probe.patterns import burn_timecode
from syncsummoner.probe.archive import GAP, CHANNELS, ArchiveError, FrameRow
from syncsummoner.probe.runner import ANALYSIS_WIDTH
from syncsummoner.probe.store import slug
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

    def passes(self, *, width=None, native=(), into=None, strip_px=0, strips=None):
        """Every frame once, the native ones asked for held in ``into`` and any strip kept in ``strips``."""
        self.widths.append(width)
        keep = set(native)
        for i, frame in enumerate(self.frames):
            if i in keep:
                into[i] = frame
            if strip_px and strips is not None:
                strips.append(frame[-strip_px:])
            yield frame[: frame.shape[0] - strip_px]


class Broken(Reader):
    """A reader whose decoder fails part way through the archive."""

    def passes(self, *, width=None, native=(), into=None, strip_px=0, strips=None):
        del width, native, into, strip_px, strips
        yield self.frames[0]
        raise ArchiveError("decoding gave 0 bytes")


class Archive:
    """Frame archive stub holding one reader per program."""

    def __init__(self, readers, directory="."):
        self.readers = dict(readers)
        self.directory = directory

    def committed(self):
        return {name: reader.meta for name, reader in self.readers.items()}

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
    assert not {"si_ratio", "ti_ratio", "motion_follow"} & set(record.metrics)


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


def test_an_undeclared_program_is_refitted_unnamed_with_the_reason_noted(loop, monkeypatch):
    def undeclared(meta, program):
        raise R.UndeclaredProgramError(f"no single .vmprog declares {program!r}")

    monkeypatch.setattr(R, "archived_specs", undeclared)
    result = R.replay_program(Archive({"p": sweep()}), "p", analyzer=FakeAnalyzer(), loop=loop)
    assert result.profile is not None and "p: no single .vmprog declares 'p'" in result.notes


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


STRIP, PERIOD, FPS, DWELL = 8, 48, 30.0, 30


def clip_frames():
    """A looping clip of a gray texture panning at a varying speed, each frame's index burnt in its strip."""
    width, height = INSPECT_SIZE
    rng = np.random.default_rng(1)
    field = np.cumsum(np.cumsum(rng.standard_normal((height, 4 * width)), axis=0), axis=1)
    field = (field - field.min()) / np.ptp(field)
    shift = np.cumsum(rng.integers(0, 6, size=PERIOD))
    frames = []
    for n in range(PERIOD):
        body = np.repeat(np.roll(field, -int(shift[n]), axis=1)[:, :width, None], CHANNELS, axis=2)
        full = np.concatenate([body, np.zeros((STRIP, width, CHANNELS))])
        frames.append(np.rint(255 * burn_timecode(full, n, bits=16, strip_px=STRIP)).astype(np.uint8))
    return frames


@pytest.fixture(name="clip", scope="module")
def clip_fixture():
    """The clip's frames and the SI and TI of its pictures, as the source series measures them."""
    frames = clip_frames()
    return frames, aesthetics.information(f[:-STRIP, :, 0].astype(np.float32) / 255.0 for f in frames)


def footage_reader(clip, programs, *, wipe=False, captures=None, rate=1):
    """A card capturing the played loop ``rate`` times a clip frame, through one program per setpoint.

    ``programs`` maps each setpoint to ``"pass"`` or ``"freeze"``; a frozen dwell holds its first
    frame, strip and all; ``wipe`` blanks every strip.
    """
    frames, setpoints = [], []
    for setpoint, program in enumerate(programs):
        start = len(frames) + 2
        for i in range(len(frames), start + DWELL):
            shown = clip[((start if program == "freeze" and i >= start else i) // rate) % PERIOD].copy()
            if wipe:
                shown[-STRIP:] = 0
            frames.append(shown)
            setpoints.append(GAP if i < start else setpoint)
    frames, setpoints = frames[:captures], setpoints[:captures]
    reader = Reader(frames, setpoints=setpoints)
    height, width = frames[0].shape[:2]
    stimulus = {"kind": "footage", "clip": "stimulus.mkv", "fps": FPS, "frames": PERIOD, "bits": 16}
    reader.meta.update(width=width, height=height, stimulus={**stimulus, "strip_px": STRIP})
    reader.rows = [replace_captured(row, (row.frame + 0.37) / (rate * FPS)) for row in reader.rows]
    return reader


def replace_captured(row, captured):
    return FrameRow(
        frame=row.frame, program=row.program, params=row.params, setpoint=row.setpoint, captured=captured
    )


def holds_for(directory, settle=4, **named):
    """``probe hold`` records settling each named program in ``settle`` frames, as ``hold-<slug>.json``."""
    for program in named or {"p": settle}:
        path = Path(directory) / f"hold-{slug(program)}.json"
        path.write_text(
            json.dumps({"program": program, "settle": named.get(program, settle)}), encoding="utf-8"
        )
    return directory


def survival_of(clip, reader, holds):
    result = R.replay_program(
        Archive({"p": reader}), "p", analyzer=FakeAnalyzer(), series=clip[1], holds=holds_for(holds)
    )
    return result, [r.metrics for r in result.records]


def test_footage_setpoints_are_scored_on_what_they_kept_of_the_clip(clip, tmp_path):
    """Passthrough keeps the clip's picture, motion and its timing; a frozen dwell keeps no motion."""
    result, metrics = survival_of(
        clip, footage_reader(clip[0], ["pass", "freeze", "pass", "freeze"]), tmp_path
    )
    assert "p: footage clock from strip" in result.notes
    assert [r.stimulus for r in result.records] == ["footage"] * 4
    for kept in metrics[::2]:
        assert kept["si_ratio"] == pytest.approx(1.0, abs=1e-6)
        assert kept["ti_ratio"] == pytest.approx(1.0, abs=1e-6)
        assert kept["motion_follow"] > 0.99
    for kept in metrics[1::2]:
        assert kept["ti_ratio"] == 0.0 and kept["motion_follow"] == 0.0
    assert all("concentration" in kept for kept in metrics), "the analyzer's own metrics stay"


def test_a_clip_whose_strip_never_reads_is_aligned_by_its_motion(clip, tmp_path):
    """Captured faster than it plays, a clip's motion fixes where each of its frames went up."""
    result, metrics = survival_of(
        clip, footage_reader(clip[0], ["pass", "pass"], wipe=True, rate=2), tmp_path
    )
    assert "p: footage clock from motion" in result.notes
    for kept in metrics:
        assert kept["ti_ratio"] == pytest.approx(1.0, abs=1e-6) and kept["motion_follow"] > 0.99


def test_with_no_clock_survival_is_left_out_rather_than_zero(clip, tmp_path):
    result, metrics = survival_of(clip, footage_reader(clip[0], ["pass"], wipe=True, captures=4), tmp_path)
    assert "p: footage clock from none" in result.notes
    assert metrics and not {"si_ratio", "ti_ratio", "motion_follow"} & set(metrics[0])


def test_the_refit_reads_each_played_clip_once_for_every_footage_program(clip, monkeypatch, tmp_path):
    seen = []

    def series(path, *, strip_px):
        seen.append((str(path), strip_px))
        return clip[1]

    monkeypatch.setattr(R, "source_series", series)
    archive = Archive({name: footage_reader(clip[0], ["pass"]) for name in "ab"}, directory="/archive")
    holds = holds_for(tmp_path, a=4, b=4)
    out = R.replay(archive, analyzer=FakeAnalyzer(), executor=ThreadPoolExecutor, holds=holds)
    assert seen == [("/archive/stimulus.mkv", STRIP)]
    assert all(out[name].records[0].metrics["ti_ratio"] == pytest.approx(1.0, abs=1e-6) for name in "ab")
    result = R.replay_program(archive, "a", analyzer=FakeAnalyzer(), holds=holds)
    assert len(seen) == 2 and "si_ratio" in result.records[0].metrics


def test_a_footage_profile_settles_by_its_programs_hold_record_on_every_parameter(clip, tmp_path):
    """On moving footage the frame-difference plateau measures the footage; the hold measured the program."""
    result, _ = survival_of(clip, footage_reader(clip[0], ["pass", "freeze", "pass", "freeze"]), tmp_path)
    assert result.profile.settle_frames and set(result.profile.settle_frames.values()) == {4}
    assert {r.settle_frames for r in result.records} != {4}


def test_a_footage_archive_without_its_hold_record_is_refused_before_any_decode(clip, tmp_path):
    archive = Archive({name: footage_reader(clip[0], ["pass"]) for name in "ab"}, directory="/archive")
    holds = holds_for(tmp_path, a=4)
    with pytest.raises(FileNotFoundError, match=r"b: .*hold-b\.json is missing"):
        R.replay(archive, analyzer=FakeAnalyzer(), executor=ThreadPoolExecutor, holds=holds)
    with pytest.raises(FileNotFoundError, match="hold record hold-a.json; give its directory"):
        R.replay_program(archive, "a", analyzer=FakeAnalyzer())
    holds_for(tmp_path, b=4)
    (tmp_path / "hold-b.json").write_text(json.dumps({"program": "a", "settle": 4}), encoding="utf-8")
    with pytest.raises(ValueError, match="measured a, not b"):
        R.hold_settle(archive.reader("b").meta, "b", tmp_path)
    assert R.hold_settle({}, "codeframes", None) is None


def delayed(delay, count=150, seed=3):
    """A moving source and an output showing it ``delay`` frames late, as 8-bit planes and source SI/TI."""
    rng = np.random.default_rng(seed)
    steps = rng.normal(0.0, 1.0, (count, 16, 16)) * rng.uniform(0.2, 3.0, count)[:, None, None]
    source = np.clip(128 + np.cumsum(steps, axis=0), 0, 255).astype(np.uint8)
    out = [source[max(n - delay, 0)] for n in range(count)]
    return out, aesthetics.information(source.astype(np.float32) / 255.0)


def kept_of(delay, ambiguity):
    frames, series = delayed(delay)
    n = np.arange(len(frames))
    return R._kept(frames, n, n, series, len(frames), ambiguity)  # pylint: disable=protected-access


def test_a_delay_every_candidate_lead_finds_is_kept_with_its_lead_as_an_interval():
    got = kept_of(1, 1)
    assert got["motion_follow"] > 0.99 and (got["motion_lead_lo"], got["motion_lead_hi"]) == (0.0, 1.0)
    sure = kept_of(1, 0)
    assert (sure["motion_lead_lo"], sure["motion_lead_hi"]) == (1.0, 1.0)


def test_a_delay_only_one_candidate_can_reach_is_attribution_ambiguous():
    assert kept_of(2, 1) == {"attribution_ambiguous": 1.0}
    unique = kept_of(2, 0)
    assert "attribution_ambiguous" not in unique and unique["motion_lead_lo"] == unique["motion_lead_hi"]


def test_the_union_of_candidate_leads_is_searched_at_a_bound_sized_to_it(monkeypatch):
    asked = []
    real = aesthetics.critical_z

    def spy(*args, **kwargs):
        asked.append(args[0] if args else kwargs["leads"])
        return real(*args, **kwargs)

    monkeypatch.setattr(R.aesthetics, "critical_z", spy)
    kept_of(1, 0)
    kept_of(1, 2)
    assert asked == [len(aesthetics.LEADS), len(aesthetics.LEADS) + 2]


def followed(lead, settings=6, n=300, seed=3):
    """Per setting, source and output SI/TI whose output TI follows the source's ``lead`` frames later."""
    rng = np.random.default_rng(seed)
    out = []
    for _ in range(settings):
        src_ti = np.abs(np.cumsum(rng.normal(0, 1, n))) + 1.0
        out_ti = np.r_[np.full(lead, np.nan), src_ti[: n - lead]] + rng.normal(0, 0.05, n)
        out.append((np.ones(n), src_ti, np.ones(n), out_ti))
    return out


def test_pooled_follow_selects_the_shift_whose_own_leads_reach_the_programs_delay():
    pooled = R.pooled_follow(followed(2), 1)
    said = R.select_shift(pooled)
    assert pooled[1] > pooled[0] and said["shift"] == 1 and said["margin_z"] > said["critical_z"]


def test_a_delay_every_shift_reaches_selects_none():
    said = R.select_shift(R.pooled_follow(followed(1), 1))
    assert said["shift"] is None and abs(said["pooled_z"][0] - said["pooled_z"][1]) < 1e-9
    assert R.select_shift(np.array([4.0]))["shift"] == 0


def test_a_locked_takes_rows_are_scored_against_the_clip_frames_they_played(clip):
    """Rows name the played frame, lead-in included; the source series starts after it."""
    frames, (si, ti) = clip[0], clip[1]
    lead = 7
    rows = [FrameRow(frame=j, program="p", params=(0,) * 12, setpoint=0, shown=lead + j) for j in range(40)]
    stimulus = {"frames": PERIOD, "bits": 16, "strip_px": STRIP, "lead": lead}
    played = [f[:-STRIP, :, 0] for f in frames[:40]]
    (kept,), how = R.survival_metrics(rows, played, [None] * 40, stimulus, (si, ti))
    assert how == "take" and kept["motion_follow"] > 0.99
    assert kept["si_ratio"] == pytest.approx(1.0, abs=1e-3)

"""Harvest against natural footage: the prepared clip, its record, and resuming against it."""

# pylint: disable=missing-function-docstring

import dataclasses
import hashlib
import json
from pathlib import Path


from syncsummoner.compose.render import RenderConfig
from syncsummoner.probe import harvest as H
from syncsummoner.probe.store import KeyKind, ProgramKey
from tests.device.conftest import FakeClock
from tests.probe.test_harvest import CONFIG, FIRMWARE, KEY, Archive, Player, Port, Rec, Sess, no_ffmpeg, run

RENDER = RenderConfig.for_format(CONFIG.timing)

__all__ = ["no_ffmpeg"]


class Writer:
    """Stands in for the timecoded encoder, writing ``frames`` bytes and recording each call."""

    def __init__(self, frames=90):
        self.frames = frames
        self.calls = []

    def __call__(self, source, out, *, config, seconds, start):
        self.calls.append((source, Path(out).name, (config.width, config.height, config.fps), seconds, start))
        Path(out).write_bytes(bytes(range(self.frames % 256)) * 3)
        return self.frames


def test_footage_is_written_at_session_timing_with_a_record_of_the_clip(tmp_path):
    writer = Writer()
    record = H.prepare_footage("film.mp4", tmp_path, render=RENDER, start=5.0, seconds=3.0, write=writer)
    assert writer.calls == [("film.mp4", "stimulus.mkv", (1920, 1080, 30.0), 3.0, 5.0)]
    assert record == json.loads((tmp_path / "stimulus.json").read_text(encoding="utf-8"))
    assert record["sha256"] == hashlib.sha256((tmp_path / "stimulus.mkv").read_bytes()).hexdigest()
    assert (record["kind"], record["clip"], record["frames"], record["fps"]) == (
        "footage",
        "stimulus.mkv",
        90,
        30.0,
    )
    assert (record["bits"], record["strip_px"], record["source"]) == (16, 8, "film.mp4")
    assert (record["start"], record["seconds"]) == (5.0, 3.0)


def test_prepared_footage_is_reused_only_for_the_same_excerpt_and_an_intact_clip(tmp_path):
    writer = Writer()
    first = H.prepare_footage("film.mp4", tmp_path, render=RENDER, write=writer)
    assert H.prepare_footage("film.mp4", tmp_path, render=RENDER, write=writer) == first
    assert len(writer.calls) == 1, "a resumed run plays the clip it began with"
    H.prepare_footage("film.mp4", tmp_path, render=RENDER, start=1.0, write=writer)
    assert len(writer.calls) == 2, "another excerpt is another stimulus"
    (tmp_path / "stimulus.mkv").write_bytes(b"truncated")
    H.prepare_footage("film.mp4", tmp_path, render=RENDER, start=1.0, write=writer)
    assert len(writer.calls) == 3, "a clip that no longer hashes as recorded is rewritten"


def test_a_codeframe_run_names_its_loop_by_seed():
    assert H.run_stimulus(CONFIG) == {"kind": "codeframes", "loop_seed": 7, "loop_frames": 2}


def footage_config(tmp_path):
    """A config playing a prepared clip, and that clip's record."""
    record = H.prepare_footage("film.mp4", tmp_path / "out", render=RENDER, write=Writer())
    return dataclasses.replace(CONFIG, footage=str(tmp_path / "out" / "stimulus.mkv"), loop_fps=30.0), record


def test_a_footage_run_uploads_the_clip_and_archives_against_it(tmp_path):
    config, record = footage_config(tmp_path)
    player, archive = Player(running=False), Archive(tmp_path / "raw")
    run(archive, player=player, config=config)
    assert player.sent == config.footage and player.uploaded == 1
    assert H.run_stimulus(config) == record and player.rates and set(player.rates) == {30.0}
    assert archive.commits == ["Alpha", "Beta"] and all(archive.stimuli[p] == record for p in archive.commits)


def test_a_resume_re_archives_what_was_recorded_against_another_stimulus(tmp_path):
    config, record = footage_config(tmp_path)
    keys = [("Alpha", KEY), ("Beta", ProgramKey("Beta", FIRMWARE, KeyKind.NAME_FIRMWARE))]
    archive = Archive(tmp_path / "raw", stored=keys)
    archive.stimuli["Beta"] = record
    report = run(archive, config=config)
    assert [r.cached for r in report.results] == [False, True], "Alpha was recorded against codeframes"
    assert archive.commits == ["Alpha"] and archive.stimuli["Alpha"] == record


def test_harvest_program_alone_names_the_runs_stimulus(tmp_path):
    clock, archive, port = FakeClock(), Archive(tmp_path), Port()
    H.harvest_program(
        Sess(port),
        Rec(clock),
        archive,
        port,
        "Alpha",
        KEY,
        config=CONFIG,
        plan=lambda info: None,
        sleep=clock.sleep,
        clock=clock,
    )
    assert archive.stimuli["Alpha"] == H.run_stimulus(CONFIG)

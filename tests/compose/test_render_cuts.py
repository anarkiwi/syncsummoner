"""Cutting a run of takes: picture starts, reused takes, take names."""

# pylint: disable=missing-function-docstring

from pathlib import Path

import pytest

from syncsummoner.compose import render as R
from syncsummoner.compose.score import Layer, Score, Section


def _cut_run(monkeypatch, tmp_path, starts, *, reuse=False, **kwargs):
    """Cut a run whose takes measure ``starts``, returning what was rendered and assembled."""
    seen = {"rendered": [], "starts": None}

    def fake_render(score, source, out, **kw):
        del source, kw
        seen["rendered"].append(score.layers[0].program)
        Path(out).write_bytes(b"take")
        return R.TakeReport(frames=10, picture=10, frozen=False, luma=0.4)

    monkeypatch.setattr(R, "picture_start", lambda path, **kw: starts[Path(path).name])
    monkeypatch.setattr(R, "assemble", lambda plan, paths, out, **kw: seen.update(kw, paths=paths))
    names = [name.removeprefix("take-").removesuffix(".mkv") for name in starts]
    R.render_cuts(
        Score(
            duration=30.0,
            sections=[Section(10.0 * i, 10.0 * (i + 1), "A") for i in range(len(starts))],
            layers=[Layer(names[0], 0, [])],
        ),
        "clip.mkv",
        tmp_path / "out.mkv",
        profiles={},
        programs=names,
        takes=tmp_path,
        prepared=True,
        reuse=reuse,
        pass_render=kwargs.pop("pass_render", fake_render),
        **kwargs,
    )
    return seen


def test_a_take_whose_strip_is_lost_is_cut_at_the_runs_median_start(monkeypatch, tmp_path, caplog):
    starts = {
        "take-Derez.mkv": 1.9,
        "take-Pinwheel.mkv": None,
        "take-Lorenz.mkv": 1.4,
        "take-Sfumato.mkv": 2.0,
    }
    with caplog.at_level("WARNING", logger="syncsummoner"):
        seen = _cut_run(monkeypatch, tmp_path, starts)
    assert seen["starts"] == {"Derez": 1.9, "Pinwheel": pytest.approx(1.9), "Lorenz": 1.4, "Sfumato": 2.0}
    warning = [r.getMessage() for r in caplog.records if r.levelname == "WARNING"]
    assert warning == ["Pinwheel: timecode strip did not survive; using the run's median picture start 1.90s"]


def test_a_run_with_no_legible_strip_is_refused_rather_than_cut_from_lead_in(monkeypatch, tmp_path):
    with pytest.raises(R.LostTimecodeError, match="Pinwheel: the timecode strip did not survive"):
        _cut_run(monkeypatch, tmp_path, {"take-Pinwheel.mkv": None})


def test_reused_takes_skip_their_pass_and_are_cut_like_fresh_ones(monkeypatch, tmp_path, caplog):
    (tmp_path / "take-Derez.mkv").write_bytes(b"old")
    with caplog.at_level("INFO", logger="syncsummoner"):
        seen = _cut_run(monkeypatch, tmp_path, {"take-Derez.mkv": 1.0, "take-Lorenz.mkv": 2.0}, reuse=True)
    assert seen["rendered"] == ["Lorenz"]
    assert seen["starts"] == {"Derez": 1.0, "Lorenz": 2.0}
    assert f"Derez: reusing take {tmp_path / 'take-Derez.mkv'}" in caplog.messages


def test_without_reuse_an_existing_take_is_rendered_again(monkeypatch, tmp_path):
    (tmp_path / "take-Derez.mkv").write_bytes(b"old")
    assert _cut_run(monkeypatch, tmp_path, {"take-Derez.mkv": 1.0})["rendered"] == ["Derez"]


def test_a_fully_reused_render_never_touches_the_rig(monkeypatch, tmp_path):
    """Re-cutting or re-mastering must cost no rig time and need no rig at all."""

    def touched(*args, **kwargs):
        raise AssertionError(f"rig touched: {args} {kwargs}")

    for name in ("open_rig", "device_programs", "write_timecoded", "render_played"):
        monkeypatch.setattr(R, name, touched)
    starts = {"take-Derez.mkv": 1.0, "take-Lorenz.mkv": 1.2}
    for name in starts:
        (tmp_path / name).write_bytes(b"old")
    seen = _cut_run(
        monkeypatch, tmp_path, starts, reuse=True, pass_render=touched, available=R.device_programs
    )
    assert seen["starts"] == {"Derez": 1.0, "Lorenz": 1.2}


def test_the_device_is_asked_for_its_programs_only_when_a_pass_is_due(monkeypatch, tmp_path):
    asked = []
    seen = _cut_run(
        monkeypatch, tmp_path, {"take-Derez.mkv": 1.0}, available=lambda: asked.append(1) or ["Derez"]
    )
    assert asked == [1] and seen["rendered"] == ["Derez"]


def test_an_author_qualified_program_gets_a_take_file_not_a_subdirectory(monkeypatch, tmp_path):
    """``author/Name`` once named a directory that did not exist, and the pass died opening it."""
    program = "czartroniks/Roll the Bones"
    seen = {}
    monkeypatch.setattr(R, "picture_start", lambda path, **kw: 1.0)
    monkeypatch.setattr(R, "assemble", lambda plan, paths, out, **kw: seen.update(paths))

    def fake_render(score, source, out, **kw):
        del score, source, kw
        Path(out).write_bytes(b"take")

    for reuse in (False, True):
        R.render_cuts(
            Score(duration=10.0, sections=[Section(0.0, 10.0, "A")], layers=[Layer(program, 0, [])]),
            "clip.mkv",
            tmp_path / "out.mkv",
            profiles={},
            programs=[program],
            takes=tmp_path,
            prepared=True,
            reuse=reuse,
            pass_render=fake_render if not reuse else None,
        )
        take = Path(seen[program])
        assert take.parent == tmp_path and take.name.startswith("take-czartroniks_Roll_the_Bones-")

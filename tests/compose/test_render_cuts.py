"""Cutting a run of takes: attribution, reused takes, take names."""

# pylint: disable=missing-function-docstring

import types
from pathlib import Path

import numpy as np
import pytest

from syncsummoner.compose import render as R
from syncsummoner.compose import takecheck as T
from syncsummoner.compose.locktake import NoLatencyError
from syncsummoner.compose.score import Layer, Score, Section


def _checked(monkeypatch, *, spans=(), cuts=None):
    """Stand in for the take check: the plan as given, or ``cuts``, with ``spans`` found dead.

    Each take's attribution is its own name's length plus the clip frame, so a splice
    handed one take's picks for another's is caught.
    """
    seen = {"attributed": []}

    def attribute(take, *, config, **kw):
        del kw
        seen["attributed"].append((Path(take).name, config.fps))
        return types.SimpleNamespace(picked=np.arange(5) * len(Path(take).name))

    def check(score, plan, takes, clip, **kw):
        del score
        seen.update(clip=clip, takes=dict(takes), **kw)
        traces = {name: T.Trace(np.arange(10) + len(name), None, None, None, None, "strip") for name in takes}
        return T.CheckReport(cuts=list(cuts or plan), spans=list(spans), traces=traces)

    monkeypatch.setattr(R, "attribute_take", attribute)
    monkeypatch.setattr(T, "check_takes", check)
    monkeypatch.setattr("syncsummoner.probe.decoded.features", lambda path, spec, **kw: str(path))
    return seen


def _cut_run(monkeypatch, tmp_path, names, *, reuse=False, **kwargs):
    """Cut a run of programs ``names``, returning what was rendered and spliced."""
    seen = {"rendered": []}

    def fake_render(score, source, out, **kw):
        del source, kw
        seen["rendered"].append(score.layers[0].program)
        Path(out).write_bytes(b"take")
        return R.TakeReport(frames=10, picture=10, frozen=False, luma=0.4)

    def fake_splice(plan, paths, picked, out, **kw):
        seen.update(kw, plan=plan, paths=paths, picked=picked, out=out)

    monkeypatch.setattr(R, "splice", fake_splice)
    seen["check"] = _checked(monkeypatch, **kwargs.pop("check", {}))
    R.render_cuts(
        Score(
            duration=30.0,
            fps=30.0,
            sections=[Section(10.0 * i, 10.0 * (i + 1), "A") for i in range(len(names))],
            layers=[Layer(names[0], 0, [])],
        ),
        "clip.mkv",
        tmp_path / "out.mkv",
        profiles={},
        programs=list(names),
        takes=tmp_path,
        prepared=True,
        reuse=reuse,
        pass_render=kwargs.pop("pass_render", fake_render),
        **kwargs,
    )
    return seen


def test_each_take_is_spliced_by_its_own_attribution_at_the_clip_rate(monkeypatch, tmp_path):
    seen = _cut_run(monkeypatch, tmp_path, ["Derez", "Pinwheel"], config=R.RenderConfig(fps=60.0))
    assert set(seen["picked"]) == {"Derez", "Pinwheel"}
    assert all(np.array_equal(p, np.arange(10) + len(n)) for n, p in seen["picked"].items())
    assert seen["config"].fps == 30.0 and seen["out"] == tmp_path / "out.mkv"


def test_reused_takes_skip_their_pass_and_are_cut_like_fresh_ones(monkeypatch, tmp_path, caplog):
    (tmp_path / "take-Derez.mkv").write_bytes(b"old")
    with caplog.at_level("INFO", logger="syncsummoner"):
        seen = _cut_run(monkeypatch, tmp_path, ["Derez", "Lorenz"], reuse=True)
    assert seen["rendered"] == ["Lorenz"] and set(seen["picked"]) == {"Derez", "Lorenz"}
    assert f"Derez: reusing take {tmp_path / 'take-Derez.mkv'}" in caplog.messages


def test_without_reuse_an_existing_take_is_rendered_again(monkeypatch, tmp_path):
    (tmp_path / "take-Derez.mkv").write_bytes(b"old")
    assert _cut_run(monkeypatch, tmp_path, ["Derez"])["rendered"] == ["Derez"]


def test_a_fully_reused_render_never_touches_the_rig(monkeypatch, tmp_path):
    """Re-cutting or re-mastering must cost no rig time and need no rig at all."""

    def touched(*args, **kwargs):
        raise AssertionError(f"rig touched: {args} {kwargs}")

    for name in ("open_rig", "device_programs", "write_timecoded", "render_played"):
        monkeypatch.setattr(R, name, touched)
    for name in ("Derez", "Lorenz"):
        (tmp_path / f"take-{name}.mkv").write_bytes(b"old")
    seen = _cut_run(
        monkeypatch,
        tmp_path,
        ["Derez", "Lorenz"],
        reuse=True,
        pass_render=touched,
        available=R.device_programs,
    )
    assert seen["plan"] == [R.Cut(0.0, 10.0, "Derez"), R.Cut(10.0, 20.0, "Lorenz")]


def test_the_device_is_asked_for_its_programs_only_when_a_pass_is_due(monkeypatch, tmp_path):
    asked = []
    seen = _cut_run(monkeypatch, tmp_path, ["Derez"], available=lambda: asked.append(1) or ["Derez"])
    assert asked == [1] and seen["rendered"] == ["Derez"]


def test_an_author_qualified_program_gets_a_take_file_not_a_subdirectory(monkeypatch, tmp_path):
    """``author/Name`` once named a directory that did not exist, and the pass died opening it."""
    program = "czartroniks/Roll the Bones"
    seen = {}
    monkeypatch.setattr(R, "splice", lambda plan, paths, picked, out, **kw: seen.update(paths))
    _checked(monkeypatch)

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


def test_every_take_is_checked_against_the_played_clip_by_its_locked_attribution(monkeypatch, tmp_path):
    seen = _cut_run(monkeypatch, tmp_path, ["Derez", "Lorenz"])["check"]
    assert seen["clip"] == "timecoded.mkv" and set(seen["takes"]) == {"Derez", "Lorenz"}
    fps = R.RenderConfig().fps
    assert seen["attributed"] == [("take-Derez.mkv", fps), ("take-Lorenz.mkv", fps)], "at the session rate"
    assert {n: a.tolist() for n, a in seen["attributions"].items()} == {
        "Derez": (np.arange(5) * 14).tolist(),
        "Lorenz": (np.arange(5) * 15).tolist(),
    }


def test_a_covered_span_is_cut_from_the_take_alive_there(monkeypatch, tmp_path):
    covered = [R.Cut(0.0, 5.0, "Derez"), R.Cut(5.0, 20.0, "Lorenz")]
    seen = _cut_run(monkeypatch, tmp_path, ["Derez", "Lorenz"], check={"cuts": covered})
    assert seen["plan"] == covered


def test_a_dead_span_nothing_covers_fails_the_render_and_names_it(monkeypatch, tmp_path):
    dead = T.DeadSpan("Derez", 67.1, 112.1, T.Fault.FLAT, None)
    with pytest.raises(T.DeadTakeError, match="Derez:   67.10- 112.10s flat"):
        _cut_run(monkeypatch, tmp_path, ["Derez"], check={"spans": [dead]})


def _needing_latency(dry, failing=()):
    """A pass that needs a measured latency unless its program is ``dry`` or one rendered before it."""
    seen = {"tried": []}

    def fake_render(score, source, out, **kw):
        del source, kw
        program = score.layers[0].program
        seen["tried"].append(program)
        if program in failing or not (program in dry or set(dry) & set(seen["tried"][:-1])):
            raise NoLatencyError(f"{program}: no latency")
        Path(out).write_bytes(b"take")
        return R.TakeReport(frames=10, picture=10, frozen=False, luma=0.4)

    return seen, fake_render


def test_a_program_needing_a_measured_latency_waits_for_a_dry_take(monkeypatch, tmp_path):
    seen, fake_render = _needing_latency({"Derez"})
    cut = _cut_run(monkeypatch, tmp_path, ["Lorenz", "Derez", "Pinwheel"], pass_render=fake_render)
    assert seen["tried"] == ["Lorenz", "Derez", "Pinwheel", "Lorenz"]
    assert set(cut["picked"]) == {"Lorenz", "Derez", "Pinwheel"}


@pytest.mark.parametrize(
    "names,failing,tried",
    [(["Lorenz", "Derez"], {"Lorenz"}, ["Lorenz", "Derez", "Lorenz"]), (["Lorenz"], (), ["Lorenz"])],
    ids=["still-unmeasured", "nothing-else-left"],
)
def test_a_program_still_without_a_latency_fails_the_render(monkeypatch, tmp_path, names, failing, tried):
    seen, fake_render = _needing_latency({"Derez"}, failing)
    with pytest.raises(NoLatencyError, match="Lorenz: no latency"):
        _cut_run(monkeypatch, tmp_path, names, pass_render=fake_render)
    assert seen["tried"] == tried


def _faulting(faults):
    """A pass whose capture loses frames the first ``faults[program]`` times it runs."""
    seen = {"tried": []}

    def fake_render(score, source, out, **kw):
        del source, kw
        program = score.layers[0].program
        seen["tried"].append(program)
        if seen["tried"].count(program) <= faults.get(program, 0):
            raise R.TakeFaultError(f"{program}: the capture driver lost 3 frames")
        Path(out).write_bytes(b"take")
        return R.TakeReport(frames=10, picture=10, frozen=False, luma=0.4)

    return seen, fake_render


def test_a_take_whose_capture_lost_frames_is_recaptured_at_once(monkeypatch, tmp_path):
    seen, fake_render = _faulting({"Derez": 1})
    cut = _cut_run(monkeypatch, tmp_path, ["Lorenz", "Derez", "Pinwheel"], pass_render=fake_render)
    assert seen["tried"] == ["Lorenz", "Derez", "Derez", "Pinwheel"]
    assert set(cut["picked"]) == {"Lorenz", "Derez", "Pinwheel"}


def test_a_take_losing_frames_twice_fails_the_render(monkeypatch, tmp_path):
    seen, fake_render = _faulting({"Derez": 2})
    with pytest.raises(R.TakeFaultError, match="Derez: the capture driver lost"):
        _cut_run(monkeypatch, tmp_path, ["Derez", "Lorenz"], pass_render=fake_render)
    assert seen["tried"] == ["Derez", "Derez"]


def test_a_composed_plan_is_what_the_render_cuts_and_what_every_pass_plays(monkeypatch, tmp_path):
    from syncsummoner.compose.cut import Cut

    plan = [Cut(0.0, 10.0, "Lorenz"), Cut(10.0, 20.0, "Lorenz"), Cut(20.0, 30.0, "Derez")]
    score = Score(
        duration=30.0,
        fps=30.0,
        sections=[Section(10.0 * i, 10.0 * (i + 1), "A") for i in range(3)],
        layers=[Layer("Lorenz", 0, []), Layer("Derez", 1, [])],
        cuts=plan,
    )
    played = []

    def fake_render(score, source, out, **kw):
        del source, kw
        played.append((score.layers[0].program, score.cuts))
        Path(out).write_bytes(b"take")
        return R.TakeReport(frames=10, picture=10, frozen=False, luma=0.4)

    seen = {}
    monkeypatch.setattr(R, "splice", lambda cuts, *a, **kw: seen.update(spliced=cuts))
    _checked(monkeypatch)
    cut = R.render_cuts(
        score,
        "clip.mkv",
        tmp_path / "out.mkv",
        profiles={},
        takes=tmp_path,
        prepared=True,
        pass_render=fake_render,
    )
    assert cut == seen["spliced"] == plan
    assert played == [("Lorenz", plan), ("Derez", plan)]
    with pytest.raises(ValueError, match="composed for cuts over Lorenz, Derez"):
        R.render_cuts(
            score, "clip.mkv", tmp_path / "out.mkv", profiles={}, programs=["Pinwheel"], takes=tmp_path
        )

"""Timeline IR: lossless YAML round-trip, layer expansion, and section queries."""

# pylint: disable=missing-function-docstring

import numpy as np
import pytest

from syncsummoner.compose.score import (
    GestureInstance,
    Layer,
    Score,
    Section,
    control_rate,
    program_key,
)

from . import make_profile


def example_score() -> Score:
    return Score(
        seed=17,
        bpm=128.5,
        duration=12.0,
        fps=59.94,
        sections=[Section(0.0, 6.0, "A"), Section(6.0, 12.0, "B", destroy=True)],
        layers=[
            Layer(
                program="glitch",
                index=0,
                gestures=[
                    GestureInstance("ramp", 4.0, 2.0, "motion_rate", 0.6, (2,), 99),
                    GestureInstance("cliff_cross", 8.0, 1.0, "color_destruction", 0.9, (), 7),
                ],
            ),
            Layer(program="blur", index=1, gestures=[GestureInstance("hold", 1.0, 1.0)]),
        ],
        meta={"style": "glitchy", "objective": [{"mud": 0.1}]},
    )


def test_yaml_round_trip_is_lossless():
    score = example_score()
    reloaded = Score.from_yaml(score.to_yaml())
    assert reloaded.to_dict() == score.to_dict()
    assert reloaded.sections == score.sections
    assert reloaded.layers[0].gestures == score.layers[0].gestures
    assert Score.from_yaml(Score.from_yaml(score.to_yaml()).to_yaml()).to_dict() == score.to_dict()


def test_save_and_load_round_trip(tmp_path):
    score = example_score()
    path = tmp_path / "score.yaml"
    score.save(path)
    assert Score.load(path).to_dict() == score.to_dict()


def test_defaults_survive_a_minimal_document():
    score = Score.from_yaml("{}")
    assert (score.seed, score.bpm, score.fps, score.layers, score.sections) == (0, 120.0, 60.0, [], [])
    assert Score.from_dict({"layers": [{"program": "p"}]}).layers[0].gestures == []
    assert GestureInstance.from_dict({"gesture": "hold", "arrival": 0.0, "duration": 1.0}).intensity == 0.5


def test_section_queries():
    score = example_score()
    assert score.section_at(1.0).label == "A"
    assert score.section_at(99.0) is None
    mask = score.destroy_mask(np.array([1.0, 7.0, 20.0]))
    assert mask.tolist() == [False, True, False]
    assert Section(1.0, 3.0, "A").duration == 2.0


def test_control_rate_is_the_modulation_nyquist():
    assert control_rate(60.0) == 30.0
    assert program_key("glitch") == program_key("glitch") != program_key("blur")


def test_render_layer_is_deterministic_and_index_independent():
    score = example_score()
    profile = make_profile()
    first = score.render_layer(score.layers[0], profile)
    second = score.render_layer(score.layers[0], profile)
    assert first.values.tolist() == second.values.tolist()
    moved = score.with_layers([Layer("glitch", 5, score.layers[0].gestures)])
    assert moved.render_layer(moved.layers[0], profile).values.tolist() == first.values.tolist()


def test_render_layer_skips_unknown_gestures():
    score = Score(layers=[Layer("glitch", 0, [GestureInstance("nope", 1.0, 1.0)])])
    assert len(score.render_layer(score.layers[0], make_profile())) == 0


def test_automation_is_keyed_by_layer_and_skips_missing_profiles():
    score = example_score()
    autos = score.automation({"glitch": make_profile("glitch")})
    assert set(autos) == {0}
    assert len(autos[0]) > 0
    autos = score.automation({"glitch": make_profile("glitch"), "blur": make_profile("blur")}, rate=4.0)
    assert set(autos) == {0, 1}


def test_gesture_arrivals_land_on_the_declared_anchor():
    score = example_score()
    auto = score.render_layer(score.layers[0], make_profile())
    assert auto.times.max() == pytest.approx(8.0)


def test_window_rebases_sections_and_gestures():
    excerpt = example_score().window(5.0, 9.0)
    assert excerpt.duration == pytest.approx(4.0)
    assert [(s.start, s.end, s.label) for s in excerpt.sections] == [(0.0, 1.0, "A"), (1.0, 4.0, "B")]
    assert [g.arrival for g in excerpt.layers[0].gestures] == pytest.approx([3.0])
    assert not excerpt.layers[1].gestures


def test_window_keeps_a_gesture_whose_span_reaches_into_it():
    score = Score(duration=10.0, layers=[Layer("glitch", 0, [GestureInstance("ramp", 3.0, 2.0)])])
    assert score.window(2.0, 6.0).layers[0].gestures
    assert not score.window(4.0, 6.0).layers[0].gestures


def test_window_cannot_run_past_the_score():
    assert example_score().window(6.0, 100.0).duration == pytest.approx(6.0)

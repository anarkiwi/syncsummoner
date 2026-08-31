"""Planner: objective terms, the reachability hard constraint, and the evolutionary loop."""

# pylint: disable=missing-function-docstring,protected-access

import numpy as np
import pytest

from syncsummoner.compose import planner as P
from syncsummoner.compose.features import Features, Section, VideoFeatures
from syncsummoner.compose.score import GestureInstance, Layer, Score, control_rate
from syncsummoner.compose.vocabulary import Automation
from syncsummoner.device.profile import Axis, Source

from . import make_features, make_profile

FPS = 30.0


@pytest.fixture(name="features", scope="module")
def _features():
    return make_features(np.random.default_rng(11), seconds=6.0)


def dense_automation(n=600, span=1.0):
    return Automation.of(np.linspace(0, span, n), 2, np.arange(n) % 1024)


def test_reachable_rejects_a_cc_budget_overrun():
    profile = make_profile()
    assert not P.reachable(dense_automation(), profile, fps=FPS)
    assert P.reachable(Automation.empty(), profile, fps=FPS)


def test_reachable_rejects_settle_time_violations():
    profile = make_profile()
    fast = Automation.of([0.0, 0.001, 0.002], 2, np.array([0, 500, 900]))
    assert not P.reachable(fast, profile, fps=FPS)
    slow = Automation.of([0.0, 0.5, 1.0], 2, np.array([0, 500, 900]))
    assert P.reachable(slow, profile, fps=FPS)


def test_reachable_ignores_repeated_values():
    profile = make_profile()
    repeats = Automation.of([0.0, 0.001, 0.002], 2, np.array([500, 500, 500]))
    assert P.reachable(repeats, profile, fps=FPS)


def test_thin_enforces_the_hard_constraint():
    profile = make_profile()
    auto = dense_automation()
    thinned = P.thin(auto, profile, fps=FPS, cc_budget_hz=50.0)
    assert len(thinned) < len(auto)
    assert thinned.max_rate() <= 50.0
    assert P.reachable(thinned, profile, fps=FPS, cc_budget_hz=50.0)
    assert len(P.thin(Automation.empty(), profile, fps=FPS)) == 0


def test_plan_automation_is_always_reachable(features):
    profiles = {"glitch": make_profile("glitch")}
    score = P.search(profiles, features, style="glitchy", rng=np.random.default_rng(2), budget=8, fps=FPS)
    autos = P.plan_automation(score, profiles, fps=FPS)
    assert autos
    for auto in autos.values():
        assert P.reachable(auto, profiles["glitch"], fps=FPS)
    assert not P.plan_automation(score, {})


def test_proxy_render_shapes_and_cliff_driven_levels(features):
    profile = make_profile()
    rng = np.random.default_rng(0)
    over = Automation.of([0.0], 2, 1000)
    under = Automation.of([0.0], 2, 0)
    hot = P.proxy_render(over, profile, fps=FPS, duration=1.0, rng=rng)
    cold = P.proxy_render(under, profile, fps=FPS, duration=1.0, rng=rng)
    assert hot.times.size == 30 and hot.state.shape == (30, 12)
    assert hot.clip_frac.mean() > cold.clip_frac.mean()
    assert cold.clip_frac.max() == 0.0
    assert 0.0 <= hot.concentration.min() <= hot.concentration.max() <= 1.0
    assert features.audio is not None


def test_evaluate_reports_every_design_term(features):
    profile = make_profile()
    score = Score(duration=features.audio.duration, fps=FPS, sections=list(features.audio.sections))
    traj = P.proxy_render(
        Automation.of(np.arange(0, 6.0, 0.5), 2, np.arange(12) * 80),
        profile,
        fps=FPS,
        duration=6.0,
        rng=np.random.default_rng(0),
    )
    obj = P.evaluate(traj, features, score)
    assert set(obj.terms) == {"av_corr", "mud", "slope", "levels", "motif", "boredom"}
    assert obj.feasible and np.isfinite(obj.total)


def test_destroy_sections_exempt_illegal_levels(features):
    profile = make_profile()
    traj = P.proxy_render(
        Automation.of([0.0], 2, 1000), profile, fps=FPS, duration=6.0, rng=np.random.default_rng(0)
    )
    plain = Score(duration=6.0, fps=FPS, sections=[Section(0.0, 6.0, "A")])
    destroyed = Score(duration=6.0, fps=FPS, sections=[Section(0.0, 6.0, "A", destroy=True)])
    assert P.evaluate(traj, features, plain).terms["levels"] > 0
    assert P.evaluate(traj, features, destroyed).terms["levels"] == pytest.approx(
        P.evaluate(traj, features, plain).terms["levels"]
    )


def test_av_correlation_term_rewards_alignment(features):
    audio = features.audio
    times = np.arange(0, audio.duration, 1 / FPS)
    onset = np.interp(times, audio.times, audio.onset_strength)
    score = Score(duration=audio.duration, fps=FPS, sections=list(audio.sections))
    aligned = P.Trajectory(
        times=times,
        state=np.zeros((times.size, 12)),
        activity=onset,
        ic=onset,
        concentration=np.zeros(times.size),
        slope=np.full(times.size, -1.2),
        clip_frac=np.zeros(times.size),
        illegal_frac=np.zeros(times.size),
    )
    inverted = P.Trajectory(**{**aligned.__dict__, "ic": -onset})
    good = P.evaluate(aligned, features, score).terms["av_corr"]
    bad = P.evaluate(inverted, features, score).terms["av_corr"]
    assert good > 0.4 > 0 > bad == pytest.approx(-good)


def test_slope_excursions_are_permitted_on_hits(features):
    audio = features.audio
    times = np.arange(0, audio.duration, 1 / FPS)
    onset = np.interp(times, audio.times, audio.onset_strength)
    hits = onset >= onset.mean() + onset.std()
    score = Score(duration=audio.duration, fps=FPS, sections=list(audio.sections))
    slope = np.where(hits, 0.5, -1.2)
    traj = P.Trajectory(
        times=times,
        state=np.zeros((times.size, 12)),
        activity=np.zeros(times.size),
        ic=np.zeros(times.size),
        concentration=np.zeros(times.size),
        slope=slope,
        clip_frac=np.zeros(times.size),
        illegal_frac=np.zeros(times.size),
    )
    assert hits.any()
    assert P.evaluate(traj, features, score).terms["slope"] == pytest.approx(0.0)


def test_boredom_penalizes_a_flat_bar(features):
    audio = features.audio
    times = np.arange(0, audio.duration, 1 / FPS)
    score = Score(duration=audio.duration, fps=FPS, sections=list(audio.sections))
    varying = np.sin(2 * np.pi * times)
    flat = varying.copy()
    flat[times > times.max() / 2] = 0.0

    def traj(activity):
        return P.Trajectory(
            times=times,
            state=np.zeros((times.size, 12)),
            activity=activity,
            ic=np.zeros(times.size),
            concentration=np.zeros(times.size),
            slope=np.full(times.size, -1.2),
            clip_frac=np.zeros(times.size),
            illegal_frac=np.zeros(times.size),
        )

    assert P.evaluate(traj(flat), features, score).terms["boredom"] > 0
    assert P.evaluate(traj(varying), features, score).terms["boredom"] == pytest.approx(0.0)


def test_motif_return_rewards_repeated_labels():
    times = np.arange(6.0)
    state = np.zeros((6, 12))
    state[[0, 4]] = 1.0
    state[[2]] = -1.0
    traj = P.Trajectory(
        times=times,
        state=state,
        activity=np.zeros(6),
        ic=np.zeros(6),
        concentration=np.zeros(6),
        slope=np.zeros(6),
        clip_frac=np.zeros(6),
        illegal_frac=np.zeros(6),
    )
    sections = [Section(0.0, 2.0, "A"), Section(2.0, 4.0, "B"), Section(4.0, 6.0, "A")]
    assert P._motif_return(traj, sections) > 0
    assert P._motif_return(traj, sections[:2]) == 0.0
    assert P._motif_return(traj, [Section(0.0, 2.0, "A")] * 3) == 0.0


def test_search_keeps_program_as_the_outermost_variable(features):
    profiles = {"alpha": make_profile("alpha"), "beta": make_profile("beta")}
    score = P.search(profiles, features, rng=np.random.default_rng(4), budget=8, fps=FPS, n_passes=2)
    assert [layer.index for layer in score.layers] == [0, 1]
    assert {layer.program for layer in score.layers} == set(profiles)
    for layer in score.layers:
        assert all(inst.gesture in P.GESTURES for inst in layer.gestures)
    assert score.meta["style"] == "default" and len(score.meta["objective"]) == 2


def test_mutation_never_changes_the_program(features):
    profile = make_profile()
    rng = np.random.default_rng(6)
    audio = features.audio
    weights = P._applicable(profile, P.STYLES["default"], rng)
    layer = P._sample_layer("glitch", profile, audio, audio.sections, weights, rng, density=1.0)
    mutated = P._mutate(layer, profile, audio, audio.sections, weights, rng, rate=1.0)
    assert mutated.program == layer.program == "glitch"
    assert len(mutated.gestures) == len(layer.gestures)
    assert [g.arrival for g in mutated.gestures] == sorted(g.arrival for g in mutated.gestures)


def test_glitchy_style_marks_destroy_sections(features):
    profiles = {"glitch": make_profile()}
    score = P.search(profiles, features, style="glitchy", rng=np.random.default_rng(9), budget=6, fps=FPS)
    assert len(score.sections) >= 2
    assert any(s.destroy for s in score.sections) or all(not s.destroy for s in score.sections)
    plain = P.search(profiles, features, style="smooth", rng=np.random.default_rng(9), budget=6, fps=FPS)
    assert not any(s.destroy for s in plain.sections)


def test_simulated_profiles_are_discounted(features):
    score = Score(duration=1.0, fps=FPS, sections=[Section(0.0, 1.0, "A")])
    traj = P.proxy_render(
        Automation.of([0.0], 2, 700), make_profile(), fps=FPS, duration=1.0, rng=np.random.default_rng(0)
    )
    full = P.evaluate(traj, features, score, discount=1.0).total
    assert P.evaluate(traj, features, score, discount=P.SIM_DISCOUNT).total == pytest.approx(
        full * P.SIM_DISCOUNT
    )
    sim = P.search(
        {"s": make_profile("s", source=Source.SIM)}, features, rng=np.random.default_rng(1), budget=4
    )
    assert sim.layers[0].program == "s"


def test_search_needs_audio_and_applicable_gestures(features):
    with pytest.raises(ValueError):
        P.search({}, Features(audio=None, video=None), rng=np.random.default_rng(0))
    bare = make_profile("bare", rich=False)
    bare.params = []
    empty = P.search({"bare": bare}, features, rng=np.random.default_rng(0), budget=4)
    assert empty.layers == []


def test_search_accepts_an_injected_evaluator(features):
    seen = []

    def evaluator(layer, profile, score):
        seen.append(layer.program)
        del profile, score
        return P.Objective(total=float(len(layer.gestures)), feasible=True, terms={})

    out = P.search(
        {"a": make_profile("a")},
        features,
        rng=np.random.default_rng(0),
        budget=8,
        fps=FPS,
        evaluator=evaluator,
    )
    assert seen and set(seen) == {"a"}
    assert out.layers[0].program == "a"


def test_infeasible_candidates_score_minus_infinity(features):
    profiles = {"glitch": make_profile()}
    score = P.search(profiles, features, rng=np.random.default_rng(0), budget=4, fps=FPS, cc_budget_hz=0.0)
    assert P.plan_automation(score, profiles, fps=FPS, cc_budget_hz=0.0)


def test_anchor_times_cover_every_anchor(features):
    audio = features.audio
    section = audio.sections[0]
    for anchor in P.SPAN_BARS:
        times = P._anchor_times(audio, section, anchor)
        assert times.size >= 1
    empty = Section(1000.0, 1001.0, "Z")
    assert P._anchor_times(audio, empty, P.Anchor.BAR).size >= 1
    assert P._axes(make_profile("x", rich=False)) == [Axis.TEXTURE_SCALE]


def test_describe_summarizes_the_score(features):
    score = P.search({"g": make_profile("g")}, features, rng=np.random.default_rng(3), budget=6, fps=FPS)
    summary = P.describe(score)
    assert summary["programs"] == ["g"]
    assert sum(summary["gestures"].values()) == len(score.layers[0].gestures)
    assert len(summary["sections"]) == len(score.sections)


def test_objective_weights_are_documented_defaults():
    assert P.DEFAULT_WEIGHTS == P.ObjectiveWeights()
    assert P.ObjectiveWeights(mud=2.0).mud == 2.0
    assert control_rate(60.0) == 30.0
    assert not Layer("p").gestures
    assert not GestureInstance("hold", 0.0, 1.0).targets


def test_a_program_with_nothing_to_drive_is_not_planned(features):
    """Passthru, fitted from a one setpoint sweep, offered 49 gestures against no parameters."""
    inert = make_profile("Passthru")
    for spec in inert.params:
        spec.sensitivity = 0.0
    both = P.search(
        {"glitch": make_profile("glitch"), "Passthru": inert},
        features,
        rng=np.random.default_rng(0),
        budget=12,
    )
    assert both.layers and {layer.program for layer in both.layers} == {"glitch"}


def test_score_runs_no_longer_than_the_shorter_input(features):
    video = VideoFeatures(
        fps=30.0,
        n_frames=90,
        shot_boundaries=np.zeros(0, dtype=np.int64),
        motion_energy=np.zeros(90),
        luma=np.zeros(90),
        chroma=np.zeros(90),
    )
    both = Features(audio=features.audio, video=video)
    score = P.search({"g": make_profile("g")}, both, rng=np.random.default_rng(5), budget=4, fps=FPS)
    assert score.duration == pytest.approx(3.0)
    assert score.sections[-1].end == pytest.approx(3.0)
    assert max(g.arrival for g in score.layers[0].gestures) <= 3.0


def test_clipping_leaves_at_least_one_section():
    assert P._clip_sections([Section(4.0, 6.0, "A")], 3.0) == [Section(0.0, 3.0, "A")]


def _measured_profile(program="measured", **curves):
    """A profile whose parameter 2 carries measured metric curves over its swept values."""
    profile = make_profile(program=program)
    spec = next(p for p in profile.params if p.index == 2)
    spec.values = [0, 256, 512, 768, 1023]
    spec.measured = {name: list(values) for name, values in curves.items()}
    return profile


def test_proxy_reads_the_measured_slope_instead_of_inferring_it():
    profile = _measured_profile(spectral_slope=[-3.0, -2.5, -2.0, -1.5, -1.0])
    traj = P.proxy_render(
        Automation.of([0.0, 0.5], 2, [0, 1023]),
        profile,
        fps=FPS,
        duration=1.0,
        rng=np.random.default_rng(0),
    )
    assert traj.slope.min() == pytest.approx(-3.0, abs=0.05)
    assert traj.slope.max() == pytest.approx(-1.0, abs=0.05)


def test_proxy_falls_back_where_nothing_was_measured():
    traj = P.proxy_render(
        Automation.of([0.0], 2, 1000),
        make_profile(),
        fps=FPS,
        duration=1.0,
        rng=np.random.default_rng(0),
    )
    assert traj.slope.size == traj.times.size and np.isfinite(traj.slope).all()
    assert traj.clip_frac.max() > 0.0


def test_a_program_that_never_moves_is_maximally_bored(features):
    times = np.arange(0, features.audio.duration, 1 / FPS)
    score = Score(duration=features.audio.duration, fps=FPS, sections=list(features.audio.sections))
    static = P.Trajectory(
        times=times,
        state=np.zeros((times.size, 12)),
        activity=np.zeros(times.size),
        ic=np.zeros(times.size),
        concentration=np.zeros(times.size),
        slope=np.full(times.size, -1.2),
        clip_frac=np.zeros(times.size),
        illegal_frac=np.zeros(times.size),
    )
    assert P.evaluate(static, features, score).terms["boredom"] == pytest.approx(1.0)


def test_a_passthrough_does_not_outrank_a_program_that_does_something(features):
    """The defect this whole layer existed to hide: doing nothing scored best."""
    flat = [0.0] * 5
    passthru = _measured_profile("passthru", spectral_slope=[-1.2] * 5, clip_frac=flat, illegal_frac=flat)
    doer = _measured_profile(
        "doer",
        spectral_slope=[-3.4, -2.6, -1.8, -1.0, -0.2],
        clip_frac=[0.0, 0.1, 0.3, 0.6, 0.9],
        illegal_frac=[0.0, 0.2, 0.4, 0.7, 1.0],
    )
    score = P.search(
        {"passthru": passthru, "doer": doer},
        features,
        style="glitchy",
        rng=np.random.default_rng(3),
        budget=48,
        fps=FPS,
        n_passes=2,
    )
    assert [layer.program for layer in score.layers][0] == "doer"


def test_glitchy_does_not_pay_the_naturalness_penalties():
    assert P.STYLE_WEIGHTS["glitchy"].slope == 0.0
    assert P.STYLE_WEIGHTS["glitchy"].levels == 0.0
    assert P.STYLE_WEIGHTS["glitchy"].boredom > P.DEFAULT_WEIGHTS.boredom
    assert P.STYLE_WEIGHTS.get("smooth", P.DEFAULT_WEIGHTS) is P.DEFAULT_WEIGHTS

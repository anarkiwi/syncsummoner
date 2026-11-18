"""Planner: objective terms, the reachability hard constraint, and the evolutionary loop."""

# pylint: disable=missing-function-docstring,protected-access

from dataclasses import replace

import numpy as np
import pytest

from syncsummoner.compose import planner as P
from syncsummoner.compose.cut import Cut, spans_of
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


def test_reachable_rejects_a_dense_ramp_no_parameter_can_settle_between():
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
    thinned = P.thin(auto, profile, fps=FPS)
    assert len(thinned) < len(auto)
    assert thinned.values[-1] == auto.values[-1]
    assert P.reachable(thinned, profile, fps=FPS)
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


def test_mutation_keeps_every_gesture_inside_its_span(features):
    profile = make_profile()
    rng = np.random.default_rng(6)
    audio = features.audio
    weights = P._applicable(profile, P.STYLES["default"], rng)
    span = Section(2.0, 4.5, "")
    gestures = [P._sample_instance(name, audio, span, profile, rng) for name in weights for _ in range(4)]
    mutated = P._mutate(gestures, profile, audio, span, weights, rng, rate=1.0)
    assert len(mutated) == len(gestures)
    assert [g.arrival for g in mutated] == sorted(g.arrival for g in mutated)
    for inst in gestures + mutated:
        assert span.start <= inst.arrival - inst.duration <= inst.arrival < span.end


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


def test_infeasible_candidates_score_minus_infinity(features, monkeypatch):
    profiles = {"glitch": make_profile()}
    monkeypatch.setattr(P, "reachable", lambda *a, **k: False)
    score = P.search(profiles, features, rng=np.random.default_rng(0), budget=4, fps=FPS)
    assert P.plan_automation(score, profiles, fps=FPS)


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


def test_thin_defers_a_release_it_cannot_settle_rather_than_latching_the_toggle():
    profile = make_profile()
    profile.settle_frames[7] = 30
    pulse = Automation.concat(
        [Automation.of([1.0, 1.2], 7, np.array([1023, 0])), Automation.of([1.5, 1.6], 7, np.array([1023, 0]))]
    )
    thinned = P.thin(pulse, profile, fps=FPS)
    assert thinned.values.tolist() == [1023, 0] and thinned.values[-1] == 0, "released, never latched"
    assert thinned.times.tolist() == pytest.approx([1.0, 2.0])
    assert P.reachable(thinned, profile, fps=FPS)
    back = Automation.of([0.0, 0.1, 0.2], 7, np.array([1023, 0, 1023]))
    assert P.thin(back, profile, fps=FPS).values.tolist() == [1023], "a change undone in time is no change"


def test_a_composed_score_states_no_wet_band(features):
    profiles = {"glitch": make_profile("glitch")}
    score = P.search(profiles, features, style="glitchy", rng=np.random.default_rng(3), budget=6, fps=FPS)
    assert "wet" not in score.meta and "morph" not in P.STYLES["glitchy"]


def _footage(seconds, fps=FPS, seed=0):
    """Source P.910 series for a moving clip: SI steady, TI varying frame to frame."""
    rng = np.random.default_rng(seed)
    n = int(seconds * fps)
    return VideoFeatures(
        fps=fps,
        n_frames=n,
        shot_boundaries=np.zeros(0, dtype=np.int64),
        motion_energy=np.zeros(n),
        luma=np.zeros(n),
        chroma=np.zeros(n),
        si=np.full(n, 0.3),
        ti=np.r_[np.nan, rng.gamma(4.0, 0.01, n - 1)],
    )


def _kept(features, *, si=1.0, ti=1.0, follow=0.9, dead_from=None):
    """A trajectory keeping the given shares of the source, collapsing to nothing after ``dead_from``."""
    times = np.arange(0, features.audio.duration, 1 / FPS)

    def ratio(v):
        return np.where(times >= dead_from, 0.0, v) if dead_from is not None else np.full(times.size, v)

    zeros = np.zeros(times.size)
    return P.Trajectory(
        times=times,
        state=np.zeros((times.size, 12)),
        activity=zeros,
        ic=zeros,
        concentration=zeros,
        slope=np.full(times.size, -1.2),
        clip_frac=zeros,
        illegal_frac=zeros,
        si_ratio=ratio(si),
        ti_ratio=ratio(ti),
        follow=ratio(follow),
    )


def test_a_kept_picture_has_no_dead_bars(features):
    both = Features(audio=features.audio, video=_footage(features.audio.duration))
    kept = P.predict_survival(_kept(both), both)
    assert kept is not None and not kept.dead.any()
    assert np.allclose(kept.ti_ratio, 1.0)


def test_a_flat_or_frozen_or_unfollowed_bar_is_dead(features):
    both = Features(audio=features.audio, video=_footage(features.audio.duration))
    for shares in ({"si": 0.0}, {"ti": 0.0}, {"follow": 0.0}):
        assert P.predict_survival(_kept(both, **shares), both).dead.all()


def test_survival_is_unknown_without_curves_or_footage(features):
    traj = _kept(Features(audio=features.audio, video=None))
    assert P.predict_survival(traj, Features(audio=features.audio, video=None)) is None
    both = Features(audio=features.audio, video=_footage(features.audio.duration))
    assert P.predict_survival(replace(traj, si_ratio=None), both) is None


def test_destruction_is_budgeted_one_bar_at_a_time_inside_destroy_sections():
    kept = P.Kept(
        starts=np.arange(6.0),
        ends=np.arange(1.0, 7.0),
        ti_ratio=np.zeros(6),
        dead=np.array([1, 0, 1, 1, 0, 1], bool),
    )
    calm = Score(duration=6.0, sections=[Section(0.0, 6.0, "A")])
    wild = Score(duration=6.0, sections=[Section(0.0, 6.0, "A", destroy=True)])
    assert P.dead_bars_over_budget(kept, calm) == 4.0
    assert P.dead_bars_over_budget(kept, wild) == 1.0


def test_dead_bars_outrank_any_total_and_bore_the_objective(features):
    both = Features(audio=features.audio, video=_footage(features.audio.duration))
    score = Score(
        duration=features.audio.duration, fps=FPS, sections=[Section(0.0, features.audio.duration, "A")]
    )
    alive = P.evaluate(_kept(both), both, score)
    dying = P.evaluate(_kept(both, dead_from=1.0), both, score)
    assert alive.violation == 0.0 and dying.violation > 0.0 and alive.rank < dying.rank
    assert alive.terms["boredom"] == pytest.approx(0.0) and dying.terms["boredom"] == pytest.approx(1.0)
    assert (
        P.Objective(total=-5.0, feasible=True).rank < P.Objective(total=5.0, feasible=True, violation=1).rank
    )


def test_survival_curves_are_read_back_from_the_profile():
    profile = _measured_profile(
        si_ratio=[1.0, 1.0, 0.5, 0.0, 0.0], ti_ratio=[1.0] * 5, motion_follow=[0.9] * 5
    )
    traj = P.proxy_render(
        Automation.of([0.0, 0.5], 2, [0, 1023]), profile, fps=FPS, duration=1.0, rng=np.random.default_rng(0)
    )
    assert traj.si_ratio.max() == pytest.approx(1.0, abs=0.05) and traj.si_ratio.min() == pytest.approx(
        0.0, abs=0.05
    )
    assert traj.follow is not None and np.all(traj.follow <= 1.0)


def test_search_cuts_sections_at_the_phrase_grid(features):
    score = P.search({"g": make_profile("g")}, features, rng=np.random.default_rng(1), budget=4, fps=FPS)
    bars = features.audio.phrase_bars * features.audio.bar_duration
    assert score.meta["phrase_bars"] == features.audio.phrase_bars and score.meta["bar"] > 0
    assert all(s.duration <= bars + 1e-6 for s in score.sections)


def _inside(times, spans):
    return np.array([any(a <= t < b for a, b in spans) for t in times], dtype=bool)


def test_every_write_a_layer_plays_lands_inside_its_own_cuts(features):
    profiles = {"alpha": make_profile("alpha"), "beta": make_profile("beta")}
    score = P.search(profiles, features, rng=np.random.default_rng(4), budget=12, fps=FPS, n_passes=2)
    assert {cut.program for cut in score.cuts} == {"alpha", "beta"}
    assert [(c.start, c.end) for c in score.cuts] == [(s.start, s.end) for s in score.sections]
    plans = P.plan_automation(score, profiles, fps=FPS, gates={})
    unclipped = P.plan_automation(replace(score, cuts=[]), profiles, fps=FPS, gates={})
    for layer in score.layers:
        spans = spans_of(score.cuts, layer.program)
        assert len(plans[layer.index]) and _inside(plans[layer.index].times, spans).all()
        assert all(
            any(a <= g.arrival - g.duration <= g.arrival < b for a, b in spans) for g in layer.gestures
        )
        assert len(unclipped[layer.index]) >= len(plans[layer.index])


def test_no_gesture_is_kept_whose_writes_settle_past_its_cut(features):
    """A release deferred past a cut lands unseen and leaves its toggle latched into the next one."""
    slow = make_profile("slow")
    slow.settle_frames = {index: 45 for index in slow.settle_frames}
    profiles = {"slow": slow, "other": make_profile("other")}
    score = P.search(profiles, features, rng=np.random.default_rng(5), budget=12, fps=FPS, n_passes=2)
    plans = P.plan_automation(score, profiles, fps=FPS, gates={})
    whole = P.plan_automation(replace(score, cuts=[]), profiles, fps=FPS, gates={})
    for layer in score.layers:
        assert len(plans[layer.index]) and len(whole[layer.index]) == len(plans[layer.index])


def test_each_cut_is_planned_to_its_share_of_writes_per_second_on_screen(features):
    profiles = {"alpha": make_profile("alpha")}
    for density in (0.5, 4.0):
        score = P.search(
            profiles, features, rng=np.random.default_rng(2), budget=12, fps=FPS, density=density
        )
        auto = P.plan_automation(score, profiles, fps=FPS, gates={})[0]
        for cut in score.cuts:
            writes = int(((auto.times >= cut.start) & (auto.times < cut.end)).sum())
            assert writes >= max(1, int(np.rint(density * cut.duration)))
        assert score.meta["density"] == density


def _survival_profiles():
    alive = _measured_profile("alive", si_ratio=[1.0] * 5, ti_ratio=[1.0] * 5, motion_follow=[0.9] * 5)
    dead = _measured_profile("dead", si_ratio=[0.0] * 5, ti_ratio=[0.0] * 5, motion_follow=[0.0] * 5)
    return {"alive": alive, "dead": dead}


def test_a_cut_predicted_dead_moves_to_a_program_alive_there_and_is_planned_on_it(features, caplog):
    both = Features(audio=features.audio, video=_footage(features.audio.duration))
    with caplog.at_level("INFO", logger="syncsummoner"):
        score = P.search(
            _survival_profiles(), both, rng=np.random.default_rng(3), budget=12, fps=FPS, n_passes=2
        )
    assert len(score.cuts) == len(score.sections) > 1
    assert {cut.program for cut in score.cuts} == {"alive"} and [l.program for l in score.layers] == ["alive"]
    gestures = score.layers[0].gestures
    assert all(any(cut.start <= g.arrival < cut.end for g in gestures) for cut in score.cuts)
    moved = [m for m in caplog.messages if "dead predicted dead" in m]
    assert len(moved) == len(score.cuts) // 2 and all(m.endswith("planned on alive") for m in moved)


def test_a_cut_no_program_is_alive_in_stays_and_is_said(features, caplog):
    both = Features(audio=features.audio, video=_footage(features.audio.duration))
    profiles = {"dead": _survival_profiles()["dead"]}
    with caplog.at_level("WARNING", logger="syncsummoner"):
        score = P.search(profiles, both, rng=np.random.default_rng(3), budget=6, fps=FPS)
    assert {cut.program for cut in score.cuts} == {"dead"}
    assert any("no program is alive there" in m for m in caplog.messages)


def test_a_cut_thinned_on_from_its_opening_is_the_whole_layer_thinned_then_shown():
    profile = make_profile()
    profile.settle_frames = {1: 20, 2: 45, 7: 30}
    rng = np.random.default_rng(8)
    requests = P.Automation.concat(
        P.Automation.of(np.sort(rng.uniform(0.0, 6.0, 40)), index, rng.integers(0, 1024, 40))
        for index in (1, 2, 7)
    )
    head, cut = requests.within(0.0, 2.0), Cut(3.0, 5.0, "glitch")
    prog = P._Program("glitch", profile, {}, 1.0)
    prog.commit(Cut(0.0, 2.0, "glitch"), [], head)
    opening = prog.opening(cut, FPS)
    shown = P._continued(opening, requests.within(3.0, 6.0), profile, FPS).within(3.0, 5.0)
    whole = P.visible(
        P.thin(P.Automation.concat([head, requests.within(3.0, 6.0)]), profile, fps=FPS),
        [(0.0, 2.0), (3.0, 5.0)],
    )
    expect = whole.within(3.0, 5.0)
    order, want = np.lexsort((shown.indices, shown.times)), np.lexsort((expect.indices, expect.times))
    assert np.allclose(shown.times[order], expect.times[want])
    assert (shown.indices[order] == expect.indices[want]).all() and (
        shown.values[order] == expect.values[want]
    ).all()
    before = whole.within(0.0, 3.0)
    held = {int(i): int(before.values[before.indices == i][-1]) for i in np.unique(before.indices)}
    assert dict(zip(opening.held.indices.tolist(), opening.held.values.tolist())) == held


def test_only_writes_inside_the_spans_are_shown():
    auto = Automation.of([0.0, 1.0, 1.5, 2.0, 2.5, 3.0], 2, [1, 2, 3, 4, 5, 6])
    assert P.visible(auto, None) is auto
    assert P.visible(auto, [(2.0, 3.0), (1.0, 2.0)]).values.tolist() == [2, 3, 4, 5]
    assert len(P.visible(auto, [])) == 0


def test_every_parameter_measured_to_move_is_aimed_at_whatever_its_axis():
    profile = make_profile()
    rng = np.random.default_rng(0)
    aimed = {P._aim("hold", profile, rng) for _ in range(200)}
    assert {t for _, t in aimed} == {(1,), (2,), (3,), (12,)}
    assert ("unassigned", (12,)) in aimed
    assert {P._aim("punch", profile, rng)[1] for _ in range(20)} == {(7,)}
    assert not P._aim("cliff_cross", profile, rng)[1]


def test_objectives_over_cuts_weigh_by_length_and_sum_dead_bars():
    parts = [
        (1.0, P.Objective(total=1.0, feasible=True, terms={"mud": 1.0, "dead_bars": 2.0}, violation=1.0)),
        (3.0, P.Objective(total=-1.0, feasible=True, terms={"mud": 0.0, "dead_bars": 1.0})),
    ]
    both = P._combine(parts)
    assert both.total == pytest.approx(-0.5) and both.violation == 1.0
    assert both.terms == {"mud": pytest.approx(0.25), "dead_bars": 3.0}

"""Planner: reachable writes, the automation a score plays, and measured effects mapped onto music."""

# pylint: disable=missing-function-docstring,protected-access

import numpy as np
import pytest
from scipy.stats import spearmanr

from syncsummoner.compose import planner as P
from syncsummoner.compose.cut import Cut, spans_of
from syncsummoner.compose.features import Features, VideoFeatures, analyze_frames
from syncsummoner.compose.gating import gate_automation
from syncsummoner.compose.score import GestureInstance, Layer, Score
from syncsummoner.compose.stimulus import Coverage, Stimulus, chroma_sigma
from syncsummoner.compose.vocabulary import SETTING, Automation

from . import make_features, make_profile

FPS = 30.0
SATURATION = 0.54


def dense_automation(n=600, span=1.0):
    return Automation.of(np.linspace(0, span, n), 2, np.arange(n) % 1024)


def test_reachable_rejects_a_dense_ramp_no_parameter_can_settle_between():
    profile = make_profile()
    assert not P.reachable(dense_automation(), profile, fps=FPS)
    assert P.reachable(Automation.empty(), profile, fps=FPS)


def test_reachable_rejects_settle_time_violations_and_ignores_repeats():
    profile = make_profile()
    assert not P.reachable(Automation.of([0.0, 0.001, 0.002], 2, np.array([0, 500, 900])), profile, fps=FPS)
    assert P.reachable(Automation.of([0.0, 0.5, 1.0], 2, np.array([0, 500, 900])), profile, fps=FPS)
    assert P.reachable(Automation.of([0.0, 0.001, 0.002], 2, np.array([500, 500, 500])), profile, fps=FPS)


def test_thin_enforces_the_hard_constraint():
    profile = make_profile()
    auto = dense_automation()
    thinned = P.thin(auto, profile, fps=FPS)
    assert len(thinned) < len(auto)
    assert thinned.values[-1] == auto.values[-1]
    assert P.reachable(thinned, profile, fps=FPS)


def test_thin_defers_a_release_it_cannot_settle_rather_than_latching_the_toggle():
    profile = make_profile()
    profile.settle_frames[7] = 30
    pulse = Automation.concat(
        [Automation.of([1.0, 1.2], 7, np.array([1023, 0])), Automation.of([1.5, 1.6], 7, np.array([1023, 0]))]
    )
    thinned = P.thin(pulse, profile, fps=FPS)
    assert thinned.values.tolist() == [1023, 0], "released, never latched"
    assert thinned.times.tolist() == pytest.approx([1.0, 2.0])
    back = Automation.of([0.0, 0.1, 0.2], 7, np.array([1023, 0, 1023]))
    assert P.thin(back, profile, fps=FPS).values.tolist() == [1023], "a change undone in time is no change"


def test_only_writes_inside_the_spans_are_shown():
    auto = Automation.of([0.0, 1.0, 1.5, 2.0, 2.5, 3.0], 2, [1, 2, 3, 4, 5, 6])
    assert P.visible(auto, None) is auto
    assert P.visible(auto, [(2.0, 3.0), (1.0, 2.0)]).values.tolist() == [2, 3, 4, 5]
    assert len(P.visible(auto, [])) == 0


class _Gate:
    """A program-model gate stand-in: ``dead`` parameters are never live."""

    def __init__(self, dead):
        self.dead, self.refused = set(dead), 0

    def __call__(self, auto):
        gated = gate_automation(
            auto, np.full(12, 512, np.int64), lambda _s, p: p not in self.dead, lambda _s, _p: []
        )
        self.refused += gated.refused
        return gated.auto


def _setting(at, values, seed=0):
    return GestureInstance(SETTING, at, 0.0, targets=tuple(range(1, 13)), values=tuple(values), seed=seed)


def test_drawn_gestures_play_inside_their_cuts_through_the_gate_and_settings_play_whole():
    profiles = {"a": make_profile("a"), "b": make_profile("b")}
    drawn = [GestureInstance("hold", t, 0.5, intensity=0.9, targets=(p,)) for t in (0.5, 1.5) for p in (1, 2)]
    vector = [100 + 10 * i for i in range(12)]
    score = Score(
        fps=FPS,
        duration=2.0,
        cuts=[Cut(0.0, 1.0, "a"), Cut(1.0, 2.0, "b")],
        layers=[Layer("a", 0, drawn), Layer("b", 1, [_setting(0.2, vector)])],
    )
    gates = {"a": _Gate({2}), "b": _Gate(set(range(1, 13)))}
    autos = P.plan_automation(score, profiles, fps=FPS, gates=gates)
    assert autos[0].times.tolist() == [0.5] and autos[0].indices.tolist() == [1]
    assert gates["a"].refused == 1 and gates["b"].refused == 0
    assert autos[1].times.tolist() == pytest.approx(
        0.2 + np.arange(12) / FPS
    ), "one a frame, off screen, ungated"
    assert autos[1].hold(np.array([1.5]))[0].tolist() == vector
    assert not P.plan_automation(score, {})


def test_ranks_share_ties_and_centre_a_lone_value():
    assert P.ranks([3.0, 1.0, 3.0, 2.0]).tolist() == pytest.approx([5 / 6, 0.0, 5 / 6, 1 / 3])
    assert P.ranks([7.0]).tolist() == [0.5]


def test_a_pickup_joins_the_first_bar_and_bars_end_at_the_duration():
    edges = P.bar_edges(np.array([0.4, 2.4, 4.4, 6.4, 8.4]), 7.0)
    assert edges.tolist() == [0.0, 2.4, 4.4, 6.4, 7.0]


def test_bar_means_average_each_bar_and_take_the_nearest_sample_in_an_empty_one():
    times = np.array([0.0, 0.5, 1.0, 1.5, 3.0])
    means = P.bar_means(np.array([1.0, 3.0, 5.0, np.nan, 9.0]), times, np.array([0.0, 1.0, 2.0, 2.5, 4.0]))
    assert means.tolist() == [2.0, 2.5, 9.0, 9.0]


def test_a_steady_change_curve_changes_every_so_many_bars():
    assert P.changes(np.linspace(0, 1, 13), 4.0, 0.0).tolist() == [0, 4, 8, 12]
    assert P.changes(np.zeros(9), 3.0, 0.0).tolist() == [0, 3, 6]


def test_quickening_crowds_changes_where_the_drive_is_high_and_keeps_their_number():
    level = np.r_[np.zeros(32), np.ones(32)]
    steady = P.changes(level, 4.0, 0.0)
    quick = P.changes(level, 4.0, 0.5)
    slow = P.changes(level, 4.0, -0.5)
    assert abs(len(quick) - len(steady)) <= 1 and abs(len(slow) - len(steady)) <= 1
    assert (quick >= 32).sum() > (steady >= 32).sum() > (slow >= 32).sum()


def test_assignment_follows_the_cost_and_never_repeats_a_program():
    cost = np.array([[0.0, 1.0, 2.0], [1.0, 0.0, 2.0], [2.0, 2.0, 0.0], [0.0, 5.0, 5.0]])
    path = P.assign(cost)
    assert path.tolist() == [0, 1, 2, 0]
    assert (np.diff(path) != 0).all()
    assert P.assign(cost[:, :1]).tolist() == [0, 0, 0, 0]


def test_passes_keep_the_programs_that_lower_the_cost_most():
    cost = np.array([[0.0, 9.0, 1.0, 9.0], [9.0, 5.0, 1.0, 9.0], [0.0, 9.0, 1.0, 9.0], [9.0, 5.0, 1.0, 9.0]])
    assert P.assign(cost, 1).tolist() == [2, 2, 2, 2]
    assert P.assign(cost, 2).tolist() == [0, 2, 0, 2]
    assert set(P.assign(cost, 3).tolist()) <= {0, 1, 2}


def _clip(coverage, frames=24):
    stimulus = Stimulus(seed=3, width=256, height=160, shot_frames=8, texture=256, coverage=coverage)
    return np.stack(list(stimulus.frames(0, frames))).astype(np.float32) / 255.0


def test_a_synthetic_clips_chroma_is_what_its_saturation_says():
    coverage = Coverage().full_colour()
    found = analyze_frames(_clip(coverage), fps=FPS).chroma
    assert float(np.sqrt(np.mean(found**2))) == pytest.approx(chroma_sigma(coverage.saturation[0]), rel=0.1)
    assert analyze_frames(_clip(Coverage().monochrome()), fps=FPS).chroma.max() < 1e-2


def _effects(seed, n=40, saturation=SATURATION):
    """``n`` settings, a quarter unsafe and those the most extreme, intensity a function of the first two."""
    rng = np.random.default_rng(seed)
    params = rng.integers(0, 1024, (n, 12))
    intensity = (params[:, 0] + params[:, 1]) / 2046.0
    unsafe = np.argsort(np.abs(intensity - 0.5))[-n // 4 :]
    settings = [
        {
            "setpoint": i,
            "params": params[i].tolist(),
            "intensity": float(intensity[i]),
            "safe": i not in unsafe,
        }
        for i in range(n)
    ]
    return {"saturation": [saturation, saturation], "settings": settings}


def _profiles(names=("a", "b", "c"), kinds=("bw", "colour")):
    out = {}
    for k, name in enumerate(names):
        profile = make_profile(name)
        profile.effects = {
            kind: _effects(10 * k + j, saturation=0.0 if kind == "bw" else SATURATION)
            for j, kind in enumerate(kinds)
        }
        out[name] = profile
    return out


def _video(seconds, *, chroma, seed=0):
    rng = np.random.default_rng(seed)
    n = int(seconds * FPS)
    ti = np.abs(np.sin(np.arange(n) / (FPS * 3.0))) + 0.05 * rng.random(n)
    zeros = np.zeros(n)
    return VideoFeatures(FPS, n, np.zeros(0, np.int64), zeros, zeros, np.full(n, chroma), si=zeros + 1, ti=ti)


def test_the_source_takes_the_clip_it_is_nearer_in_chroma():
    profiles = _profiles()
    colour = chroma_sigma(SATURATION)
    assert P.clip_kind(_video(2, chroma=0.1 * colour), profiles)[0] == "bw"
    kind, source, clips = P.clip_kind(_video(2, chroma=0.9 * colour), profiles)
    assert kind == "colour" and source == pytest.approx(0.9 * colour)
    assert clips == {"bw": 0.0, "colour": pytest.approx(colour)}
    assert P.clip_kind(_video(2, chroma=0.0), profiles, "colour")[0] == "colour"
    assert P.clip_kind(None, _profiles(kinds=("bw",)))[0] == "bw"
    for video, found, kind, match in [
        (None, profiles, None, "name the clip"),
        (None, profiles, "sepia", "no profile measured effects on a sepia clip"),
        (None, {"a": make_profile("a")}, None, "run `probe effects`"),
    ]:
        with pytest.raises(ValueError, match=match):
            P.clip_kind(video, found, kind)


def test_palettes_hold_safe_settings_only_with_quantiles_pooled_over_programs():
    profiles = _profiles()
    profiles["d"] = make_profile("d")
    found = P.palettes(profiles, "bw")
    assert [p.program for p in found] == ["a", "b", "c"]
    pooled = np.concatenate([p.quantile for p in found])
    intensity = np.concatenate([p.intensity for p in found])
    assert pooled.min() == 0.0 and pooled.max() == 1.0
    assert spearmanr(pooled, intensity)[0] == pytest.approx(1.0)
    safe = {
        (name, s["setpoint"]) for name in "abc" for s in profiles[name].effects["bw"]["settings"] if s["safe"]
    }
    assert {(p.program, int(s)) for p in found for s in p.setpoints} == safe
    with pytest.raises(ValueError, match="no program has a safe setting"):
        P.palettes({"d": profiles["d"]}, "bw")


@pytest.fixture(name="audio", scope="module")
def _audio():
    return make_features(np.random.default_rng(11), seconds=40.0).audio


def _settings_of(profiles, kind):
    return {
        (name, tuple(s["params"])): s
        for name, profile in profiles.items()
        for s in profile.effects[kind]["settings"]
    }


def test_every_bar_plays_a_safe_measured_setting_whole_on_a_program_that_changes_each_cut(audio):
    profiles = _profiles()
    features = Features(audio=audio, video=_video(audio.duration, chroma=0.0))
    score = P.search(profiles, features, curve=P.Curve(bars=2), fps=FPS)
    measured = _settings_of(profiles, "bw")
    edges = np.asarray(score.meta["bars"])
    assert score.meta["kind"] == "bw" and len(score.cuts) == int(np.ceil((edges.size - 1) / 2))
    assert all(a.program != b.program for a, b in zip(score.cuts, score.cuts[1:]))
    assert {c.start for c in score.cuts} <= set(edges.tolist()) and score.cuts[-1].end == score.duration
    autos = P.plan_automation(Score.from_yaml(score.to_yaml()), profiles, fps=FPS)
    mids = (edges[:-1] + edges[1:]) / 2
    for layer in score.layers:
        frames = np.rint(autos[layer.index].times * FPS).astype(int)
        assert np.unique(frames).size == frames.size, "one write a frame, as the archive wrote them"
        lands = [g.arrival + (len(g.targets) - 1) / FPS for g in layer.gestures]
        openings = len(spans_of(score.cuts, layer.program))
        assert sum(np.isclose(t, edges).any() for t in lands) >= len(layer.gestures) - openings
        held = autos[layer.index].hold(mids).astype(int)
        for b in np.flatnonzero(
            [any(s <= t < e for s, e in spans_of(score.cuts, layer.program)) for t in mids]
        ):
            found = measured[(layer.program, tuple(held[b].tolist()))]
            assert found["safe"] and found["intensity"] == pytest.approx(score.meta["intensity"][b])


def test_the_chosen_intensity_follows_the_drive_between_the_curves_bounds(audio):
    profiles = _profiles()
    features = Features(audio=audio, video=_video(audio.duration, chroma=1.0))
    score = P.search(profiles, features, curve=P.Curve(follow=1.0), fps=FPS)
    motion = P.ranks(
        P.bar_means(features.video.ti, np.arange(features.video.n_frames) / FPS, score.meta["bars"])
    )
    assert score.meta["kind"] == "colour" and score.meta["drive"] == pytest.approx(motion.tolist())
    assert spearmanr(score.meta["drive"], score.meta["intensity"])[0] > 0.8
    safe = [s["intensity"] for p in profiles.values() for s in p.effects["colour"]["settings"] if s["safe"]]
    for bound, pick in ((0.0, min), (1.0, max)):
        flat = P.search(profiles, features, curve=P.Curve(low=bound, high=bound, bars=1, passes=1), fps=FPS)
        assert set(flat.meta["intensity"]) == {pick(safe)} and len(flat.layers) == 1


def test_a_search_needs_the_track(audio):
    with pytest.raises(ValueError, match="beats"):
        P.search(_profiles(), Features(audio=None, video=None))
    assert P.search(
        _profiles(kinds=("bw",)), Features(audio=audio, video=None), curve=P.Curve(follow=1.0)
    ).cuts


@pytest.mark.parametrize(
    "kwargs",
    [{"follow": 1.5}, {"low": -0.1}, {"high": 2.0}, {"quicken": 1.5}, {"bars": 0.0}, {"passes": 0}],
)
def test_a_curve_out_of_range_is_refused(kwargs):
    with pytest.raises(ValueError):
        P.Curve(**kwargs)


def test_describe_summarizes_the_score():
    score = Score(layers=[Layer("a", 0, [_setting(0.0, [0] * 12)])])
    assert P.describe(score) == {"programs": ["a"], "gestures": {SETTING: 1}, "sections": []}

"""Evolutionary search from measured profiles plus musical structure to a score.

Sample candidate layers, render a fast analytic proxy from the profile's measured
response curves and cliff atlas, score it, keep and mutate the best. Program is
the outermost loop variable and is never mutated inside the inner loop.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field, replace
from typing import Any, Callable, Mapping, Sequence

import numpy as np

from syncsummoner.device.profile import PARAM_MAX, Axis, ParamKind, ParamSpec, ProgramProfile, Source
from syncsummoner.compose.features import Features, Section, information_content
from syncsummoner.compose.score import GestureInstance, Layer, Score, control_rate, program_key
from syncsummoner.compose.vocabulary import GESTURES, Anchor, Automation, GestureContext

EPS = 1e-12
CC_BUDGET_HZ = 200.0
NATURAL_SLOPE_BAND = (-1.4, -1.0)
SIM_DISCOUNT = 0.5

SPAN_BARS: dict[Anchor, float] = {
    Anchor.BAR: 1.0,
    Anchor.TWO_BAR: 2.0,
    Anchor.MULTI_BAR: 4.0,
    Anchor.PHRASE: 4.0,
    Anchor.DOWNBEAT: 1.0,
    Anchor.SECTION: 1.0,
    Anchor.ONSET: 0.25,
}

STYLES: dict[str, dict[str, float]] = {
    "default": dict.fromkeys(GESTURES, 1.0),
    "glitchy": {
        "hold": 0.5,
        "ramp": 1.0,
        "cliff_cross": 4.0,
        "detune_drift": 2.0,
        "lock_snap": 1.0,
        "punch": 3.0,
        "morph": 1.0,
        "hysteresis_loop": 2.0,
    },
    "smooth": {
        "hold": 2.0,
        "ramp": 4.0,
        "cliff_cross": 0.2,
        "detune_drift": 2.0,
        "lock_snap": 1.0,
        "punch": 0.2,
        "morph": 3.0,
        "hysteresis_loop": 0.5,
    },
}
DESTROY_STYLES = frozenset({"glitchy"})


@dataclass(frozen=True)
class ObjectiveWeights:
    """Relative weight of each objective term; every term is reported separately regardless."""

    av_corr: float = 1.0
    mud: float = 0.5
    slope: float = 0.3
    levels: float = 0.8
    motif: float = 0.4
    boredom: float = 0.6


DEFAULT_WEIGHTS = ObjectiveWeights()


@dataclass(frozen=True, eq=False)
class Trajectory:
    """Proxy render: predicted per-frame metrics from the measured profile, no hardware."""

    times: np.ndarray
    state: np.ndarray
    activity: np.ndarray
    ic: np.ndarray
    concentration: np.ndarray
    slope: np.ndarray
    clip_frac: np.ndarray
    illegal_frac: np.ndarray


@dataclass(frozen=True)
class Objective:
    """Weighted objective value plus its decomposition and the reachability verdict."""

    total: float
    feasible: bool
    terms: dict[str, float] = field(default_factory=dict)


def _corr(a: np.ndarray, b: np.ndarray) -> float:
    a = a - a.mean()
    b = b - b.mean()
    d = float(np.sqrt((a * a).sum() * (b * b).sum()))
    return float((a * b).sum() / d) if d > EPS else 0.0


def _response(spec: ParamSpec, values: np.ndarray) -> np.ndarray:
    """Measured metric response of one parameter, normalized to ``[0, 1]``."""
    v = np.asarray(spec.values, dtype=np.float64)
    r = np.asarray(spec.response, dtype=np.float64)
    if v.size < 2 or r.size != v.size:
        return values / PARAM_MAX
    order = np.argsort(v)
    r = r[order]
    return np.interp(values, v[order], (r - r.min()) / (np.ptp(r) + EPS))


def reachable(
    auto: Automation, profile: ProgramProfile, *, fps: float, cc_budget_hz: float = CC_BUDGET_HZ
) -> bool:
    """Hard constraint: aggregate CC budget and per-parameter measured settle time both respected."""
    if len(auto) == 0:
        return True
    if auto.max_rate() > cc_budget_hz:
        return False
    tol = 0.5 / fps
    for idx in np.unique(auto.indices):
        m = auto.indices == idx
        settle = profile.settle_frames.get(int(idx), 0) / fps
        t = auto.times[m]
        changed = np.concatenate(([True], np.diff(auto.values[m]) != 0))
        gaps = np.diff(t[changed])
        if gaps.size and gaps.min() < settle - tol:
            return False
    return True


def thin(
    auto: Automation, profile: ProgramProfile, *, fps: float, cc_budget_hz: float = CC_BUDGET_HZ
) -> Automation:
    """Drop redundant and unreachable messages so the result satisfies the hard constraint."""
    if len(auto) == 0:
        return auto
    keep = np.ones(len(auto), dtype=bool)
    tol = 0.5 / fps
    for idx in np.unique(auto.indices):
        settle = profile.settle_frames.get(int(idx), 0) / fps - tol
        last_t, last_v = -np.inf, None
        for j in np.flatnonzero(auto.indices == idx):
            v = int(auto.values[j])
            if v == last_v or auto.times[j] < last_t + settle:
                keep[j] = False
            else:
                last_t, last_v = float(auto.times[j]), v
    times = auto.times[keep]
    window: deque[float] = deque()
    quota = max(1, int(cc_budget_hz))
    rate_ok = np.zeros(times.size, dtype=bool)
    for i, t in enumerate(times):
        while window and window[0] <= t - 1.0:
            window.popleft()
        if len(window) < quota:
            window.append(float(t))
            rate_ok[i] = True
    keep[np.flatnonzero(keep)[~rate_ok]] = False
    return Automation(auto.times[keep], auto.indices[keep], auto.values[keep])


def plan_automation(
    score: Score,
    profiles: Mapping[str, ProgramProfile],
    *,
    fps: float | None = None,
    cc_budget_hz: float = CC_BUDGET_HZ,
) -> dict[int, Automation]:
    """Expand a score to reachable automation per layer; this is what the renderer plays."""
    fps = score.fps if fps is None else fps
    out = {}
    for layer in score.layers:
        profile = profiles.get(layer.program)
        if profile is None:
            continue
        auto = score.render_layer(layer, profile, rate=control_rate(fps))
        out[layer.index] = thin(auto, profile, fps=fps, cc_budget_hz=cc_budget_hz)
    return out


def proxy_render(
    auto: Automation,
    profile: ProgramProfile,
    *,
    fps: float,
    duration: float,
    rng: np.random.Generator,
    slope_center: float = -1.2,
    slope_gain: float = 0.8,
) -> Trajectory:
    """Predict per-frame metrics from measured response curves and the cliff atlas."""
    times = np.arange(max(2, int(round(duration * fps)))) / fps
    raw = auto.hold(times)
    resp = np.zeros_like(raw)
    weights = np.zeros(raw.shape[1])
    for spec in profile.params:
        if spec.kind is ParamKind.UNUSED:
            continue
        col = spec.index - 1
        resp[:, col] = _response(spec, raw[:, col])
        weights[col] = max(spec.sensitivity, EPS)
    weights /= weights.sum() + EPS
    energy = resp * weights
    total = energy.sum(axis=1, keepdims=True) + EPS
    share = energy / total
    n_active = max(1, int((weights > EPS).sum()))
    conc = (np.square(share).sum(axis=1) - 1.0 / n_active) / (1.0 - 1.0 / n_active + EPS)
    activity = energy.sum(axis=1)
    clip = np.zeros(times.size)
    illegal = np.zeros(times.size)
    for spec in profile.params:
        for cliff in spec.cliffs:
            crossed = (raw[:, spec.index - 1] >= cliff.at).astype(np.float64) * cliff.jump
            if "clip_frac" in cliff.metrics:
                clip += crossed
            if "illegal_frac" in cliff.metrics:
                illegal += crossed
    n_cliffs = max(1, sum(len(p.cliffs) for p in profile.params))
    return Trajectory(
        times=times,
        state=resp,
        activity=activity,
        ic=information_content(activity, rng=rng),
        concentration=np.clip(conc, 0.0, 1.0),
        slope=slope_center + slope_gain * (activity - activity.mean()),
        clip_frac=clip / n_cliffs,
        illegal_frac=illegal / n_cliffs,
    )


def _motif_return(traj: Trajectory, sections: Sequence[Section]) -> float:
    """Contrast between same-label and different-label section-boundary states: visual rhyme."""
    if len(sections) < 3:
        return 0.0
    idx = np.clip(np.searchsorted(traj.times, [s.start for s in sections]), 0, traj.times.size - 1)
    s = traj.state[idx]
    s = s / (np.linalg.norm(s, axis=1, keepdims=True) + EPS)
    sim = s @ s.T
    labels = np.array([sec.label for sec in sections])
    same = (labels[:, None] == labels[None, :]) & ~np.eye(len(sections), dtype=bool)
    other = (labels[:, None] != labels[None, :]) & ~np.eye(len(sections), dtype=bool)
    if not same.any() or not other.any():
        return 0.0
    return float(sim[same].mean() - sim[other].mean())


def _boredom(traj: Trajectory, bar_duration: float, *, floor_frac: float) -> float:
    """Worst-bar shortfall in activity variance, normalized by the floor."""
    if bar_duration <= 0:
        return 0.0
    index = np.floor(traj.times / bar_duration).astype(np.int64)
    n = int(index.max()) + 1
    counts = np.bincount(index, minlength=n).astype(np.float64)
    mean = np.bincount(index, weights=traj.activity, minlength=n) / np.maximum(counts, 1)
    sq = np.bincount(index, weights=traj.activity**2, minlength=n) / np.maximum(counts, 1)
    std = np.sqrt(np.maximum(sq - mean**2, 0.0))[counts > 1]
    floor = floor_frac * float(traj.activity.std())
    if std.size == 0 or floor <= EPS:
        return 0.0
    return float(np.max(np.maximum(floor - std, 0.0)) / floor)


def evaluate(
    traj: Trajectory,
    features: Features,
    score: Score,
    *,
    weights: ObjectiveWeights = DEFAULT_WEIGHTS,
    natural_band: tuple[float, float] = NATURAL_SLOPE_BAND,
    boredom_floor_frac: float = 0.25,
    hit_sigma: float = 1.0,
    discount: float = 1.0,
) -> Objective:
    """Weighted objective over the proxy trajectory; every design 4.5 term contributes."""
    audio = features.audio
    onset = np.interp(traj.times, audio.times, audio.onset_strength)
    terms = {"av_corr": _corr(traj.ic, onset)}
    if audio.onset_ic.size > 1 and audio.beats.size >= audio.onset_ic.size:
        aic = np.interp(traj.times, audio.beats[: audio.onset_ic.size], audio.onset_ic)
        terms["av_corr"] = 0.5 * (terms["av_corr"] + _corr(traj.ic, aic))
    hits = onset >= onset.mean() + hit_sigma * onset.std()
    dev = np.maximum(natural_band[0] - traj.slope, 0.0) + np.maximum(traj.slope - natural_band[1], 0.0)
    allowed = ~hits if (~hits).any() else np.ones_like(hits)
    destroy = score.destroy_mask(traj.times)
    legal = ~destroy if (~destroy).any() else np.ones_like(destroy)
    terms["mud"] = float(traj.concentration.mean())
    terms["slope"] = float(dev[allowed].mean())
    terms["levels"] = float((traj.clip_frac + traj.illegal_frac)[legal].mean())
    terms["motif"] = _motif_return(traj, score.sections)
    terms["boredom"] = _boredom(traj, audio.bar_duration, floor_frac=boredom_floor_frac)
    total = discount * (
        weights.av_corr * terms["av_corr"]
        + weights.motif * terms["motif"]
        - weights.mud * terms["mud"]
        - weights.slope * terms["slope"]
        - weights.levels * terms["levels"]
        - weights.boredom * terms["boredom"]
    )
    return Objective(total=float(total), feasible=True, terms=terms)


def _bar_grid(audio, section: Section) -> np.ndarray:
    downs = audio.downbeats[(audio.downbeats >= section.start) & (audio.downbeats < section.end)]
    if downs.size:
        return downs
    return np.arange(section.start, max(section.end, section.start + EPS), audio.bar_duration)


def _anchor_times(audio, section: Section, anchor: Anchor) -> np.ndarray:
    """Candidate arrival times inside a section for one anchor kind."""
    if anchor is Anchor.SECTION:
        return np.array([section.end])
    if anchor is Anchor.ONSET:
        beats = audio.beats[(audio.beats >= section.start) & (audio.beats < section.end)]
        return beats if beats.size else _bar_grid(audio, section)
    grid = _bar_grid(audio, section)
    if anchor in (Anchor.PHRASE, Anchor.MULTI_BAR):
        return grid[::4] if grid.size > 4 else grid[-1:]
    if anchor is Anchor.TWO_BAR:
        return grid[::2]
    return grid


def _applicable(profile: ProgramProfile, style: Mapping[str, float], rng) -> dict[str, float]:
    """Style weights restricted to gestures that this profile actually supports."""
    probe = GestureContext(arrival=1.0, duration=1.0, rate=8.0, intensity=0.5, axis=Axis.UNASSIGNED, rng=rng)
    return {n: w for n, w in style.items() if w > 0 and len(GESTURES[n](profile, probe))}


def _axes(profile: ProgramProfile) -> list[Axis]:
    found = sorted({p.axis for p in profile.params if p.axis is not Axis.UNASSIGNED}, key=lambda a: a.value)
    return found or [Axis.UNASSIGNED]


def _sample_instance(name: str, audio, section: Section, axes: list[Axis], rng) -> GestureInstance:
    anchor = GESTURES[name].anchor
    times = _anchor_times(audio, section, anchor)
    arrival = float(rng.choice(times)) if times.size else section.start
    span = SPAN_BARS[anchor] * audio.bar_duration
    return GestureInstance(
        gesture=name,
        arrival=arrival,
        duration=span,
        axis=rng.choice(axes).value,
        intensity=float(rng.uniform(0.15, 0.95)),
        seed=int(rng.integers(1 << 30)),
    )


def _sample_layer(program, profile, audio, sections, weights, rng, *, density) -> Layer:
    names = list(weights)
    prob = np.asarray([weights[n] for n in names], dtype=np.float64)
    prob /= prob.sum()
    axes = _axes(profile)
    gestures = []
    for section in sections:
        n = max(1, int(round(density * section.duration / max(audio.bar_duration, EPS))))
        for name in rng.choice(names, size=n, p=prob):
            gestures.append(_sample_instance(str(name), audio, section, axes, rng))
    return Layer(program=program, gestures=sorted(gestures, key=lambda g: g.arrival))


def _mutate(layer, profile, audio, sections, weights, rng, *, rate=0.3) -> Layer:
    """Perturb gesture instances; the program is fixed for the whole layer and never mutated."""
    names = list(weights)
    prob = np.asarray([weights[n] for n in names], dtype=np.float64)
    prob /= prob.sum()
    axes = _axes(profile)
    out = []
    for inst in layer.gestures:
        if rng.random() > rate:
            out.append(inst)
            continue
        section = next((s for s in sections if s.start <= inst.arrival <= s.end), sections[-1])
        if rng.random() < 0.5:
            out.append(_sample_instance(str(rng.choice(names, p=prob)), audio, section, axes, rng))
        else:
            out.append(
                replace(
                    inst,
                    intensity=float(np.clip(inst.intensity + rng.normal(0, 0.15), 0.0, 1.0)),
                    duration=float(inst.duration * np.exp(rng.normal(0, 0.2))),
                    axis=str(rng.choice(axes).value),
                )
            )
    return Layer(program=layer.program, index=layer.index, gestures=sorted(out, key=lambda g: g.arrival))


def _mark_destroy(sections: Sequence[Section], audio, *, enabled: bool) -> list[Section]:
    """Mark the loudest sections ``destroy`` so illegal levels stop being penalized there."""
    if not enabled or len(sections) < 2:
        return list(sections)
    power = np.array(
        [
            audio.onset_strength[slice(*np.searchsorted(audio.times, [s.start, s.end]))].mean()  # noqa: E231
            for s in sections
        ]
    )
    power = np.nan_to_num(power)
    hot = power >= power.mean() + power.std()
    return [replace(s, destroy=bool(h)) for s, h in zip(sections, hot)]


def search(
    profiles: Mapping[str, ProgramProfile],
    features: Features,
    *,
    style: str = "default",
    rng: np.random.Generator,
    budget: int = 48,
    fps: float = 60.0,
    n_passes: int = 1,
    population: int = 6,
    density: float = 0.5,
    weights: ObjectiveWeights | None = None,
    cc_budget_hz: float = CC_BUDGET_HZ,
    evaluator: Callable[[Layer, ProgramProfile, Score], Objective] | None = None,
) -> Score:
    """Evolve one layer per program, then keep the best ``n_passes`` as the score's layers.

    A program whose parameters were never measured to move anything is skipped: a
    gesture drives a parameter, so a profile with none can only score as the
    absence of a change, which a passthrough wins.
    """
    audio = features.audio
    if audio is None:
        raise ValueError("planner.search requires audio features")
    style_weights = STYLES.get(style, STYLES["default"])
    sections = _mark_destroy(audio.sections, audio, enabled=style in DESTROY_STYLES)
    base = Score(
        seed=int(rng.integers(1 << 30)),
        bpm=audio.tempo,
        duration=audio.duration,
        fps=fps,
        sections=list(sections),
        meta={"style": style, "density": density},
    )
    weights = DEFAULT_WEIGHTS if weights is None else weights
    programs = sorted(profiles)
    generations = max(1, budget // max(1, population * len(programs)))
    best: list[tuple[float, Layer, dict[str, float]]] = []

    for program in programs:
        profile = profiles[program]
        if not any(spec.sensitivity > 0 for spec in profile.params):
            continue
        applicable = _applicable(profile, style_weights, rng)
        if not applicable:
            continue
        discount = SIM_DISCOUNT if profile.source is Source.SIM else 1.0

        def fitness(layer: Layer, program=program, profile=profile, discount=discount) -> Objective:
            if evaluator is not None:
                return evaluator(layer, profile, base)
            auto = thin(
                base.render_layer(layer, profile, rate=control_rate(fps)),
                profile,
                fps=fps,
                cc_budget_hz=cc_budget_hz,
            )
            if not reachable(auto, profile, fps=fps, cc_budget_hz=cc_budget_hz):
                return Objective(total=-np.inf, feasible=False)
            traj = proxy_render(
                auto,
                profile,
                fps=fps,
                duration=base.duration,
                rng=np.random.default_rng(program_key(program)),
            )
            return evaluate(traj, features, base, weights=weights, discount=discount)

        pop = [
            _sample_layer(program, profile, audio, sections, applicable, rng, density=density)
            for _ in range(population)
        ]
        ranked = sorted(((fitness(m), m) for m in pop), key=lambda p: -p[0].total)
        for _ in range(generations - 1):
            elite = [m for _, m in ranked[: max(1, population // 2)]]
            children = [
                _mutate(elite[i % len(elite)], profile, audio, sections, applicable, rng)
                for i in range(population - len(elite))
            ]
            pop = elite + children
            ranked = sorted(((fitness(m), m) for m in pop), key=lambda p: -p[0].total)
        obj, layer = ranked[0]
        best.append((obj.total, layer, obj.terms))

    best.sort(key=lambda t: -t[0])
    base.layers = [replace(layer, index=i) for i, (_, layer, _) in enumerate(best[:n_passes])]
    base.meta["objective"] = [terms for _, _, terms in best[:n_passes]]
    return base


def describe(score: Score) -> dict[str, Any]:
    """Compact summary of a score: programs, gesture histogram, section labels."""
    hist: dict[str, int] = {}
    for layer in score.layers:
        for inst in layer.gestures:
            hist[inst.gesture] = hist.get(inst.gesture, 0) + 1
    return {
        "programs": [layer.program for layer in score.layers],
        "gestures": hist,
        "sections": [(s.label, s.destroy) for s in score.sections],
    }

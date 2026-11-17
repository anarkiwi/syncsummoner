"""Evolutionary search from measured profiles plus musical structure to a score.

Sample candidate layers, render a fast analytic proxy from the profile's measured
response curves and cliff atlas, score it, keep and mutate the best. Program is
the outermost loop variable and is never mutated inside the inner loop.
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from typing import Any, Callable, Mapping, Sequence

import numpy as np

from syncsummoner.device.profile import PARAM_MAX, Axis, ParamKind, ParamSpec, ProgramProfile, Source
from syncsummoner.progress import LOG, track
from syncsummoner import aesthetics
from syncsummoner.compose.cut import Cut, cut_plan, spans_of
from syncsummoner.compose.gating import model_gate
from syncsummoner.compose.features import Features, Section, information_content, phrases
from syncsummoner.compose.score import GestureInstance, Layer, Score, control_rate, program_key
from syncsummoner.compose.thinning import flush, reachable, settle_of, settled, thin, visible
from syncsummoner.compose.vocabulary import GESTURES, Anchor, Automation, GestureContext, candidates

EPS = 1e-12
NATURAL_SLOPE_BAND = (-1.4, -1.0)
SIM_DISCOUNT = 0.5
#: Physical span of each metric the objective scores, so movement in one is
#: comparable to movement in another and to the same metric in another program.
#: The three fractions are bounded by construction; the slope's span is its
#: measured range over every archived program.
METRIC_SPAN = {"spectral_slope": 4.2, "concentration": 1.0, "clip_frac": 1.0, "illegal_frac": 1.0}
#: Stasis floor for profiles that measured no survival, as a fraction of each metric's span.
BOREDOM_FLOOR = 0.02
#: Survival responses a footage archive measures, as each parameter's share of what the source keeps.
SURVIVAL = ("si_ratio", "ti_ratio", "motion_follow")

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
        "hysteresis_loop": 2.0,
    },
    "smooth": {
        "hold": 2.0,
        "ramp": 4.0,
        "cliff_cross": 0.2,
        "detune_drift": 2.0,
        "lock_snap": 1.0,
        "punch": 0.2,
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
#: Per-style weights; ``glitchy`` leaves the natural slope band and legal levels on purpose, so neither costs.
STYLE_WEIGHTS: dict[str, ObjectiveWeights] = {
    "glitchy": ObjectiveWeights(slope=0.0, levels=0.0, boredom=1.0),
}


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
    #: Predicted share of the source's SI and TI kept, and how far output motion follows it; None unmeasured.
    si_ratio: np.ndarray | None = None
    ti_ratio: np.ndarray | None = None
    follow: np.ndarray | None = None


@dataclass(frozen=True)
class Objective:
    """Weighted objective value, its decomposition, the reachability verdict and dead bars over budget.

    ``violation`` counts bars predicted dead beyond what destroy-marked sections allow; the search
    ranks by it before ``total``, so no weighting trades a dead picture for anything else.
    """

    total: float
    feasible: bool
    terms: dict[str, float] = field(default_factory=dict)
    violation: float = 0.0

    @property
    def rank(self) -> tuple[float, float]:
        """Sort key: fewer dead bars first, then the higher total."""
        return (self.violation, -self.total)


def _corr(a: np.ndarray, b: np.ndarray) -> float:
    a = a - a.mean()
    b = b - b.mean()
    d = float(np.sqrt((a * a).sum() * (b * b).sum()))
    return float((a * b).sum() / d) if d > EPS else 0.0


def _response(spec: ParamSpec, values: np.ndarray) -> np.ndarray:
    """The profiled metric response of one parameter, normalized to ``[0, 1]``."""
    v = np.asarray(spec.values, dtype=np.float64)
    r = np.asarray(spec.response, dtype=np.float64)
    if v.size < 2 or r.size != v.size:
        return values / PARAM_MAX
    order = np.argsort(v)
    r = r[order]
    return np.interp(values, v[order], (r - r.min()) / (np.ptp(r) + EPS))


def score_spans(score: Score, program: str) -> list[tuple[float, float]] | None:
    """The spans a program's take is shown in, or ``None`` when the score carries no cut plan."""
    return spans_of(score.cuts, program) if score.cuts else None


def plan_automation(
    score: Score,
    profiles: Mapping[str, ProgramProfile],
    *,
    fps: float | None = None,
    gates: Mapping[str, Callable[[Automation], Automation] | None] | None = None,
) -> dict[int, Automation]:
    """Expand a score to reachable automation per layer; this is what the renderer plays.

    A score carrying a cut plan sends each layer's writes only inside its program's cuts. Each
    layer's writes then pass its program's gate (default: from the program model), which pairs
    a write to a dead parameter with the move that makes it live or refuses it.
    """
    fps = score.fps if fps is None else fps
    out = {}
    for layer in score.layers:
        profile = profiles.get(layer.program)
        if profile is None:
            continue
        auto = score.render_layer(layer, profile, rate=control_rate(fps))
        auto = visible(thin(auto, profile, fps=fps), score_spans(score, layer.program))
        gate = model_gate(layer.program) if gates is None else gates.get(layer.program)
        out[layer.index] = auto if gate is None else gate(auto)
    return out


def _marginal(profile: ProgramProfile, raw: np.ndarray, name: str) -> np.ndarray | None:
    """Superpose each parameter's measured curve for one metric over the timeline.

    ``ParamSpec.measured`` holds each parameter's first-order marginal, so the prediction at a
    vector is the sum of their departures from their medians (first-order HDMR); ``None``
    where nothing measured the metric.
    """
    deltas, bases = [], []
    for spec in profile.params:
        curve = spec.measured.get(name)
        if not curve or len(curve) != len(spec.values) or len(curve) < 2:
            continue
        values = np.asarray(spec.values, dtype=np.float64)
        measured = np.asarray(curve, dtype=np.float64)
        order = np.argsort(values)
        base = float(np.median(measured))
        deltas.append(np.interp(raw[:, spec.index - 1], values[order], measured[order]) - base)
        bases.append(base)
    if not deltas:
        return None
    return np.sum(deltas, axis=0) + float(np.mean(bases))


def _cliff_fractions(profile: ProgramProfile, raw: np.ndarray, size: int) -> tuple[np.ndarray, np.ndarray]:
    """Clip and illegal fractions inferred from the cliff atlas, for profiles with no measured curves."""
    clip, illegal = np.zeros(size), np.zeros(size)
    for spec in profile.params:
        for cliff in spec.cliffs:
            crossed = (raw[:, spec.index - 1] >= cliff.at).astype(np.float64) * cliff.jump
            if "clip_frac" in cliff.metrics:
                clip += crossed
            if "illegal_frac" in cliff.metrics:
                illegal += crossed
    n_cliffs = max(1, sum(len(p.cliffs) for p in profile.params))
    return clip / n_cliffs, illegal / n_cliffs


def proxy_render(
    auto: Automation,
    profile: ProgramProfile,
    *,
    fps: float,
    duration: float,
    rng: np.random.Generator,
    start: float = 0.0,
    slope_center: float = -1.2,
    slope_gain: float = 0.8,
) -> Trajectory:
    """Predict per-frame metrics from the measured curves, the cliff atlas and the response.

    Every metric the objective scores is read back from what the probe measured at
    each setpoint. Where a profile predates those curves the older inferences stand
    in, so an unrefitted profile still composes. The frames are those of
    ``[start, start + duration)``, frame ``n`` at ``n / fps``.
    """
    first = int(np.rint(start * fps))
    times = (first + np.arange(max(2, int(np.rint((start + duration) * fps)) - first))) / fps
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

    measured = {name: _marginal(profile, raw, name) for name in METRIC_SPAN}
    moved = [(m - np.median(m)) / METRIC_SPAN[name] for name, m in measured.items() if m is not None]
    if moved:
        activity = np.sqrt(np.mean(np.square(moved), axis=0))
    else:
        activity = energy.sum(axis=1)

    if measured["concentration"] is not None:
        conc = measured["concentration"]
    else:
        total = energy.sum(axis=1, keepdims=True) + EPS
        share = energy / total
        n_active = max(1, int((weights > EPS).sum()))
        conc = (np.square(share).sum(axis=1) - 1.0 / n_active) / (1.0 - 1.0 / n_active + EPS)

    slope = measured["spectral_slope"]
    if slope is None:
        slope = slope_center + slope_gain * (activity - activity.mean())

    clip, illegal = measured["clip_frac"], measured["illegal_frac"]
    if clip is None or illegal is None:
        inferred_clip, inferred_illegal = _cliff_fractions(profile, raw, times.size)
        clip = inferred_clip if clip is None else clip
        illegal = inferred_illegal if illegal is None else illegal

    kept = {name: _marginal(profile, raw, name) for name in SURVIVAL}
    return Trajectory(
        times=times,
        state=resp,
        activity=activity,
        ic=information_content(activity, rng=rng),
        concentration=np.clip(conc, 0.0, 1.0),
        slope=slope,
        clip_frac=np.clip(clip, 0.0, 1.0),
        illegal_frac=np.clip(illegal, 0.0, 1.0),
        si_ratio=None if kept["si_ratio"] is None else np.maximum(kept["si_ratio"], 0.0),
        ti_ratio=None if kept["ti_ratio"] is None else np.maximum(kept["ti_ratio"], 0.0),
        follow=None if kept["motion_follow"] is None else np.clip(kept["motion_follow"], -1.0, 1.0),
    )


def _motif_return(traj: Trajectory, sections: Sequence[Section]) -> float:
    """Contrast between same-label and different-label section-boundary states: visual rhyme."""
    sections = [sec for sec in sections if traj.times[0] <= sec.start <= traj.times[-1]]
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


def _boredom(traj: Trajectory, bar_duration: float, *, floor: float) -> float:
    """Worst-bar shortfall in activity variance against an absolute floor.

    The floor is fixed rather than a fraction of the trajectory's own spread: a
    program that never moves has no spread, so a self-normalized floor collapses to
    zero and scores stasis as perfect.
    """
    if bar_duration <= 0:
        return 0.0
    index = np.floor(traj.times / bar_duration).astype(np.int64)
    n = int(index.max()) + 1
    counts = np.bincount(index, minlength=n).astype(np.float64)
    mean = np.bincount(index, weights=traj.activity, minlength=n) / np.maximum(counts, 1)
    sq = np.bincount(index, weights=traj.activity**2, minlength=n) / np.maximum(counts, 1)
    std = np.sqrt(np.maximum(sq - mean**2, 0.0))[counts > 1]
    if std.size == 0 or floor <= EPS:
        return 0.0
    return float(np.max(np.maximum(floor - std, 0.0)) / floor)


def bar_edges(features: Features, end: float, start: float = 0.0) -> np.ndarray:
    """Bar boundaries in seconds over ``[start, end]``: the downbeats, else the bar period."""
    audio = features.audio
    marks = audio.downbeats if audio.downbeats.size > 1 else np.arange(0.0, end, audio.bar_duration)
    return np.unique(np.r_[start, marks[(marks > start) & (marks < end)], end])


@dataclass(frozen=True)
class Kept:
    """Predicted survival per bar: how much picture and motion the output keeps, and which bars die."""

    starts: np.ndarray
    ends: np.ndarray
    ti_ratio: np.ndarray
    dead: np.ndarray


def predict_survival(traj: Trajectory, features: Features) -> Kept | None:
    """Judge each bar of a predicted trajectory against the same bar of the source footage.

    A bar is dead where the output's predicted SI or TI falls below
    :func:`~syncsummoner.aesthetics.live_floor` of the source's, or where its predicted
    following would not reach :func:`~syncsummoner.aesthetics.critical_z` over the bar's frames.
    """
    video = features.video
    if traj.si_ratio is None or traj.ti_ratio is None or video is None or video.si is None:
        return None
    edges = bar_edges(features, float(traj.times[-1]) + (traj.times[1] - traj.times[0]), float(traj.times[0]))
    frame = np.clip(np.rint(traj.times * video.fps).astype(np.int64), 0, video.si.size - 1)
    source_si, source_ti = video.si[frame], np.nan_to_num(video.ti[frame])
    follow = traj.follow if traj.follow is not None else np.ones_like(traj.ti_ratio)
    owner = np.clip(np.searchsorted(edges, traj.times, side="right") - 1, 0, edges.size - 2)
    ratio, dead = np.zeros(edges.size - 1), np.zeros(edges.size - 1, dtype=bool)
    for b in range(edges.size - 1):
        m = owner == b
        if not m.any():
            continue
        lo, hi = np.clip(np.rint(edges[b : b + 2] * video.fps).astype(np.int64), 0, video.ti.size)
        series = np.nan_to_num(video.ti[lo:hi])
        si, ti = float(source_si[m].mean()), float(source_ti[m].mean())
        ratio[b] = float(traj.ti_ratio[m].mean())
        flat = traj.si_ratio[m].mean() * si < aesthetics.live_floor(si, aesthetics.quantisation_si())
        frozen = ratio[b] * ti < aesthetics.live_floor(ti, aesthetics.quantisation_ti())
        unfollowed = (
            series.size > 3
            and aesthetics.follow_z(float(follow[m].mean()), aesthetics.effective_samples(series, series))
            <= aesthetics.critical_z()
        )
        dead[b] = bool(flat or frozen or unfollowed)
    return Kept(edges[:-1], edges[1:], ratio, dead)


def dead_bars_over_budget(kept: Kept, score: Score) -> float:
    """Dead bars the score cannot afford, destruction being budgeted one bar at a time.

    Any dead bar outside a destroy-marked section counts, and inside one every bar of a dead
    run beyond its first.
    """
    inside = score.destroy_mask(0.5 * (kept.starts + kept.ends))
    edges = np.flatnonzero(np.diff(np.r_[0, kept.dead.astype(np.int8), 0]))
    over = 0
    for a, b in zip(edges[::2], edges[1::2]):
        over += int(np.count_nonzero(~inside[a:b])) + max(0, int(np.count_nonzero(inside[a:b])) - 1)
    return float(over)


def evaluate(
    traj: Trajectory,
    features: Features,
    score: Score,
    *,
    weights: ObjectiveWeights = DEFAULT_WEIGHTS,
    natural_band: tuple[float, float] = NATURAL_SLOPE_BAND,
    boredom_floor: float = BOREDOM_FLOOR,
    hit_sigma: float = 1.0,
    discount: float = 1.0,
) -> Objective:
    """Weighted objective over the proxy trajectory; every design 4.5 term contributes."""
    audio = features.audio
    onset = np.interp(traj.times, audio.times, audio.onset_strength)
    terms = {"av_corr": _corr(traj.ic, onset)}
    if 1 < audio.onset_ic.size <= audio.beats.size:
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
    kept = predict_survival(traj, features)
    if kept is None:
        terms["boredom"] = _boredom(traj, audio.bar_duration, floor=boredom_floor)
        violation = 0.0
    else:
        terms["boredom"] = float(np.max(1.0 - np.minimum(kept.ti_ratio, 1.0)))
        violation = dead_bars_over_budget(kept, score)
        terms["dead_bars"] = float(kept.dead.sum())
    total = discount * (
        weights.av_corr * terms["av_corr"]
        + weights.motif * terms["motif"]
        - weights.mud * terms["mud"]
        - weights.slope * terms["slope"]
        - weights.levels * terms["levels"]
        - weights.boredom * terms["boredom"]
    )
    return Objective(total=float(total), feasible=True, terms=terms, violation=violation)


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


def _aim(name: str, profile: ProgramProfile, rng) -> tuple[str, tuple[int, ...]]:
    """The axis and parameter a gesture is aimed at, drawn alike from every parameter it can drive.

    A gesture left to its axis drives only that axis's most sensitive parameter, so every other
    parameter, and every one assigned no axis, would never move.
    """
    pool = candidates(profile, name)
    if not pool:
        return str(rng.choice(_axes(profile)).value), ()
    spec = pool[int(rng.integers(len(pool)))]
    return spec.axis.value, (spec.index,)


def _fit_span(inst: GestureInstance, span: Section) -> GestureInstance:
    """The instance with its approach shortened to start inside ``span``."""
    return replace(inst, duration=float(np.clip(inst.duration, 0.0, inst.arrival - span.start)))


def _sample_instance(name: str, audio, span: Section, profile: ProgramProfile, rng) -> GestureInstance:
    """One gesture arriving on its anchor inside ``span``, its approach inside it too."""
    anchor = GESTURES[name].anchor
    times = _anchor_times(audio, span, anchor)
    times = times[(times >= span.start) & (times < span.end)]
    axis, aimed = _aim(name, profile, rng)
    inst = GestureInstance(
        gesture=name,
        arrival=float(rng.choice(times)) if times.size else span.start,
        duration=SPAN_BARS[anchor] * audio.bar_duration,
        axis=axis,
        intensity=float(rng.uniform(0.15, 0.95)),
        targets=aimed,
        seed=int(rng.integers(1 << 30)),
    )
    return _fit_span(inst, span)


def _mutate(gestures, profile, audio, span, weights, rng, *, rate=0.3) -> list[GestureInstance]:
    """Perturb gesture instances inside one span; the program is fixed and never mutated."""
    names = list(weights)
    prob = np.asarray([weights[n] for n in names], dtype=np.float64)
    prob /= prob.sum()
    out = []
    for inst in gestures:
        if rng.random() > rate:
            out.append(inst)
        elif rng.random() < 0.5:
            out.append(_sample_instance(str(rng.choice(names, p=prob)), audio, span, profile, rng))
        else:
            axis, aimed = _aim(inst.gesture, profile, rng)
            moved = replace(
                inst,
                intensity=float(np.clip(inst.intensity + rng.normal(0, 0.15), 0.0, 1.0)),
                duration=float(inst.duration * np.exp(rng.normal(0, 0.2))),
                axis=axis,
                targets=aimed,
            )
            out.append(_fit_span(moved, span))
    return sorted(out, key=lambda g: g.arrival)


def _fill(gestures, sample, writes, target: int, rng) -> list[GestureInstance]:
    """Gestures making at least ``target`` writes in the span and none past it, none of which
    could go and leave enough.

    ``writes`` counts a candidate's writes inside the span and past its end. One that thinning
    defers past the end lands unseen, leaving the parameter where the span left it, such as a
    toggle latched on, so a gesture making one is dropped and none is drawn that would. Each
    kept draw lands at least one write, so ``target`` draws bound the fill; where they fall
    short, only the draws that added a write stay.
    """
    out = list(gestures)
    for inst in list(out):
        rest = [g for g in out if g is not inst]
        if writes(rest)[1] < writes(out)[1]:
            out = rest
    for _ in range(target):
        inside, spilled = writes(out)
        if inside >= target:
            break
        drawn = out + [sample()]
        if writes(drawn)[1] <= spilled:
            out = drawn
    inside, spilled = writes(out)
    need = min(target, inside)
    for inst in [out[i] for i in rng.permutation(len(out))]:
        rest = [g for g in out if g is not inst]
        left = writes(rest)
        if rest and left[0] >= need and left[1] <= spilled:
            out = rest
    return sorted(out, key=lambda g: g.arrival)


def _combine(parts: Sequence[tuple[float, Objective]]) -> Objective:
    """One objective over several spans: totals and terms weighed by span length, dead bars summed."""
    weights = np.array([w for w, _ in parts], dtype=np.float64)
    weights = (
        weights / weights.sum() if weights.sum() > 0 else np.full(weights.size, 1.0 / max(1, weights.size))
    )
    keys = dict.fromkeys(k for _, obj in parts for k in obj.terms)
    terms = {
        k: float(
            sum(
                obj.terms.get(k, 0.0) * (1.0 if k == "dead_bars" else w)
                for w, (_, obj) in zip(weights, parts)
            )
        )
        for k in keys
    }
    return Objective(
        total=float(sum(w * obj.total for w, (_, obj) in zip(weights, parts))),
        feasible=all(obj.feasible for _, obj in parts),
        terms=terms,
        violation=float(sum(obj.violation for _, obj in parts)),
    )


@dataclass(frozen=True)
class _Opening:
    """What a program's committed plan leaves at a cut: its thinning scans, the requests still
    to come at or after the cut, and the value each parameter holds on screen as the cut opens.
    """

    scans: dict[int, tuple]
    tail: Automation
    held: Automation


@dataclass
class _Program:
    """One program being planned: its profile, the gestures and cuts committed to it so far."""

    name: str
    profile: ProgramProfile
    weights: dict[str, float]
    discount: float
    gestures: list[GestureInstance] = field(default_factory=list)
    raw: Automation = field(default_factory=Automation.empty)
    spans: list[tuple[float, float]] = field(default_factory=list)

    def opening(self, cut: Cut, fps: float) -> _Opening:
        """Scan every committed request before ``cut``; what a candidate for it continues from."""
        scans, held_t, held_i, held_v = {}, [], [], []
        before = self.raw.times < cut.start
        for idx in np.unique(self.raw.indices[before]).tolist():
            m = before & (self.raw.indices == idx)
            out_t, out_v, scan = settled(
                self.raw.times[m], self.raw.values[m], settle_of(self.profile, idx, fps)
            )
            if scan[2] is not None and scan[0] < cut.start:
                late_t, late_v = flush(scan)
                out_t, out_v, scan = (
                    out_t + late_t,
                    out_v + late_v,
                    (scan[0] + settle_of(self.profile, idx, fps), scan[2], None),
                )
            shown = visible(
                Automation(
                    np.asarray(out_t), np.full(len(out_t), idx, np.int32), np.asarray(out_v, np.int32)
                ),
                self.spans,
            )
            scans[idx] = scan
            if len(shown):
                held_t.append(cut.start - 1.0)
                held_i.append(idx)
                held_v.append(int(shown.values[-1]))
        tail = Automation(self.raw.times[~before], self.raw.indices[~before], self.raw.values[~before])
        held = Automation(
            np.asarray(held_t, np.float64), np.asarray(held_i, np.int32), np.asarray(held_v, np.int32)
        )
        return _Opening(scans, tail, held)

    def commit(self, cut: Cut, gestures: list[GestureInstance], raw: Automation) -> None:
        """Take on ``cut`` with the gestures planned for it and what they render."""
        self.gestures += gestures
        self.raw = Automation.concat([self.raw, raw])
        self.spans.append((cut.start, cut.end))


def _continued(opening: _Opening, raw: Automation, profile: ProgramProfile, fps: float) -> Automation:
    """The writes from a cut on: the committed tail and ``raw`` thinned on from the opening scans."""
    requests = Automation.concat([opening.tail, raw])
    parts = []
    for idx in np.unique(np.r_[requests.indices, list(opening.scans)]).astype(np.int32).tolist():
        m = requests.indices == idx
        scan = opening.scans.get(idx, (-np.inf, None, None))
        t, v, scan = settled(requests.times[m], requests.values[m], settle_of(profile, idx, fps), scan)
        late_t, late_v = flush(scan)
        parts.append(
            Automation(
                np.asarray(t + late_t),
                np.full(len(t) + len(late_t), idx, np.int32),
                np.asarray(v + late_v, np.int32),
            )
        )
    return Automation.concat(parts)


@dataclass(frozen=True)
class _Plan:
    """How one cut is planned: the search's settings and what every candidate is judged against."""

    base: Score
    features: Features
    weights: ObjectiveWeights
    density: float
    population: int
    generations: int
    evaluator: Callable[[Layer, ProgramProfile, Score], Objective] | None = None

    def cut(self, prog: _Program, cut: Cut, rng) -> tuple[Objective, list[GestureInstance], Automation]:
        """Evolve ``prog``'s gestures for one cut, judged on that cut's frames alone.

        Returns the best objective, its gestures, and what they render.
        """
        audio, fps = self.features.audio, self.base.fps
        span = Section(cut.start, cut.end, "")
        rate = control_rate(fps)
        opening = prog.opening(cut, fps)
        target = max(1, int(np.rint(self.density * cut.duration)))
        names = list(prog.weights)
        prob = np.asarray([prog.weights[n] for n in names], dtype=np.float64)
        prob /= prob.sum()

        def render(gestures) -> Automation:
            layer = Layer(prog.name, gestures=list(gestures))
            return self.base.render_layer(layer, prog.profile, rate=rate, offset=len(prog.gestures))

        counted: dict[tuple, tuple[int, int]] = {}

        def writes(gestures) -> tuple[int, int]:
            key = tuple(gestures)
            if key not in counted:
                after = _continued(opening, render(gestures), prog.profile, fps).times
                counted[key] = (int((after < cut.end).sum()), int((after >= cut.end).sum()))
            return counted[key]

        def sample() -> GestureInstance:
            return _sample_instance(str(rng.choice(names, p=prob)), audio, span, prog.profile, rng)

        def fitness(gestures) -> Objective:
            if self.evaluator is not None:
                layer = Layer(prog.name, gestures=prog.gestures + list(gestures))
                return self.evaluator(layer, prog.profile, replace(self.base, cuts=[cut]))
            shown = _continued(opening, render(gestures), prog.profile, fps).within(cut.start, cut.end)
            if not reachable(shown, prog.profile, fps=fps):
                return Objective(total=-np.inf, feasible=False, violation=np.inf)
            traj = proxy_render(
                Automation.concat([opening.held, shown]),
                prog.profile,
                fps=fps,
                start=cut.start,
                duration=cut.duration,
                rng=np.random.default_rng(program_key(prog.name)),
            )
            return evaluate(traj, self.features, self.base, weights=self.weights, discount=prog.discount)

        pop = [_fill([], sample, writes, target, rng) for _ in range(self.population)]
        ranked = sorted(((fitness(m), m) for m in pop), key=lambda p: p[0].rank)
        for _ in range(self.generations - 1):
            elite = [m for _, m in ranked[: max(1, self.population // 2)]]
            children = [
                _fill(
                    _mutate(elite[i % len(elite)], prog.profile, audio, span, prog.weights, rng),
                    sample,
                    writes,
                    target,
                    rng,
                )
                for i in range(self.population - len(elite))
            ]
            ranked = sorted(
                ranked[: len(elite)] + [(fitness(m), m) for m in children], key=lambda p: p[0].rank
            )
        obj, gestures = ranked[0]
        return obj, gestures, render(gestures)


def _standing(plan: _Plan, prog: _Program, sections: Sequence[Section], rng) -> Objective:
    """How a program fares over the whole piece: one generation of it showing every section."""
    trial = replace(plan, generations=1)
    scratch = replace(prog, gestures=[], raw=Automation.empty(), spans=[])
    parts = []
    for section in sections:
        cut = Cut(section.start, section.end, prog.name)
        obj, gestures, raw = trial.cut(scratch, cut, rng)
        scratch.commit(cut, gestures, raw)
        parts.append((cut.duration, obj))
    return _combine(parts)


def _assign(
    plan: _Plan, cuts: Sequence[Cut], programs: Mapping[str, _Program], rng
) -> tuple[list[Cut], dict[str, Objective]]:
    """Plan every cut in time order on its program, moving a cut predicted dead to one alive there.

    A cut is predicted dead when its bars die beyond the destruction budget; every other
    program is then planned for it, and the cut goes to the best ranked, its own program first
    on a tie, so it moves only to a program that keeps more of the picture.
    """
    out, parts = [], {name: [] for name in programs}
    for cut in track(sorted(cuts, key=lambda c: c.start), desc="cuts", unit="cut"):
        options = {cut.program: plan.cut(programs[cut.program], cut, rng)}
        own = options[cut.program][0]
        if own.violation > 0:
            options.update(
                {name: plan.cut(programs[name], cut, rng) for name in programs if name != cut.program}
            )
        name = min(options, key=lambda n, found=options: found[n][0].rank)
        obj, gestures, raw = options[name]
        if name != cut.program:
            LOG.info(
                "%7.2f-%7.2fs: %s predicted dead over %g bars, planned on %s",
                cut.start,
                cut.end,
                cut.program,
                own.violation,
                name,
            )
        elif obj.violation:
            LOG.warning(
                "%7.2f-%7.2fs: %s predicted dead over %g bars and no program is alive there",
                cut.start,
                cut.end,
                cut.program,
                obj.violation,
            )
        moved = replace(cut, program=name)
        programs[name].commit(moved, gestures, raw)
        parts[name].append((cut.duration, obj))
        out.append(moved)
    return out, {name: _combine(p) for name, p in parts.items() if p}


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


def _clip_sections(sections: Sequence[Section], duration: float) -> list[Section]:
    """Keep only what the render covers: sections past the end go, the last one is truncated."""
    kept = [replace(s, end=min(s.end, duration)) for s in sections if s.start < duration]
    return kept or [Section(start=0.0, end=duration, label="A")]


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
    evaluator: Callable[[Layer, ProgramProfile, Score], Objective] | None = None,
) -> Score:
    """Rank the programs, cut the best ``n_passes`` in rotation, and evolve each cut on its program.

    Each program first shows every section for one generation, which ranks it; the kept ones
    are cut in rank order over the sections by :func:`~syncsummoner.compose.cut.cut_plan`,
    and that plan is the score's. Every cut is then evolved on its own frames, from the state
    its program's earlier cuts leave, toward ``density`` writes per second on screen; a cut
    predicted dead moves to a program alive there (:func:`_assign`). A program whose
    parameters were never measured to move anything is skipped: a gesture drives a parameter,
    so a profile with none can only score as the absence of a change, which a passthrough wins.
    """
    audio = features.audio
    if audio is None:
        raise ValueError("planner.search requires audio features")
    style_weights = STYLES.get(style, STYLES["default"])
    duration = features.duration or audio.duration
    marked = _mark_destroy(audio.sections, audio, enabled=style in DESTROY_STYLES)
    sections = _clip_sections(list(phrases(marked, audio.downbeats, audio.phrase_bars)), duration)
    base = Score(
        seed=int(rng.integers(1 << 30)),
        bpm=audio.tempo,
        duration=duration,
        fps=fps,
        sections=list(sections),
        meta={
            "style": style,
            "density": density,
            "bar": audio.bar_duration,
            "phrase_bars": audio.phrase_bars,
        },
    )
    plan = _Plan(
        base=base,
        features=features,
        weights=STYLE_WEIGHTS.get(style, DEFAULT_WEIGHTS) if weights is None else weights,
        density=density,
        population=population,
        generations=max(1, budget // max(1, population * len(profiles))),
        evaluator=evaluator,
    )
    LOG.info(
        "composing %.1fs over %d programs: %d %s of %d candidates a cut, %d sections at %.1f bpm",
        duration,
        len(profiles),
        plan.generations,
        "generation" if plan.generations == 1 else "generations",
        population,
        len(sections),
        audio.tempo,
    )
    standing = []
    for program in track(sorted(profiles), desc="ranking", unit="program"):
        profile = profiles[program]
        if not any(spec.sensitivity > 0 for spec in profile.params):
            continue
        applicable = _applicable(profile, style_weights, rng)
        if not applicable:
            continue
        prog = _Program(program, profile, applicable, SIM_DISCOUNT if profile.source is Source.SIM else 1.0)
        obj = _standing(plan, prog, sections, rng)
        LOG.debug("%s: standing %.3f", program, obj.total)
        if "dead_bars" not in obj.terms:
            LOG.warning("%s: no survival measured, so its picture is unverified; refit from footage", program)
        standing.append((obj, prog))
    standing.sort(key=lambda t: t[0].rank)
    kept = {prog.name: prog for _, prog in standing[:n_passes]}
    if not kept:
        return base
    base.cuts, objectives = _assign(plan, cut_plan(base, list(kept)), kept, rng)
    shown = [name for name in kept if name in objectives]
    base.layers = [Layer(program=name, index=i, gestures=kept[name].gestures) for i, name in enumerate(shown)]
    base.meta["objective"] = [objectives[name].terms for name in shown]
    for name in shown:
        LOG.info(
            "kept %s: score %.3f, %d gestures over %d cuts",
            name,
            objectives[name].total,
            len(kept[name].gestures),
            len(kept[name].spans),
        )
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

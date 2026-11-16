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
from syncsummoner.device.session import CC_BUDGET_HZ
from syncsummoner.progress import LOG, track
from syncsummoner import aesthetics
from syncsummoner.compose.features import Features, Section, information_content, phrases
from syncsummoner.compose.score import DEFAULT_WET, GestureInstance, Layer, Score, control_rate, program_key
from syncsummoner.compose.vocabulary import GESTURES, Anchor, Automation, GestureContext

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
#: Wet range per style, as fractions of the wet control's travel. Neither end blacks the
#: output on any archived program, so the range is a look: ``glitchy`` reaches fully wet.
STYLE_WET: dict[str, tuple[float, float]] = {"glitchy": (DEFAULT_WET[0], 1.0)}


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
    """Hard constraint: aggregate parameter writes per second and per-parameter settle time both respected."""
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


def _settled(times: np.ndarray, values: np.ndarray, settle: float) -> tuple[list[float], list[int]]:
    """One parameter's writes no closer together than ``settle``, each the latest value asked for.

    A write that comes too soon is deferred rather than dropped: dropping the release
    of a pulse leaves the parameter latched wherever the pulse put it.
    """
    out_t: list[float] = []
    out_v: list[int] = []
    due, pending = -np.inf, None
    for t, v in zip(times.tolist(), values.tolist()):
        if pending is not None and due <= t:
            out_t.append(due)
            out_v.append(pending)
            due, pending = due + settle, None
        if t >= due:
            if not out_v or v != out_v[-1]:
                out_t.append(t)
                out_v.append(v)
                due = t + settle
            pending = None
        else:
            pending = v if v != out_v[-1] else None
    if pending is not None:
        out_t.append(due)
        out_v.append(pending)
    return out_t, out_v


def thin(
    auto: Automation, profile: ProgramProfile, *, fps: float, cc_budget_hz: float = CC_BUDGET_HZ
) -> Automation:
    """Defer and drop messages so the result satisfies the hard constraint.

    Writes a parameter cannot settle between are deferred to the latest value, so the
    held state matches what the gestures asked for; the write budget, in parameter writes per
    second, then drops the excess.
    """
    if len(auto) == 0:
        return auto
    parts = []
    for idx in np.unique(auto.indices):
        m = auto.indices == idx
        t, v = _settled(auto.times[m], auto.values[m], profile.settle_frames.get(int(idx), 0) / fps)
        parts.append(Automation(np.asarray(t), np.full(len(t), idx, np.int32), np.asarray(v, np.int32)))
    auto = Automation.concat(parts)
    window: deque[float] = deque()
    quota = max(1, int(cc_budget_hz))
    keep = np.zeros(len(auto), dtype=bool)
    for i, t in enumerate(auto.times.tolist()):
        while window and window[0] <= t - 1.0:
            window.popleft()
        if len(window) < quota:
            window.append(t)
            keep[i] = True
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
    slope_center: float = -1.2,
    slope_gain: float = 0.8,
) -> Trajectory:
    """Predict per-frame metrics from the measured curves, the cliff atlas and the response.

    Every metric the objective scores is read back from what the probe measured at
    each setpoint. Where a profile predates those curves the older inferences stand
    in, so an unrefitted profile still composes.
    """
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


def bar_edges(features: Features, duration: float) -> np.ndarray:
    """Bar boundaries in seconds over ``[0, duration]``: the downbeats, else the bar period."""
    audio = features.audio
    marks = audio.downbeats if audio.downbeats.size > 1 else np.arange(0.0, duration, audio.bar_duration)
    return np.unique(np.r_[0.0, marks[(marks > 0.0) & (marks < duration)], duration])


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
    edges = bar_edges(features, float(traj.times[-1]) + (traj.times[1] - traj.times[0]))
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
        z = aesthetics.follow_z(float(follow[m].mean()), aesthetics.effective_samples(series, series))
        dead[b] = bool(flat or frozen or (series.size > 3 and z <= aesthetics.critical_z()))
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
    wet: tuple[float, float] | None = None,
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
            "wet": list(wet or STYLE_WET.get(style, DEFAULT_WET)),
            "bar": audio.bar_duration,
            "phrase_bars": audio.phrase_bars,
        },
    )
    weights = STYLE_WEIGHTS.get(style, DEFAULT_WEIGHTS) if weights is None else weights
    programs = sorted(profiles)
    generations = max(1, budget // max(1, population * len(programs)))
    best: list[tuple[float, Layer, dict[str, float]]] = []
    LOG.info(
        "composing %.1fs over %d programs: %d %s of %d candidates, %d sections at %.1f bpm",
        duration,
        len(programs),
        generations,
        "generation" if generations == 1 else "generations",
        population,
        len(sections),
        audio.tempo,
    )

    for program in track(programs, desc="evolving", unit="program"):
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
                return Objective(total=-np.inf, feasible=False, violation=np.inf)
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
        ranked = sorted(((fitness(m), m) for m in pop), key=lambda p: p[0].rank)
        for _ in range(generations - 1):
            elite = [m for _, m in ranked[: max(1, population // 2)]]
            children = [
                _mutate(elite[i % len(elite)], profile, audio, sections, applicable, rng)
                for i in range(population - len(elite))
            ]
            pop = elite + children
            ranked = sorted(((fitness(m), m) for m in pop), key=lambda p: p[0].rank)
        obj, layer = ranked[0]
        LOG.debug("%s: score %.3f over %d gestures", program, obj.total, len(layer.gestures))
        if "dead_bars" not in obj.terms:
            LOG.warning("%s: no survival measured, so its picture is unverified; refit from footage", program)
        elif obj.violation:
            LOG.warning("%s: %d bars predicted dead outside the destruction budget", program, obj.violation)
        best.append((obj, layer))

    best.sort(key=lambda t: t[0].rank)
    base.layers = [replace(layer, index=i) for i, (_, layer) in enumerate(best[:n_passes])]
    base.meta["objective"] = [obj.terms for obj, _ in best[:n_passes]]
    for layer, (obj, _) in zip(base.layers, best):
        LOG.info("kept %s: score %.3f, %d gestures", layer.program, obj.total, len(layer.gestures))
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

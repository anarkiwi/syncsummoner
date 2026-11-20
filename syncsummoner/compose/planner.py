"""Measured effects mapped onto the music: each bar a safe measured setting, each cut a program.

The source takes the clip it is nearer in chroma (:func:`clip_kind`); each bar's drive blends the source's
motion and the track's onsets by rank (:func:`drive`) into a quantile of the safe intensities; effects change
on downbeats (:func:`changes`), each run of bars on the program whose settings meet it best (:func:`assign`).
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Callable, Mapping, Sequence

import numpy as np
from scipy.stats import rankdata

from syncsummoner.device.profile import ProgramProfile
from syncsummoner.progress import LOG, track
from syncsummoner.compose.cut import Cut, spans_of
from syncsummoner.compose.features import Features, Section, VideoFeatures
from syncsummoner.compose.gating import model_gate
from syncsummoner.compose.score import GestureInstance, Layer, Score, control_rate
from syncsummoner.compose.stimulus import chroma_sigma
from syncsummoner.compose.thinning import reachable, thin, visible
from syncsummoner.compose.vocabulary import SETTING, Automation

__all__ = [
    "Curve",
    "Palette",
    "assign",
    "bar_edges",
    "changes",
    "clip_kind",
    "drive",
    "palettes",
    "plan_automation",
    "ranks",
    "reachable",
    "search",
    "thin",
    "visible",
]


@dataclass(frozen=True)
class Curve:
    """How the chosen effects follow the source and the music.

    Attributes:
        follow: weight of the source's motion against the track's onsets in each bar's drive, 0 to 1.
        low: quantile of the safe measured intensities the least driven bar gets.
        high: quantile the most driven bar gets; ``(0, 1)`` spans the subtlest to the most exaggerated.
        bars: mean bars between effect changes; ``None`` for the track's phrase.
        quicken: -1 to 1, how far the drive speeds changes where it is high; 0 a steady grid.
        passes: most programs used, one capture each; ``None`` for every program with a safe setting.
        kind: ``bw`` or ``colour`` to force the clip; ``None`` for the one the source is nearer.
    """

    follow: float = 0.5
    low: float = 0.0
    high: float = 1.0
    bars: float | None = None
    quicken: float = 0.0
    passes: int | None = None
    kind: str | None = None

    def __post_init__(self) -> None:
        for name in ("follow", "low", "high"):
            if not 0.0 <= getattr(self, name) <= 1.0:
                raise ValueError(f"{name} must lie in [0, 1], got {getattr(self, name)}")
        if not -1.0 <= self.quicken <= 1.0:
            raise ValueError(f"quicken must lie in [-1, 1], got {self.quicken}")
        if self.bars is not None and self.bars <= 0:
            raise ValueError(f"bars must be positive, got {self.bars}")
        if self.passes is not None and self.passes < 1:
            raise ValueError(f"passes must be at least 1, got {self.passes}")


@dataclass(frozen=True, eq=False)
class Palette:
    """One program's safe settings on one clip, with each intensity's quantile among every program's."""

    program: str
    params: np.ndarray
    setpoints: np.ndarray
    intensity: np.ndarray
    quantile: np.ndarray

    def nearest(self, targets: np.ndarray) -> np.ndarray:
        """Index of the setting whose quantile is nearest each target, the first on a tie."""
        return np.argmin(np.abs(self.quantile[None, :] - np.asarray(targets)[:, None]), axis=1)

    def cost(self, targets: np.ndarray) -> float:
        """Total distance, in quantile, from each target to its nearest setting."""
        return float(np.abs(self.quantile[self.nearest(targets)] - targets).sum())


def ranks(values: Sequence[float] | np.ndarray) -> np.ndarray:
    """Each value's mid-rank as a fraction of ``[0, 1]``, ties sharing one; a lone value is 0.5."""
    x = np.asarray(values, dtype=np.float64)
    if x.size < 2:
        return np.full(x.size, 0.5)
    return (rankdata(x) - 1.0) / (x.size - 1.0)


def bar_edges(downbeats: np.ndarray, duration: float) -> np.ndarray:
    """Bar boundaries from zero to ``duration``: every downbeat after the first, so a pickup joins bar one."""
    inner = np.asarray(downbeats, dtype=np.float64)[1:]
    inner = inner[(inner > 0.0) & (inner < duration)]
    return np.unique(np.r_[0.0, inner, duration])


def bar_means(series: np.ndarray, times: np.ndarray, edges: np.ndarray) -> np.ndarray:
    """Mean of ``series`` sampled at ``times`` over each bar, the nearest sample where a bar holds none."""
    x = np.nan_to_num(np.asarray(series, dtype=np.float64))
    at = np.searchsorted(np.asarray(times, dtype=np.float64), edges)
    sums = np.diff(np.r_[0.0, np.cumsum(x)][at])
    counts = np.diff(at)
    near = x[np.clip(at[:-1], 0, x.size - 1)] if x.size else np.zeros(counts.size)
    return np.where(counts > 0, sums / np.maximum(counts, 1), near)


def drive(features: Features, edges: np.ndarray, follow: float) -> np.ndarray:
    """Each bar's drive in ``[0, 1]``: the rank of the ``follow``-weighted blend of the source motion's rank
    and the onset strength's, so the most driven bar is 1 whatever the blend.

    The source's motion is its P.910 temporal information; without it the music drives alone.
    """
    audio, video = features.audio, features.video
    music = ranks(bar_means(audio.onset_strength, audio.times, edges))
    if video is None or video.ti is None or follow == 0.0:
        return music
    motion = ranks(bar_means(video.ti, np.arange(video.ti.size) / video.fps, edges))
    return ranks(follow * motion + (1.0 - follow) * music)


def changes(level: np.ndarray, bars: float, quicken: float) -> np.ndarray:
    """The bars a new effect starts on, bar 0 always: each bar counts ``(1 + quicken (2 level - 1)) / bars``,
    and an effect starts where the count passes a whole number, so ``quicken`` 0 changes every ``bars``.
    """
    rate = 1.0 + quicken * (2.0 * np.asarray(level, dtype=np.float64) - 1.0)
    count = np.floor_divide(np.r_[0.0, np.cumsum(rate)][:-1], bars)
    return np.flatnonzero(np.r_[True, np.diff(count) > 0])


def _route(cost: np.ndarray) -> tuple[float, np.ndarray]:
    """The least total cost through ``cost[run, program]`` with no program twice running, and its path."""
    runs, programs = cost.shape
    if programs == 1:
        return float(cost.sum()), np.zeros(runs, np.int64)
    barred = np.where(np.eye(programs, dtype=bool), np.inf, 0.0)
    total, back = cost[0].copy(), []
    for row in cost[1:]:
        step = total[:, None] + barred
        back.append(np.argmin(step, axis=0))
        total = step.min(axis=0) + row
    path = [int(np.argmin(total))]
    for came in reversed(back):
        path.append(int(came[path[-1]]))
    return float(total.min()), np.asarray(path[::-1], np.int64)


def assign(cost: np.ndarray, passes: int | None = None) -> np.ndarray:
    """A program for each run of bars, from ``cost[run, program]``, never the same one twice running.

    With ``passes``, programs are kept one at a time, each the one that lowers the least total most.
    """
    cost = np.asarray(cost, dtype=np.float64)
    programs = cost.shape[1]
    if passes is None or passes >= programs:
        return _route(cost)[1]
    kept: list[int] = []
    for _ in range(passes):
        rest = [p for p in range(programs) if p not in kept]
        kept.append(min(rest, key=lambda p: _route(cost[:, kept + [p]])[0]))
    return np.asarray(kept)[_route(cost[:, kept])[1]]


def clip_kind(video: VideoFeatures | None, profiles: Mapping[str, ProgramProfile], kind: str | None = None):
    """The measured clip the source is nearer in RMS chroma, with the source's chroma and each clip's.

    A clip's chroma follows from its saturation (:func:`~syncsummoner.compose.stimulus.chroma_sigma`), the
    colour clip's set by the gamut, so the midpoint between the clips is the only threshold.
    """
    sigma: dict[str, list[float]] = {}
    for profile in profiles.values():
        for name, entry in profile.effects.items():
            sigma.setdefault(name, []).append(chroma_sigma(max(entry["saturation"])))
    clips = {name: float(np.median(found)) for name, found in sorted(sigma.items())}
    if not clips:
        raise ValueError("no profile carries effects; run `probe effects` on its archive")
    source = None
    if video is not None and video.chroma is not None and video.chroma.size:
        source = float(np.sqrt(np.mean(np.square(video.chroma, dtype=np.float64))))
    if kind is not None:
        if kind not in clips:
            raise ValueError(f"no profile measured effects on a {kind} clip; measured: {', '.join(clips)}")
        return kind, source, clips
    if source is None:
        if len(clips) > 1:
            raise ValueError("without the source's chroma, name the clip to compose for")
        return next(iter(clips)), source, clips
    return min(clips, key=lambda name: (abs(clips[name] - source), name)), source, clips


def palettes(profiles: Mapping[str, ProgramProfile], kind: str) -> list[Palette]:
    """Every program's safe settings measured on the ``kind`` clip, quantiles taken over them all."""
    found = []
    for name in sorted(profiles):
        settings = [s for s in profiles[name].effects.get(kind, {}).get("settings", ()) if s["safe"]]
        if not settings:
            LOG.info("%s: no safe setting measured on the %s clip", name, kind)
            continue
        found.append((name, settings))
    if not found:
        raise ValueError(f"no program has a safe setting measured on the {kind} clip")
    pooled = ranks([s["intensity"] for _, settings in found for s in settings])
    out, at = [], 0
    for name, settings in found:
        out.append(
            Palette(
                program=name,
                params=np.array([s["params"] for s in settings], np.int64),
                setpoints=np.array([s["setpoint"] for s in settings], np.int64),
                intensity=np.array([s["intensity"] for s in settings], np.float64),
                quantile=pooled[at : at + len(settings)],
            )
        )
        at += len(settings)
    return out


def _gestures(
    palette: Palette,
    picks: np.ndarray,
    starts: np.ndarray,
    opening: float,
    held: np.ndarray | None,
    fps: float,
) -> list[GestureInstance]:
    """A setting gesture per change of setting over a run's bars, writing the slots that change one a frame.

    Each one's last write lands as its bar starts; the run's first starts at ``opening``. ``held`` is the
    setting the program last held, ``None`` when unknown.
    """
    out = []
    for k, (pick, start) in enumerate(zip(picks.tolist(), starts.tolist())):
        values = palette.params[pick]
        slots = np.arange(values.size) if held is None else np.flatnonzero(values != held)
        held = values
        if not slots.size:
            continue
        out.append(
            GestureInstance(
                gesture=SETTING,
                arrival=float(opening if k == 0 else start - (slots.size - 1) / fps),
                duration=0.0,
                intensity=float(palette.quantile[pick]),
                targets=tuple(int(s) + 1 for s in slots),
                seed=int(palette.setpoints[pick]),
                values=tuple(int(v) for v in values[slots]),
            )
        )
    return out


def search(
    profiles: Mapping[str, ProgramProfile],
    features: Features,
    *,
    curve: Curve | None = None,
    fps: float = 60.0,
    seed: int = 0,
) -> Score:
    """A score of measured settings, one a bar, its program changing on the curve's downbeats.

    Each run of bars goes to a program by :func:`assign`, costed by each bar's distance to that program's
    nearest safe setting, which the bar then plays. A run's first setting is written from when its program
    last left the screen, unseen, so it has settled by the cut.
    """
    curve = curve or Curve()
    audio = features.audio
    if audio is None:
        raise ValueError("composing needs the track's beats")
    duration = features.duration or audio.duration
    kind, source, clips = clip_kind(features.video, profiles, curve.kind)
    LOG.info(
        "composing for the %s clip: source chroma %s, clips %s",
        kind,
        "unmeasured" if source is None else f"{source:.4f}",
        ", ".join(f"{name} {value:.4f}" for name, value in clips.items()),
    )
    found = palettes(profiles, kind)
    edges = bar_edges(audio.downbeats, duration)
    level = drive(features, edges, curve.follow)
    target = curve.low + (curve.high - curve.low) * level
    runs = np.split(
        np.arange(level.size), changes(level, float(curve.bars or audio.phrase_bars), curve.quicken)[1:]
    )
    chosen = assign(np.array([[p.cost(target[run]) for p in found] for run in runs]), curve.passes)
    layers: dict[str, list[GestureInstance]] = {}
    cuts, left, held, shown = [], {}, {}, np.zeros(level.size)
    for run, index in track(list(zip(runs, chosen.tolist())), desc="cuts", unit="cut"):
        palette = found[index]
        name, picks = palette.program, palette.nearest(target[run])
        shown[run] = palette.intensity[picks]
        written = _gestures(palette, picks, edges[run], left.get(name, 0.0), held.get(name), fps)
        layers.setdefault(name, []).extend(written)
        cuts.append(Cut(float(edges[run[0]]), float(edges[run[-1] + 1]), name))
        left[name], held[name] = cuts[-1].end, palette.params[picks[-1]]
    names = list(dict.fromkeys(cut.program for cut in cuts))
    LOG.info("%d cuts over %d bars on %s", len(cuts), level.size, ", ".join(names))
    return Score(
        seed=int(seed),
        bpm=audio.tempo,
        duration=duration,
        fps=fps,
        sections=[Section(c.start, c.end, str(i)) for i, c in enumerate(cuts)],
        layers=[Layer(program=name, index=i, gestures=layers[name]) for i, name in enumerate(names)],
        cuts=cuts,
        meta={
            "kind": kind,
            "source_chroma": source,
            "clip_chroma": clips,
            "curve": asdict(curve),
            "bar": audio.bar_duration,
            "phrase_bars": audio.phrase_bars,
            "downbeats": [float(t) for t in audio.downbeats[audio.downbeats < duration]],
            "bars": [float(t) for t in edges],
            "drive": [float(v) for v in level],
            "target": [float(v) for v in target],
            "intensity": [float(v) for v in shown],
        },
    )


def plan_automation(
    score: Score,
    profiles: Mapping[str, ProgramProfile],
    *,
    fps: float | None = None,
    gates: Mapping[str, Callable[[Automation], Automation] | None] | None = None,
) -> dict[int, Automation]:
    """Expand a score to reachable automation per layer; this is what the renderer plays.

    A measured setting is sent whole where the score places it. Other gestures' writes are sent only inside
    their program's cuts, through its gate (default: the program model's), which pairs or refuses dead writes.
    """
    fps = score.fps if fps is None else fps
    rate = control_rate(fps)
    out = {}
    for layer in score.layers:
        profile = profiles.get(layer.program)
        if profile is None:
            continue
        measured = Layer(layer.program, layer.index, [g for g in layer.gestures if g.gesture == SETTING])
        drawn = Layer(layer.program, layer.index, [g for g in layer.gestures if g.gesture != SETTING])
        auto = Automation.empty()
        if drawn.gestures:
            spans = spans_of(score.cuts, layer.program) if score.cuts else None
            auto = visible(thin(score.render_layer(drawn, profile, rate=rate), profile, fps=fps), spans)
            gate = model_gate(layer.program) if gates is None else gates.get(layer.program)
            auto = auto if gate is None else gate(auto)
        if measured.gestures:
            exact = thin(score.render_layer(measured, profile, rate=rate), profile, fps=fps)
            auto = Automation.concat([auto, exact])
        out[layer.index] = auto
    return out


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

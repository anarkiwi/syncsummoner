"""Bar by bar survival of a predicted trajectory, judged as the take check will judge its take."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from syncsummoner import aesthetics
from syncsummoner.compose import kriged
from syncsummoner.compose.features import Features
from syncsummoner.compose.score import Score


def bar_edges(features: Features, end: float, start: float = 0.0) -> np.ndarray:
    """Bar boundaries in seconds over ``[start, end]``: the downbeats, else the bar period."""
    audio = features.audio
    marks = audio.downbeats if audio.downbeats.size > 1 else np.arange(0.0, end, audio.bar_duration)
    return np.unique(np.r_[start, marks[(marks > start) & (marks < end)], end])


@dataclass(frozen=True)
class Kept:
    """Predicted survival per bar: how much picture and motion the output keeps, and which bars die.

    ``unmeasured`` holds the settings shown in dead bars that no record decides either way.
    """

    starts: np.ndarray
    ends: np.ndarray
    ti_ratio: np.ndarray
    dead: np.ndarray
    unmeasured: frozenset[tuple[int, ...]] = frozenset()


def live_bounds(si: float, ti: float, series: np.ndarray) -> dict[str, float]:
    """The least each survival response must keep over one bar of the source, in its own units.

    SI and TI must reach :func:`~syncsummoner.aesthetics.live_floor` of the source's, and where the
    source's TI varies beyond quantisation the output's following must reach
    :func:`~syncsummoner.aesthetics.critical_z` over the bar's effective samples, as the take check asks.
    """
    bounds = {
        "si_ratio": aesthetics.live_floor(si, aesthetics.quantisation_si()) / si if si > 0 else -np.inf,
        "ti_ratio": aesthetics.live_floor(ti, aesthetics.quantisation_ti()) / ti if ti > 0 else -np.inf,
    }
    if series.size > 3 and np.std(series) > aesthetics.quantisation_ti():
        n_eff = aesthetics.effective_samples(series, series)
        bounds["motion_follow"] = (
            float(np.tanh(aesthetics.critical_z() / np.sqrt(n_eff - 3.0))) if n_eff > 3.0 else np.inf
        )
    return bounds


def _short(traj: Any, name: str, m: np.ndarray, bound: float) -> tuple[bool, bool]:
    """Whether a bar's frames ``m`` fall short of ``bound`` in one response, and if no record decides that."""
    if not np.isfinite(bound):
        return bound > 0, False
    found = traj.joint.get(name)
    if found is None:
        series = {"si_ratio": traj.si_ratio, "ti_ratio": traj.ti_ratio, "motion_follow": traj.follow}[name]
        return float(series[m].mean()) <= bound, False
    margin = found.margin(m, kriged.boundary(name, bound))
    return margin <= 0, abs(margin) < aesthetics.critical_z(1) * float(found.sd[m].mean())


def predict_survival(traj: Any, features: Features) -> Kept | None:
    """Judge each bar of a predicted trajectory against the same bar of the source footage.

    A bar is dead where a survival response falls short of its :func:`live_bounds`; a kriged
    response is judged with the bound as its prior mean, so a setting no record informs is not alive.
    """
    video = features.video
    if traj.si_ratio is None or traj.ti_ratio is None or video is None or video.si is None:
        return None
    edges = bar_edges(features, float(traj.times[-1]) + (traj.times[1] - traj.times[0]), float(traj.times[0]))
    frame = np.clip(np.rint(traj.times * video.fps).astype(np.int64), 0, video.si.size - 1)
    source_si, source_ti = video.si[frame], np.nan_to_num(video.ti[frame])
    owner = np.clip(np.searchsorted(edges, traj.times, side="right") - 1, 0, edges.size - 2)
    ratio, dead = np.zeros(edges.size - 1), np.zeros(edges.size - 1, dtype=bool)
    unmeasured: set[tuple[int, ...]] = set()
    for b in range(edges.size - 1):
        m = owner == b
        if not m.any():
            continue
        lo, hi = np.clip(np.rint(edges[b : b + 2] * video.fps).astype(np.int64), 0, video.ti.size)
        bounds = live_bounds(
            float(source_si[m].mean()), float(source_ti[m].mean()), np.nan_to_num(video.ti[lo:hi])
        )
        ratio[b] = float(traj.ti_ratio[m].mean())
        if traj.follow is None:
            bounds.pop("motion_follow", None)
        short = [_short(traj, name, m, bound) for name, bound in bounds.items()]
        dead[b] = any(fell for fell, _ in short)
        if dead[b] and traj.settings is not None and all(open_ for fell, open_ in short if fell):
            unmeasured.update(map(tuple, np.unique(traj.settings[m].astype(np.int64), axis=0).tolist()))
    return Kept(edges[:-1], edges[1:], ratio, dead, frozenset(unmeasured))


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

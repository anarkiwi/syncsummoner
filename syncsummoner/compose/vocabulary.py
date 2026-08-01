"""Gesture primitives: the composer's vocabulary, written against canonical axes.

Each primitive carries a musical anchor and consumes one measured profile field,
so new programs are absorbed by re-probing. Gestures end on their anchor, putting
the visual arrival on the beat; latency bias is applied in :mod:`.render`.
"""

from __future__ import annotations

import enum
from dataclasses import dataclass
from typing import Any, Callable, Iterable, Sequence

import numpy as np

from syncsummoner.device.profile import (
    PARAM_COUNT,
    PARAM_MAX,
    Axis,
    ParamKind,
    ParamSpec,
    ProgramProfile,
    Tongue,
)

CONTINUOUS_KINDS = (ParamKind.CONTINUOUS, ParamKind.QUANTIZED)


class Anchor(enum.Enum):
    """Musical grid position a gesture is scheduled against."""

    BAR = "bar"
    PHRASE = "phrase"
    DOWNBEAT = "downbeat"
    SECTION = "section"
    ONSET = "onset"
    MULTI_BAR = "multi_bar"
    TWO_BAR = "two_bar"


@dataclass(frozen=True, eq=False)
class Automation:
    """Timed parameter automation: parallel arrays of time, param index and raw value."""

    times: np.ndarray
    indices: np.ndarray
    values: np.ndarray

    @classmethod
    def empty(cls) -> "Automation":
        """Automation with no events."""
        return cls(np.empty(0), np.empty(0, np.int32), np.empty(0, np.int32))

    @classmethod
    def of(cls, times: Sequence[float] | np.ndarray, index: int, values) -> "Automation":
        """Automation for a single parameter index over ``times``."""
        t = np.atleast_1d(np.asarray(times, dtype=np.float64))
        v = np.clip(np.round(np.atleast_1d(np.asarray(values, dtype=np.float64))), 0, PARAM_MAX)
        return cls(t, np.full(t.size, int(index), np.int32), np.broadcast_to(v, t.shape).astype(np.int32))

    @classmethod
    def concat(cls, parts: Iterable["Automation"]) -> "Automation":
        """Merge automations into one, stably sorted by time."""
        parts = [p for p in parts if len(p) > 0]
        if not parts:
            return cls.empty()
        times = np.concatenate([p.times for p in parts])
        order = np.argsort(times, kind="stable")
        return cls(
            times[order],
            np.concatenate([p.indices for p in parts])[order],
            np.concatenate([p.values for p in parts])[order],
        )

    def __len__(self) -> int:
        return int(self.times.size)

    def shift(self, dt: float) -> "Automation":
        """Move every event by ``dt`` seconds."""
        return Automation(self.times + dt, self.indices, self.values)

    def within(self, t0: float, t1: float) -> "Automation":
        """Events in the half-open time window."""
        m = (self.times >= t0) & (self.times < t1)
        return Automation(self.times[m], self.indices[m], self.values[m])

    def max_rate(self, window_s: float = 1.0) -> float:
        """Peak message rate over any sliding window, in messages per second."""
        if len(self) == 0:
            return 0.0
        counts = np.searchsorted(self.times, self.times + window_s, side="left") - np.arange(len(self))
        return float(counts.max() / window_s)

    def hold(self, times: np.ndarray, n_params: int = PARAM_COUNT) -> np.ndarray:
        """Zero-order-hold onto a time grid as ``(len(times), n_params)`` raw values."""
        out = np.full((times.size, n_params), PARAM_MAX // 2, dtype=np.float64)
        for idx in np.unique(self.indices):
            m = self.indices == idx
            vals = self.values[m]
            pos = np.searchsorted(self.times[m], times, side="right") - 1
            out[:, int(idx) - 1] = np.where(pos >= 0, vals[np.clip(pos, 0, vals.size - 1)], PARAM_MAX // 2)
        return out


@dataclass(frozen=True, eq=False)
class GestureContext:
    """Where and how strongly to place one gesture instance."""

    arrival: float
    duration: float
    rate: float
    intensity: float
    axis: Axis
    rng: np.random.Generator
    targets: tuple[int, ...] = ()


Renderer = Callable[[ProgramProfile, GestureContext], Automation]


@dataclass(frozen=True)
class Gesture:
    """One vocabulary primitive: an anchor, the profile field it consumes, and its renderer."""

    name: str
    anchor: Anchor
    profile_field: str
    fn: Renderer

    def __call__(self, profile: ProgramProfile, ctx: GestureContext) -> Automation:
        return self.fn(profile, ctx)


GESTURES: dict[str, Gesture] = {}


def _register(name: str, anchor: Anchor, profile_field: str) -> Callable[[Renderer], Renderer]:
    def deco(fn: Renderer) -> Renderer:
        GESTURES[name] = Gesture(name, anchor, profile_field, fn)
        return fn

    return deco


def safe_range(spec: ParamSpec) -> tuple[int, int]:
    """Widest raw-value interval clear of the measured dead zone."""
    lo, hi = 0, PARAM_MAX
    if spec.dead_zone and spec.kind is not ParamKind.BOOLEAN:
        d0, d1 = spec.dead_zone
        if d0 - lo >= hi - d1:
            hi = int(d0)
        else:
            lo = int(d1)
    return lo, max(lo + 1, hi)


def monotonic_segment(spec: ParamSpec) -> tuple[int, int]:
    """Longest constant-sign run of the measured response curve, as raw values."""
    v = np.asarray(spec.values, dtype=np.float64)
    r = np.asarray(spec.response, dtype=np.float64)
    if v.size < 3 or r.size != v.size:
        return safe_range(spec)
    s = np.sign(np.diff(r))
    edges = np.concatenate(([0], np.flatnonzero(np.diff(s)) + 1, [s.size]))
    k = int(np.argmax(np.diff(edges)))
    return int(v[edges[k]]), int(v[edges[k + 1]])


def step_margin(spec: ParamSpec) -> int:
    """Half the measured sampling step: the smallest delta the probe resolved."""
    n = max(2, len(spec.values))
    return max(1, int(round(PARAM_MAX / (2 * (n - 1)))))


def targets(profile: ProgramProfile, axis: Axis, *, kinds=CONTINUOUS_KINDS, n: int = 1) -> list[ParamSpec]:
    """Most sensitive parameters on a canonical axis, falling back to any usable parameter."""
    pool = [p for p in profile.by_axis(axis) if p.kind in kinds]
    if not pool:
        pool = [p for p in profile.params if p.kind in kinds]
    return sorted(pool, key=lambda p: -p.sensitivity)[:n]


def _grid(ctx: GestureContext, span: float | None = None) -> np.ndarray:
    """Control-rate time grid ending on the anchor."""
    span = ctx.duration if span is None else span
    n = max(2, int(round(span * ctx.rate)))
    return np.linspace(ctx.arrival - span, ctx.arrival, n)


def _pick(profile: ProgramProfile, ctx: GestureContext, kinds=CONTINUOUS_KINDS) -> ParamSpec | None:
    if ctx.targets:
        chosen = [p for p in profile.params if p.index in ctx.targets and p.kind in kinds]
        if chosen:
            return chosen[0]
    found = targets(profile, ctx.axis, kinds=kinds)
    return found[0] if found else None


@_register("hold", Anchor.BAR, "safe_range")
def _hold(profile: ProgramProfile, ctx: GestureContext) -> Automation:
    spec = _pick(profile, ctx)
    if spec is None:
        return Automation.empty()
    lo, hi = safe_range(spec)
    return Automation.of([ctx.arrival], spec.index, lo + ctx.intensity * (hi - lo))


@_register("ramp", Anchor.PHRASE, "monotonic_segments")
def _ramp(profile: ProgramProfile, ctx: GestureContext) -> Automation:
    spec = _pick(profile, ctx)
    if spec is None:
        return Automation.empty()
    lo, hi = monotonic_segment(spec)
    t = _grid(ctx)
    return Automation.of(t, spec.index, np.linspace(lo + (1.0 - ctx.intensity) * (hi - lo), hi, t.size))


@_register("cliff_cross", Anchor.DOWNBEAT, "cliff_atlas")
def _cliff_cross(profile: ProgramProfile, ctx: GestureContext) -> Automation:
    atlas = profile.cliff_atlas()
    if not atlas:
        return Automation.empty()
    index, cliff = atlas[min(int(ctx.intensity * len(atlas)), len(atlas) - 1)]
    spec = next(p for p in profile.params if p.index == index)
    margin = step_margin(spec)
    approach = _grid(ctx)[:-1]
    below = np.linspace(cliff.at - 4 * margin, cliff.at - margin, approach.size)
    return Automation.concat(
        [
            Automation.of(approach, index, below),
            Automation.of([ctx.arrival], index, cliff.at + margin),
        ]
    )


def _best_tongue(profile: ProgramProfile) -> tuple[Any, Tongue] | None:
    """Strongest measured lock region across every parameter pair."""
    pairs = [(lock, tongue) for lock in profile.lock_maps for tongue in lock.tongues]
    return max(pairs, key=lambda pair: pair[1].strength) if pairs else None


@_register("detune_drift", Anchor.MULTI_BAR, "lock_map")
def _detune_drift(profile: ProgramProfile, ctx: GestureContext) -> Automation:
    best = _best_tongue(profile)
    if best is None:
        return Automation.empty()
    lock, tongue = best
    t = _grid(ctx)
    edge = tongue.center + tongue.width / 2.0
    return Automation.of(t, lock.b, np.linspace(edge, edge + ctx.intensity * tongue.width, t.size))


@_register("lock_snap", Anchor.SECTION, "tongue_centers")
def _lock_snap(profile: ProgramProfile, ctx: GestureContext) -> Automation:
    best = _best_tongue(profile)
    if best is None:
        return Automation.empty()
    lock, tongue = best
    return Automation.of([ctx.arrival], lock.b, tongue.center)


@_register("punch", Anchor.ONSET, "effect_buttons")
def _punch(profile: ProgramProfile, ctx: GestureContext) -> Automation:
    spec = _pick(profile, ctx, kinds=(ParamKind.BOOLEAN,))
    if spec is None:
        return Automation.empty()
    dwell = max(ctx.duration, 2.0 / ctx.rate)
    return Automation.concat(
        [
            Automation.of([ctx.arrival], spec.index, PARAM_MAX),
            Automation.of([ctx.arrival + dwell], spec.index, 0),
        ]
    )


@_register("morph", Anchor.PHRASE, "crossfader")
def _morph(profile: ProgramProfile, ctx: GestureContext) -> Automation:
    fader = next((p for p in profile.params if p.index == PARAM_COUNT), None)
    if fader is None:
        return Automation.empty()
    t = _grid(ctx)
    parts = [Automation.of(t, fader.index, np.linspace(0, PARAM_MAX, t.size))]
    for spec in targets(profile, ctx.axis, n=2):
        lo, hi = safe_range(spec)
        parts.append(Automation.of([t[0]], spec.index, ctx.rng.integers(lo, hi)))
    return Automation.concat(parts)


@_register("hysteresis_loop", Anchor.TWO_BAR, "hysteresis")
def _hysteresis_loop(profile: ProgramProfile, ctx: GestureContext) -> Automation:
    pool = [p for p in profile.params if p.hysteresis and p.kind in CONTINUOUS_KINDS]
    spec = pool[0] if pool else _pick(profile, ctx)
    if spec is None:
        return Automation.empty()
    lo, hi = safe_range(spec)
    t = _grid(ctx)
    up = np.linspace(lo, hi, max(2, t.size // 2))
    loop = np.r_[up, up[::-1]]
    return Automation.of(t, spec.index, np.interp(t, np.linspace(t[0], t[-1], loop.size), loop))

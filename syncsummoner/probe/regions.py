"""An archive sweep over regions: settings spread jointly through each gating regime, held to judge motion.

Within each chosen regime the stepped parameters are sampled together by a scrambled Sobol design over
the settings their read bits tell apart; each setting is held as long as the follow test needs on the
footage it plays over (:func:`holds`), and a time budget then sets how many there are.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Mapping, Sequence

import numpy as np
from scipy.stats import qmc

from syncsummoner.aesthetics.survival import SIGMA, critical_z, effective_samples
from syncsummoner.compose.locktake import PLAN
from syncsummoner.device.profile import PARAM_COUNT, snap_toggles
from syncsummoner.probe.locked import Chunk, SweepError

__all__ = [
    "Slot",
    "allot",
    "chunks_of",
    "design",
    "distinct",
    "region_dims",
    "second_order",
    "without_bypass",
    "effect_sizes",
    "guard_frames",
    "hold_at",
    "holds",
    "slots",
]


def effect_sizes(ti: np.ndarray, edges: np.ndarray) -> np.ndarray:
    """Per frame, the least follow the planner's dead-bar test can tell from none over the bar it is in.

    Over a bar of ``n`` effective samples of the footage's TI, a follow is called once its Fisher z
    passes :func:`critical_z`, so the least is ``tanh(z / sqrt(n - 3))``; NaN where a bar is too short.
    """
    ti = np.asarray(ti, dtype=np.float64)
    edges = np.unique(np.clip(np.r_[0, np.asarray(edges, dtype=np.int64), ti.size], 0, ti.size))
    out = np.full(ti.size, np.nan)
    for a, b in zip(edges[:-1], edges[1:]):
        n = effective_samples(ti[a:b], ti[a:b]) if b - a > 2 else 0.0
        if n > 3.0:
            out[a:b] = math.tanh(critical_z() / math.sqrt(n - 3.0))
    return out


def _lag1_growing(y: np.ndarray) -> np.ndarray:
    """Lag-one autocorrelation of ``y[:h]`` for every ``h`` from 1, as :mod:`survival` measures it."""
    n = np.arange(1, y.size + 1, dtype=np.float64)
    s, q = np.cumsum(y), np.cumsum(y * y)
    c = np.r_[0.0, np.cumsum(y[:-1] * y[1:])]
    mean = s / n
    den = q - n * mean**2
    num = c - mean * ((s - y) + (s - y[0])) + (n - 1) * mean**2
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.where((den > 0) & (n > 2), num / den, 0.0)


def hold_at(ti: np.ndarray, effect: float, start: int, sigma: float = SIGMA) -> int:
    """Frames from ``start`` the follow test needs to see a follow of ``effect`` at the evidence's power.

    That is ``atanh(effect) * sqrt(n_eff - 3)`` reaching ``critical_z() + sigma``, a miss as rare as a
    false call, ``n_eff`` the footage TI's own effective samples; 0 if the clip ends first.
    """
    y = np.asarray(ti[start:], dtype=np.float64)
    if not y.size or not np.isfinite(effect) or effect <= 0:
        return 0
    a = _lag1_growing(y)
    n_eff = np.arange(1, y.size + 1) * (1.0 - a * a) / (1.0 + a * a)
    z = math.atanh(min(effect, math.tanh(critical_z() + sigma))) * np.sqrt(np.maximum(n_eff - 3.0, 0.0))
    enough = np.flatnonzero(z >= critical_z() + sigma)
    return int(enough[0]) + 1 if enough.size else 0


def holds(ti: np.ndarray, edges: np.ndarray, starts: Sequence[int] | None = None) -> np.ndarray:
    """:func:`hold_at` for each start, ``effect`` the bar's (:func:`effect_sizes`); every frame by default."""
    effect = effect_sizes(ti, edges)
    starts = np.arange(np.size(ti)) if starts is None else np.asarray(starts, dtype=np.int64)
    return np.array([hold_at(ti, float(effect[t]), int(t)) for t in starts], dtype=np.int64)


def guard_frames(settle: int, band_ns: float, period_ns: float) -> int:
    """Frames after a setting's writes before its picture is measured: its landing window, then ``settle``.

    A write lands on one of the ticks its latency band spans, one more for a tick inside its echo
    bracket and one more for an echo taking up to a frame.
    """
    return int(settle) + int(math.ceil(band_ns / period_ns)) + 2


@dataclass(frozen=True)
class Slot:
    """One setting's place in a take: its first frame, the guard before it is measured, and its hold."""

    take: int
    start: int
    guard: int
    hold: int

    @property
    def frames(self) -> int:
        """Frames the setting occupies."""
        return self.guard + self.hold


def slots(
    ti: np.ndarray,
    edges: np.ndarray,
    *,
    guard: int,
    budget_s: float = math.inf,
    overhead_s: float,
    fps: float,
    limit: int | None = None,
) -> list[Slot]:
    """Settings placed one after another on the clip, a new take each time it runs out, within ``budget_s``
    and at most ``limit`` of them; one of the two must bound it.

    The takes play back to back in one session, which costs ``overhead_s`` beside their frames; each
    setting is held for what the footage after its ``guard`` needs (:func:`hold_at`).
    """
    if limit is None and not math.isfinite(budget_s):
        raise ValueError("slots needs a budget or a limit")
    effect = _filled(effect_sizes(ti, edges))
    out: list[Slot] = []
    take, at, spent = 0, 0, overhead_s
    while True:
        need = hold_at(ti, float(effect[at + guard]), at + guard) if at + guard < np.size(ti) else 0
        if not need:
            if at == 0:
                raise SweepError("no setting's follow test reaches its power anywhere on this clip")
            take, at = take + 1, 0
            continue
        if len(out) == limit or spent + (guard + need) / fps > budget_s:
            return out
        out.append(Slot(take=take, start=at, guard=guard, hold=need))
        at, spent = at + guard + need, spent + (guard + need) / fps


def _filled(x: np.ndarray) -> np.ndarray:
    """``x`` with each NaN taken from the nearest finite value before it, else after it."""
    ok = np.isfinite(x)
    if not ok.any():
        return x
    at = np.maximum.accumulate(np.where(ok, np.arange(x.size), -1))
    return x[np.where(at < 0, np.flatnonzero(ok)[0], at)]


def distinct(params: Sequence[int], values: Mapping[int, np.ndarray]) -> int:
    """The largest power of two no more than the distinct settings ``params`` have together.

    A design asking for more repeats settings, and a repeat that follows itself writes nothing.
    """
    total = math.prod(len(values[p]) for p in params) if params else 1
    return int(2 ** math.floor(math.log2(total)))


def second_order(dims: int) -> int:
    """Settings a regime sampling ``dims`` parameters needs: the least power of two at which its design can
    resolve a second-order model, an intercept, each parameter's effect and each pair's interaction."""
    return 1 << math.ceil(math.log2(1 + dims + dims * (dims - 1) // 2))


def allot(total: int, dims: Sequence[int]) -> list[int]:
    """Settings per regime: shares of ``total`` by the dimensions each samples, each a power of two.

    A Sobol design is balanced in every dimension only at a power of two points.
    """
    weight = np.asarray(dims, dtype=np.float64)
    share = total * weight / weight.sum() if weight.sum() else np.zeros_like(weight)
    return [int(2 ** math.floor(math.log2(s))) if s >= 1 else 0 for s in share]


def region_dims(record: Mapping, values: Mapping[int, np.ndarray]) -> list[list[int]]:
    """Per chosen regime of a cover record, the parameters a design samples: those stepped there that
    the program reads at more than one of ``values`` (from :func:`without_bypass`, so never its bypass).
    """
    return [
        [int(p) for p in record["stepped"].get(str(i), []) if len(values.get(p, ())) > 1]
        for i in record["chosen"]
    ]


def without_bypass(values: Mapping[int, np.ndarray], bypass: Any) -> dict[int, np.ndarray]:
    """``values`` less the bypass's proven settings, which leave every other control dead.

    ``bypass`` is a :class:`~syncsummoner.program.model.Bypass` or None.
    """
    out = dict(values)
    if bypass is not None and bypass.parameter_id in out:
        mine = np.asarray(out[bypass.parameter_id])
        out[bypass.parameter_id] = mine[~bypass.holds(mine)]
    return out


def design(
    base: np.ndarray, params: Sequence[int], values: Mapping[int, np.ndarray], count: int, seed: int
) -> np.ndarray:
    """``count`` states sampling ``params`` jointly from ``base`` by a scrambled Sobol design.

    Each coordinate picks among ``values[p]``, the settings the program tells apart, by the stratum it
    falls in, so each of a few settings comes up equally often; other parameters keep ``base``.
    """
    params = [int(p) for p in params if p in values and len(values[p])]
    out = np.repeat(np.asarray(base, dtype=np.int64)[None, :], count, axis=0)
    if not params or not count:
        return out
    u = qmc.Sobol(d=len(params), scramble=True, seed=seed).random_base2(int(math.log2(count)))
    for j, p in enumerate(params):
        reps = np.asarray(values[p], dtype=np.int64)
        out[:, p - 1] = reps[np.minimum((u[:, j] * reps.size).astype(np.int64), reps.size - 1)]
    return out


def chunks_of(
    states: np.ndarray, placed: Sequence[Slot], *, settle: int, toggles: Sequence[int]
) -> list[Chunk]:
    """The takes playing each state on its slot, every slot a setting moves written at its start."""
    states = snap_toggles(np.arange(1, PARAM_COUNT + 1)[None, :], np.asarray(states, dtype=np.int64), toggles)
    if len(states) != len(placed):
        raise SweepError(f"{len(states)} settings for {len(placed)} slots")
    out, k = [], 0
    for take in sorted({s.take for s in placed}):
        mine = [i for i, s in enumerate(placed) if s.take == take]
        rows = [(0, p + 1, int(v)) for p, v in enumerate(states[mine[0]])]
        owner = [0] * PARAM_COUNT
        for j, i in enumerate(mine[1:], start=1):
            moved = np.flatnonzero(states[i] != states[i - 1]) + 1
            rows += [(placed[i].start, int(p), int(states[i][p - 1])) for p in moved]
            owner += [j] * moved.size
        end = placed[mine[-1]]
        out.append(
            Chunk(
                first=k,
                states=states[mine],
                plan=np.array(rows, dtype=PLAN),
                owner=np.asarray(owner, dtype=np.int64),
                frames=end.start + end.frames,
                settle=int(settle),
                hold=max(placed[i].hold for i in mine),
            )
        )
        k += len(mine)
    return out

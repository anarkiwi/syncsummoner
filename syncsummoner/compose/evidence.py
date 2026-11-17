"""Frame-indexed evidence that every parameter of a take was sent, applied and acted upon.

The device latches CCs on a frame tick; capture timestamps say when each clip frame reached the
card. A CC lands on the first tick after the device took it, as its echo bounds, so sends,
readbacks and pictures are joined on one clip-frame index (docs/evidence.md).
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass, field
from typing import Any, Mapping, Sequence

import numpy as np

from syncsummoner.compose.matched import Regime, critical, fisher_z, frame_change, matched_change, tail_p
from syncsummoner.device.wire import Received
from syncsummoner.program.predict import DEAD, LIVE, NOOP, UNCHANGED

#: Target of a calibration write, which no clip frame schedules.
CALIBRATION = int(np.iinfo(np.int64).min)
#: Target of a write setting the lead-in's starting state, which no clip frame schedules either.
START = CALIBRATION + 1
#: Lags, in frames after the modelled landing, at which a step's picture is tested.
LAGS = (0, 1)
#: Luma range, in 8-bit codes, at or below which a frame is a uniform field (docs/hardware.md).
FLAT_CODES = 1


class EvidenceError(RuntimeError):
    """The take's records cannot be joined on clip frames, or contradict the landing model."""


@dataclass(frozen=True)
class FrameTimes:
    """When each played frame started on the capture clock: ``origin + j * period`` for clip frame ``j``.

    The take's frame line (:mod:`~syncsummoner.compose.frameline`), so it carries no capture phase.
    """

    lead: int
    origin: float
    period: float

    def at(self, frame: int | np.ndarray) -> np.ndarray:
        """When ``frame`` started."""
        return self.origin + np.asarray(frame, dtype=np.float64) * self.period

    def frame_of(self, t_ns: float | np.ndarray) -> np.ndarray:
        """The frame whose start lies nearest ``t_ns``."""
        return np.rint((np.asarray(t_ns, dtype=np.float64) - self.origin) / self.period).astype(np.int64)


def frame_times(found: Any) -> FrameTimes:
    """Frame start times from an attribution: its frame line, clip frame 0 at ``frame0_ns``."""
    return FrameTimes(lead=int(found.lead), origin=float(found.frame0_ns), period=float(found.period_ns))


@dataclass(frozen=True)
class Ticks:
    """The device's frame tick: one lies in ``(lo, hi] + n * period`` for every integer ``n``."""

    lo: float
    hi: float
    period: float
    count: int

    @property
    def phase(self) -> float:
        """Midpoint of the bracket."""
        return 0.5 * (self.lo + self.hi)

    @property
    def width(self) -> float:
        """How far the bracket leaves a tick unknown, in ns."""
        return self.hi - self.lo

    def after(self, t_ns: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Index of the first tick at or after each time, and how far that tick's bracket clears it.

        A negative clearance means the tick may fall before the time, so the landing is ambiguous.
        """
        t = np.asarray(t_ns, dtype=np.float64)
        n = np.ceil((t - self.hi) / self.period).astype(np.int64)
        return n, self.lo + n * self.period - t

    def at(self, n: np.ndarray) -> np.ndarray:
        """Midpoint time of tick ``n``."""
        return self.phase + np.asarray(n, dtype=np.float64) * self.period


def frame_of_tick(n: np.ndarray, ticks: Ticks, times: FrameTimes, delta: float) -> np.ndarray:
    """The clip frame tick ``n`` latches: the one whose first capture lands ``delta`` after it."""
    return times.frame_of(ticks.at(n) + delta)


def send_time(frames: np.ndarray, ticks: Ticks, times: FrameTimes, delta: float) -> np.ndarray:
    """When to send a CC so it lands on each frame: midway between the ticks either side of its own.

    The clock's prediction can err either way and a send can wake late, so half a frame each side.
    """
    tick = np.rint((times.at(frames) - delta - ticks.phase) / ticks.period)
    return ticks.phase + (tick - 0.5) * ticks.period


def toggle_landings(
    images: np.ndarray,
    t_ns: np.ndarray,
    starts: np.ndarray,
    shown: np.ndarray,
    toggles: np.ndarray,
    until: float | None = None,
) -> np.ndarray:
    """The clip frame each calibration write first shows in; :data:`CALIBRATION` where the picture can't say.

    Each write's window runs to the next; a state's picture is the mean of its windows' second
    halves, every capture goes to the nearer state, and a window must switch once, on a frame start.
    The last window ends at ``until``, the next write of any kind.
    """
    t = np.asarray(t_ns, dtype=np.float64)
    flat = np.asarray(images, dtype=np.float32).reshape(len(images), -1)
    sent = toggles["t1_ns"].astype(np.float64)
    ends = np.r_[toggles["t0_ns"][1:].astype(np.float64), t[-1] + 1 if until is None else until]
    windows = [np.flatnonzero((t > a) & (t < b)) for a, b in zip(sent, ends)]
    values = np.unique(toggles["value"])
    if values.size != 2:
        return np.full(len(toggles), CALIBRATION, dtype=np.int64)
    centre = {}
    for value in values:
        tails = [w[w.size // 2 :] for w, v in zip(windows, toggles["value"]) if v == value and w.size]
        centre[value] = flat[np.concatenate(tails)].mean(axis=0) if tails else None
    if any(c is None for c in centre.values()):
        return np.full(len(toggles), CALIBRATION, dtype=np.int64)
    near = np.abs(flat - centre[values[1]]).mean(axis=1) < np.abs(flat - centre[values[0]]).mean(axis=1)
    out = np.full(len(toggles), CALIBRATION, dtype=np.int64)
    for m, (w, value) in enumerate(zip(windows, toggles["value"])):
        state = near[w] == (value == values[1])
        if not w.size or not state[-1]:
            continue
        k = w[int(np.flatnonzero(~state)[-1]) + 1] if (~state).any() else w[0]
        if (~state).any() and np.flatnonzero(~state)[-1] != np.flatnonzero(~state).size - 1:
            continue
        if starts[k]:
            out[m] = shown[k]
    return out


#: What a latency is measured to: the start of the frame a write lands on, on the frame line.
BASIS = "start"


@dataclass(frozen=True)
class Latency:
    """From a write leaving the host to the start of the frame it lands on, per take.

    Each calibration write that showed in the picture bounds it: after the start of the frame
    before the one it landed on, at or before its own. ``lo`` and ``hi`` are those bounds
    intersected; where writes disagree by a little, they are the band they disagree over.
    ``delta`` is the band's middle, in ns. Frame starts lie ``line_lo..line_hi`` off the take's line.
    """

    lo: float
    hi: float
    samples: int
    period: float
    line_lo: float = 0.0
    line_hi: float = 0.0

    @property
    def bounds(self) -> tuple[float, float]:
        """The band the device's own latency lies in, whichever line the take's captures admit."""
        return self.lo + self.line_lo, self.hi + self.line_hi

    @property
    def delta(self) -> float:
        """The middle of the band."""
        return 0.5 * (self.lo + self.hi)

    @property
    def spread(self) -> float:
        """Half the band's width."""
        return 0.5 * abs(self.hi - self.lo)

    @property
    def frames(self) -> float:
        """Send to capture, in frames."""
        return self.delta / self.period

    def as_dict(self) -> dict:
        """JSON-ready."""
        return {
            "lo": self.lo,
            "hi": self.hi,
            "samples": self.samples,
            "period": self.period,
            "delta": self.delta,
            "spread": self.spread,
            "frames": self.frames,
            "line_lo": self.line_lo,
            "line_hi": self.line_hi,
            "basis": BASIS,
        }


def latency(sent: np.ndarray, shown_at: np.ndarray, times: FrameTimes) -> Latency:
    """The send-to-capture latency from where each calibration write first showed in the picture.

    A write taken at ``t`` and first shown in frame ``F`` puts it in ``(T(F-1) - t, T(F) - t]``; with
    ``(n, 2)`` brackets on ``t`` (``wire.echoed``) the later end bounds it below and the earlier above.
    Bounds that cross by more than half a frame mean no single latency holds, which is an error.
    """
    t = np.asarray(sent, dtype=np.float64)
    first, last = (t, t) if t.ndim == 1 else (t[:, 0], t[:, 1])
    ok = (shown_at >= -times.lead) & np.isfinite(last)
    if not ok.any():
        raise EvidenceError(
            "no calibration write showed in the picture, so the CC-to-picture latency is unknown"
        )
    lo = float((times.at(shown_at[ok] - 1) - last[ok]).max())
    hi = float((times.at(shown_at[ok]) - first[ok]).min())
    if lo - hi >= times.period / 2:
        raise EvidenceError(
            f"calibration writes disagree on the CC-to-picture latency by {(lo - hi) / 1e6:.2f} ms, "
            "half a frame or more, so no single latency holds"
        )
    return Latency(lo=min(lo, hi), hi=max(lo, hi), samples=int(ok.sum()), period=times.period)


def grid(times: FrameTimes, lat: Latency) -> Ticks:
    """The instants a write must precede to land on each frame: every frame's capture less the latency.

    The bracket is the latency band; tick ``n`` is frame ``n``'s, on the frame line.
    """
    return Ticks(lo=times.origin - lat.hi, hi=times.origin - lat.lo, period=times.period, count=lat.samples)


@dataclass(frozen=True)
class Landed:
    """Intended against realized: each scheduled write matched to its send and the frame it took effect on."""

    rows: np.ndarray
    missing: int
    duplicated: int
    out_of_order: int
    late: int
    early: int
    ambiguous: int
    wake_ms: float = float("nan")
    unechoed: int = 0
    slack_ms: float = float("nan")

    @property
    def ok(self) -> bool:
        """Every write sent once, in order, taken by the device, and its landing window holding its target.

        A window wider than one frame (``ambiguous``) is counted, not failed: a setting's measured
        frames start after the window's end.
        """
        return not (
            self.missing or self.duplicated or self.out_of_order or self.late or self.early or self.unechoed
        )

    def summary(self) -> str:
        """One line of counts."""
        return (
            f"{len(self.rows)} writes scheduled: {self.missing} missing, {self.unechoed} never echoed, "
            f"{self.duplicated} duplicated, {self.out_of_order} out of order, {self.late} late, "
            f"{self.early} early, {self.ambiguous} over more than one frame; "
            f"sends woke at worst {self.wake_ms:.2f} ms "
            f"after due; least slack, echo to its target's tick, {self.slack_ms:.2f} ms"
        )


LANDED = np.dtype(
    [
        ("target", "<i8"),
        ("param", "<i8"),
        ("value", "<i8"),
        ("wire", "<i8"),
        ("sent", "<i8"),
        ("t1_ns", "<i8"),
        ("tick", "<i8"),
        ("seen", "<i8"),
        ("clear_ns", "<f8"),
        ("landed", "<i8"),
        ("echo_ns", "<i8"),
        ("first", "<i8"),
        ("last", "<i8"),
    ]
)


def taken(sends: np.ndarray, echo: np.ndarray | None) -> np.ndarray:
    """``(n, 2)`` brackets on when the device took each write: its echo's, else the send interval's."""
    if echo is None:
        return np.c_[sends["t0_ns"], sends["t1_ns"]].astype(np.float64)
    return np.asarray(echo, dtype=np.float64).reshape(len(sends), 2)


def landing(bracket: np.ndarray, ticks: Ticks) -> tuple[np.ndarray, np.ndarray]:
    """The tick each bracketed write takes effect on, and the least clearance of either end from a tick.

    The tick is the first after the bracket's end; a negative clearance means a tick's bracket
    overlaps the write's, so it may have taken effect a tick earlier or later.
    """
    n, after = ticks.after(bracket[:, 1])
    before = bracket[:, 0] - (ticks.hi + (n - 1) * ticks.period)
    return n, np.minimum(after, before)


def window(
    bracket: np.ndarray,
    ticks: Ticks,
    times: FrameTimes,
    delta: float,
    spread: tuple[float, float] = (0.0, 0.0),
) -> tuple[np.ndarray, np.ndarray]:
    """The earliest and latest clip frame each bracketed write may take effect on.

    The device takes it somewhere in its bracket and applies it on the first tick after. A tick lies
    in its bracket widened by ``spread``, how far the frame starts the take's captures admit lie
    either side of its line, so the window holds every landing the echo and the line allow.
    """
    lo, hi = ticks.lo + spread[0], ticks.hi + spread[1]
    early = np.ceil((bracket[:, 0] - hi) / ticks.period)
    late = np.ceil((bracket[:, 1] - lo) / ticks.period)
    return frame_of_tick(early, ticks, times, delta), frame_of_tick(
        np.maximum(late, early), ticks, times, delta
    )


def landed(
    plan: np.ndarray,
    sends: np.ndarray,
    ticks: Ticks,
    times: FrameTimes,
    delta: float,
    seen: np.ndarray | None = None,
    *,
    echo: np.ndarray | None = None,
    resent: np.ndarray | None = None,
    spread: tuple[float, float] = (0.0, 0.0),
) -> Landed:
    """Match scheduled ``(target, param, value)`` rows to sends in order, and land each where it took effect.

    The k-th send of a ``(target, param)`` pair answers its k-th scheduled row, resent copies aside; a
    send nothing scheduled is a duplicate. ``echo`` (``wire.echoed``) brackets when the device took each
    send, else its send interval does; ``seen`` is each send's tick as read back, which lands it when known.
    ``first`` and ``last`` are each write's landing window (:func:`window`, widened by ``spread``); a write
    is late only when its window starts after its target, early only when it ends before.
    """
    rows = np.zeros(len(plan), dtype=LANDED)
    rows["target"], rows["param"], rows["value"] = plan["target"], plan["param"], plan["value"]
    rows["sent"], rows["wire"], rows["tick"], rows["seen"], rows["landed"] = -1, -1, -1, -1, CALIBRATION
    rows["first"], rows["last"] = CALIBRATION, CALIBRATION
    rows["clear_ns"], rows["echo_ns"] = np.nan, -1
    keep = sends["target"] > START
    copy = np.isin(np.arange(len(sends)), [] if resent is None else resent)[keep]
    live, bracket = sends[keep], taken(sends, echo)[keep]
    live_seen = np.full(len(live), -1, dtype=np.int64) if seen is None else np.asarray(seen)[keep]
    keys = {}
    for i, (target, param) in enumerate(zip(plan["target"], plan["param"])):
        keys.setdefault((int(target), int(param)), []).append(i)
    used = {key: 0 for key in keys}
    duplicated = 0
    hit = np.full(len(plan), -1, dtype=np.int64)
    for j, (target, param) in enumerate(zip(live["target"], live["param"])):
        key = (int(target), int(param))
        slots = keys.get(key, [])
        if copy[j]:
            continue
        if used.get(key, 0) >= len(slots):
            duplicated += 1
            continue
        i = slots[used[key]]
        used[key] += 1
        hit[i] = j
        rows["sent"][i], rows["wire"][i], rows["t1_ns"][i] = j, live["value"][j], live["t1_ns"][j]
        rows["seen"][i] = live_seen[j]
    sent = hit >= 0
    took = bracket[hit[sent]]
    heard = np.isfinite(took[:, 1])
    ok = np.flatnonzero(sent)[heard]
    n, clear = landing(took[heard], ticks)
    rows["tick"][ok], rows["clear_ns"][ok], rows["echo_ns"][ok] = n, clear, took[heard, 1]
    seen_ok = rows["seen"][ok]
    rows["landed"][ok] = frame_of_tick(np.where(seen_ok >= 0, seen_ok, n), ticks, times, delta)
    first, last = window(took[heard], ticks, times, delta, spread)
    rows["first"][ok] = np.minimum(first, rows["landed"][ok])
    rows["last"][ok] = np.maximum(last, rows["landed"][ok])
    due = np.rint((times.at(rows["target"][ok]) - delta - ticks.phase) / ticks.period)
    slack = ticks.lo + due * ticks.period - took[heard, 1]
    order = rows["sent"][sent]
    return Landed(
        rows=rows,
        missing=int((~sent).sum()),
        duplicated=duplicated,
        out_of_order=int(np.count_nonzero(np.diff(order) < 0)),
        late=int(np.count_nonzero(rows["first"][ok] > rows["target"][ok])),
        early=int(np.count_nonzero(rows["last"][ok] < rows["target"][ok])),
        ambiguous=int(np.count_nonzero(rows["last"][ok] > rows["first"][ok])),
        unechoed=int((~heard).sum()),
        slack_ms=float(slack.min()) / 1e6 if slack.size else float("nan"),
        wake_ms=(
            float((live["t0_ns"] - live["due_ns"]).max()) / 1e6
            if "due_ns" in (live.dtype.names or ()) and live.size
            else float("nan")
        ),
    )


DISAGREE = np.dtype([("read", "<i8"), ("slot", "<i8"), ("frame", "<i8"), ("want", "<i8"), ("got", "<i8")])


@dataclass(frozen=True)
class Applied:
    """Sent against applied: every readback slot checked against the writes landed before it."""

    reads: int
    checked: int
    ambiguous: int
    disagree: np.ndarray
    confirmed: int
    unconfirmed: int

    @property
    def ok(self) -> bool:
        """The device read back what was sent, every time it was asked."""
        return not self.disagree.size

    def summary(self) -> str:
        """One line of counts."""
        return (
            f"{self.reads} readbacks: {self.checked} slot reads, {self.disagree.size} disagree, "
            f"{self.ambiguous} straddle a tick; {self.confirmed} writes read back on their tick, "
            f"{self.unconfirmed} changed nothing to read"
        )


def applied(
    reads: np.ndarray,
    sends: np.ndarray,
    initial: Sequence[int],
    ticks: Ticks,
    times: FrameTimes,
    delta: float,
    seen: np.ndarray | None = None,
    *,
    echo: np.ndarray | None = None,
    resent: np.ndarray | None = None,
) -> Applied:
    """Check every readback slot against the values its frames held while the read was in flight.

    A read's request interval, widened by a frame each side for the readback's own lag, spans the
    frames it can report; any value the slot held over them agrees. A write lands on its ``seen``
    tick where one is given, else where :func:`landing` puts it; one never echoed held nothing.
    """
    o = np.asarray(reads["o"], dtype=np.int64)
    t0, t1 = reads["t0_ns"].astype(np.float64), reads["t1_ns"].astype(np.float64)
    disagree, checked, ambiguous, confirmed, unconfirmed = [], 0, 0, 0, 0
    bracket = taken(sends, echo)
    took = np.isfinite(bracket[:, 1]) & ~np.isin(np.arange(len(sends)), [] if resent is None else resent)
    n_send, _ = landing(np.where(took[:, None], bracket, 0.0), ticks)
    if seen is not None:
        n_send = np.where(np.asarray(seen) >= 0, seen, n_send)
    lands = ticks.at(n_send)
    for slot in range(1, 13):
        mine = np.flatnonzero((sends["param"] == slot) & took)
        values = np.r_[int(initial[slot - 1]), sends["value"][mine]]
        at = lands[mine]
        before = np.searchsorted(at, t0 - ticks.period, side="right")
        after = np.searchsorted(at, t1 + ticks.period, side="right")
        got = o[:, slot - 1]
        ok = values[before] == got
        span = after != before
        for k in np.flatnonzero(~ok & span):
            ok[k] = got[k] in values[before[k] : after[k] + 1]
        checked += int(o.shape[0])
        ambiguous += int(span.sum())
        for k in np.flatnonzero(~ok):
            disagree.append(
                (k, slot, int(times.frame_of(t0[k] + delta)), int(values[before[k]]), int(got[k]))
            )
        changed = np.r_[True, values[1:] != values[:-1]][1:]
        first = np.searchsorted(t0, at + ticks.period, side="left")
        shown = (first < o.shape[0]) & (got[np.clip(first, 0, o.shape[0] - 1)] == values[1:])
        confirmed += int((shown & changed).sum())
        unconfirmed += int((~changed).sum())
    return Applied(
        reads=int(o.shape[0]),
        checked=checked,
        ambiguous=ambiguous,
        disagree=np.array(disagree, dtype=DISAGREE),
        confirmed=confirmed,
        unconfirmed=unconfirmed,
    )


def realized(rows: np.ndarray, start: Sequence[int], frames: int) -> np.ndarray:
    """The ``(frames, 12)`` parameter state in effect over clip frames ``0..frames-1``.

    Each slot holds ``start``, then each :data:`LANDED` row's sent value from the frame it took effect
    on; a write landing before frame 0 holds from frame 0, and one never taken holds nowhere.
    """
    base = np.asarray(start, dtype=np.int64)
    out = np.repeat(base[None, :], int(frames), axis=0)
    took = rows[(rows["sent"] >= 0) & (rows["landed"] != CALIBRATION)]
    took = took[np.lexsort((took["sent"], took["landed"]))]
    clip = np.arange(int(frames))
    for slot in range(base.size):
        mine = took[took["param"] == slot + 1]
        held = np.searchsorted(np.maximum(mine["landed"], 0), clip, side="right")
        out[:, slot] = np.r_[base[slot], mine["wire"]][held]
    return out


#: Steps the model does not prove leave the picture alone: the ones a parameter is tested on.
TESTED = (LIVE,)
#: The pool of a parameter's steps a test ran on: those predicted to change the picture, those
#: proven not to, or every step when no model judged them.
LIVE_POOL, DEAD_POOL, ALL_POOL = "live", "dead", "all"


@dataclass(frozen=True)
class Moved:
    """Applied against acted upon, for one parameter's pool of steps: did they move the picture?

    Each step's output change is scored against frames of matched source motion with no step and
    placed among the scores of all such frames; Fisher's combination at the best lag decides ``moved``.
    ``effect`` and ``scatter`` are the steps' mean and the null's spread of that change, in its own units.
    """

    param: int
    name: str
    steps: int
    effective: float
    least_p: float
    z: float
    lag: int
    critical: float
    flat: float
    jointly: int
    partners: tuple[int, ...] = ()
    effect: float = float("nan")
    scatter: float = float("nan")
    pool: str = ALL_POOL
    #: Steps left untested because their regime has too few step-free frames for a null.
    unmatched: int = 0

    @property
    def moved(self) -> bool:
        """Whether the steps moved the picture beyond chance, family-wise."""
        return self.z > self.critical

    @property
    def detectable(self) -> float:
        """The least mean output change these steps would tell from none at the family-wise bound."""
        if self.effective > 0 and np.isfinite(self.scatter):
            return self.critical * self.scatter / math.sqrt(self.effective)
        return math.inf

    @property
    def verdict(self) -> str:
        """``moved`` or ``not seen``: a statistic never shows that a parameter has no effect."""
        return "moved" if self.moved else "not seen"

    @property
    def steps_needed(self) -> int:
        """Steps at which the observed mean change would clear the bound; 0 if moved or none was seen."""
        if self.moved or not (np.isfinite(self.effect) and self.effect > 0 and np.isfinite(self.scatter)):
            return 0
        need = (self.critical * self.scatter / self.effect) ** 2
        return int(math.ceil(need * self.steps / max(self.effective, 1.0)))


def _cover(first: np.ndarray, last: np.ndarray, count: int) -> np.ndarray:
    """How many of the windows ``[first, last]`` cover each of ``count`` frames."""
    marks = np.zeros(count + 1, dtype=np.int64)
    np.add.at(marks, np.clip(first, 0, count), 1)
    np.add.at(marks, np.clip(last + 1, 0, count), -1)
    return np.cumsum(marks)[:count]


def _within(cover: np.ndarray, first: np.ndarray, last: np.ndarray) -> np.ndarray:
    """The total of ``cover`` over each window ``[first, last]``."""
    total = np.r_[0, np.cumsum(cover)]
    return total[np.clip(last + 1, 0, cover.size)] - total[np.clip(first, 0, cover.size)]


def _groups(
    frames: np.ndarray, spans: np.ndarray, params: np.ndarray, kinds: np.ndarray | None, reach: int
) -> list[tuple]:
    """``(param, pool, steps, alone, partners)`` for every pool of steps to test, steps as ``(last, span)``.

    A step takes effect somewhere in its window, the ``span`` frames ending at ``last``. A live step is
    tested where no other parameter's live step's window meets its own, so proven no-ops beside it do
    not count as partners. A dead pool keeps only windows more than ``reach`` from every live window,
    its own parameter's included, so no live step's picture can reach it. Steps that never land alone
    are tested with their partners.
    """
    count = int((frames.max() if frames.size else 0) + reach + 2)
    first = frames - spans + 1
    tested = np.ones(frames.size, dtype=bool) if kinds is None else np.isin(kinds, TESTED)
    proven = np.zeros(frames.size, dtype=bool) if kinds is None else np.isin(kinds, (NOOP, DEAD))
    near_live = _within(_cover(first[tested] - reach, frames[tested] + reach, count), first, frames) > 0
    out = []
    for param in sorted(set(int(p) for p in params[tested | proven])):
        for pool, mask, rivals in (
            (ALL_POOL if kinds is None else LIVE_POOL, tested, tested),
            (DEAD_POOL, proven & ~near_live, tested | proven),
        ):
            pick = mask & (params == param)
            if not pick.any():
                continue
            mine = np.unique(np.c_[frames[pick], spans[pick]], axis=0)
            other = rivals & (params != param)
            busy = _cover(first[other], frames[other], count)
            lo, hi = mine[:, 0] - mine[:, 1] + 1, mine[:, 0]
            free = _within(busy, lo, hi) == 0
            alone = mine[free]
            partners: tuple[int, ...] = ()
            if not alone.size:
                alone = mine
                near = _within(_cover(lo, hi, count), first[other], frames[other]) > 0
                partners = tuple(sorted({int(p) for p in params[other][near]}))
            out.append((param, pool, mine, alone, partners))
    return out


class _Scorer:
    """Steps' picture changes across their windows, scored against step-free spans as long, per regime."""

    def __init__(self, out: np.ndarray, src: np.ndarray, stepped: np.ndarray, regime: Regime):
        self.out, self.src, self.stepped, self.regime = out, src, stepped.astype(np.int64), regime
        self.quantum = 1.0 / float(np.prod(np.shape(out)[1:]))
        self.changes: dict[int, tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]] = {}
        self.nulls: dict[tuple[int, int, int], tuple[np.ndarray, np.ndarray, float]] = {}

    def change(self, span: int) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Output and source change over ``span`` frames, where both are measured, and the step-free ends."""
        if span not in self.changes:
            d_out, d_src = frame_change(self.out, span), frame_change(self.src, span)
            measured = np.isfinite(d_out) & np.isfinite(d_src)
            end = np.arange(self.stepped.size)
            clear = _within(self.stepped, end - span + 1, end) == 0
            self.changes[span] = (d_out, d_src, measured, np.flatnonzero(clear & measured))
        return self.changes[span]

    def null_of(self, param: int, key: int, span: int) -> tuple[np.ndarray, np.ndarray, float]:
        """The step-free controls of a regime and span, their matched scores and their scatter."""
        mask = self.regime.own.get(param, 0)
        if (mask, key, span) not in self.nulls:
            d_out, d_src, _, controls = self.change(span)
            mine = controls[self.regime.of(param, controls) == key]
            res, spread = (
                matched_change(d_out, d_src, mine, mine, self.quantum) if mine.size > 1 else (mine, mine)
            )
            scatter = float(np.std(res, ddof=1)) if res.size > 1 else math.nan
            self.nulls[mask, key, span] = (mine, res / spread, scatter)
        return self.nulls[mask, key, span]

    def tested(
        self, param: int, x: np.ndarray, span: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray, float, int]:
        """Each step ending at ``x``: its tail p, its residual, the pooled scatter and the steps measured."""
        p_e, residual, weight, var, kept = [], [], 0, 0.0, 0
        for s in np.unique(span):
            d_out, d_src, measured, _ = self.change(int(s))
            at = x[span == s]
            at = at[measured[at]]
            kept += at.size
            keys = self.regime.of(param, at)
            for key in np.unique(keys):
                mine, null, scatter = self.null_of(param, int(key), int(s))
                here = at[keys == key]
                if null.size:
                    r, spread = matched_change(d_out, d_src, here, mine, self.quantum)
                    p_e.append(tail_p(r / spread, null))
                    residual.append(r)
                    weight, var = weight + here.size, var + here.size * scatter**2
        p_e = np.concatenate(p_e) if p_e else np.zeros(0)
        residual = np.concatenate(residual) if residual else np.zeros(0)
        ok = np.isfinite(p_e)
        return p_e[ok], residual[ok], math.sqrt(var / weight) if weight else math.nan, kept


def acted_upon(
    out: np.ndarray,
    src: np.ndarray,
    luma_range: np.ndarray,
    landed_frames: np.ndarray,
    params: np.ndarray,
    names: Mapping[int, str],
    *,
    lags: Sequence[int] = LAGS,
    kinds: Sequence[str] | None = None,
    regime: Regime | None = None,
    spans: np.ndarray | None = None,
) -> list[Moved]:
    """Test every parameter's steps against frames of matched source motion with no step.

    ``out`` and ``src`` are per clip frame; ``landed_frames``, ``params`` and ``kinds`` give each
    write's frame and the model's verdict on its step, and ``spans`` its landing window's length in
    frames, ending at its frame (one each when not given). A step is scored by the picture's change
    across its window, from the frame before it to its end plus the lag, against step-free spans as
    long. With ``kinds``, a parameter's steps the model leaves live and those it proves no-ops are
    tested apart; steps that change no value are not tested. With ``regime``, each step is scored
    against the step-free frames of its own regime (:class:`Regime`), each regime's controls forming
    its own null. Family-wise over all tests.
    """
    count = len(out)
    frames = np.asarray(landed_frames, dtype=np.int64)
    spans = (
        np.ones(frames.size, dtype=np.int64) if spans is None else np.maximum(np.asarray(spans, np.int64), 1)
    )
    kinds = None if kinds is None else np.asarray(kinds, dtype=object)
    inside = (frames - spans >= 0) & (frames < count - max(lags))
    stepped = _cover(frames[inside] - spans[inside] + 1, frames[inside] + max(0, *lags), count) > 0
    if kinds is not None:
        inside &= kinds != UNCHANGED
        kinds = kinds[inside]
    frames, params, spans = frames[inside], np.asarray(params)[inside], spans[inside]
    scored = _Scorer(out, src, stepped, Regime(np.zeros(count, dtype=np.int64)) if regime is None else regime)
    groups = _groups(frames, spans, params, kinds, max(lags))
    limit = critical(len(groups) * len(lags))
    verdicts = []
    for param, pool, mine, alone, partners in groups:
        best = (np.zeros(0), 0.0, -math.inf, lags[0], np.zeros(0), math.nan, 0)
        for lag in lags:
            p_e, residual, scatter, kept = scored.tested(param, alone[:, 0] + lag, alone[:, 1])
            if not p_e.size:
                best = best if best[0].size else (*best[:6], kept)
                continue
            z, n_eff = fisher_z(p_e)
            if z > best[2]:
                best = (p_e, n_eff, z, lag, residual, scatter, kept - p_e.size)
        p_e, n_eff, z, lag, residual, scatter, unmatched = best
        verdicts.append(
            Moved(
                param=param,
                name=names.get(param, f"P{param}"),
                steps=int(p_e.size),
                effective=float(n_eff),
                least_p=float(p_e.min()) if p_e.size else float("nan"),
                z=float(z) if p_e.size else 0.0,
                lag=int(lag),
                critical=limit,
                flat=float(np.mean(luma_range[alone[:, 0]] <= FLAT_CODES)) if alone.size else float("nan"),
                jointly=int(len(mine) - len(alone)) if not partners else int(len(mine)),
                partners=partners,
                effect=float(residual.mean()) if residual.size else float("nan"),
                scatter=scatter,
                pool=pool,
                unmatched=int(unmatched),
            )
        )
    return verdicts


def why_not(moved: Moved, name: str, applied_ok: bool, landed_ok: bool) -> str:
    """Why a parameter's steps did not show, from the earliest link of the chain that broke.

    A picture that was a uniform field at most of the steps, as a take's ``mostly_picture`` counts,
    had nothing to move. Past the chain, the least change the steps could have shown, in its own units.
    """
    partners = ", P".join(map(str, moved.partners))
    reasons = (
        (not landed_ok, "not sent on time"),
        (not applied_ok, "the firmware did not take it"),
        (name.strip() in ("-", ""), "the program leaves the slot unused"),
        (
            moved.steps == 0 and moved.unmatched > 0,
            f"its {moved.unmatched} steps' switch regimes have too few step-free frames for a null",
        ),
        (moved.steps == 0, "no step landed inside the clip's measured frames"),
        (2 * moved.flat > 1, f"the picture was a uniform field at {moved.flat:.0%} of its steps"),
    )
    broke = next((why for broken, why in reasons if broken), None)
    if broke is not None:
        return broke
    joint = f", even with P{partners} changing at once" if moved.partners else ""
    need = (
        f"; about {moved.steps_needed} steps would show its observed mean change of {moved.effect:.3g}"
        if moved.steps_needed
        else "; its steps show no mean change to build on"
    )
    return (
        f"not seen{joint}: {moved.steps} steps detect only a mean change of {moved.detectable:.3g} "
        f"(minimum detectable effect, against step-free scatter {moved.scatter:.3g}){need}"
    )


@dataclass
class Evidence:
    """The three links for one take, and what was loaded before its clip began."""

    program: str
    latency: dict
    ticks: dict
    landed: Landed
    applied: Applied
    received: Received | None
    moved: list[Moved]
    loaded: dict = field(default_factory=dict)
    #: Per parameter, how many of its steps the program model judged each way (:data:`LIVE` ...).
    predicted: dict = field(default_factory=dict)

    @property
    def ok(self) -> bool:
        """The hard check: every write landed on its frame, read back, and the program was up first."""
        return (
            self.landed.ok
            and self.applied.ok
            and bool(self.loaded.get("ok", False))
            and (self.received is None or self.received.ok)
        )

    def pools(self) -> dict[int, dict[str, Moved]]:
        """Each parameter's tests by pool."""
        out: dict[int, dict[str, Moved]] = {int(p): {} for p in self.predicted}
        for m in self.moved:
            out.setdefault(m.param, {})[m.pool] = m
        return dict(sorted(out.items()))

    @staticmethod
    def verdict(pools: Mapping[str, Moved], counts: Mapping[str, int] = ()) -> str:
        """``dead, moved`` (the model or firmware is wrong), ``live, moved``, ``live, not seen``,
        ``never live`` or ``untested`` (live steps, none inside the clip); ``moved`` or ``not seen``
        when no model judged the steps."""
        dead, live = pools.get(DEAD_POOL), pools.get(LIVE_POOL)
        if dead is not None and dead.moved:
            return "dead, moved"
        if live is not None:
            return f"live, {live.verdict}"
        if ALL_POOL in pools:
            return pools[ALL_POOL].verdict
        return "untested" if dict(counts).get(LIVE) else "never live"

    def _why(
        self, verdict: str, test: Moved | None, dead: Moved | None, counts: Mapping, ok: tuple
    ) -> str | None:
        if verdict == "dead, moved":
            return (
                f"contradiction: {dead.steps} steps the model proves cannot change the picture moved it "
                f"(z {dead.z:+.2f}, critical {dead.critical:.2f}); the model or the firmware is wrong"
            )
        if verdict == "never live":
            return (
                f"never live in this take: {counts.get(DEAD, 0)} steps dead, {counts.get(NOOP, 0)} proven "
                "no-ops in the states it was stepped from; not tested, so this says nothing of its effect"
            )
        if verdict == "untested":
            return f"untested: its {counts[LIVE]} live steps all landed outside the clip's measured frames"
        if test is None or test.moved:
            return None
        return why_not(test, test.name, *ok)

    def table(self) -> list[dict]:
        """Per parameter: writes, landings, readback, the model's step counts, the verdict, and why."""
        rows = self.landed.rows
        bad = set(int(s) for s in self.applied.disagree["slot"]) if self.applied.disagree.size else set()
        out = []
        for param, pools in self.pools().items():
            test = pools.get(LIVE_POOL) or pools.get(ALL_POOL)
            dead = pools.get(DEAD_POOL)
            shown = test or dead
            mine = rows[rows["param"] == param]
            on_time = bool(np.all(mine["landed"] == mine["target"])) and bool(np.all(mine["sent"] >= 0))
            agree = param not in bad
            counts = {k: int(v) for k, v in self.predicted.get(param, {}).items()}
            verdict = self.verdict(pools, counts)
            out.append(
                {
                    "param": param,
                    "name": shown.name if shown else f"P{param}",
                    "writes": int(mine.size),
                    "sent": int(np.count_nonzero(mine["sent"] >= 0)),
                    "on_target": int(np.count_nonzero(mine["landed"] == mine["target"])),
                    "readback": "agrees" if agree else "disagrees",
                    "model_steps": counts,
                    "steps_tested": test.steps if test else 0,
                    "steps_unmatched": test.unmatched if test else 0,
                    "z": round(test.z, 2) if test else None,
                    "critical_z": round(test.critical, 2) if test else None,
                    "lag": test.lag if test else None,
                    "acted_upon": bool(test and test.moved),
                    "verdict": verdict,
                    "effect": round(test.effect, 4) if test else None,
                    "mde": round(test.detectable, 4) if test else None,
                    "scatter": round(test.scatter, 4) if test else None,
                    "steps_needed": test.steps_needed if test else 0,
                    "with": list(test.partners) if test else [],
                    "dead_steps_tested": dead.steps if dead else 0,
                    "dead_steps_unmatched": dead.unmatched if dead else 0,
                    "dead_z": round(dead.z, 2) if dead else None,
                    "why_not": self._why(verdict, test, dead, counts, (agree, on_time)),
                }
            )
        return out

    @property
    def contradictions(self) -> list[int]:
        """Parameters whose steps the model proves cannot change the picture, that moved it anyway."""
        return [row["param"] for row in self.table() if row["verdict"] == "dead, moved"]

    def schedule(self) -> list[dict]:
        """Parameters this take could not show: steps a sweep needs for each, 0 for every level where live."""
        return [
            {
                "program": self.program,
                "param": row["param"],
                "name": row["name"],
                "steps": row["steps_needed"],
            }
            for row in self.table()
            if row["verdict"] in ("never live", "untested") or (not row["acted_upon"] and row["steps_needed"])
        ]

    def lines(self) -> list[str]:
        """The report as log lines."""
        lat, tick = self.latency, self.ticks
        said = [
            f"{self.program}: CC-to-picture {lat.get('frames', float('nan')):.3f} frames send to capture, "
            f"known to {tick.get('width_ms', float('nan')):.2f} ms from {lat.get('samples', 0)} calibration "
            "writes (0: the dry takes' range)",
            f"{self.program}: loaded before clip frame 0: {self.loaded.get('summary', 'not checked')}",
            f"{self.program}: {self.landed.summary()}",
            f"{self.program}: {self.applied.summary()}",
        ]
        if self.received is not None:
            said.append(f"{self.program}: {self.received.summary()}")
        for row in self.table():
            verdict = row["verdict"] if row["why_not"] is None else row["why_not"]
            if row["acted_upon"] and row["with"]:
                verdict += f" only together with P{', P'.join(map(str, row['with']))}"
            model = ", ".join(f"{v} {k}" for k, v in sorted(row["model_steps"].items()))
            said.append(
                f"{self.program}:   P{row['param']:<2} {row['name'][:16]:16s} "
                f"{row['sent']}/{row['writes']} sent, {row['on_target']} on target, "
                f"readback {row['readback']}"
                + (f", model: {model}" if model else "")
                + (
                    f", z {row['z']:+.2f} (lag {row['lag']}, critical {row['critical_z']:.2f})"
                    if row["z"] is not None
                    else ""
                )
                + (
                    f", {row['steps_unmatched'] + row['dead_steps_unmatched']} steps with no regime null"
                    if row["steps_unmatched"] + row["dead_steps_unmatched"]
                    else ""
                )
                + f": {verdict}"
            )
        return said

    def to_json(self) -> str:
        """The whole report as JSON."""
        return json.dumps(
            {
                "program": self.program,
                "ok": self.ok,
                "latency": self.latency,
                "ticks": self.ticks,
                "loaded": self.loaded,
                "landed": self.landed.summary(),
                "applied": self.applied.summary(),
                "received": None if self.received is None else self.received.summary(),
                "missing": [] if self.received is None else self.received.gaps(),
                "disagree": self.applied.disagree.tolist(),
                "late": self.landed.rows[self.landed.rows["landed"] > self.landed.rows["target"]].tolist(),
                "parameters": self.table(),
                "schedule": self.schedule(),
            },
            indent=1,
            default=float,
        )

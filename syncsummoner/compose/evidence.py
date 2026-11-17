"""Frame-indexed evidence that every parameter of a take was sent, applied and acted upon.

The device latches CCs on a frame tick, which its own readback shows; capture timestamps say
when each clip frame reached the card. A CC lands on the frame of the first tick after it is
sent, so sends, readbacks and pictures are joined on one clip-frame index (docs/evidence.md).
"""

from __future__ import annotations

import difflib
import json
import math
from dataclasses import dataclass, field
from typing import Any, Mapping, Sequence

import numpy as np
from scipy.stats import chi2, norm

from syncsummoner.aesthetics.survival import SIGMA
from syncsummoner.device.session import LSB_OFFSET, wire_messages
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
    """When each played frame's first capture reached the host, ``ns[j + lead]`` for frame ``j``.

    NaN where no capture starts the frame; ``origin`` and ``period`` are the least-squares line.
    """

    ns: np.ndarray
    lead: int
    origin: float
    period: float

    def at(self, frame: int | np.ndarray) -> np.ndarray:
        """Measured time of ``frame``'s first capture, or the line's where none was measured."""
        frame = np.asarray(frame, dtype=np.int64)
        index = frame + self.lead
        inside = (index >= 0) & (index < self.ns.size)
        measured = np.where(inside, self.ns[np.clip(index, 0, self.ns.size - 1)], np.nan)
        return np.where(np.isfinite(measured), measured, self.origin + frame * self.period)

    def frame_of(self, t_ns: float | np.ndarray) -> np.ndarray:
        """The frame whose first capture lies nearest ``t_ns`` on the line."""
        return np.rint((np.asarray(t_ns, dtype=np.float64) - self.origin) / self.period).astype(np.int64)


def frame_times(
    t_ns: np.ndarray, shown: np.ndarray, runs: Sequence[Any], per_frame: int, lead: int
) -> FrameTimes:
    """Frame start times from an attribution: each run's captures on its cadence's first copy.

    The card relocks at a new phase after a gap, so the period is the slope common to every run,
    and the line runs through the last run, the one the clip plays in.
    """
    t = np.asarray(t_ns, dtype=np.float64)
    last = int(shown.max()) if shown.size else -lead
    ns = np.full(last + lead + 1, np.nan)
    dx, dt, line = [], [], None
    for run in runs:
        k = np.arange(run.first, run.last + 1)
        start = k[(run.slot + k - run.first) % per_frame == 0]
        frame = shown[start] + lead
        ok = (frame >= 0) & (frame < ns.size)
        ns[frame[ok]] = t[start[ok]]
        if ok.sum() > 1:
            f, s = frame[ok].astype(np.float64) - lead, t[start[ok]]
            dx.append(f - f.mean())
            dt.append(s - s.mean())
            line = (f.mean(), s.mean())
    if line is None:
        raise EvidenceError("no run captured two frame starts, so the frame period is unknown")
    x, y = np.concatenate(dx), np.concatenate(dt)
    slope = float(x @ y / (x @ x))
    return FrameTimes(ns=ns, lead=lead, origin=float(line[1] - slope * line[0]), period=slope)


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


@dataclass(frozen=True)
class Latency:
    """From a write leaving the host to the card's first capture of the frame it lands on, per take.

    Each calibration write that showed in the picture bounds it: after the start of the frame
    before the one it landed on, at or before its own. ``lo`` and ``hi`` are those bounds
    intersected; where writes disagree by a little, they are the band they disagree over.
    ``delta`` is the band's middle, in ns.
    """

    lo: float
    hi: float
    samples: int
    period: float

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
        }


def latency(sent: np.ndarray, shown_at: np.ndarray, times: FrameTimes) -> Latency:
    """The send-to-capture latency from where each calibration write first showed in the picture.

    A write sent at ``t`` and first shown in frame ``F`` puts the latency in ``(T(F-1) - t, T(F) - t]``.
    Bounds that cross by more than half a frame mean no single latency holds, which is an error.
    """
    ok = shown_at >= -times.lead
    if not ok.any():
        raise EvidenceError(
            "no calibration write showed in the picture, so the CC-to-picture latency is unknown"
        )
    t = np.asarray(sent, dtype=np.float64)[ok]
    lo = float((times.at(shown_at[ok] - 1) - t).max())
    hi = float((times.at(shown_at[ok]) - t).min())
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
    """Intended against sent: every scheduled write matched to its send and the frame it landed on."""

    rows: np.ndarray
    missing: int
    duplicated: int
    out_of_order: int
    late: int
    early: int
    ambiguous: int
    wake_ms: float = float("nan")

    @property
    def ok(self) -> bool:
        """Every write sent once, in order, landing unambiguously on its target."""
        return not (
            self.missing or self.duplicated or self.out_of_order or self.late or self.early or self.ambiguous
        )

    def summary(self) -> str:
        """One line of counts."""
        return (
            f"{len(self.rows)} writes scheduled: {self.missing} missing, {self.duplicated} duplicated, "
            f"{self.out_of_order} out of order, {self.late} late, {self.early} early, "
            f"{self.ambiguous} on a tick; sends woke at worst {self.wake_ms:.2f} ms after due"
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
    ]
)


def landed(
    plan: np.ndarray,
    sends: np.ndarray,
    ticks: Ticks,
    times: FrameTimes,
    delta: float,
    seen: np.ndarray | None = None,
) -> Landed:
    """Match scheduled ``(target, param, value)`` rows to sends in order, and land each on its frame.

    The k-th send of a ``(target, param)`` pair answers its k-th scheduled row; a send nothing
    scheduled is a duplicate. ``seen`` is each send's tick as read back, which lands it when known.
    """
    rows = np.zeros(len(plan), dtype=LANDED)
    rows["target"], rows["param"], rows["value"] = plan["target"], plan["param"], plan["value"]
    rows["sent"], rows["wire"], rows["tick"], rows["seen"], rows["landed"] = -1, -1, -1, -1, CALIBRATION
    rows["clear_ns"] = np.nan
    keep = sends["target"] > START
    live = sends[keep]
    live_seen = np.full(len(live), -1, dtype=np.int64) if seen is None else np.asarray(seen)[keep]
    keys = {}
    for i, (target, param) in enumerate(zip(plan["target"], plan["param"])):
        keys.setdefault((int(target), int(param)), []).append(i)
    used = {key: 0 for key in keys}
    duplicated = 0
    for j, (target, param) in enumerate(zip(live["target"], live["param"])):
        key = (int(target), int(param))
        slots = keys.get(key, [])
        if used.get(key, 0) >= len(slots):
            duplicated += 1
            continue
        i = slots[used[key]]
        used[key] += 1
        rows["sent"][i], rows["wire"][i], rows["t1_ns"][i] = j, live["value"][j], live["t1_ns"][j]
        rows["seen"][i] = live_seen[j]
    sent = rows["sent"] >= 0
    n, clear = ticks.after(rows["t1_ns"][sent].astype(np.float64))
    rows["tick"][sent], rows["clear_ns"][sent] = n, clear
    at = np.where(rows["seen"][sent] >= 0, rows["seen"][sent], n)
    rows["landed"][sent] = frame_of_tick(at, ticks, times, delta)
    order = rows["sent"][sent]
    return Landed(
        rows=rows,
        missing=int((~sent).sum()),
        duplicated=duplicated,
        out_of_order=int(np.count_nonzero(np.diff(order) < 0)),
        late=int(np.count_nonzero(rows["landed"][sent] > rows["target"][sent])),
        early=int(np.count_nonzero(rows["landed"][sent] < rows["target"][sent])),
        ambiguous=int(
            np.count_nonzero((clear < 0) | ((rows["seen"][sent] >= 0) & (rows["seen"][sent] != n)))
        ),
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
) -> Applied:
    """Check every readback slot against the values its frames held while the read was in flight.

    A read's request interval, widened by a frame each side for the readback's own lag, spans the
    frames it can report; any value the slot held over them agrees. A write lands on its ``seen``
    tick where one is given, else on the first tick at or after it.
    """
    o = np.asarray(reads["o"], dtype=np.int64)
    t0, t1 = reads["t0_ns"].astype(np.float64), reads["t1_ns"].astype(np.float64)
    disagree, checked, ambiguous, confirmed, unconfirmed = [], 0, 0, 0, 0
    n_send, _ = ticks.after(sends["t1_ns"].astype(np.float64))
    if seen is not None:
        n_send = np.where(np.asarray(seen) >= 0, seen, n_send)
    lands = ticks.at(n_send)
    for slot in range(1, 13):
        mine = np.flatnonzero(sends["param"] == slot)
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


#: One CC message put on the wire: controller, 7-bit value, and the index of the send that carried it.
MESSAGE = np.dtype([("cc", "<i8"), ("value", "<i8"), ("send", "<i8")])
#: A sent message no receipt answers: the send's index, interval start, parameter, target and the message.
MISSING = np.dtype(
    [("send", "<i8"), ("t0_ns", "<i8"), ("param", "<i8"), ("target", "<i8"), ("cc", "<i8"), ("value", "<i8")]
)


def expected_messages(
    sends: np.ndarray, cc_map: Sequence[int], *, high_resolution: bool = True
) -> np.ndarray:
    """The :data:`MESSAGE` records the sends put on the wire, in order, as pyvmancer encodes them."""
    cc = np.asarray(cc_map, dtype=np.int64)
    per = wire_messages(cc, high_resolution=high_resolution)
    param = np.asarray(sends["param"], dtype=np.int64) - 1
    value = np.asarray(sends["value"], dtype=np.int64)
    pair = per[param] == 2
    word = np.rint(value * 16383 / 1023).astype(np.int64)
    out = np.zeros(int(per[param].sum()), dtype=MESSAGE)
    out["send"] = np.repeat(np.arange(value.size), per[param])
    head = np.r_[0, np.cumsum(per[param])[:-1]].astype(np.int64)
    out["cc"][head] = cc[param]
    out["value"][head] = np.where(pair, word >> 7, np.rint(value * 127 / 1023).astype(np.int64))
    tail = head[pair] + 1
    out["cc"][tail] = cc[param][pair] + LSB_OFFSET
    out["value"][tail] = word[pair] & 0x7F
    return out


@dataclass(frozen=True)
class Received:
    """Sent against received: the firmware's monitor receipts aligned to the messages put on the wire."""

    sent: int
    received: int
    matched: int
    missing: np.ndarray = field(default_factory=lambda: np.zeros(0, dtype=MISSING))

    @property
    def ok(self) -> bool:
        """Every message received, in order, and nothing else."""
        return self.sent == self.received == self.matched

    @property
    def extra(self) -> int:
        """Receipts no sent message answers."""
        return self.received - self.matched

    def gaps(self) -> list[dict]:
        """Runs of consecutive missing messages: how many, the sends that carried them and when."""
        miss = self.missing
        if not miss.size:
            return []
        cut = np.flatnonzero(np.diff(miss["send"]) > 1) + 1
        return [
            {
                "messages": int(run.size),
                "sends": [int(run["send"][0]), int(run["send"][-1])],
                "params": sorted({int(p) for p in run["param"]}),
                "targets": sorted({int(t) for t in run["target"]}),
                "t0_ns": [int(run["t0_ns"][0]), int(run["t0_ns"][-1])],
            }
            for run in np.split(miss, cut)
        ]

    def summary(self) -> str:
        """One line of counts, and each run of missing messages."""
        said = (
            f"{self.sent} messages sent, {self.received} received, {self.matched} aligned, "
            f"{self.sent - self.matched} missing, {self.extra} unexpected"
        )
        for gap in self.gaps():
            span = (gap["t0_ns"][1] - gap["t0_ns"][0]) / 1e6
            said += (
                f"; {gap['messages']} missing from sends {gap['sends'][0]}..{gap['sends'][1]} "
                f"(P{', P'.join(map(str, gap['params']))} for frames "
                f"{gap['targets'][0]}..{gap['targets'][-1]}, sent over {span:.2f} ms)"
            )
        return said


def received(expected: np.ndarray, got: np.ndarray, sends: np.ndarray | None = None) -> Received:
    """Receipts aligned against the expected messages, so a gap costs only the messages it lost.

    The alignment is difflib's matching blocks over ``(cc, value)``, a common subsequence in which
    every receipt answers at most one message in order. ``sends`` names the missing messages' writes.
    """
    exp = np.asarray(expected["cc"], dtype=np.int64) * 128 + np.asarray(expected["value"], dtype=np.int64)
    rec = np.asarray(got["cc"], dtype=np.int64) * 128 + np.asarray(got["value"], dtype=np.int64)
    blocks = difflib.SequenceMatcher(None, exp.tolist(), rec.tolist(), autojunk=False).get_matching_blocks()
    hit = np.zeros(exp.size, dtype=bool)
    for block in blocks:
        hit[block.a : block.a + block.size] = True
    lost = np.flatnonzero(~hit)
    missing = np.zeros(lost.size, dtype=MISSING)
    missing["cc"], missing["value"] = expected["cc"][lost], expected["value"][lost]
    missing["send"] = expected["send"][lost]
    if sends is not None and lost.size:
        carried = sends[missing["send"]]
        missing["t0_ns"], missing["param"], missing["target"] = (
            carried["t0_ns"],
            carried["param"],
            carried["target"],
        )
    return Received(sent=int(exp.size), received=int(rec.size), matched=int(hit.sum()), missing=missing)


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


def frame_change(stack: np.ndarray) -> np.ndarray:
    """Mean absolute difference of every frame from the one before, all planes; NaN for the first."""
    flat = np.asarray(stack, dtype=np.float32).reshape(len(stack), -1)
    return np.r_[np.nan, np.abs(np.diff(flat, axis=0)).mean(axis=1)]


def matched_change(
    d_out: np.ndarray, d_src: np.ndarray, frames: np.ndarray, controls: np.ndarray, quantum: float
) -> tuple[np.ndarray, np.ndarray]:
    """Each frame's output change less the median of the controls nearest it in source change, and their MAD.

    The ``sqrt(n)`` nearest controls (Loftsgaarden and Quesenberry), the frame itself left out;
    ``quantum``, the measure's resolution, stands in for a MAD of zero.
    """
    k = max(1, int(round(math.sqrt(controls.size))))
    order = np.argsort(d_src[controls], kind="stable")
    ranked, values = controls[order], d_src[controls][order]
    residual, spread = np.full(frames.size, np.nan), np.full(frames.size, np.nan)
    for i, frame in enumerate(frames):
        at = int(np.searchsorted(values, d_src[frame]))
        lo, hi = max(0, at - k - 1), min(values.size, at + k + 1)
        window = ranked[lo:hi]
        window = window[window != frame]
        near = window[np.argsort(np.abs(d_src[window] - d_src[frame]), kind="stable")[:k]]
        ref = d_out[near]
        centre = float(np.median(ref))
        residual[i] = d_out[frame] - centre
        spread[i] = max(1.4826 * float(np.median(np.abs(ref - centre))), quantum)
    return residual, spread


def matched_scores(
    d_out: np.ndarray, d_src: np.ndarray, frames: np.ndarray, controls: np.ndarray, quantum: float
) -> np.ndarray:
    """Each frame's output change against the controls nearest it in source change, in robust units."""
    residual, spread = matched_change(d_out, d_src, frames, controls, quantum)
    return residual / spread


def tail_p(scores: np.ndarray, null: np.ndarray) -> np.ndarray:
    """One-sided p of each score in the empirical distribution of the null scores (mid-p)."""
    ordered = np.sort(null)
    above = null.size - np.searchsorted(ordered, scores, side="left")
    ties = np.searchsorted(ordered, scores, side="right") - np.searchsorted(ordered, scores, side="left")
    return (above - 0.5 * ties + 0.5) / (null.size + 1)


def fisher_z(p: np.ndarray) -> tuple[float, float]:
    """Fisher's combination of one-sided p values as a z score, and the independent samples behind it.

    Serial dependence shrinks the statistic and its degrees of freedom by ``n_eff / n`` (Brown).
    """
    if not p.size:
        return 0.0, 0.0
    terms = -2.0 * np.log(p)
    n_eff = effective(terms)
    stat = float(terms.sum()) * n_eff / p.size
    return float(norm.isf(chi2.sf(stat, 2.0 * n_eff))), n_eff


def effective(u: np.ndarray) -> float:
    """Independent samples behind a mean of a first-order autoregressive series."""
    if u.size < 3:
        return float(u.size)
    if np.ptp(u) == 0:
        return float(u.size)
    d = u - u.mean()
    rho = float(d[:-1] @ d[1:] / float(d @ d))
    return float(np.clip(u.size * (1 - rho) / (1 + rho), 1.0, u.size))


def critical(tests: int, sigma: float = SIGMA) -> float:
    """The one-sided z a test must pass for a family-wise error of the ``sigma`` tail over ``tests``."""
    return float(norm.isf(norm.sf(sigma) / max(1, int(tests))))


def _groups(frames: np.ndarray, params: np.ndarray, kinds: np.ndarray | None, reach: int) -> list[tuple]:
    """``(param, pool, frames, alone, partners)`` for every pool of steps to test.

    A live step is tested on frames where no other step the model leaves live lands, so proven
    no-ops beside it do not count as partners. A dead pool keeps only frames more than ``reach``
    from every live step, its own parameter's included, so no live step's picture can reach it.
    Steps that never land alone are tested with their partners.
    """
    tested = np.ones(frames.size, dtype=bool) if kinds is None else np.isin(kinds, TESTED)
    proven = np.zeros(frames.size, dtype=bool) if kinds is None else np.isin(kinds, (NOOP, DEAD))
    live = np.unique(frames[tested])
    near_live = np.zeros(frames.size, dtype=bool)
    if live.size:
        at = np.clip(np.searchsorted(live, frames), 1, live.size) - 1
        gap = np.minimum(np.abs(frames - live[at]), np.abs(frames - live[np.clip(at + 1, 0, live.size - 1)]))
        near_live = gap <= reach
    out = []
    for param in sorted(set(int(p) for p in params[tested | proven])):
        for pool, mask, rivals in (
            (ALL_POOL if kinds is None else LIVE_POOL, tested, tested),
            (DEAD_POOL, proven & ~near_live, tested | proven),
        ):
            mine = np.unique(frames[mask & (params == param)])
            if not mine.size:
                continue
            others = {(int(f), int(p)) for f, p in zip(frames[rivals], params[rivals]) if p != param}
            busy = np.array(sorted({f for f, _ in others}), dtype=np.int64)
            alone = mine[~np.isin(mine, busy)]
            partners: tuple[int, ...] = ()
            if not alone.size:
                alone, near = mine, set(mine.tolist())
                partners = tuple(sorted({p for f, p in others if f in near}))
            out.append((param, pool, mine, alone, partners))
    return out


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
) -> list[Moved]:
    """Test every parameter's steps against frames of matched source motion with no step.

    ``out`` and ``src`` are per clip frame; ``landed_frames``, ``params`` and ``kinds`` give each
    write's frame and the model's verdict on its step. With ``kinds``, a parameter's steps the
    model leaves live and those it proves no-ops are tested apart; steps that change no value are
    not tested. Family-wise over all tests.
    """
    d_out, d_src = frame_change(out), frame_change(src)
    count = d_out.size
    quantum = 1.0 / float(np.prod(np.shape(out)[1:]))
    frames = np.asarray(landed_frames, dtype=np.int64)
    kinds = None if kinds is None else np.asarray(kinds, dtype=object)
    inside = (frames >= 1) & (frames < count - max(lags))
    stepped = np.zeros(count, dtype=bool)
    for lag in (0, *lags):
        stepped[np.clip(frames[inside] + lag, 0, count - 1)] = True
    if kinds is not None:
        inside &= kinds != UNCHANGED
        kinds = kinds[inside]
    frames, params = frames[inside], np.asarray(params)[inside]
    measured = np.isfinite(d_out) & np.isfinite(d_src)
    controls = np.flatnonzero(~stepped & measured)
    null_res, null_spread = (
        matched_change(d_out, d_src, controls, controls, quantum)
        if controls.size > 1
        else (np.zeros(0), np.zeros(0))
    )
    null = null_res / null_spread if null_res.size else np.zeros(0)
    groups = _groups(frames, params, kinds, max(lags))
    limit = critical(len(groups) * len(lags))
    scatter = float(np.std(null_res, ddof=1)) if null_res.size > 1 else float("nan")
    verdicts = []
    for param, pool, mine, alone, partners in groups:
        best = (np.zeros(0), 0.0, -math.inf, lags[0], np.zeros(0))
        for lag in lags:
            x = alone + lag
            x = x[measured[x]]
            if not (x.size and null.size):
                continue
            residual, spread = matched_change(d_out, d_src, x, controls, quantum)
            p_e = tail_p(residual / spread, null)
            z, n_eff = fisher_z(p_e)
            if z > best[2]:
                best = (p_e, n_eff, z, lag, residual)
        p_e, n_eff, z, lag, residual = best
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
                flat=float(np.mean(luma_range[alone] <= FLAT_CODES)) if alone.size else float("nan"),
                jointly=int(mine.size - alone.size) if not partners else int(mine.size),
                partners=partners,
                effect=float(residual.mean()) if residual.size else float("nan"),
                scatter=scatter,
                pool=pool,
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

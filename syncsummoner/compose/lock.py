"""Exact attribution of a locked take: which capture shows each clip frame.

Each gap-free run's frame starts lie on a straight line on the capture clock (:mod:`frameline`),
named by the strips the dry lead-in and lead-out carry and fitted through every start the strips
or the captures' bytes show; each capture shows the frame its time falls in on that line.
"""

from __future__ import annotations

import json
import subprocess
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, Sequence

import numpy as np

from syncsummoner.compose import frameline, relock
from syncsummoner.probe import decoded
from syncsummoner.probe.patterns import read_codes
from syncsummoner.progress import LOG


class LockError(RuntimeError):
    """A take's captures cannot be attributed to clip frames exactly."""


class LostTimecodeError(RuntimeError):
    """A take carries no lock record, so nothing says which capture shows which clip frame."""


class AnchorError(LockError):
    """The lead-in's unprocessed captures name too few frames by strip to fix the frame line."""


#: Frames an unprocessed stretch must show for its strips to fix the frame line: a start is strip-named
#: only when the capture before it shows the frame before (:func:`anchors`), and a run is fixed only by
#: named starts on two frames (:func:`_lines`), so the frame before the first named one counts too.
ANCHOR_FRAMES = 3


def require_anchor(found: "Attribution", what: str) -> "Attribution":
    """``found`` once its lead-in's strips fixed its frame line; an :class:`AnchorError` naming ``what`` else.

    Nothing else numbers a session's frames, so a lead-in that leaves it to the Pi's flips is refused.
    """
    if not found.anchored:
        raise AnchorError(
            f"{what}: the lead-in's unprocessed captures name no two frames by strip, so they do not fix "
            f"the frame line; it must show {ANCHOR_FRAMES} consecutive legible strip-coded frames"
        )
    return found


@dataclass(frozen=True)
class Attribution:
    """Per clip frame, the capture that shows it; per capture, the clip frame it shows.

    ``picked[j]`` is the capture nearest clip frame ``j``'s centre on the line (the later on a tie),
    -1 where none shows it; ``shown[k]`` is the clip frame capture ``k`` shows, negative in the lead-in.
    ``faults`` are legible strips a frame off where a boundary may cross: what attribution can get wrong.
    ``span`` is the first and last capture of the run ``line`` is fitted to. ``ambiguity`` is, per capture,
    how many frames earlier than ``shown`` the clock also admits (:mod:`relock`), per run in ``clock``.
    """

    picked: np.ndarray
    shown: np.ndarray
    starts: np.ndarray
    per_frame: int
    lead: int
    line: frameline.Line
    capture_ns: float
    jitter_ns: float
    frame0_ns: float
    runs: int
    slips: int
    uncertain: int
    tied: int
    drops: int
    read: int
    agreed: int
    ahead: int
    faults: int
    strays: int
    rejected: int = 0
    span: tuple[int, int] = (0, -1)
    resolution: float = 0.0
    ambiguity: np.ndarray | None = None
    clock: tuple = ()
    #: Whether the strips fixed the first run's line, the lead-in's.
    anchored: bool = True
    #: The clip file's index of clip frame 0, when its lead-in is not ``lead``.
    source_lead: int | None = None

    @property
    def source_offset(self) -> int:
        """The clip file's index of clip frame 0: its own lead-in, which a session plays only the end of."""
        return self.lead if self.source_lead is None else self.source_lead

    @property
    def period_ns(self) -> float:
        """The frame period on the capture clock."""
        return self.line.period

    @property
    def drift_ppm(self) -> float:
        """How far the frame period runs from a whole number of capture periods, in parts per million."""
        return (self.line.period / (self.per_frame * self.capture_ns) - 1.0) * 1e6

    @property
    def drift_range_ppm(self) -> tuple[float, float]:
        """The least and greatest drift any line the brackets admit has."""
        least, most = self.line.polygon.periods
        unit = self.per_frame * self.capture_ns
        return (least / unit - 1.0) * 1e6, (most / unit - 1.0) * 1e6

    @property
    def spread_ms(self) -> float:
        """How far apart the lines the brackets admit put clip frame 0's start, in ms."""
        lo, hi = self.line.polygon.at(np.array([self.lead]))
        return float(hi[0] - lo[0]) / 1e6

    @property
    def behind(self) -> int:
        """Legible strips showing an earlier frame than the line: the program showing past frames."""
        return self.read - self.agreed - self.ahead

    @property
    def first_capture(self) -> int:
        """The first capture showing clip frame 0."""
        return int(np.flatnonzero(self.shown == 0)[0])

    def record(self) -> dict:
        """The frame line and how the take bears it out, JSON-ready."""
        least, most = self.drift_range_ppm
        return {
            "drift_ppm": self.drift_ppm,
            "drift_range_ppm": [least, most],
            "period_ns": self.period_ns,
            "capture_ns": self.capture_ns,
            "residual_ms": self.line.residual / 1e6,
            "jitter_ms": self.jitter_ns / 1e6,
            "frame0_spread_ms": self.spread_ms,
            "runs": self.runs,
            "slips": self.slips,
            "uncertain": self.uncertain,
            "tied": self.tied,
            "strips_read": self.read,
            "strips_agree": self.agreed,
            "strips_ahead": self.ahead,
            "strips_behind": self.behind,
            "faults": self.faults,
            "strays": self.strays,
            "rejected": self.rejected,
            "clock": list(self.clock),
        }

    @property
    def complete(self) -> bool:
        """Whether every clip frame has a capture."""
        return bool(self.picked.size) and bool(np.all(self.picked >= 0))

    def summary(self) -> str:
        """One line: frames covered, the line against the jitter, slips, boundary captures, strips, drops."""
        covered = int(np.count_nonzero(self.picked >= 0))
        return (
            f"{covered}/{self.picked.size} clip frames attributed over {self.runs} runs; frame line "
            f"{self.drift_ppm:+.1f} ppm ({self.drift_range_ppm[0]:+.1f}..{self.drift_range_ppm[1]:+.1f}) "
            f"off {self.per_frame} captures a frame, residual "
            f"{self.line.residual / 1e6:.3f} of {self.jitter_ns / 1e6:.3f} ms jitter, frame 0 within "
            f"{self.spread_ms:.3f} ms; {self.slips} slips; {self.uncertain} captures on a boundary, "
            f"{self.tied} tied; strips agree {self.agreed}/{self.read}, {self.ahead} ahead, "
            f"{self.behind} behind, {self.faults} a boundary off, {self.rejected} lead strips off the line; "
            f"{self.strays} byte changes off the line in the clip; {self.drops} frames lost by the driver"
        )


@dataclass(frozen=True)
class View:
    """One take of a session: its clip frames ``offset..offset+count-1`` of the session's, and the
    session captures it holds, ``captures``, rising.
    """

    offset: int
    count: int
    captures: np.ndarray

    def picked(self, picked: np.ndarray) -> np.ndarray:
        """The take's own capture showing each of its clip frames, -1 where the take holds none."""
        mine = picked[self.offset : self.offset + self.count]
        if not self.captures.size:
            return np.full(mine.size, -1, dtype=np.int64)
        at = np.clip(np.searchsorted(self.captures, mine), 0, self.captures.size - 1)
        return np.where((mine >= 0) & (self.captures[at] == mine), at, -1)

    def span(self, run: np.ndarray) -> tuple[int, int]:
        """The take's own first and last capture of a run given by session captures."""
        lo = int(np.searchsorted(self.captures, run[0])) if run.size else 0
        hi = int(np.searchsorted(self.captures, run[-1], side="right")) - 1 if run.size else -1
        return lo, hi


def displays(flips: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray, float]:
    """Each stretch a frame was up: start times, frames and ends, plus the vblank period, from a flip log."""
    t = flips["t_ns"].astype(np.float64)
    seg = (flips["flags"] & 1) != 0
    steps = np.diff(t)[~seg[1:]]
    vblank = float(np.median(steps)) if steps.size else float("nan")
    start = seg | np.r_[True, flips["frame"][1:] != flips["frame"][:-1]]
    idx = np.flatnonzero(start)
    ends = np.r_[t[idx[1:] - 1], t[-1]] + vblank
    return t[idx], flips["frame"][idx].astype(np.int64), ends, vblank


def signed(codes: np.ndarray, bits: int) -> np.ndarray:
    """Strip codes as clip frame indices, the lead-in's wrapped ones negative."""
    half = 1 << (bits - 1)
    return np.where(codes >= half, codes - (1 << bits), codes)


def capture_times(t_ns: np.ndarray, sequence: np.ndarray, seg: np.ndarray, period: float) -> np.ndarray:
    """Each capture's time on its run's capture line, its timestamp's jitter removed."""
    t, s = np.asarray(t_ns, dtype=np.float64), np.asarray(sequence, dtype=np.float64)
    counts = np.maximum(np.bincount(seg), 1)
    return (np.bincount(seg, t - period * s) / counts)[seg] + period * s


def changes(digest: np.ndarray) -> np.ndarray:
    """Captures whose bytes differ from the capture before's, outside any run of three such in a row.

    A frame lasting over one and a half captures cannot be captured once twice running, so a longer
    run of changes is the card's picture settling, not frames starting.
    """
    new = np.r_[False, digest[1:] != digest[:-1]]
    edges = np.flatnonzero(np.diff(np.r_[0, new.astype(np.int8), 0]))
    length = np.zeros(new.size + 1, dtype=np.int64)
    np.add.at(length, edges[::2], edges[1::2] - edges[::2])
    np.subtract.at(length, edges[1::2], edges[1::2] - edges[::2])
    return new & (np.cumsum(length)[:-1] < 3)


def anchors(played: np.ndarray, seg: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Strip-named frame starts, and each one's group: the run of starts on consecutive frames it is in.

    A program showing past frames' strips makes a group of its own, which the frame line then refuses.
    """
    k = np.arange(1, played.size)
    k = k[(seg[k] == seg[k - 1]) & (played[k - 1] >= 0) & (played[k - 1] == played[k] - 1)]
    out, group = np.full(played.size, -1, dtype=np.int64), np.full(played.size, -1, dtype=np.int64)
    out[k] = played[k]
    group[k] = np.cumsum(np.r_[True, (np.diff(played[k]) != 1) | (np.diff(seg[k]) != 0)])
    return out, group


def run_line(
    lo: np.ndarray,
    hi: np.ndarray,
    named: np.ndarray,
    jitter: float,
    period: tuple[float, float] | None,
    group: np.ndarray | None = None,
    resolution: float = 0.0,
) -> frameline.Line:
    """A run's frame line; strips missing every line past the capture jitter are a :class:`LockError`."""
    try:
        return frameline.fit(lo, hi, named, jitter, period=period, group=group, resolution=resolution)
    except frameline.LineError as err:
        raise LockError(str(err)) from err


def attribute(
    codes: np.ndarray,
    frames: np.ndarray,
    flips: np.ndarray,
    *,
    hold: int,
    lead: int,
    count: int,
    tail: int = 0,
    bits: int = 16,
    trusted: np.ndarray | None = None,
    view: View | None = None,
) -> Attribution:
    # A ``view`` attributes one take of the session whose captures these are.
    """Attribute every capture of a take to the clip frame it shows.

    ``codes`` are the captures' strip codes, ``frames`` their V4L2 records and ``flips`` the Pi's
    flip log; the clip played ``lead`` lead-in frames, ``count`` clip frames and ``tail`` lead-out.
    Only strips on captures ``trusted`` (all by default) are evidence: a program processing its input shows
    its own picture there, so those strips neither anchor the line nor count against it. A run with no strip
    the Pi's flips cannot number is left unattributed; the take stays exact only if the clip needs none of it.
    """
    seg = frameline.segments(frames["t_ns"], frames["sequence"])
    resolution = float(np.gcd.reduce(frames["t_ns"])) if frames.size else 0.0
    capture, jitter = frameline.capture_line(frames["t_ns"], frames["sequence"], seg)
    tau = capture_times(frames["t_ns"], frames["sequence"], seg, capture)
    starts, up, _, vblank = displays(flips)
    shown_at = np.full(int(up.max()) + 1 if up.size else 0, np.nan)
    shown_at[up[::-1]] = starts[::-1]
    clip = signed(np.asarray(codes, dtype=np.int64), bits)
    legible = (codes >= 0) & (clip >= -lead) & (clip < count + tail)
    if trusted is not None:
        legible &= np.asarray(trusted, dtype=bool)
    played = np.where(legible, clip + lead, -1)
    dry = np.where((clip < 0) | (clip >= count), played, -1)
    named, group = anchors(dry, seg)
    digest = frames["digest"]
    k = np.flatnonzero(np.r_[False, seg[1:] == seg[:-1]] & (changes(digest) | (named >= 0)))
    lines, clocks, fixed = _lines(
        tau, k, seg, (named, group), (jitter, resolution), (shown_at, hold * vblank)
    )
    main_run = max(lines, key=lambda s: int(np.count_nonzero(seg == s)))
    main = lines[main_run]
    shown = np.full(tau.size, -1 - lead, dtype=np.int64)
    uncertain = np.zeros(tau.size, dtype=bool)
    tie = np.zeros(tau.size, dtype=bool)
    for s, line in lines.items():
        at = np.flatnonzero(seg == s)
        n, on, ties = _frames(line, tau[at], digest[at], resolution)
        shown[at], uncertain[at], tie[at] = n - lead, on, ties
    begins = np.r_[True, (shown[1:] != shown[:-1]) | (seg[1:] != seg[:-1])]
    per_frame = max(1, int(round(main.period / capture)))
    picked = _pick(shown, tau, seg, lines, lead, count, resolution)
    diff = np.where(legible, clip - shown, 0)
    stray = np.zeros(tau.size, dtype=bool)
    stray[np.concatenate([k[seg[k] == s][line.strays] for s, line in lines.items()]).astype(np.int64)] = True
    view = View(0, count, np.arange(tau.size)) if view is None else view
    mine, first = view.captures, view.offset
    stray &= (shown >= first) & (shown < first + view.count)
    run_of = seg[mine]
    in_main = mine[seg[mine] == main_run]
    return Attribution(
        picked=view.picked(picked),
        shown=shown[mine] - first,
        starts=begins[mine],
        per_frame=per_frame,
        lead=lead + first,
        line=main,
        capture_ns=capture,
        jitter_ns=jitter,
        frame0_ns=float(main.origin + (lead + first) * main.period),
        runs=int(np.unique(run_of).size),
        slips=_slips({s: lines[s] for s in np.unique(run_of) if s in lines}, tau[mine], run_of, per_frame),
        uncertain=int(uncertain[mine].sum()),
        tied=int(tie[mine].sum()),
        drops=int(np.clip(np.diff(frames["sequence"][mine]) - 1, 0, None).sum()),
        read=int(legible[mine].sum()),
        agreed=int(np.count_nonzero(legible[mine] & (diff[mine] == 0))),
        ahead=int(np.count_nonzero(diff[mine] > 0)),
        faults=int(np.count_nonzero(uncertain[mine] & (np.abs(diff[mine]) == 1))),
        strays=int(np.count_nonzero(stray[mine])),
        rejected=int(sum(line.rejected for line in lines.values())),
        span=view.span(np.flatnonzero(seg == main_run) if not in_main.size else in_main),
        resolution=resolution,
        ambiguity=_ambiguity(seg, clocks)[mine],
        clock=tuple(
            {**said, "run": int(said["run"] - run_of[0])} for s, said in clocks.items() if s in set(run_of)
        ),
        anchored=seg[0] in fixed,
    )


def _ambiguity(seg: np.ndarray, clocks: dict) -> np.ndarray:
    """Per capture, how far below its run's numbering, the highest candidate, the lowest one lies."""
    out = np.zeros(seg.size, dtype=np.int64)
    for s, said in clocks.items():
        out[seg == s] = max(said["candidates"]) - min(said["candidates"])
    return out


def _lines(
    tau: np.ndarray, k: np.ndarray, seg: np.ndarray, anchored: tuple, error: tuple, pi: tuple
) -> tuple:
    """Each run's frame line through its starts ``k``, within ``error``, its clock numbering, the runs fixed.

    ``pi`` is ``(shown_at, period)``: when the Pi first showed each frame, and its frame period.

    The run with most strip-named frames fixes the period an outage leaves alone and the lag from the
    Pi showing a frame to its start; another run keeps its strips only if their largest group spans two
    frames and keeps that lag too, and is otherwise numbered by the lag.
    """
    named, group = anchored
    jitter, resolution = error
    shown_at, pi_period = pi
    runs = [int(s) for s in np.unique(seg[k])]
    own = {s: named[k[seg[k] == s]] for s in runs}

    def spans(s: int) -> bool:
        mine = k[seg[k] == s]
        ok = named[mine] >= 0
        if not ok.any():
            return False
        ids, sizes = np.unique(group[mine][ok], return_counts=True)
        largest = ok & (group[mine] == ids[np.argmax(sizes)])
        return np.unique(named[mine][largest]).size > 1

    fixed = sorted((s for s in runs if spans(s)), key=lambda s: -int((own[s] >= 0).sum()))
    if not fixed:
        raise AnchorError(
            "no run of captures carries two legible strip-coded frames, so nothing fixes the line; "
            f"an unprocessed stretch must show {ANCHOR_FRAMES} consecutive legible strip-coded frames"
        )

    def fitted(s: int, period: tuple[float, float] | None) -> frameline.Line:
        mine = k[seg[k] == s]
        return run_line(tau[mine - 1], tau[mine], own[s], jitter, period, group[mine], resolution)

    lines = {s: fitted(s, None) for s in fixed}
    period = lines[fixed[0]].polygon.periods
    for s in runs:
        if s not in lines:
            own[s] = np.full_like(own[s], -1)
            lines[s] = fitted(s, period)
    between = [s for s in runs if s not in fixed and min(fixed) < s < max(fixed)]
    rate = min(lines.values(), key=lambda line: np.ptp(line.polygon.periods)).period / pi_period
    lag = _lag(lines[fixed[0]], shown_at, rate)
    for s in fixed[1:]:
        mine = _lag(lines[s], shown_at, rate)
        if max(lag[0], mine[0]) <= min(lag[1], mine[1]):
            lag = (max(lag[0], mine[0]), min(lag[1], mine[1]))
            continue
        if between:
            LOG.warning(
                "captures in run %d carry strips off the Pi lag of the rest; the clock numbers the take", s
            )
            continue
        LOG.warning(
            "captures in run %d carry strips off the Pi lag of the rest; counting its frames instead", s
        )
        own[s] = np.full_like(own[s], -1)
        lines[s] = fitted(s, period)
    clocks = _clocked(lines, between, fixed, [tau, k, seg, own], resolution)
    anchored = [s for s in fixed if (own[s] >= 0).any()]
    for s in runs:
        if not (own[s] >= 0).any() and s not in clocks:
            try:
                lines[s] = _bridge(lines[s], lag, rate, shown_at)
            except LockError as err:
                LOG.warning("captures in run %d are left unattributed: %s", s, err)
                del lines[s]
    return lines, clocks, anchored


def _clocked(lines: dict, between: list[int], fixed: list[int], seen: list, resolution: float) -> dict:
    """Number each strip-less run between strip-fixed ones by the capture clock (:mod:`relock`).

    ``seen`` is ``[tau, k, seg, own]``: capture times, frame starts, runs, strip-named frames. The period
    range is the take's, every run's admitted range intersected. Each run is numbered by its highest
    candidate; its record says every candidate, the latencies at the middle period, and the frame a gap's
    rounding would have named, which is reported only. A run no shift fits fails the take if it is the
    longest such run, and is otherwise left unattributed.
    """
    tau, k, seg, own = seen
    lows, highs = zip(*(lines[s].polygon.periods for s in [*fixed, *between]))
    periods = (max(lows), min(highs))
    if periods[0] > periods[1]:
        raise LockError("the runs' frame lines admit no common period")
    t = 0.5 * (periods[0] + periods[1])

    def brackets(s: int, n: np.ndarray) -> relock.Brackets:
        mine = k[seg[k] == s]
        return relock.Brackets.of(tau[mine - 1] + resolution, tau[mine], n)

    out = {}
    longest = max(between, key=lambda s: int(np.count_nonzero(seg == s)), default=None)
    for s in between:
        ends = [brackets(f, own[f]) for f in (max(f for f in fixed if f < s), min(f for f in fixed if f > s))]
        line = lines[s]
        mid = brackets(s, line.frame)
        found = relock.candidates(ends, mid, periods)
        said = {
            "run": s,
            "candidates": found,
            "periods_ns": list(periods),
            "latency_before_ns": list(relock.latency(ends[0], t)),
            "latency_after_ns": list(relock.latency(ends[1], t)),
            "latency_ns": list(relock.latency(mid, t)),
            "gap_count": int(ends[0].n.max() + np.rint((mid.lo.min() - ends[0].hi.max()) / t)),
        }
        if not found:
            if s == longest:
                raise LockError(f"the capture clock numbers run {s} no way: {said}")
            LOG.warning("captures in run %d are left unattributed: the capture clock numbers it no way", s)
            del lines[s]
            continue
        shift = max(found)
        said["latency_ns"] = list(relock.latency(mid, t, shift))
        ok = line.frame >= 0
        lines[s] = frameline.Line(
            origin=line.origin - shift * line.period,
            period=line.period,
            slack=line.slack,
            residual=line.residual,
            frame=np.where(ok, line.frame + shift, line.frame),
            polygon=frameline.Polygon(line.polygon.ref + shift, line.polygon.vertices),
            tied=line.tied,
        )
        out[s] = said
    return out


def _lag(line: frameline.Line, shown_at: np.ndarray, rate: float) -> tuple[float, float]:
    """The lag from the Pi showing a frame to its start that a line admits over its frames.

    ``rate`` turns the Pi's clock into the capture clock.
    """
    up = np.flatnonzero(np.isfinite(shown_at))
    n = up[(up >= line.frame[line.frame >= 0].min()) & (up <= line.frame.max())]
    least, most = line.polygon.at(n)
    return float(np.max(least - rate * shown_at[n], initial=-np.inf)), float(
        np.min(most - rate * shown_at[n], initial=np.inf)
    )


def _bridge(
    line: frameline.Line, lag: tuple[float, float], rate: float, shown_at: np.ndarray
) -> frameline.Line:
    """A run with no strip, counted from 0, renumbered by the one shift whose Pi lag meets ``lag``."""
    lo, hi = lag
    ok = line.frame >= 0
    span = np.arange(line.frame[ok].min(), line.frame[ok].max() + 1)
    least, most = line.polygon.at(span)
    up = np.flatnonzero(np.isfinite(shown_at))
    centre = int(up[np.argmin(np.abs(rate * shown_at[up] + 0.5 * (lo + hi) - 0.5 * (least[0] + most[0])))])
    shifts = []
    for shift in range(centre - int(span[0]) - 1, centre - int(span[0]) + 2):
        n = span + shift
        inside = (n >= 0) & (n < shown_at.size)
        inside[inside] = np.isfinite(shown_at[n[inside]])
        p = rate * shown_at[n[inside]]
        if inside.any() and np.max(least[inside] - p) <= hi and np.min(most[inside] - p) > lo:
            shifts.append(shift)
    if len(shifts) != 1:
        raise LockError(f"the Pi's flip log numbers a run with no strip {len(shifts)} ways across its outage")
    shift = shifts[0]
    return frameline.Line(
        origin=line.origin - shift * line.period,
        period=line.period,
        slack=line.slack,
        residual=line.residual,
        frame=np.where(ok, line.frame + shift, line.frame),
        polygon=frameline.Polygon(line.polygon.ref + shift, line.polygon.vertices),
        tied=line.tied,
    )


def _frames(
    line: frameline.Line, tau: np.ndarray, digest: np.ndarray, resolution: float
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Each capture's frame: the side of its nearest boundary every admitted line puts it, else its bytes'.

    The admitted range is widened by the line's slack, and a capture within the timestamps' ``resolution``
    of it is on its side. Bytes equal to the capture before's are its frame, to the next's the next;
    equal to both or neither, any choice measures alike and the line's stands, marked in the mask returned.
    """
    b = np.rint((tau - line.origin) / line.period).astype(np.int64)
    lo, hi = line.polygon.at(b)
    new = hi + line.slack <= tau + resolution
    old = lo - line.slack >= tau - resolution
    on = ~(new | old)
    same_before = np.r_[False, digest[1:] == digest[:-1]]
    same_after = np.r_[digest[:-1] == digest[1:], False]
    sure = on & (same_before != same_after)
    n = np.where(
        new | (sure & same_after), b, np.where(old | (sure & same_before), b - 1, line.frame_of(tau))
    )
    return n, on, on & ~sure


def _pick(
    shown: np.ndarray, tau: np.ndarray, seg: np.ndarray, lines: dict, lead: int, count: int, resolution: float
) -> np.ndarray:
    """Per clip frame, the capture showing it nearest its centre on its run's line.

    The later wins a tie within the timestamps' ``resolution``.
    """
    picked = np.full(count, -1, dtype=np.int64)
    k = np.flatnonzero((shown >= 0) & (shown < count) & np.isin(seg, list(lines)))
    if not k.size:
        return picked
    origin = np.array([lines[int(s)].origin for s in seg[k]])
    period = np.array([lines[int(s)].period for s in seg[k]])
    away = np.abs(tau[k] - (origin + (shown[k] + lead + 0.5) * period))
    away = np.floor(away / resolution) if resolution > 0 else away
    best = k[np.lexsort((-k, away, shown[k]))]
    first = np.r_[True, shown[best][1:] != shown[best][:-1]]
    picked[shown[best][first]] = best[first]
    return picked


def _slips(lines: dict, tau: np.ndarray, seg: np.ndarray, per_frame: int) -> int:
    """Frames a run's line captures other than ``per_frame`` times, its end frames aside: the card's slips."""
    total = 0
    for s, line in lines.items():
        _, counts = np.unique(line.frame_of(tau[seg == s]), return_counts=True)
        total += int(np.count_nonzero(counts[1:-1] != per_frame))
    return total


def lock_path(take: str | Path) -> Path:
    """Where a take's lock record sits: the Pi's flip log and what the clip played."""
    return Path(f"{take}.lock.npz")


def picked_path(take: str | Path) -> Path:
    """Where a take's attribution sits: per clip frame, the capture index showing it."""
    return Path(f"{take}.picked.npy")


def save_lock(
    take: str | Path,
    flips: np.ndarray,
    *,
    hold: int,
    lead: int,
    count: int,
    tail: int = 0,
    wet: bool = False,
    **session: Any,
) -> None:
    """Keep what attribution needs from the Pi next to the take; ``wet`` when a program played the clip.

    A take of a session also keeps the session's own (:func:`session_record`), ``lead``, ``count``
    and ``tail`` then being the session's.
    """
    np.savez(lock_path(take), flips=flips, hold=hold, lead=lead, count=count, tail=tail, wet=wet, **session)


def session_record(said: dict, log: np.ndarray, codes: np.ndarray, span: tuple[float, float] | None) -> dict:
    """A session's share of each take's lock record: ``said`` (its id, this take's index, every take's
    clip frame offset and count, the clip's own lead-in), the session's whole capture log, the strip code
    read at each capture (its anchors; -1 where none was read), and when a program processed the picture.
    """
    return {
        "session": np.array(json.dumps({**said, "processed": span})),
        "session_log": log,
        "session_codes": np.asarray(codes, dtype=np.int64),
    }


def session_log(logs: Sequence[np.ndarray]) -> np.ndarray:
    """One capture log from a session's takes' logs: every capture once, in the card's order.

    Takes recorded back to back share the captures either side of each roll (:mod:`~.v4l2rec`).
    """
    whole = np.concatenate(list(logs)) if logs else np.zeros(0)
    return whole[np.unique(whole["sequence"], return_index=True)[1]] if whole.size else whole


def save_clock(take: str | Path, clock: Sequence[dict]) -> None:
    """Keep the capture clock's numbering of the take's runs in its lock record, every other key as it was."""
    with np.load(lock_path(take)) as record:
        kept = {key: record[key] for key in record.files if key != "clock"}
    np.savez(lock_path(take), **kept, clock=np.array(json.dumps(list(clock))))


def read_clock(take: str | Path) -> list[dict]:
    """The capture clock's numbering of the take's runs as its lock record keeps it; none if never judged."""
    with np.load(lock_path(take)) as record:
        return json.loads(str(record["clock"])) if "clock" in record.files else []


def _trusted(take: str | Path, record: Any, frames: np.ndarray) -> np.ndarray | None:
    if "session" in record.files:
        span = json.loads(str(record["session"]))["processed"]
    else:
        span = processed(take, record, frames)
    if span is None:
        return None
    t = frames["t_ns"].astype(np.float64)
    return (t < span[0]) | (t > span[1])


def processed(take: str | Path, record: Any, frames: np.ndarray) -> tuple[float, float] | None:
    """When a program processed the take's picture, ``(from, to)`` on the host clock; None if it never did.

    From the first write leaving the dry state, or the program's load command, to the close of the
    landing window of the write or reload making it dry again: that write's echo, or the reload's
    return, then the latency band and the frame it lands on. A take without those times on file, a wet
    lockcheck's, is processed from the end of its first run, the load's outage, to the start of its last
    when it played a lead-out, the reload's, and to its end when it did not.
    """
    from syncsummoner.device.wire import load_wire, wire_path  # pylint: disable=import-outside-toplevel

    wet = bool(record["wet"]) if "wet" in record else wire_path(take).is_file()
    if not wet:
        return None
    if wire_path(take).is_file():
        wire = load_wire(take)
        span = _written_span(wire, int(record["count"]))
        if span is not None:
            return span
    seg = frameline.segments(frames["t_ns"], frames["sequence"])
    t = frames["t_ns"].astype(np.float64)
    reloaded = int(record["tail"]) > 0 if "tail" in record else False
    return float(t[seg == seg[0]][-1]), (
        float(t[seg == seg[-1]][0]) if reloaded and seg[-1] != seg[0] else np.inf
    )


def _written_span(wire: Any, count: int) -> tuple[float, float] | None:
    """:func:`processed` from a wire record: the dry control's writes, else the loads it recorded."""
    from syncsummoner.compose.evidence import START  # pylint: disable=import-outside-toplevel
    from syncsummoner.device.wire import echoed  # pylint: disable=import-outside-toplevel

    lat = json.loads(str(wire["latency"])) if "latency" in wire else {}
    if "hi" not in lat:
        return None
    after = float(lat["hi"]) + float(lat["period"])
    loads = json.loads(str(wire["loads"])) if "loads" in wire else []
    if json.loads(str(wire["dry"])) is None:
        if len(loads) < 2:
            return None
        return float(loads[0]["t0_ns"]), float(loads[-1]["t1_ns"]) + after
    sends = wire["sends"]
    moved = np.flatnonzero(sends["target"] != START)
    back = np.flatnonzero(sends["target"] == count)
    if not moved.size or not back.size:
        return None
    took = echoed(sends, wire["receipts"], wire["reads"], wire["cc_map"], read_errors=wire.get("read_errors"))
    end = took[back[-1], 1]
    end = float(sends["t1_ns"][back[-1]]) if not np.isfinite(end) else float(end)
    return float(sends["t0_ns"][moved[0]]), end + after


def attribute_take(
    take: str | Path,
    *,
    config: Any,
    codes: np.ndarray | None = None,
    features: decoded.Features | None = None,
) -> Attribution:
    """The take's :class:`~syncsummoner.compose.lock.Attribution`, checked exact; ``picked`` saved beside it.

    Strips read as ``codes``, else as the take's :class:`~syncsummoner.probe.decoded.Features` hold them.
    Every clip frame must have a capture, and no strip at a boundary capture nor byte change in the
    clip may contradict the line; anything less is a :class:`~syncsummoner.compose.lock.LockError`.
    """
    from syncsummoner.device.v4l2rec import read_log  # pylint: disable=import-outside-toplevel

    try:
        record = np.load(lock_path(take))
    except FileNotFoundError as err:
        raise LostTimecodeError(f"{take}: no lock record, so it was not played by the locked player") from err
    if codes is None:
        if features is None:
            features = decoded.features(
                take, decoded.Spec.of(config), ffmpeg=getattr(config, "ffmpeg", "ffmpeg")
            )
        codes = features.codes
    frames = read_log(take)
    if "session" in record.files:
        found = _session_take(take, record, codes, frames, bits=config.bits)
    else:
        found = attribute(
            codes,
            frames,
            record["flips"],
            hold=int(record["hold"]),
            lead=int(record["lead"]),
            count=int(record["count"]),
            tail=int(record["tail"]) if "tail" in record else 0,
            bits=config.bits,
            trusted=_trusted(take, record, frames),
        )
    LOG.info("%s: %s", take, found.summary())
    if not found.complete:
        raise LockError(f"{take}: {found.summary()}")
    if found.faults or found.strays:
        raise LockError(f"{take}: the strips or bytes contradict the frame line: {found.summary()}")
    np.save(picked_path(take), found.picked)
    return found


def _session_take(
    take: str | Path, record: Any, codes: np.ndarray, frames: np.ndarray, *, bits: int
) -> Attribution:
    """A session take's attribution: the session's frame line, through every capture of the session.

    The take's own strips stand where the session read none; a lead-out strip, read as the last take's
    clip frame, is renumbered to the session's. The lead-in's strips must fix it (:func:`require_anchor`).
    """
    said = json.loads(str(record["session"]))
    log = record["session_log"]
    whole = np.array(record["session_codes"], dtype=np.int64)
    at = np.searchsorted(log["sequence"], frames["sequence"])
    if at.size and (at.max() >= log.size or not np.array_equal(log["sequence"][at], frames["sequence"])):
        raise LockError(f"{take}: its captures are not the session's {said['id']}")
    own = np.asarray(codes, dtype=np.int64)[: at.size]
    whole[at[: own.size]] = np.where(own >= 0, own, whole[at[: own.size]])
    t = log["t_ns"].astype(np.float64)
    span = said["processed"]
    trusted = None if span is None else (t < span[0]) | (t > span[1])
    after = (whole >= 0) & (signed(whole, bits) >= 0) & (t > (np.inf if span is None else span[1]))
    whole[after] = (whole[after] + said["offsets"][-1]) & ((1 << bits) - 1)
    index = int(said["index"])
    found = attribute(
        whole,
        log,
        record["flips"],
        hold=int(record["hold"]),
        lead=int(record["lead"]),
        count=int(record["count"]),
        tail=int(record["tail"]),
        bits=bits,
        trusted=trusted,
        view=View(int(said["offsets"][index]), int(said["counts"][index]), at),
    )
    return replace(
        require_anchor(found, f"{take} (session {said['id']})"), source_lead=int(said["source_lead"])
    )


def read_live(
    path: str | Path, *, strip_px: int, bits: int, size: tuple[int, int], ffmpeg: str = "ffmpeg"
) -> tuple[np.ndarray, np.ndarray]:
    """One pass over a recording as far as it is written: strip codes, and the body small in planar YUV.

    The strip is area-scaled to the body's width, which keeps every cell several samples wide.
    """
    width, height = size
    graph = (
        f"[0:v]split[a][b];[a]crop=iw:ih-{strip_px}:0:0,scale={width}:{height}:flags=area[s];"
        f"[b]crop=iw:{strip_px}:0:ih-{strip_px},scale={width}:{strip_px}:flags=area[t];"
        "[s][t]vstack,format=yuv444p[o]"
    )
    argv = [ffmpeg, "-loglevel", "quiet", "-i", str(path), "-filter_complex", graph, "-map", "[o]"]
    argv += ["-fps_mode", "passthrough", "-f", "rawvideo", "pipe:1"]
    data = subprocess.run(argv, check=False, capture_output=True).stdout
    rows = height + strip_px
    count = len(data) // (3 * width * rows)
    stack = np.frombuffer(data, np.uint8, count * 3 * width * rows).reshape(count, 3, rows, width)
    strip = np.repeat(stack[:, 0, height:, :, None], 3, axis=3) / np.float32(255.0)
    return read_codes(strip, bits=bits), stack[:, :, :height]


def live_attribution(
    take: str | Path, flips: np.ndarray, codes: np.ndarray, *, config: Any, hold: int, lead: int, count: int
) -> tuple:
    """A take's attribution so far, read while it records: ``(attribution, frame log)``.

    The captures the recording has stored are attributed as a finished take's would be; the
    frame log is cut to them, since the card's sidecar runs ahead of the encoder's file.
    """
    from syncsummoner.device.v4l2rec import read_log  # pylint: disable=import-outside-toplevel

    frames = read_log(take)
    size = min(codes.size, frames.size)
    LOG.info(
        "%s: live attribution over %d captures (%d decoded, %d logged), %d flips",
        take,
        size,
        codes.size,
        frames.size,
        flips.size,
    )
    found = attribute(codes[:size], frames[:size], flips, hold=hold, lead=lead, count=count, bits=config.bits)
    return require_anchor(found, str(take)), frames[:size]


def take_start(take: str | Path, *, config: Any) -> float:
    """Seconds into a take at which clip frame 0's first capture lands, from its attribution."""
    found = attribute_take(take, config=config)
    return float(found.first_capture) / (found.per_frame * config.fps)

"""Exact attribution of a locked take: which capture shows each clip frame.

Within a run of captures with no gap the card's cadence says which frame each shows; the
strip-coded lead-in fixes the first run's offset, and capture timestamps against the Pi's
logged flip times carry it across the outage a program load causes.
"""

from __future__ import annotations

import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from syncsummoner.probe import decoded
from syncsummoner.probe.patterns import read_codes
from syncsummoner.progress import LOG

#: A capture interval past the midpoint between one period and two is a lost frame or an outage.
GAP_PERIODS = 1.5


class LockError(RuntimeError):
    """A take's captures cannot be attributed to clip frames exactly."""


class LostTimecodeError(RuntimeError):
    """A take carries no lock record, so nothing says which capture shows which clip frame."""


@dataclass(frozen=True)
class Run:
    """Captures ``first..last`` (inclusive) with no gap between them, so one cadence holds them.

    ``slot`` is the sub-frame slot capture ``first`` shows, ``per_frame * played + copy``.
    ``repeats`` is the fraction of the card's frame changes on that cadence, ``agree`` the
    fraction of frame starts whose timestamp gives that slot, ``worst_ms`` their largest error.
    """

    first: int
    last: int
    slot: int
    repeats: float
    agree: float
    worst_ms: float

    def __len__(self) -> int:
        return self.last - self.first + 1


@dataclass(frozen=True)
class Attribution:
    """Per clip frame, the capture that shows it; per capture, the clip frame it shows.

    ``picked[j]`` is the capture nearest clip frame ``j``'s centre (the later on a tie), -1
    where none shows it. ``shown[k]`` is the clip frame capture ``k`` shows, negative in the
    lead-in. ``frame0_ns`` is when clip frame 0 reached the card, on the capture clock.
    """

    picked: np.ndarray
    shown: np.ndarray
    per_frame: int
    lead: int
    runs: tuple[Run, ...]
    anchor: int
    offset_ns: float
    period_ns: float
    frame0_ns: float
    drops: int
    read: int
    agreed: int

    @property
    def run(self) -> Run:
        """The run clip frame 0 is shown in."""
        k = int(self.picked[0]) if self.picked.size and self.picked[0] >= 0 else -1
        return next((r for r in self.runs if r.first <= k <= r.last), self.runs[-1])

    @property
    def complete(self) -> bool:
        """Whether every clip frame has a capture."""
        return bool(self.picked.size) and bool(np.all(self.picked >= 0))

    @property
    def margin_ms(self) -> float:
        """How far a frame start's timestamp may stray before it names the wrong frame: half a frame."""
        return self.period_ns / 2e6

    def summary(self) -> str:
        """One line: frames covered, runs, bridge margin, strip agreement and drops."""
        covered = int(np.count_nonzero(self.picked >= 0))
        run = self.run
        return (
            f"{covered}/{self.picked.size} clip frames attributed at {self.per_frame} captures a frame, "
            f"{len(self.runs)} runs, clip in captures {run.first}..{run.last} (repeats on cadence "
            f"{run.repeats:.1%}, timestamps agree {run.agree:.1%}, worst {run.worst_ms:.2f} of "
            f"{self.margin_ms:.2f} ms), strips agree "
            f"{self.agreed}/{self.read}, {self.drops} frames lost by the driver"
        )


def runs_of(t_ns: np.ndarray, sequence: np.ndarray) -> tuple[list[tuple[int, int]], float]:
    """Gap-free runs of captures, and the capture period; a sequence gap or a long interval breaks one."""
    dt = np.diff(t_ns.astype(np.float64))
    if not dt.size:
        return ([(0, t_ns.size - 1)] if t_ns.size else []), float("nan")
    period = float(np.median(dt))
    breaks = np.flatnonzero((dt > GAP_PERIODS * period) | (np.diff(sequence) != 1)) + 1
    edges = np.r_[0, breaks, t_ns.size]
    return [(int(a), int(b) - 1) for a, b in zip(edges[:-1], edges[1:])], period


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


def anchor_run(runs: list[tuple[int, int]], played: np.ndarray, per_frame: int) -> tuple[int, int]:
    """The run with most strips on one cadence, and the slot its first capture shows."""
    best = (-1, 0, 0)
    for index, (first, last) in enumerate(runs):
        ok = played[first : last + 1] >= 0
        if not ok.any():
            continue
        q = np.arange(first, last + 1)[ok] - per_frame * played[first : last + 1][ok]
        values, counts = np.unique(q, return_counts=True)
        window = np.array([counts[(values >= v) & (values < v + per_frame)].sum() for v in values])
        at = int(window.argmax())
        if window[at] > best[2]:
            best = (index, first - int(values[at]), int(window[at]))
    if best[0] < 0:
        raise LockError("no run of captures carries a legible strip, so nothing fixes the take's offset")
    return best[0], best[1]


def repeat_phase(new: np.ndarray, per_frame: int) -> tuple[int | None, float]:
    """The copy index a run's frames change at, and the fraction of changes there; None if none change."""
    at = np.flatnonzero(new)
    if not at.size:
        return None, float("nan")
    counts = np.bincount(at % per_frame, minlength=per_frame)
    return int(counts.argmax()), float(counts.max()) / at.size


def anchor_offset(
    t: np.ndarray, run: tuple[int, int], slot: int, per_frame: int, first_up: dict, phase: int | None
) -> float:
    """Capture time minus Pi flip time, for the frame starts of the anchoring run."""
    first, last = run
    if phase is not None and (phase + 1 + slot) % per_frame:
        raise LockError(
            f"the strips put frame starts at copy {(-slot) % per_frame} of the anchoring run, "
            f"the card's repeats at copy {(phase + 1) % per_frame}"
        )
    n = np.arange(last - first + 1) + slot
    starting = n % per_frame == 0
    t_frame = np.array([first_up.get(int(p), np.nan) for p in n[starting] // per_frame])
    x = t[first : last + 1][starting] - t_frame
    if not np.isfinite(x).any():
        raise LockError("the anchoring run shows no frame the Pi logged a flip for")
    return float(np.nanmedian(x))


def bridge(
    span: tuple[int, int],
    guess: np.ndarray,
    error: np.ndarray,
    new: np.ndarray,
    per_frame: int,
    period: float,
) -> tuple[int, float, np.ndarray, np.ndarray]:
    """A run's slot from its frame-starting captures' timestamps, the card's repeats saying which those are.

    Returns the slot, the fraction of repeats on its cadence, and each frame start's slot
    offset and timing error.
    """
    first, last = span
    if per_frame < 2:
        raise LockError("the card takes one capture a frame, so nothing bridges a gap in the captures")
    phase, repeats = repeat_phase(new[first + 1 : last + 1], per_frame)
    k = np.arange(first, last + 1)
    starting = (k - first) % per_frame == ((phase + 1) % per_frame if phase is not None else 0)
    rel = guess[first : last + 1][starting] - (k - first)[starting]
    err = error[first : last + 1][starting]
    live = np.abs(err) < period
    values, counts = np.unique(rel[live] if live.any() else rel, return_counts=True)
    return int(values[counts.argmax()]), repeats, rel, err


def attribute(
    codes: np.ndarray,
    frames: np.ndarray,
    flips: np.ndarray,
    *,
    hold: int,
    lead: int,
    count: int,
    bits: int = 16,
) -> Attribution:
    """Attribute every capture of a take to the clip frame it shows.

    ``codes`` are the captures' strip codes, ``frames`` their V4L2 records and ``flips`` the
    Pi's flip log; the clip played ``lead`` lead-in frames then ``count`` clip frames.
    """
    t = frames["t_ns"].astype(np.float64)
    runs, capture_ns = runs_of(frames["t_ns"], frames["sequence"])
    starts, up, _, vblank = displays(flips)
    period = hold * vblank
    per_frame = max(1, int(round(period / capture_ns)))
    clip = signed(np.asarray(codes, dtype=np.int64), bits)
    played = np.where((codes >= 0) & (clip >= -lead) & (clip < count), clip + lead, -1)
    led = np.where(clip < 0, played, -1)
    anchor, slot0 = anchor_run(runs, led if (led >= 0).any() else played, per_frame)
    new = np.r_[False, frames["digest"][1:] != frames["digest"][:-1]]
    first_up = {int(f): s for f, s in zip(up[::-1], starts[::-1])}
    a_first, a_last = runs[anchor]
    phase, a_repeats = repeat_phase(new[a_first + 1 : a_last + 1], per_frame)
    offset = anchor_offset(t, runs[anchor], slot0, per_frame, first_up, phase)
    tau = t - offset
    right = np.clip(np.searchsorted(starts, tau), 1, starts.size - 1)
    nearest = np.where(np.abs(tau - starts[right - 1]) <= np.abs(starts[right] - tau), right - 1, right)
    error, guess = tau - starts[nearest], per_frame * up[nearest]
    error[(tau < starts[0] - period / 2) | (tau > starts[-1] + period / 2)] = np.inf
    out_runs, shown = [], np.empty(t.size, dtype=np.int64)
    for index, (first, last) in enumerate(runs):
        if index == anchor:
            slot, repeats, agree, worst = slot0, a_repeats, 1.0, 0.0
        else:
            slot, repeats, rel, err = bridge((first, last), guess, error, new, per_frame, period)
            live = np.abs(err) < period
            same = live & (rel == slot)
            agree = float(same.sum()) / max(1, int(live.sum()))
            worst = float(np.abs(err[same]).max()) / 1e6 if same.any() else 0.0
        out_runs.append(Run(first, last, slot, repeats, agree, worst))
        shown[first : last + 1] = (slot + np.arange(last - first + 1)) // per_frame - lead
    picked = np.full(count, -1, dtype=np.int64)
    j = np.arange(count)
    for run in out_runs:
        k = run.first + per_frame * (j + lead) + per_frame // 2 - run.slot
        inside = (k >= run.first) & (k <= run.last)
        picked[inside] = k[inside]
    readable = played >= 0
    return Attribution(
        picked=picked,
        shown=shown,
        per_frame=per_frame,
        lead=lead,
        runs=tuple(out_runs),
        anchor=anchor,
        offset_ns=offset,
        period_ns=period,
        frame0_ns=float(first_up.get(lead, np.nan) + offset),
        drops=int(np.clip(np.diff(frames["sequence"]) - 1, 0, None).sum()),
        read=int(readable.sum()),
        agreed=int(np.count_nonzero(readable & (shown == clip))),
    )


def lock_path(take: str | Path) -> Path:
    """Where a take's lock record sits: the Pi's flip log and what the clip played."""
    return Path(f"{take}.lock.npz")


def picked_path(take: str | Path) -> Path:
    """Where a take's attribution sits: per clip frame, the capture index showing it."""
    return Path(f"{take}.picked.npy")


def save_lock(take: str | Path, flips: np.ndarray, *, hold: int, lead: int, count: int) -> None:
    """Keep what attribution needs from the Pi next to the take."""
    np.savez(lock_path(take), flips=flips, hold=hold, lead=lead, count=count)


def attribute_take(
    take: str | Path,
    *,
    config: Any,
    codes: np.ndarray | None = None,
    features: decoded.Features | None = None,
) -> Attribution:
    """The take's :class:`~syncsummoner.compose.lock.Attribution`, checked exact; ``picked`` saved beside it.

    Strips read as ``codes``, else as the take's :class:`~syncsummoner.probe.decoded.Features` hold them.
    Every clip frame must have a capture, inside one gap-free run whose frame starts all agree on
    its cadence; anything less is a :class:`~syncsummoner.compose.lock.LockError`.
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
    found = attribute(
        codes,
        read_log(take),
        record["flips"],
        hold=int(record["hold"]),
        lead=int(record["lead"]),
        count=int(record["count"]),
        bits=config.bits,
    )
    LOG.info("%s: %s", take, found.summary())
    if not found.complete:
        raise LockError(f"{take}: {found.summary()}")
    if found.run.agree < 1.0:
        raise LockError(f"{take}: the clip's frame starts disagree on its cadence: {found.run}")
    np.save(picked_path(take), found.picked)
    return found


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
    return found, frames[:size]


def take_start(take: str | Path, *, config: Any) -> float:
    """Seconds into a take at which clip frame 0's first capture lands, from its attribution."""
    found = attribute_take(take, config=config)
    rate = found.per_frame * config.fps
    return float(found.picked[0] - found.per_frame // 2) / rate

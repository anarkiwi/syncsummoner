"""The hard constraint on a plan's writes: each parameter settles between them; only shown ones are sent."""

from __future__ import annotations

from typing import Sequence

import numpy as np

from syncsummoner.device.profile import ProgramProfile
from syncsummoner.compose.vocabulary import Automation


def reachable(auto: Automation, profile: ProgramProfile, *, fps: float) -> bool:
    """Hard constraint: every parameter's writes no closer together than its settle time."""
    if len(auto) == 0:
        return True
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


def settled(
    times: np.ndarray, values: np.ndarray, settle: float, state: tuple = (-np.inf, None, None)
) -> tuple[list[float], list[int], tuple]:
    """One parameter's writes no closer together than ``settle``, each the latest value asked for.

    A write that comes too soon is deferred rather than dropped: dropping the release
    of a pulse leaves the parameter latched wherever the pulse put it. ``state`` is the
    scan's ``(due, last written, pending)``, returned too so a later run of requests can
    continue it; the pending write it may hold is :func:`flush`'s.
    """
    out_t: list[float] = []
    out_v: list[int] = []
    due, last, pending = state
    for t, v in zip(times.tolist(), values.tolist()):
        if pending is not None and due <= t:
            out_t.append(due)
            out_v.append(pending)
            due, last, pending = due + settle, pending, None
        if t >= due:
            if v != last:
                out_t.append(t)
                out_v.append(v)
                due, last = t + settle, v
            pending = None
        else:
            pending = v if v != last else None
    return out_t, out_v, (due, last, pending)


def flush(state: tuple) -> tuple[list[float], list[int]]:
    """The write a scan still holds back, at the time it may land."""
    due, _, pending = state
    return ([due], [pending]) if pending is not None else ([], [])


def settle_of(profile: ProgramProfile, index: int, fps: float) -> float:
    """A parameter's measured settle time in seconds."""
    return profile.settle_frames.get(int(index), 0) / fps


def thin(auto: Automation, profile: ProgramProfile, *, fps: float) -> Automation:
    """Defer writes so the result satisfies the hard constraint.

    Writes a parameter cannot settle between are deferred to the latest value, so the held state
    matches what the gestures asked for. The link is paced by its echoes, not by a rate here.
    """
    if len(auto) == 0:
        return auto
    parts = []
    for idx in np.unique(auto.indices):
        m = auto.indices == idx
        t, v, state = settled(auto.times[m], auto.values[m], settle_of(profile, idx, fps))
        late_t, late_v = flush(state)
        t, v = t + late_t, v + late_v
        parts.append(Automation(np.asarray(t), np.full(len(t), idx, np.int32), np.asarray(v, np.int32)))
    return Automation.concat(parts)


def visible(auto: Automation, spans: Sequence[tuple[float, float]] | None) -> Automation:
    """Only the writes inside ``spans``, half-open and disjoint; ``None`` keeps every write.

    A take is on screen only in its cuts, so a write anywhere else lands unseen and changes
    nothing the next cut opens on; it is not sent.
    """
    if spans is None or len(auto) == 0:
        return auto
    edges = np.asarray(sorted(spans), dtype=np.float64).reshape(-1)
    m = np.searchsorted(edges, auto.times, side="right") % 2 == 1
    return Automation(auto.times[m], auto.indices[m], auto.values[m])

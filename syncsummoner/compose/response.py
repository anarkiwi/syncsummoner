"""How many frames a program's picture keeps moving after a step lands: the hold an archive level needs.

Frames are scored as the evidence scores them; each step's early lags are compared with the later half
of its own quiet period, where it settled (signed-rank, step by step). The settle is the last lag
at which a parameter, or all steps pooled, still departs from that, family-wise.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, Sequence

import numpy as np
from scipy.stats import norm, wilcoxon

from syncsummoner.compose import evidence as ev
from syncsummoner.compose import matched

__all__ = ["MEASURED", "POOLED", "Response", "differences", "response", "steps_of", "take_scores"]

#: Settled frames an archive level keeps: the pair a frame-to-frame measure needs.
MEASURED = 2
#: The row of a :class:`Response` that pools every step of every parameter.
POOLED = 0


def steps_of(rows: np.ndarray, start: Sequence[int]) -> tuple[np.ndarray, np.ndarray]:
    """Each landed write that changed its slot: its landing frame and parameter, in landing order."""
    sent = rows[(rows["sent"] >= 0) & (rows["landed"] > ev.START)]
    sent = sent[np.argsort(sent["landed"], kind="stable")]
    held = np.asarray(start, dtype=np.int64).copy()
    changed = np.zeros(sent.size, dtype=bool)
    for i, (param, value) in enumerate(zip(sent["param"], sent["wire"])):
        changed[i] = held[param - 1] != value
        held[param - 1] = value
    return sent["landed"][changed].astype(np.int64), sent["param"][changed].astype(np.int64)


def differences(scores: np.ndarray, frames: np.ndarray, params: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Per step frame, its score at each lag less the median of where it settled, and its sole parameter.

    A step's quiet period ``w`` runs to the next landing (the take's end for the last); lags below
    ``w // 2`` are compared with the median score over the rest of it, NaN elsewhere. The parameter
    is 0 where several landed together.
    """
    count = scores.size
    at, first = np.unique(frames, return_index=True)
    solo = np.bincount(np.searchsorted(at, frames), minlength=at.size) == 1
    sole = np.where(solo, params[first], 0)
    width = np.r_[at[1:], count] - at
    lags = int(width.max() // 2) if at.size else 0
    index = at[:, None] + np.arange(lags)
    diff = np.where(np.arange(lags) < (width // 2)[:, None], scores[np.clip(index, 0, count - 1)], np.nan)
    settled = np.array(
        [
            (
                np.nanmedian(scores[f + w // 2 : f + w])
                if np.isfinite(scores[f + w // 2 : f + w]).any()
                else np.nan
            )
            for f, w in zip(at, width)
        ]
    )
    return diff - settled[:, None], sole


@dataclass(frozen=True)
class Response:
    """Per parameter, and pooled in row :data:`POOLED`, each lag's z of departure from the settled state."""

    program: str
    params: tuple[int, ...]
    z: np.ndarray
    steps: np.ndarray
    critical: float
    takes: tuple[str, ...] = field(default_factory=tuple)

    def moving(self, row: int) -> int:
        """The last lag ``row`` departs at, in a fixed sequence: the landing lags, then each next in turn.

        The evidence's landing lags are one step, since the change shows on either; past them each
        lag is tested only if the one before departed, so no lag spends the error rate. -1 if none.
        """
        z = np.where(np.isfinite(self.z[row]), self.z[row], -np.inf) > self.critical
        landing = [k for k in ev.LAGS if k < z.size and z[k]]
        if not landing:
            return -1
        past = max(ev.LAGS) + 1
        k = past
        while k < z.size and z[k]:
            k += 1
        return k - 1 if k > past else max(landing)

    @property
    def settle(self) -> int:
        """Frames after a step's landing frame that still move: the latest any row departs at, else 0."""
        return max(0, *(self.moving(r) for r in range(self.z.shape[0])))

    @property
    def least(self) -> int:
        """Steps a two-sided signed-rank test needs before it can depart beyond the bound at all."""
        return int(np.ceil(1 - np.log2(2 * norm.sf(self.critical))))

    @property
    def next_steps(self) -> int:
        """Steps, pooled, that reached the lag after the settle."""
        k = self.settle + 1
        return int(self.steps[POOLED, k]) if k < self.steps.shape[1] else 0

    @property
    def bounded(self) -> bool:
        """Whether the lag after the settle had steps enough, pooled, to have shown a departure."""
        return self.next_steps >= self.least

    @property
    def hold(self) -> int:
        """Frames from one level's landing to the next's: the settle, then the :data:`MEASURED` frames."""
        return self.settle + MEASURED

    def as_dict(self) -> dict:
        """JSON-ready, NaN as null."""
        return {
            "program": self.program,
            "params": list(self.params),
            "settle": self.settle,
            "least": self.least,
            "bounded": self.bounded,
            "hold": self.hold,
            "critical": self.critical,
            "z": [[round(float(v), 3) if np.isfinite(v) else None for v in row] for row in self.z],
            "steps": self.steps.tolist(),
            "takes": list(self.takes),
        }

    def table(self) -> str:
        """Plain text: per row, the lags departing beyond chance, then the first lags' z and steps."""
        lines = [
            f"{self.program}: settles {self.settle} frames after landing "
            f"({'bounded' if self.bounded else 'NOT bounded'}: {self.next_steps} steps at the next lag, "
            f"{self.least} needed), hold {self.hold}, critical z {self.critical:.2f}"
        ]
        for row, param in enumerate((POOLED, *self.params)):
            z, n = self.z[row], self.steps[row]
            head = " ".join(f"{v:.1f}/{c}" for v, c in zip(z[:6], n[:6]) if c)
            lines.append(
                f"  {'all' if param == POOLED else f'P{param}':>4}: to lag {self.moving(row)}  {head}"
            )
        return "\n".join(lines)

    def save(self, path: str | Path) -> None:
        """Write :meth:`as_dict` as JSON."""
        Path(path).write_text(json.dumps(self.as_dict(), indent=1), encoding="utf-8")


def response(
    program: str, parts: Iterable[tuple[np.ndarray, np.ndarray]], takes: Sequence[str] = ()
) -> Response:
    """The settle of ``program`` from its takes' :func:`differences`, tested lag by lag.

    Each test is a two-sided signed-rank test over the steps reaching that lag; the bound is the
    evidence's family-wise one over the rows, each row's lags a fixed sequence (:meth:`Response.moving`).
    """
    parts = list(parts)
    lags = max((d.shape[1] for d, _ in parts), default=0)
    diff = np.concatenate(
        [np.pad(d, ((0, 0), (0, lags - d.shape[1])), constant_values=np.nan) for d, _ in parts]
        or [np.zeros((0, lags))]
    )
    sole = np.concatenate([s for _, s in parts] or [np.zeros(0, dtype=np.int64)])
    params = tuple(int(p) for p in np.unique(sole[sole > 0]))
    rows = (np.ones(sole.size, dtype=bool), *(sole == p for p in params))
    z = np.full((len(rows), lags), np.nan)
    steps = np.zeros((len(rows), lags), dtype=np.int64)
    for r, mask in enumerate(rows):
        for k in range(lags):
            d = diff[mask, k]
            d = d[np.isfinite(d)]
            if np.count_nonzero(d):
                steps[r, k] = d.size
                z[r, k] = float(norm.isf(wilcoxon(d).pvalue / 2))
    return Response(program, params, z, steps, matched.critical(len(rows)), tuple(takes))


def take_scores(judged: Any, start: Sequence[int]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Every clip frame's matched score in a judged take, and its value-changing steps' frames and parameters.

    ``judged`` carries ``found``, ``features``, ``source`` and ``proof``; ``start`` is the take's
    start state. Controls are frames no step lands on, nor on the frame before, as in the evidence.
    """
    found, proof = judged.found, judged.proof
    count = found.picked.size
    out = judged.features.small[found.picked].astype(np.float32)
    src = judged.source.small[found.source_offset : found.source_offset + count].astype(np.float32)
    d_out, d_src = matched.frame_change(out), matched.frame_change(src)
    frames, params = steps_of(proof.landed.rows, start)
    keep = (frames >= 0) & (frames < count)
    frames, params = frames[keep], params[keep]
    stepped = np.zeros(count, dtype=bool)
    stepped[frames] = True
    stepped[np.clip(frames + 1, 0, count - 1)] = True
    measured = np.isfinite(d_out) & np.isfinite(d_src)
    controls = np.flatnonzero(~stepped & measured)
    scores = np.full(count, np.nan)
    at = np.flatnonzero(measured)
    scores[at] = matched.matched_scores(d_out, d_src, at, controls, 1.0 / float(np.prod(out.shape[1:])))
    return scores, frames, params


def judge_offline(take: str | Path, *, clip: str | Path, config: Any, workers: int | None = None) -> Any:
    """A take attributed and its writes landed, decodes cached, nothing written beside it."""
    from types import SimpleNamespace  # pylint: disable=import-outside-toplevel

    from syncsummoner.compose.lock import attribute_take  # pylint: disable=import-outside-toplevel
    from syncsummoner.compose.locktake import take_evidence  # pylint: disable=import-outside-toplevel
    from syncsummoner.device.wire import load_wire  # pylint: disable=import-outside-toplevel
    from syncsummoner.probe import decoded  # pylint: disable=import-outside-toplevel

    spec = decoded.Spec.of(config)
    features = decoded.features(take, spec, workers=workers, ffmpeg=getattr(config, "ffmpeg", "ffmpeg"))
    source = decoded.features(clip, spec, workers=workers, ffmpeg=getattr(config, "ffmpeg", "ffmpeg"))
    wire = load_wire(take)
    found = attribute_take(take, config=config, features=features)
    proof = take_evidence(
        take,
        program=str(wire["program"]),
        clip=clip,
        config=config,
        predict=False,
        found=found,
        features=features,
        source=source,
    )
    return SimpleNamespace(
        found=found,
        features=features,
        source=source,
        proof=proof,
        start=wire["start"],
        program=str(wire["program"]),
    )


def of_takes(
    takes: Sequence[str | Path], *, clip: str | Path, config: Any, workers: int | None = None
) -> dict[str, Response]:
    """Each program's :class:`Response` over every take of it among ``takes``."""
    from syncsummoner.progress import track  # pylint: disable=import-outside-toplevel

    parts: dict[str, list] = {}
    names: dict[str, list[str]] = {}
    for take in track(list(takes), desc="response", unit="take"):
        seen = judge_offline(take, clip=clip, config=config, workers=workers)
        scores, frames, params = take_scores(seen, seen.start)
        parts.setdefault(seen.program, []).append(differences(scores, frames, params))
        names.setdefault(seen.program, []).append(str(take))
    return {program: response(program, found, names[program]) for program, found in parts.items()}

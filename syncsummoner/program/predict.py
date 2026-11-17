"""Predictions for a planned run of CC steps: which steps can reach the picture, per parameter.

A step is dead when its two values differ only in bits no video output depends on with every
other control at its value then, which the fold decides exactly; any other step is live.
"""

from __future__ import annotations

import os
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass

import numpy as np

from syncsummoner.progress import track

from .model import ProgramModel


@dataclass(frozen=True)
class Prediction:
    """One parameter's planned steps split into those that reach the picture and those that cannot."""

    param: int
    live: int
    dead: int

    @property
    def verdict(self) -> str:
        """``live`` if any step reaches the picture, ``dead`` if none can, ``untested`` with no steps."""
        return "live" if self.live else ("dead" if self.dead else "untested")

    def check(self, moved: bool) -> str:
        """The prediction set against a measured verdict; ``contradiction`` flags a dead step that moved."""
        if self.verdict == "untested":
            return "no step to predict"
        if self.verdict == "dead":
            return "contradiction: predicted dead, but moved" if moved else "predicted dead, and did not move"
        return "predicted live, and moved" if moved else "predicted live, did not move"


def steps(start: np.ndarray, plan: np.ndarray) -> list[tuple[np.ndarray, int, int, int]]:
    """``(state before, param, a, b)`` for each plan row that changes its parameter, in frame order."""
    state = np.asarray(start, dtype=np.int64).copy()
    out = []
    for row in plan[np.argsort(plan["target"], kind="stable")]:
        p, b = int(row["param"]), int(row["value"])
        if 1 <= p <= state.size and state[p - 1] != b:
            out.append((state.copy(), p, int(state[p - 1]), b))
            state[p - 1] = b
    return out


def predict(model: ProgramModel, start: np.ndarray, plan: np.ndarray, timing: int | None = None) -> dict:
    """Per parameter, how many of the plan's steps can reach the picture and how many cannot."""
    todo = steps(start, plan)
    with ThreadPoolExecutor(max_workers=os.cpu_count()) as pool:
        seen = list(
            track(
                pool.map(lambda s: model.visible(s[0], s[1], s[2], s[3], timing), todo),
                total=len(todo),
                desc=f"{model.program.name} predict",
                unit="step",
            )
        )
    out = {}
    for (_, p, _, _), live in zip(todo, seen):
        prev = out.get(p, Prediction(p, 0, 0))
        out[p] = Prediction(p, prev.live + live, prev.dead + (not live))
    return out

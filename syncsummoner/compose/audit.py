"""What a score will actually drive: every written parameter by name, kind and range.

The render plays :func:`~syncsummoner.compose.planner.plan_automation`, not the
gestures, so this reads the same plan and attributes each parameter to the
gestures that wrote it, flagging a toggle the plan leaves latched on.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field
from typing import Mapping

import numpy as np

from syncsummoner.compose.planner import plan_automation
from syncsummoner.compose.score import Layer, Score, control_rate
from syncsummoner.device.profile import BOOL_THRESHOLD, ParamKind, ProgramProfile

__all__ = ["LayerAudit", "ParamUse", "audit", "format_audit"]


@dataclass(frozen=True)
class ParamUse:
    """One parameter of one layer: what the plan writes to it, and who asked."""

    index: int
    name: str
    kind: ParamKind
    writes: int = 0
    low: int | None = None
    high: int | None = None
    final: int | None = None
    gestures: Mapping[str, int] = field(default_factory=dict)
    flags: tuple[str, ...] = ()


@dataclass(frozen=True)
class LayerAudit:
    """One layer's parameters, and the gestures in it that render nothing."""

    program: str
    index: int
    params: list[ParamUse] = field(default_factory=list)
    silent: Mapping[str, int] = field(default_factory=dict)
    flags: tuple[str, ...] = ()


def _flags(spec, values: np.ndarray) -> tuple[str, ...]:
    return ("latched on",) if spec.kind is ParamKind.BOOLEAN and values[-1] >= BOOL_THRESHOLD else ()


def _layer(score: Score, layer: Layer, profile: ProgramProfile | None, plan, rate: float) -> LayerAudit:
    if profile is None:
        return LayerAudit(layer.program, layer.index, flags=("no profile: renders without automation",))
    drivers: dict[int, Counter] = {}
    silent: Counter = Counter()
    for inst, auto in score.gesture_automations(layer, profile, rate=rate):
        if len(auto) == 0:
            silent[inst.gesture] += 1
        for index in np.unique(auto.indices).tolist():
            drivers.setdefault(index, Counter())[inst.gesture] += 1
    params = []
    for spec in profile.params:
        values = plan.values[plan.indices == spec.index]
        if not values.size:
            params.append(ParamUse(spec.index, spec.name, spec.kind))
            continue
        params.append(
            ParamUse(
                spec.index,
                spec.name,
                spec.kind,
                writes=int(values.size),
                low=int(values.min()),
                high=int(values.max()),
                final=int(values[-1]),
                gestures=dict(drivers.get(spec.index, {})),
                flags=_flags(spec, values),
            )
        )
    flags = () if any(p.writes for p in params) else ("drives nothing",)
    return LayerAudit(layer.program, layer.index, params, dict(silent), flags)


def audit(
    score: Score, profiles: Mapping[str, ProgramProfile], *, fps: float | None = None
) -> list[LayerAudit]:
    """Every layer's parameter use under the plan the renderer would play."""
    fps = score.fps if fps is None else fps
    plans = plan_automation(score, profiles, fps=fps)
    rate = control_rate(fps)
    return [
        _layer(score, layer, profiles.get(layer.program), plans.get(layer.index), rate)
        for layer in score.layers
    ]


def _driven(use: ParamUse) -> str:
    if not use.writes:
        return "undriven"
    by = ", ".join(f"{name} x{count}" for name, count in sorted(use.gestures.items()))
    return f"{use.writes:>4} writes  {use.low:>4}..{use.high:<4} ends {use.final:<4}  {by}"


def format_audit(layers: list[LayerAudit]) -> str:
    """Plain-text report, one block per layer, flagged lines marked ``!``."""
    lines = []
    for layer in layers:
        lines.append(f"layer {layer.index}: {layer.program}" + "".join(f"  ! {f}" for f in layer.flags))
        for use in layer.params:
            mark = "!" if use.flags else " "
            lines.append(
                f" {mark} P{use.index:<2} {use.name[:16]:<16} {use.kind.value:<10} "
                + _driven(use)
                + "".join(f"  ! {f}" for f in use.flags)
            )
        if layer.silent:
            said = ", ".join(f"{name} x{count}" for name, count in sorted(layer.silent.items()))
            lines.append(f"   gestures that write nothing: {said}")
    return "\n".join(lines)

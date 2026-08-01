"""Sweep plans, in order of cost: OAT, Sobol, tongue raster, hysteresis.

Each plan is an iterator of parameter vectors keyed by 1-based parameter index,
with continuous values normalized to ``[0, 1]`` and booleans as ``bool``. Sampling
is type-aware via :meth:`ParamSpec.sample_values`; unused parameters never appear.
"""

from __future__ import annotations

import numpy as np
from scipy.stats import qmc

from syncsummoner.device.profile import PARAM_MAX, ParamKind, ParamSpec

__all__ = ["defaults", "oat", "sobol", "tongue_raster", "hysteresis"]


def _used(spec):
    return [p for p in spec if p.kind is not ParamKind.UNUSED]


def _value(param, raw):
    if param.kind is ParamKind.BOOLEAN:
        return bool(raw > PARAM_MAX / 2)
    return float(raw) / PARAM_MAX


def defaults(spec, *, neutral=0.5):
    """Parked reference vector: ``neutral`` for continuous, off for boolean.

    MIDI CC is an offset onto the physical knob, so the reference is a parked
    midpoint rather than an absolute zero; see ``docs/hardware.md``.
    """
    return {p.index: (False if p.kind is ParamKind.BOOLEAN else float(neutral)) for p in _used(spec)}


def _find(spec, index):
    for param in spec:
        if param.index == index:
            return param
    raise KeyError(f"no parameter with index {index}")


def oat(spec, *, steps=32, neutral=0.5):
    """One parameter at a time across its native range, others parked."""
    base = defaults(spec, neutral=neutral)
    for param in _used(spec):
        for raw in param.sample_values(steps):
            yield {**base, param.index: _value(param, raw)}


def _sobol_points(dim, n, rng):
    if dim == 0:
        return np.empty((n, 0))
    engine = qmc.Sobol(d=dim, scramble=True, seed=rng)
    return engine.random_base2(int(np.ceil(np.log2(max(n, 1)))))[:n]


def _snap(param, unit):
    if param.kind is not ParamKind.QUANTIZED:
        return float(unit)
    values = param.sample_values()
    return float(values[min(int(unit * len(values)), len(values) - 1)]) / PARAM_MAX


def sobol(spec, *, n, rng):
    """Quasi-random over the continuous cube; booleans walk corners in gray order.

    OAT structurally cannot see interactions. Boolean parameters are corners of the
    cube, never a rounded continuous draw, and step one flip at a time.
    """
    continuous = [p for p in _used(spec) if p.kind is not ParamKind.BOOLEAN]
    booleans = [p for p in _used(spec) if p.kind is ParamKind.BOOLEAN]
    n = int(n)
    if n <= 0:
        return
    points = _sobol_points(len(continuous), n, rng)
    mask = (1 << len(booleans)) - 1
    for i in range(n):
        vector = {p.index: _snap(p, u) for p, u in zip(continuous, points[i])}
        corner = (i & mask) ^ ((i & mask) >> 1)
        vector.update({p.index: bool((corner >> j) & 1) for j, p in enumerate(booleans)})
        yield vector


def tongue_raster(spec, pair, *, n, neutral=0.5):
    """Dense 2-D raster over a parameter pair, in boustrophedon order.

    Raster order minimizes parameter travel between samples; the second axis is
    the one the winding number is read against.
    """
    base = defaults(spec, neutral=neutral)
    first, second = (_find(spec, i) for i in pair)
    outer, inner = first.sample_values(n), second.sample_values(n)
    for i, a in enumerate(outer):
        for b in inner if i % 2 == 0 else inner[::-1]:
            yield {**base, first.index: _value(first, a), second.index: _value(second, b)}


def _interpolate(values, micro):
    if micro < 1 or len(values) < 2:
        return np.asarray(values)
    legs = [np.linspace(a, b, micro + 1, endpoint=False) for a, b in zip(values[:-1], values[1:])]
    return np.round(np.concatenate(legs + [values[-1:]])).astype(np.int64)


def hysteresis(spec, index, *, n, micro=3, neutral=0.5):
    """Identical setpoints approached from below and above, slowly and fast.

    Feedback programs on an FPGA are stateful; path dependence is catalogued, not
    averaged away. Slow ramps interpolate ``micro`` steps between setpoints.
    """
    base = defaults(spec, neutral=neutral)
    param = _find(spec, index)
    setpoints = param.sample_values(n)
    for rate in (0, micro):
        for path in (setpoints, setpoints[::-1]):
            for raw in _interpolate(path, rate):
                yield {**base, param.index: _value(param, raw)}


def spec_from_info(info):
    """Build ``ParamSpec`` list from a device ``program info`` payload.

    Unused parameters are named ``-``; a native range of ``0..1`` is boolean and a
    small integer range is quantized, per ``docs/hardware.md``.
    """
    specs = []
    for i, entry in enumerate(info, start=1):
        name = str(entry["name"])
        low, high = float(entry["min"]), float(entry["max"])
        span = high - low
        if name == "-":
            kind, steps = ParamKind.UNUSED, None
        elif span <= 1.0:
            kind, steps = ParamKind.BOOLEAN, 2
        elif span < 32.0 and float(span).is_integer():
            kind, steps = ParamKind.QUANTIZED, int(span) + 1
        else:
            kind, steps = ParamKind.CONTINUOUS, None
        specs.append(ParamSpec(index=i, name=name, native_min=low, native_max=high, kind=kind, steps=steps))
    return specs

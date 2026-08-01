"""Measurements to ``ProgramProfile``: response, cliffs, interactions, tongues, axes.

Axis assignment and the cliff atlas are derived here, in the profile, so gestures
stay program-agnostic and new firmware is absorbed by re-probing. Metrics are
compared in robust standardized units, and thresholds are noise-derived.
"""

from __future__ import annotations

from fractions import Fraction
from pathlib import Path

import numpy as np
import yaml
from scipy.stats import spearmanr

from syncsummoner.device.profile import (
    PARAM_COUNT,
    PARAM_MAX,
    Axis,
    Cliff,
    LockMap,
    ParamKind,
    ParamSpec,
    ProgramProfile,
    Tongue,
)
from syncsummoner.probe.runner import STABILITY_ORDER

__all__ = [
    "PROFILE_SCHEMA_VERSION",
    "AXIS_SIGNATURES",
    "fit_profile",
    "save_profile",
    "load_profile",
    "save_measurements",
    "load_measurements",
]

PROFILE_SCHEMA_VERSION = 1

AXIS_SIGNATURES = {
    Axis.TEXTURE_SCALE: {"peak_scale": 1.0, "concentration": 0.6, "spectral_slope": 0.5, "ch_*": 0.3},
    Axis.MOTION_RATE: {
        "winding_number": 1.0,
        "period_frames": 0.8,
        "periodicity_strength": 0.5,
        "framediff_energy": 0.4,
    },
    Axis.DISPLACEMENT: {"flow_magnitude": 1.0, "flow_coherence": 0.6, "passthrough_distance": 0.4},
    Axis.COLOR_DESTRUCTION: {
        "chroma_mean": 1.0,
        "colourfulness": 0.8,
        "chroma_std": 0.6,
        "illegal_frac": 0.5,
    },
    Axis.KEY_THRESHOLD: {"clip_frac": 1.0, "luma_mean": 0.6, "luma_std": 0.4},
    Axis.FEEDBACK_GAIN: {"framediff_energy": 1.0, "stability": 0.8, "periodicity_strength": 0.5},
    Axis.NOISE: {"spectral_slope": 1.0, "spectral_r2": 0.6, "fractal_dimension": 0.5, "luma_std": 0.4},
}


def _robust_scale(values, axis=0):
    mad = np.median(np.abs(values - np.median(values, axis=axis, keepdims=True)), axis=axis)
    scale = 1.4826 * mad
    fallback = values.std(axis=axis)
    return np.where(scale > 0, scale, np.where(fallback > 0, fallback, 1.0))


def _metric_matrix(records):
    names = sorted({k for r in records for k in r.metrics})
    raw = np.array([[r.metrics.get(n, np.nan) for n in names] for r in records], dtype=np.float64)
    medians = np.nanmedian(np.where(np.isfinite(raw), raw, np.nan), axis=0)
    raw = np.where(np.isfinite(raw), raw, np.nan_to_num(medians)[None, :])
    keep = np.ptp(raw, axis=0) > 0
    raw, names = raw[:, keep], [n for n, k in zip(names, keep) if k]
    if not names:
        return [], np.zeros((len(records), 0)), np.zeros(0)
    standardized = (raw - np.median(raw, axis=0)) / _robust_scale(raw)
    span = np.ptp(standardized, axis=0)
    return names, standardized, np.where(span > 0, span, 1.0)


def _noise(params, metrics, *, noise_floor=0.05):
    """Measurement noise per metric, from replicated parameter vectors."""
    floor = np.full(metrics.shape[1], noise_floor)
    if metrics.size == 0:
        return floor
    _, inverse, counts = np.unique(params, axis=0, return_inverse=True, return_counts=True)
    inverse = inverse.ravel()
    replicated = counts[inverse] > 1
    if replicated.sum() < 4:
        return floor
    sums = np.zeros((counts.size, metrics.shape[1]))
    np.add.at(sums, inverse, metrics)
    residual = metrics[replicated] - (sums / counts[:, None])[inverse[replicated]]
    return np.maximum(1.4826 * np.median(np.abs(residual), axis=0), floor)


def _mode(column):
    values, counts = np.unique(column, return_counts=True)
    return values[int(np.argmax(counts))]


def _curve(values, metrics):
    unique, inverse = np.unique(values, return_inverse=True)
    sums = np.zeros((unique.size, metrics.shape[1]))
    np.add.at(sums, inverse, metrics)
    return unique, sums / np.bincount(inverse, minlength=unique.size)[:, None]


def _longest_run(mask):
    if not mask.any():
        return None
    edges = np.flatnonzero(np.diff(np.concatenate(([0], mask.astype(np.int8), [0]))))
    starts, ends = edges[::2], edges[1::2]
    best = int(np.argmax(ends - starts))
    return int(starts[best]), int(ends[best])


def _infer_kind(column, *, max_steps=16):
    """Parameter kind from the values actually visited, ignoring the parked default."""
    unique = np.unique(column)
    if unique.size <= 1:
        return ParamKind.UNUSED, None
    swept = unique[unique != _mode(column)] if unique.size > 2 else unique
    if swept.size <= 2 and set(swept.tolist()) <= {0, PARAM_MAX}:
        return ParamKind.BOOLEAN, 2
    if swept.size <= max_steps:
        return ParamKind.QUANTIZED, int(swept.size)
    return ParamKind.CONTINUOUS, None


def _signature_weights(names, signature):
    weights = np.zeros(len(names))
    for i, name in enumerate(names):
        for key, weight in signature.items():
            if name == key or (key.endswith("*") and name.startswith(key[:-1])):
                weights[i] = max(weights[i], weight)
    return weights


def _assign_axis(effect, names, *, min_similarity=0.3):
    norm = np.linalg.norm(effect)
    if norm <= 0:
        return Axis.UNASSIGNED
    unit = effect / norm
    scores = {
        axis: float(unit @ (w / np.linalg.norm(w)))
        for axis, signature in AXIS_SIGNATURES.items()
        for w in [_signature_weights(names, signature)]
        if np.linalg.norm(w) > 0
    }
    if not scores:
        return Axis.UNASSIGNED
    best = max(scores, key=scores.get)
    return best if scores[best] >= min_similarity else Axis.UNASSIGNED


def _hysteresis(order, values, metric, floor):
    """True when ascending and descending passes disagree consistently at shared setpoints.

    The threshold is the noise floor rather than the replicate estimate: replicates
    of a hysteresis plan differ by path, which is the effect being measured.
    """
    if order.size < 4:
        return False
    direction = np.sign(np.diff(values[order]))
    up, down = order[1:][direction > 0], order[1:][direction < 0]
    if up.size == 0 or down.size == 0:
        return False
    shared = np.intersect1d(values[up], values[down])
    if shared.size == 0:
        return False
    gaps = [metric[up[values[up] == v]].mean() - metric[down[values[down] == v]].mean() for v in shared]
    return bool(abs(np.median(gaps)) > 2.0 * floor)


def _fit_param(index, spec, values, metrics, names, span, noise, *, monotonic_rho, cliff_iqr):
    """Fill one ``ParamSpec`` from its one-at-a-time subset; returns the driving metric.

    Sensitivity is a signal-to-noise ratio, not a fraction of the metric's span:
    normalising by span pins every parameter that solely owns a metric at exactly
    1.0, so the field cannot rank the parameters that matter.
    """
    unique, curve = _curve(values, metrics)
    effect = np.ptp(curve, axis=0)
    driver = int(np.argmax(effect / noise))
    response = curve[:, driver]
    spec.index = index
    spec.values = [int(v) for v in unique]
    spec.sensitivity = float(effect[driver] / noise[driver])
    spread = np.ptp(response)
    spec.response = [float(v) for v in ((response - response.min()) / (spread or 1.0))]
    moving = np.concatenate(([True], np.abs(np.diff(response)) > noise[driver]))
    rho = spearmanr(unique[moving], response[moving]).statistic if moving.sum() > 2 else 1.0
    spec.monotonic = bool(np.isfinite(rho) and abs(rho) >= monotonic_rho)
    spec.axis = _assign_axis(effect / span, names)
    if unique.size < 2:
        return driver
    diffs = np.abs(np.diff(curve, axis=0))
    flat = _longest_run((diffs <= noise[None, :]).all(axis=1))
    if flat is not None and flat[1] - flat[0] >= 2:
        spec.dead_zone = (int(unique[flat[0]]), int(unique[flat[1]]))
    low, high = np.percentile(diffs, [25, 75], axis=0)
    limit = high + cliff_iqr * (high - low) + noise
    for gap in np.flatnonzero((diffs > limit[None, :]).any(axis=1)):
        moved = np.flatnonzero(diffs[gap] > limit)
        spec.cliffs.append(
            Cliff(
                at=int(unique[gap + 1]),
                jump=float(np.clip(np.max(diffs[gap] / span), 0.0, 1.0)),
                metrics=tuple(names[m] for m in moved),
            )
        )
    return driver


def _pair_strength(response, a_bins, b_bins, shape, min_filled):
    flat = a_bins * shape[1] + b_bins
    counts = np.bincount(flat, minlength=shape[0] * shape[1]).astype(float)
    sums = np.bincount(flat, weights=response, minlength=shape[0] * shape[1])
    cells = np.where(counts > 0, sums / np.maximum(counts, 1.0), np.nan).reshape(shape)
    filled = np.isfinite(cells)
    if filled.mean() < min_filled:
        return 0.0
    rows = np.nanmean(cells, axis=1, keepdims=True)
    cols = np.nanmean(cells, axis=0, keepdims=True)
    residual = cells - (rows + cols - np.nanmean(cells))
    return float(np.sqrt(np.nanmean(residual[filled] ** 2)))


def _bins(column, q):
    ranks = np.unique(column, return_inverse=True)[1].ravel()
    return np.minimum(ranks * q // max(ranks.max() + 1, 1), q - 1)


def _interactions(params, response, varying, *, bins=4, min_strength=0.05, min_filled=0.75):
    """Pairwise non-additivity: residual of a row-plus-column model over a binned grid."""
    scale = float(np.ptp(response)) or 1.0
    found = []
    for i, a in enumerate(varying):
        for b in varying[i + 1 :]:
            strength = _pair_strength(
                response, _bins(params[:, a], bins), _bins(params[:, b], bins), (bins, bins), min_filled
            )
            if strength / scale >= min_strength:
                found.append((int(a + 1), int(b + 1), float(strength / scale)))
    return sorted(found, key=lambda t: t[2], reverse=True)


def _principal_response(metrics):
    """Scalar response: the first principal component of the standardized metrics."""
    left, singular, right = np.linalg.svd(metrics - metrics.mean(axis=0), full_matrices=False)
    component = left[:, 0] * singular[0]
    return component * np.sign(right[0, int(np.argmax(np.abs(right[0])))] or 1.0)


def _runs(labels):
    start = 0
    for i in range(1, len(labels) + 1):
        if i == len(labels) or labels[i] != labels[start]:
            if labels[start] is not None:
                yield start, i
            start = i


def _farey_tol(max_denominator):
    """Half the worst-case gap to the nearest fraction of bounded denominator.

    Any real is within ``1/(2q(q+1))`` of some ``p/q'`` with ``q' <= q``, so a looser
    tolerance would call every winding number locked.
    """
    return 0.25 / (max_denominator * (max_denominator + 1))


def _tongues(values, winding, strength, *, max_denominator, lock_tol):
    lock_tol = _farey_tol(max_denominator) if lock_tol is None else lock_tol
    labels = [
        f if np.isfinite(w) and abs(w - float(f)) <= lock_tol else None
        for w in winding
        for f in [Fraction(float(w) if np.isfinite(w) else 0.0).limit_denominator(max_denominator)]
    ]
    for start, stop in _runs(labels):
        ratio = labels[start]
        if stop - start < 2:
            continue
        yield Tongue(
            ratio=(ratio.numerator, ratio.denominator),
            center=int(round(0.5 * (values[start] + values[stop - 1]))),
            width=int(values[stop - 1] - values[start]),
            strength=float(np.nanmean(strength[start:stop])),
        )


def _lock_maps(params, records, specs, varying, *, min_levels, max_denominator, lock_tol):
    winding = np.array([r.metrics.get("winding_number", np.nan) for r in records])
    strength = np.array([r.metrics.get("periodicity_strength", np.nan) for r in records])
    if not np.isfinite(winding).any():
        return []
    dense = [j for j in varying if np.unique(params[:, j]).size >= min_levels]
    maps = []
    for i, a in enumerate(dense):
        for b in dense[i + 1 :]:
            others = [j for j in varying if j not in (a, b)]
            rows = np.flatnonzero((params[:, others] == [_mode(params[:, j]) for j in others]).all(axis=1))
            if rows.size < min_levels**2 // 2:
                continue
            tongues = []
            for level in np.unique(params[rows, a]):
                line = rows[params[rows, a] == level]
                line = line[np.argsort(params[line, b])]
                tongues += list(
                    _tongues(
                        params[line, b],
                        winding[line],
                        strength[line],
                        max_denominator=max_denominator,
                        lock_tol=lock_tol,
                    )
                )
            widest = {}
            for tongue in tongues:
                if tongue.width >= widest.get(tongue.ratio, tongue).width:
                    widest[tongue.ratio] = tongue
            if widest:
                maps.append(
                    LockMap(
                        a=specs[a].name or f"p{a + 1}",
                        b=int(b + 1),
                        tongues=sorted(widest.values(), key=lambda t: t.width, reverse=True),
                    )
                )
    return maps


def _stability_regions(params, records, varying):
    labels = np.array([r.metrics.get("stability", np.nan) for r in records])
    regions = []
    for level in np.unique(labels[np.isfinite(labels)]):
        rows = np.flatnonzero(labels == level)
        regions.append(
            {
                "stability": STABILITY_ORDER[int(level)],
                "n": int(rows.size),
                "bounds": {
                    int(j + 1): [int(params[rows, j].min()), int(params[rows, j].max())] for j in varying
                },
            }
        )
    return regions


def fit_profile(
    records,
    *,
    specs=None,
    monotonic_rho=0.9,
    cliff_iqr=1.5,
    noise_floor=0.05,
    interaction_bins=4,
    min_interaction=0.05,
    tongue_levels=4,
    max_denominator=8,
    lock_tol=None,
):
    """Fit a ``ProgramProfile`` from measurement records.

    Pass ``specs`` when device parameter metadata is available; otherwise kinds are
    inferred from the values actually visited. Thresholds are noise-derived.
    """
    if not records:
        raise ValueError("no measurement records to fit")
    params = np.array([r.params for r in records], dtype=np.int64)
    names, metrics, span = _metric_matrix(records)
    noise = _noise(params, metrics, noise_floor=noise_floor)
    varying = [j for j in range(PARAM_COUNT) if np.unique(params[:, j]).size > 1]
    modes = np.array([_mode(params[:, j]) for j in range(PARAM_COUNT)])
    fitted = []
    settle = {}
    for j in range(PARAM_COUNT):
        kind, steps = _infer_kind(params[:, j])
        spec = next((s for s in specs or [] if s.index == j + 1), None) or ParamSpec(
            index=j + 1, name=f"p{j + 1}", native_min=0.0, native_max=1.0, kind=kind, steps=steps
        )
        if j not in varying or not names:
            fitted.append(spec)
            continue
        others = [k for k in varying if k != j]
        rows = np.flatnonzero((params[:, others] == modes[others]).all(axis=1))
        if rows.size < 2:
            rows = np.arange(len(records))
        driver = _fit_param(
            j + 1,
            spec,
            params[rows, j],
            metrics[rows],
            names,
            span,
            noise,
            monotonic_rho=monotonic_rho,
            cliff_iqr=cliff_iqr,
        )
        spec.hysteresis = _hysteresis(rows, params[:, j], metrics[:, driver], noise_floor)
        settle[j + 1] = int(np.median([records[r].settle_frames for r in rows]))
        fitted.append(spec)
    response = _principal_response(metrics) if names else None
    interactions = (
        _interactions(params, response, varying, bins=interaction_bins, min_strength=min_interaction)
        if response is not None and len(varying) > 1
        else []
    )
    observed = np.array([r.settle_frames for r in records])
    return ProgramProfile(
        program=records[0].program,
        firmware=records[0].firmware,
        analyzer=records[0].analyzer,
        source=records[0].source,
        params=fitted,
        interactions=interactions,
        lock_maps=_lock_maps(
            params,
            records,
            fitted,
            varying,
            min_levels=tongue_levels,
            max_denominator=max_denominator,
            lock_tol=lock_tol,
        ),
        stability_by_region=_stability_regions(params, records, varying),
        settle_frames=settle,
        non_settling=bool(observed.max() > 0 and np.median(observed) >= 0.9 * observed.max()),
    )


def save_profile(profile, path):
    """Write a profile as versioned YAML."""
    payload = {"schema_version": PROFILE_SCHEMA_VERSION, "profile": profile.to_dict()}
    Path(path).write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")


def load_profile(path):
    """Read a versioned YAML profile, rejecting a newer schema."""
    payload = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    version = int(payload.get("schema_version", 0))
    if version > PROFILE_SCHEMA_VERSION:
        raise ValueError(f"profile schema {version} is newer than {PROFILE_SCHEMA_VERSION}")
    return ProgramProfile.from_dict(payload["profile"])


def save_measurements(records, path):
    """Write raw measurement rows to Parquet, unioning heterogeneous metric keys."""
    import pyarrow
    import pyarrow.parquet

    rows = [r.as_row() for r in records]
    columns = list(dict.fromkeys(k for row in rows for k in row))
    table = pyarrow.Table.from_pylist([{k: row.get(k) for k in columns} for row in rows])
    pyarrow.parquet.write_table(table, str(path))


def load_measurements(path):
    """Read raw measurement rows back from Parquet."""
    import pyarrow.parquet

    return pyarrow.parquet.read_table(str(path)).to_pylist()

"""Execute a plan against the device, identify frames by gray code, log records.

Frames self-identify by their burnt-in state index, so no dwell timing is assumed
anywhere: dropped and duplicated frames are tolerated, and capture-card "no signal"
splashes are skipped rather than analyzed.
"""

from __future__ import annotations

import numpy as np

from syncsummoner.device.profile import PARAM_COUNT, PARAM_MAX, MeasurementRecord, Source
from syncsummoner.probe import patterns

__all__ = ["run_plan", "frame_metrics", "measurement", "settle_frames", "raw_params", "STABILITY_ORDER"]

#: Analysis width; Gabor and optical flow on full 1080p cost far more for no gain.
ANALYSIS_WIDTH = 480
#: Setpoints re-measured per plan, so the noise floor is measured not assumed.
DEFAULT_REPLICATES = 6

STABILITY_ORDER = ("static", "periodic", "quasiperiodic", "chaotic")


def raw_params(vector):
    """Parameter vector as the device's 12 raw 0..1023 values; booleans sit at the rails."""
    raw = np.zeros(PARAM_COUNT, dtype=np.int64)
    for index, value in vector.items():
        raw[index - 1] = PARAM_MAX if value is True else 0 if value is False else round(value * PARAM_MAX)
    return tuple(int(v) for v in raw)


def _firmware(session):
    for obj in (session, getattr(session, "transport", None)):
        value = getattr(obj, "firmware", None)
        try:
            value = value() if callable(value) else value
        except Exception:
            value = None
        if value:
            return str(value)
    return "unknown"


def settle_frames(frames, *, plateau_frac=0.1):
    """Frames until the frame-to-frame difference plateaus.

    Returns ``len(frames)`` when it never plateaus, which flags the program as
    non-settling and routes it to the dynamics analyzer.
    """
    if len(frames) < 2:
        return 0
    stack = np.stack([np.asarray(f, dtype=np.float32) for f in frames])
    diffs = np.abs(np.diff(stack, axis=0)).mean(axis=(1, 2, 3))
    active = np.flatnonzero(diffs > plateau_frac * diffs.max()) if diffs.max() > 0 else np.empty(0)
    return int(active[-1] + 1) if active.size else 0


def _with_replicates(vectors, replicates, rng):
    """Append repeats of a few setpoints so ``fit`` can estimate measurement noise.

    Without replicated vectors the noise floor falls back to a constant, and a
    sensitivity or a cliff has no resolution limit to be judged against.
    """
    if replicates <= 0 or len(vectors) < 2:
        return vectors
    generator = rng if rng is not None else np.random.default_rng(0)
    picks = generator.choice(len(vectors), size=min(replicates, len(vectors)), replace=False)
    return vectors + [vectors[int(i)] for i in picks]


def downscale(frames, max_width):
    """Shrink frames before analysis, sharing the analyzer's own resampling."""
    from syncsummoner.aesthetics.levels import downscale as shrink

    return [shrink(f, max_width=max_width) for f in frames]


def frame_metrics(
    frames, analyzer, *, fps=60.0, reference=None, min_dynamics_frames=8, analysis_width=ANALYSIS_WIDTH
):
    """Per-sample metric vector for one parameter setpoint, all via ``aesthetics``.

    ``stability`` is stored as an ordinal into :data:`STABILITY_ORDER` so the row
    stays Parquet-friendly.
    """
    frames = downscale(frames, analysis_width)
    if reference is not None and analysis_width and reference.shape[1] > analysis_width:
        reference = downscale([reference], analysis_width)[0]
    last = frames[-1]
    metrics = {}
    channels = analyzer.gabor_energy(last)
    energy = np.asarray(channels.energy)
    metrics.update(
        {f"ch_s{s}_o{o}": float(v) for (s, o), v in np.ndenumerate(energy)}
        | {"concentration": float(channels.concentration)}
        | {"peak_scale": float(channels.peak[0]), "peak_orientation": float(channels.peak[1])}
    )
    spectrum = analyzer.spectral_stats(last)
    metrics.update(
        spectral_slope=float(spectrum.slope),
        spectral_r2=float(spectrum.r2),
        fractal_dimension=float(spectrum.fractal_dimension),
    )
    levels = analyzer.level_stats(last)
    metrics.update(
        {
            name: float(getattr(levels, name))
            for name in (
                "luma_mean",
                "luma_std",
                "chroma_mean",
                "chroma_std",
                "clip_frac",
                "illegal_frac",
                "colourfulness",
            )
        }
    )
    if len(frames) > 1:
        motion = [analyzer.motion_stats(a, b) for a, b in zip(frames[:-1], frames[1:])]
        for name in ("framediff_energy", "flow_magnitude", "flow_coherence"):
            metrics[name] = float(np.mean([getattr(m, name) for m in motion]))
    if len(frames) >= min_dynamics_frames:
        series = np.array([np.asarray(f, dtype=np.float32).mean() for f in frames], dtype=np.float32)
        dynamics = analyzer.analyze_dynamics(series, fps=fps)
        metrics["periodicity_strength"] = float(dynamics.periodicity_strength)
        metrics["stability"] = float(STABILITY_ORDER.index(dynamics.stability.value))
        for name in ("period_frames", "winding_number"):
            value = getattr(dynamics, name)
            if value is not None:
                metrics[name] = float(value)
    if reference is not None:
        metrics["passthrough_distance"] = float(analyzer.passthrough_distance(reference, last))
    return metrics


def measurement(frames, analyzer, *, program, firmware, source, params, state_index, stimulus, **metrics):
    """One measurement record: the metric vector over ``frames`` and its provenance.

    Shared by the hardware and simulation backends so both stamp a record the
    same way; ``metrics`` are the :func:`frame_metrics` options.
    """
    return MeasurementRecord(
        program=program,
        firmware=firmware,
        analyzer=f"aesthetics {getattr(analyzer, '__version__', 'unknown')}",
        source=source,
        params=params,
        state_index=state_index,
        stimulus=stimulus,
        metrics=frame_metrics(frames, analyzer, **metrics),
        settle_frames=settle_frames(frames),
    )


def _ahead(index, expected, capacity):
    return 0 < (index - expected) % capacity < capacity // 2


def _collect(capture, expected, *, bits, strip_px, frames_per_point, max_wait_frames, allow_untagged):
    capacity = patterns.state_index_capacity(bits)
    samples = []
    for _ in range(max_wait_frames):
        if len(samples) >= frames_per_point:
            break
        frame = capture.read()
        if frame is None or capture.is_no_signal(frame):
            continue
        index = None if allow_untagged else patterns.read_state_index(frame, bits=bits, strip_px=strip_px)
        if index is None and not allow_untagged:
            continue
        if index is not None and index != expected:
            if samples or _ahead(index, expected, capacity):
                break
            continue
        cropped = patterns.crop_strip(frame, strip_px=strip_px)
        if samples and np.array_equal(cropped, samples[-1]):
            continue
        samples.append(cropped)
    return samples


def run_plan(
    session,
    capture,
    plan,
    *,
    program,
    analyzer,
    stimulus="zoneplate",
    emit=None,
    firmware=None,
    source=Source.HW,
    bits=8,
    strip_px=8,
    fps=60.0,
    frames_per_point=8,
    max_wait_frames=None,
    reference=None,
    allow_untagged=False,
    analysis_width=ANALYSIS_WIDTH,
    replicates=DEFAULT_REPLICATES,
    rng=None,
):
    """Run ``plan`` on a live session and capture, returning measurement records.

    ``emit`` publishes the state index onto the played-out stimulus; with no
    playout, ``allow_untagged`` attributes signal frames to the current vector.
    """
    firmware = firmware or _firmware(session)
    capacity = patterns.state_index_capacity(bits)
    max_wait_frames = max_wait_frames or 8 * frames_per_point
    records = []
    plan = _with_replicates(list(plan), replicates, rng)
    for step, vector in enumerate(plan):
        state = step % capacity
        session.set_params(vector)
        if emit is not None:
            emit(state)
        samples = _collect(
            capture,
            state,
            bits=bits,
            strip_px=strip_px,
            frames_per_point=frames_per_point,
            max_wait_frames=max_wait_frames,
            allow_untagged=allow_untagged,
        )
        if not samples:
            continue
        params = raw_params(vector)
        records.append(
            measurement(
                samples,
                analyzer,
                program=program,
                firmware=firmware,
                source=source,
                params=params,
                state_index=state,
                stimulus=stimulus,
                fps=fps,
                reference=reference,
                analysis_width=analysis_width,
            )
        )
    return records

"""Redundant loop-position coding: mean hue angle and luma ramp direction.

Frame ``k`` of a ``K``-frame loop carries its index twice, in channels that fail
under different processing: mean hue survives any spatial rearrangement, the
ramp's gradient direction survives any monotone value map. Decoding classifies.
"""

# pylint: disable=no-member  ; cv2 is a compiled extension pylint cannot introspect

from __future__ import annotations

import enum
from dataclasses import dataclass
from functools import lru_cache
from typing import Iterator, Sequence

import cv2
import numpy as np

from syncsummoner.device.profile import ProgramStyle
from syncsummoner.probe import patterns

__all__ = [
    "ChannelLock",
    "ChannelState",
    "CodeLoop",
    "FrameCode",
    "LoopReport",
    "Registration",
    "Verdict",
    "analyse",
    "build_loop",
    "channel_lock",
    "classify",
    "decode_frame",
    "estimate_rate",
    "read_strip",
    "register",
    "step_rate",
]

#: BT.601 luma weights; the chroma pair used below is the matching opponent basis.
LUMA = np.array([0.299, 0.587, 0.114], dtype=np.float32)
DEFAULT_COUNT = 24
MID_LEVEL = 0.5
#: Peak amplitudes keeping every channel inside [0.12, 0.89], clear of clipping and 565 steps.
CHROMA_AMP = 0.10
RAMP_AMP = 0.20
TEXTURE_AMP = 0.12
STRIP_PX = 8
#: The ramp is fitted at its own scale, where the carrier averages away.
RAMP_GRID = 12

#: Lock, the resultant at the loop's advance rate, measures >=0.98 alive and <=0.10 dead: gap midpoint.
SURVIVAL_MIN = 0.5
#: Rate search window: wide against clock drift, narrow enough to exclude DC and the mirrored rate.
RATE_TOLERANCE = 0.2
#: A monotone value map leaves the ramp explaining >=0.80 of the variance; 2x2 tiling leaves 0.15.
LINEARITY_MIN = 0.5
#: Similarity residuals below these are the registration's own precision.
SHIFT_PX = 1.0
LOG_SCALE_MIN = 0.02
ROTATION_MIN = np.deg2rad(2.0)
#: Above the chain's own chroma demodulation error, measured at ~11 degrees on five programs.
HUE_ROTATION_MIN = np.deg2rad(20.0)
TEXTURE_CORR_MIN = 0.2
#: Log-polar grid, half a degree per angular bin.
POLAR_BINS = (256, 720)
REGISTER_SAMPLES = 4


class ChannelState(enum.Enum):
    """What one index channel is doing across a run of captures."""

    ALIVE = "alive"
    MIRRORED = "mirrored"
    FROZEN = "frozen"
    DESTROYED = "destroyed"


class Verdict(enum.Enum):
    """Class of processing implied by which index channels survived."""

    PASSTHROUGH = "passthrough"
    FROZEN = "frozen"
    GEOMETRIC = "geometric"
    VALUE = "value"
    MIXED = "mixed"
    DESTRUCTIVE = "destructive"

    @property
    def style(self) -> ProgramStyle:
        """Program style this verdict implies."""
        return _STYLE[self]


#: Verdict per (geometric evidence, value evidence), once both channels are accounted for.
_RESIDUE = {
    (True, True): Verdict.MIXED,
    (True, False): Verdict.GEOMETRIC,
    (False, True): Verdict.VALUE,
    (False, False): Verdict.PASSTHROUGH,
}

_STYLE = {
    Verdict.PASSTHROUGH: ProgramStyle.UNKNOWN,
    Verdict.FROZEN: ProgramStyle.UNKNOWN,
    Verdict.GEOMETRIC: ProgramStyle.DIGITAL,
    Verdict.VALUE: ProgramStyle.ANALOG,
    Verdict.MIXED: ProgramStyle.MIXED,
    Verdict.DESTRUCTIVE: ProgramStyle.MIXED,
}


@lru_cache(maxsize=16)
def _axes(width: int, height: int) -> tuple[np.ndarray, np.ndarray]:
    """Centred isotropic pixel axes, zero-mean and mutually orthogonal."""
    scale = max(width, height) / 2.0
    x = ((np.arange(width, dtype=np.float32) - (width - 1) / 2.0) / scale)[None, :]
    y = ((np.arange(height, dtype=np.float32) - (height - 1) / 2.0) / scale)[:, None]
    return x, y


@lru_cache(maxsize=16)
def _basis(width: int, height: int):
    """Plane-fit basis tapered over the inscribed disc, with its moments.

    A rotation about the centre maps the taper onto itself, so every global
    measurement taken under it rotates with the picture instead of mixing in
    whatever the corners became.
    """
    x, y = _axes(width, height)
    radius = np.hypot(x, y) / min(float(x.max()), float(y.max()))
    weight = np.where(radius < 1.0, 0.5 * (1.0 + np.cos(np.pi * radius)), 0.0).astype(np.float32)
    wx, wy = weight * x, weight * y
    return weight, wx, wy, float((wx * x).sum()), float((wy * y).sum()), float(weight.sum())


def _plane(field: np.ndarray, basis) -> tuple[float, float, float]:
    """Tapered least-squares plane, as gradient plus the variance it explains.

    The axes are orthogonal and zero-mean under the taper, so the fit is two
    inner products; each is divided by its own moment to drop the aspect ratio.
    """
    weight, wx, wy, xx, yy, total = basis
    gx = float((field * wx).sum()) / xx
    gy = float((field * wy).sum()) / yy
    mean = float((field * weight).sum()) / total
    variance = float((field * field * weight).sum()) - total * mean * mean
    explained = gx * gx * xx + gy * gy * yy
    return gx, gy, (explained / variance if variance > 0.0 else 0.0)


def _luma(frame: np.ndarray) -> np.ndarray:
    return np.asarray(frame, dtype=np.float32) @ LUMA


def _decimate(frame: np.ndarray, grid: int = RAMP_GRID) -> np.ndarray:
    """Area-average down to the ramp's own scale, where the carrier is not noise."""
    height, width = frame.shape[:2]
    scale = grid / min(height, width)
    if scale >= 1.0:
        return frame
    size = (max(1, round(width * scale)), max(1, round(height * scale)))
    return cv2.resize(frame, size, interpolation=cv2.INTER_AREA)


@dataclass(frozen=True)
class CodeLoop:
    """A K-frame loop whose position is encoded in both mean hue and ramp direction.

    ``texture`` is the plane-free peak-normalised carrier shared by every frame,
    which makes it a registration reference that needs no index.
    """

    width: int
    height: int
    count: int
    texture: np.ndarray
    strip_px: int = STRIP_PX
    chroma_amp: float = CHROMA_AMP
    ramp_amp: float = RAMP_AMP
    texture_amp: float = TEXTURE_AMP
    level: float = MID_LEVEL

    @property
    def body_height(self) -> int:
        """Height of the coded area, above the gray-code strip."""
        return self.height - self.strip_px

    def angle(self, index: int) -> float:
        """Loop angle for a position, shared by both index channels."""
        return 2.0 * np.pi * (int(index) % self.count) / self.count

    def frame(self, index: int) -> np.ndarray:
        """Render loop position ``index`` as ``(H, W, 3) float32`` RGB in ``[0, 1]``."""
        theta = self.angle(index)
        cos, sin = np.cos(theta), np.sin(theta)
        x, y = _axes(self.width, self.body_height)
        span = abs(cos) * float(x.max()) + abs(sin) * float(y.max())
        ramp = (cos * x + sin * y) / span
        luma = self.level + self.ramp_amp * ramp + self.texture_amp * self.texture
        cb, cr = self.chroma_amp * cos, self.chroma_amp * sin
        green = luma - (float(LUMA[0]) * cr + float(LUMA[2]) * cb) / float(LUMA[1])
        body = np.stack([luma + cr, green, luma + cb], axis=-1)
        out = np.zeros((self.height, self.width, 3), dtype=np.float32)
        out[: self.body_height] = np.clip(body, 0.0, 1.0)
        return patterns.with_state_index(out, index, strip_px=self.strip_px)

    def frames(self) -> Iterator[np.ndarray]:
        """Every loop position in order."""
        return (self.frame(k) for k in range(self.count))

    def play(self, repeats: int = 1) -> Iterator[np.ndarray]:
        """The loop repeated, for playout that must outlast dropped captures."""
        for _ in range(int(repeats)):
            yield from self.frames()


def build_loop(
    *, width: int, height: int, rng, count: int = DEFAULT_COUNT, strip_px: int = STRIP_PX
) -> CodeLoop:
    """Build the loop and its shared carrier texture at a playout geometry."""
    body = int(height) - int(strip_px)
    texture = _luma(patterns.generate("band_noise", width=width, height=body, rng=rng))
    x, y = _axes(int(width), body)
    gx, gy, _ = _plane(texture, _basis(int(width), body))
    texture = texture - gx * x - gy * y - texture.mean()
    texture /= float(np.abs(texture).max()) or 1.0
    return CodeLoop(
        width=int(width),
        height=int(height),
        count=int(count),
        texture=texture.astype(np.float32),
        strip_px=int(strip_px),
    )


@dataclass(frozen=True)
class FrameCode:
    """One captured frame's two index readings plus their signal strengths."""

    hue_angle: float
    chroma: float
    ramp_angle: float
    ramp_gain: float
    linearity: float
    strip_index: int | None


def _strip_rows(frame: np.ndarray, loop: CodeLoop) -> int:
    """Strip depth in the captured frame's own rows, which need not be the playout's."""
    if loop.strip_px <= 0:
        return 0
    return max(1, round(loop.strip_px * frame.shape[0] / loop.height))


def _body(frame: np.ndarray, loop: CodeLoop) -> np.ndarray:
    return np.asarray(patterns.crop_strip(frame, strip_px=_strip_rows(frame, loop)), dtype=np.float32)


def read_strip(frame: np.ndarray, loop: CodeLoop) -> int | None:
    """Absolute index from the gray-code strip, which only survives intact geometry."""
    if loop.strip_px <= 0:
        return None
    return patterns.read_state_index(frame, strip_px=_strip_rows(frame, loop))


def decode_frame(frame: np.ndarray, loop: CodeLoop) -> FrameCode:
    """Read both index channels from one captured frame, independently of any other.

    Mean chroma is a linear functional of the pixels, so its angle survives any
    permutation and any luma gain; the fitted plane's direction survives any
    monotone value map.
    """
    strip = read_strip(frame, loop)
    small = _decimate(_body(frame, loop))
    basis = _basis(small.shape[1], small.shape[0])
    weight, total = basis[0], basis[5]
    means = (small * weight[:, :, None]).sum(axis=(0, 1)) / total
    grey = float(means @ LUMA)
    cb, cr = float(means[2]) - grey, float(means[0]) - grey
    gx, gy, linearity = _plane(_luma(small), basis)
    return FrameCode(
        hue_angle=float(np.arctan2(cr, cb)),
        chroma=float(np.hypot(cb, cr)),
        ramp_angle=float(np.arctan2(gy, gx)),
        ramp_gain=float(np.hypot(gx, gy)),
        linearity=float(linearity),
        strip_index=strip,
    )


@dataclass(frozen=True)
class Registration:
    """Similarity transform taking the loop's carrier texture to a captured one."""

    shift: tuple[float, float]
    scale: float
    rotation: float
    response: float
    correlation: float


def _detrend(field: np.ndarray) -> np.ndarray:
    """Remove the ramp, leaving the carrier texture and whatever replaced it."""
    x, y = _axes(field.shape[1], field.shape[0])
    gx, gy, _ = _plane(field, _basis(field.shape[1], field.shape[0]))
    out = field - gx * x - gy * y - field.mean()
    return (out / (float(out.std()) or 1.0)).astype(np.float32)


def _square(field: np.ndarray) -> np.ndarray:
    """Largest centred square, which is what makes the frequency axes isotropic."""
    side = min(field.shape)
    top, left = (field.shape[0] - side) // 2, (field.shape[1] - side) // 2
    return field[top : top + side, left : left + side]


def _radial_window(side: int) -> np.ndarray:
    """Rotationally symmetric Hann window, so a rotated frame is windowed identically."""
    axis = (np.arange(side, dtype=np.float32) - (side - 1) / 2.0) / (side / 2.0)
    radius = np.hypot(axis[None, :], axis[:, None])
    return np.where(radius < 1.0, 0.5 * (1.0 + np.cos(np.pi * radius)), 0.0).astype(np.float32)


def _log_polar(field: np.ndarray, window: np.ndarray) -> tuple[np.ndarray, float]:
    spectrum = np.log1p(np.abs(np.fft.fftshift(np.fft.fft2(field * window)))).astype(np.float32)
    centre = ((spectrum.shape[1] - 1) / 2.0, (spectrum.shape[0] - 1) / 2.0)
    radius = min(centre)
    flags = cv2.INTER_LINEAR + cv2.WARP_POLAR_LOG
    return cv2.warpPolar(spectrum, POLAR_BINS, centre, radius, flags), radius


def _similarity(reference: np.ndarray, capture: np.ndarray) -> tuple[float, float]:
    """Scale and rotation from the log-polar magnitude spectra (Fourier-Mellin).

    Both frames are cropped square and radially windowed first: on a rectangular
    grid an image rotation is not a rotation of the sampled spectrum.
    """
    window = _radial_window(min(reference.shape))
    polar_ref, radius = _log_polar(_square(reference), window)
    polar_cap, _ = _log_polar(_square(capture), window)
    (d_rho, d_phi), _ = cv2.phaseCorrelate(polar_ref.astype(np.float64), polar_cap.astype(np.float64))
    scale = float(np.exp(-d_rho * np.log(radius) / POLAR_BINS[0]))
    return scale, float(2.0 * np.pi * d_phi / POLAR_BINS[1])


def _align(capture: np.ndarray, scale: float, rotation: float) -> np.ndarray:
    height, width = capture.shape
    centre = ((width - 1) / 2.0, (height - 1) / 2.0)
    matrix = cv2.getRotationMatrix2D(centre, float(np.degrees(rotation)), 1.0 / scale)
    return cv2.warpAffine(capture, matrix, (width, height), borderMode=cv2.BORDER_REFLECT_101)


def _correlation(reference: np.ndarray, field: np.ndarray) -> float:
    """Pearson correlation, zero where either field carries no variance."""
    if float(field.std()) <= 0.0:
        return 0.0
    return float(np.corrcoef(reference.ravel(), field.ravel())[0, 1])


def register(frame: np.ndarray, loop: CodeLoop) -> Registration:
    """Recover the global similarity transform from the shared carrier texture.

    Scale and rotation come from the log-polar magnitude spectrum and translation
    from phase correlation once they are undone; the spectrum's 180 degree
    ambiguity and the value map's polarity go to whichever pair correlates best.
    """
    reference = loop.texture
    capture = _detrend(_luma(_body(frame, loop)))
    if capture.shape != reference.shape:
        capture = cv2.resize(capture, (reference.shape[1], reference.shape[0]))
    window = cv2.createHanningWindow((reference.shape[1], reference.shape[0]), cv2.CV_32F)
    windowed = (reference * window).astype(np.float64)
    scale, rotation = _similarity(reference, capture)
    best = None
    for candidate in (rotation, rotation + np.pi):
        aligned = _align(capture, scale, candidate)
        for polarity in (1.0, -1.0):
            shift, response = cv2.phaseCorrelate(windowed, (polarity * aligned * window).astype(np.float64))
            settled = np.roll(aligned, (-int(round(shift[1])), -int(round(shift[0]))), axis=(0, 1))
            corr = _correlation(reference, settled)
            if best is None or abs(corr) > abs(best.correlation):
                best = Registration(
                    shift=(float(shift[0]), float(shift[1])),
                    scale=float(scale),
                    rotation=_wrap(candidate),
                    response=float(response),
                    correlation=corr,
                )
    return best


def _resultant(angles) -> tuple[float, float]:
    """Circular concentration in ``[0, 1]`` and mean angle of a set of angles."""
    mean = np.exp(1j * np.asarray(angles, dtype=np.float64)).mean()
    return float(abs(mean)), float(np.angle(mean))


def _wrap(angle: float) -> float:
    return float(np.arctan2(np.sin(angle), np.cos(angle)))


def step_rate(count: int, loop_fps: float, capture_fps: float) -> float:
    """Loop angle advanced per captured frame, from the two frame rates."""
    return 2.0 * np.pi * (float(loop_fps) / float(capture_fps)) / int(count)


def _lock(angles: np.ndarray, rate: float) -> tuple[float, float]:
    """Circular resultant and phase of angles detrended by a constant advance rate."""
    if angles.size == 0:
        return 0.0, 0.0
    mean = np.exp(1j * (angles - rate * np.arange(angles.size))).mean()
    return float(abs(mean)), float(np.angle(mean))


@dataclass(frozen=True)
class ChannelLock:
    """How one index channel tracks the loop, as resultants at three trial rates."""

    state: ChannelState
    lock: float
    phase: float
    forward: float
    reverse: float
    constant: float

    @property
    def tracking(self) -> bool:
        """Whether the channel still carries an advancing index, in either direction."""
        return self.state in (ChannelState.ALIVE, ChannelState.MIRRORED)

    @property
    def orientation(self) -> int:
        """Sign of the advance the channel locked to, zero where it locked to none."""
        return {ChannelState.ALIVE: 1, ChannelState.MIRRORED: -1}.get(self.state, 0)


def channel_lock(angles, rate: float) -> ChannelLock:
    """Classify one channel's decoded angles by how they track a known advance rate.

    Captures are in temporal order, so a surviving channel satisfies
    ``theta_n = phase + rate * n``; a destroyed one decodes to a constant, not to noise,
    which is why the constant rate is tested separately instead of scanned over.
    """
    theta = np.asarray(angles, dtype=np.float64)
    forward, phase = _lock(theta, rate)
    reverse, reverse_phase = _lock(theta, -rate)
    constant, constant_phase = _lock(theta, 0.0)
    if max(forward, reverse) < SURVIVAL_MIN:
        state = ChannelState.FROZEN if constant >= SURVIVAL_MIN else ChannelState.DESTROYED
        return ChannelLock(state, max(forward, reverse), constant_phase, forward, reverse, constant)
    if reverse > forward:
        return ChannelLock(ChannelState.MIRRORED, reverse, reverse_phase, forward, reverse, constant)
    return ChannelLock(ChannelState.ALIVE, forward, phase, forward, reverse, constant)


def estimate_rate(channels: Sequence[np.ndarray], expected: float) -> float:
    """Best-locking rate within tolerance of the expected one, which is the fallback.

    Only the best channel votes: a destroyed channel's constant leaks off DC and would
    otherwise pull a summed score away from the rate the live channel is holding. The
    grid is fine enough that its residual tilts the fitted phase by well under one step.
    """
    length = min(int(np.size(c)) for c in channels)
    span = RATE_TOLERANCE * abs(expected)
    if length < 2 or span <= 0.0:
        return expected
    steps = max(3, int(round(8.0 * RATE_TOLERANCE * length)) + 1)
    grid = np.linspace(expected - span, expected + span, steps)
    basis = np.exp(-1j * np.outer(grid, np.arange(length))) / length
    best = np.zeros(grid.size)
    for channel in channels:
        wave = np.exp(1j * np.asarray(channel[:length], dtype=np.float64))
        best = np.maximum(best, np.maximum(np.abs(basis @ wave), np.abs(basis @ wave.conj())))
    peak = int(np.argmax(best))
    return float(grid[peak]) if best[peak] >= SURVIVAL_MIN else expected


def _indices(angles: np.ndarray, offset: float, count: int) -> np.ndarray:
    """Grid positions the angles decode to, once their constant offset is removed."""
    return np.rint(count * (angles - offset) / (2.0 * np.pi)).astype(np.intp) % count


def _grid_offset(phase: float, count: int) -> float:
    """The part of a phase smaller than one grid step, which is decode bias not lag."""
    step = 2.0 * np.pi / count
    return float((phase + step / 2.0) % step - step / 2.0)


def _index_lag(angles: np.ndarray, rate: float, count: int) -> int:
    """Commonest offset in loop steps between what a capture decodes to and the loop.

    The mode, not the mean, so that a run whose first captures repeat still reports the
    lag the rest of it holds.
    """
    residual = np.asarray(angles) - rate * np.arange(np.size(angles))
    bins = np.rint(count * residual / (2.0 * np.pi)).astype(np.intp) % count
    return int((np.bincount(bins, minlength=count).argmax() + count // 2) % count - count // 2)


def _strip_agreement(strip: Sequence[int | None], indices: np.ndarray, count: int) -> float:
    """Fraction of captures whose decoded index sits at the strip's commonest offset.

    Capture starts wherever the loop happens to be, so the ground truth to test against
    is that the two indices keep step, not that they are equal.
    """
    offsets = [(int(k) - int(s)) % count for s, k in zip(strip, indices) if s is not None and s >= 0]
    if not offsets:
        return 0.0
    common = int(np.bincount(np.asarray(offsets, dtype=np.intp), minlength=count).argmax())
    return float(offsets.count(common) / len(indices))


def _sequence_stats(indices: np.ndarray, count: int, ratio: float) -> tuple[float, float]:
    """Repeat and scramble fractions of a decoded index sequence, read against the rate.

    Capture outrunning playout makes a fraction ``1 - ratio`` of steps stand still by
    design, so repeats count only the scheduled advances that failed to happen.
    """
    if indices.size < 2:
        return 0.0, 0.0
    step = (np.diff(indices) + count // 2) % count - count // 2
    idle = float(np.mean(step == 0))
    ordered = float(np.mean((step == 0) | ((step > 0) & (step <= max(1, count // 4)))))
    advance = min(1.0, max(float(ratio), 1.0 / count))
    return max(0.0, (idle - (1.0 - advance)) / advance), 1.0 - ordered


@dataclass(frozen=True)
class LoopReport:
    """Decoded loop behaviour of one program, with the residuals that size it."""

    verdict: Verdict
    style: ProgramStyle
    confidence: float
    frames: int
    rate: float
    coverage: float
    hue_state: ChannelState
    ramp_state: ChannelState
    hue_survival: float
    ramp_survival: float
    agreement: float
    orientation: int
    linearity: float
    chroma_gain: float
    hue_rotation: float
    image_rotation: float
    scale: float
    shift: tuple[float, float]
    response: float
    correlation: float
    inverting: bool
    repeat_fraction: float
    scramble_fraction: float
    index_lag: int
    delay_frames: int
    strip_agreement: float


def classify(
    hue: ChannelLock,
    ramp: ChannelLock,
    *,
    agreement: float = 1.0,
    linearity: float = 1.0,
    similarity: bool = False,
    inverting: bool = False,
    hue_rotation: float = float("nan"),
) -> tuple[Verdict, float]:
    """Verdict and confidence from the two channel states and the residual evidence.

    Each channel dies under its own class of processing, so a channel that stopped
    tracking while the other kept going is itself the evidence; two that both track
    but disagree have decoded nothing, which is the redundancy paying for itself.
    """
    if hue.state is ChannelState.FROZEN and ramp.state is ChannelState.FROZEN:
        return Verdict.FROZEN, min(hue.constant, ramp.constant)
    tracking = (hue.tracking, ramp.tracking)
    if not any(tracking):
        return Verdict.DESTRUCTIVE, 1.0 - max(hue.lock, ramp.lock)
    if all(tracking) and agreement < SURVIVAL_MIN:
        return Verdict.DESTRUCTIVE, 1.0 - agreement
    geometric = bool(
        similarity or linearity < LINEARITY_MIN or (hue.tracking and ramp.state is not ChannelState.ALIVE)
    )
    value = bool(
        inverting
        or (np.isfinite(hue_rotation) and abs(hue_rotation) > HUE_ROTATION_MIN)
        or (ramp.state is ChannelState.ALIVE and hue.state is not ChannelState.ALIVE)
    )
    exclusive = tracking[0] != tracking[1]
    confidence = abs(hue.lock - ramp.lock) if exclusive else min(hue.lock, ramp.lock, agreement)
    return _RESIDUE[(geometric, value)], confidence


def analyse(
    captures: Sequence[np.ndarray],
    loop: CodeLoop,
    *,
    samples: int = REGISTER_SAMPLES,
    rate: float | None = None,
    fps: tuple[float, float] | None = None,
) -> LoopReport:
    """Decode a run of captured frames into a labelled, quantified verdict.

    Captures are taken in temporal order at a known loop advance per frame, ``rate``
    or ``fps`` as ``(loop_fps, capture_fps)``, defaulting to one loop step per capture.
    Registration runs on a few frames only, the carrier texture being common to the loop.
    """
    codes = [decode_frame(frame, loop) for frame in captures]
    hue = np.array([c.hue_angle for c in codes])
    ramp = np.array([c.ramp_angle for c in codes])
    expected = rate if rate is not None else 2.0 * np.pi / loop.count
    if rate is None and fps is not None:
        expected = step_rate(loop.count, *fps)
    locked = estimate_rate((hue, ramp), float(expected))
    hue_lock, ramp_lock = channel_lock(hue, locked), channel_lock(ramp, locked)
    direct, _ = _resultant(hue - ramp)
    mirrored, _ = _resultant(hue + ramp)
    agreement = max(direct, mirrored)
    orientation = -1 if ChannelState.MIRRORED in (hue_lock.state, ramp_lock.state) else 1

    picked = np.unique(np.linspace(0, len(captures) - 1, min(samples, len(captures))).astype(np.intp))
    fits = [register(captures[i], loop) for i in picked]
    matched = [f for f in fits if abs(f.correlation) >= TEXTURE_CORR_MIN] or fits
    scale = float(np.median([f.scale for f in matched]))
    rotation = _wrap(_resultant([f.rotation for f in matched])[1])
    shift = (
        float(np.median([f.shift[0] for f in matched])),
        float(np.median([f.shift[1] for f in matched])),
    )
    response = float(np.median([f.response for f in matched]))
    correlation = float(np.median([f.correlation for f in matched]))

    registered = abs(correlation) >= TEXTURE_CORR_MIN
    paired = hue_lock.state is ChannelState.ALIVE and ramp_lock.state is ChannelState.ALIVE
    delta = _wrap(hue_lock.phase - ramp_lock.phase)
    hue_rotation = _wrap(delta + rotation) if paired and registered else float("nan")
    linearity = float(np.median([c.linearity for c in codes]))
    inverting = bool(correlation < -TEXTURE_CORR_MIN)
    similarity = registered and (
        abs(shift[0]) > SHIFT_PX
        or abs(shift[1]) > SHIFT_PX
        or abs(np.log(scale)) > LOG_SCALE_MIN
        or abs(rotation) > ROTATION_MIN
    )
    lead = hue_lock if hue_lock.lock >= ramp_lock.lock else ramp_lock
    sign = float(lead.orientation or 1)
    angles = sign * (hue if lead is hue_lock else ramp)
    offset = _grid_offset(sign * lead.phase, loop.count)
    indices = _indices(angles, offset, loop.count)
    repeat, scramble = _sequence_stats(indices, loop.count, loop.count * locked / (2.0 * np.pi))
    lag = _index_lag(angles - offset, locked, loop.count)
    strip = _strip_agreement([c.strip_index for c in codes], indices, loop.count)
    if len(codes) > 1 and np.unique(indices).size < 2:
        verdict, confidence = Verdict.FROZEN, repeat
    else:
        verdict, confidence = classify(
            hue_lock,
            ramp_lock,
            agreement=agreement,
            linearity=linearity,
            similarity=bool(similarity),
            inverting=inverting,
            hue_rotation=hue_rotation,
        )
    return LoopReport(
        verdict=verdict,
        style=verdict.style,
        confidence=float(confidence),
        frames=len(codes),
        rate=float(locked),
        coverage=float(np.unique(indices).size / loop.count),
        hue_state=hue_lock.state,
        ramp_state=ramp_lock.state,
        hue_survival=hue_lock.lock,
        ramp_survival=ramp_lock.lock,
        agreement=agreement,
        orientation=orientation,
        linearity=linearity,
        chroma_gain=float(np.median([c.chroma for c in codes]) / loop.chroma_amp),
        hue_rotation=hue_rotation,
        image_rotation=rotation,
        scale=scale,
        shift=shift,
        response=response,
        correlation=correlation,
        inverting=inverting,
        repeat_fraction=repeat,
        scramble_fraction=scramble,
        index_lag=lag,
        delay_frames=int(-lag % loop.count),
        strip_agreement=strip,
    )

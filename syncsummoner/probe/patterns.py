"""Synthetic stimulus battery and gray-code frame identification.

Frames are ``(H, W, 3) float32`` in ``[0, 1]``, RGB, generated with numpy only.
The state-index strip is what makes unattended sweeping possible: every captured
frame self-identifies the parameter vector that produced it.
"""

from __future__ import annotations

import numpy as np

__all__ = [
    "PATTERNS",
    "generate",
    "with_state_index",
    "read_state_index",
    "crop_strip",
    "state_index_capacity",
]

PATTERNS = (
    "zoneplate",
    "grating_sweep",
    "smpte_bars",
    "luma_ramp",
    "chroma_ramp",
    "dot_lattice",
    "siemens_star",
    "noise_1f",
    "band_noise",
    "flat_grey",
    "black",
    "motion_ball",
)

#: Band-noise blur radius as a fraction of width; chosen by measuring the
#: trade-off between misalignment robustness and displacement sensitivity.
BAND_NOISE_SIGMA = 0.025
_GUARD_CELLS = 2
_SUBSAMPLES = 5
_INNER = 0.5


def generate(name, *, width, height, frame_index=0, rng, **params):
    """Render one stimulus frame as ``(H, W, 3) float32`` in ``[0, 1]``, RGB.

    ``frame_index`` advances the time-varying patterns; ``rng`` seeds the
    stochastic ones. Extra keyword arguments are pattern-specific overrides.
    """
    try:
        fn = _GENERATORS[name]
    except KeyError as exc:
        raise ValueError(f"unknown pattern {name!r}; known: {', '.join(PATTERNS)}") from exc
    frame = fn(int(width), int(height), int(frame_index), rng, **params)
    if frame.ndim == 2:
        frame = np.repeat(frame[:, :, None], 3, axis=2)
    return np.clip(frame, 0.0, 1.0).astype(np.float32, copy=False)


def _centred(width, height):
    radius = max(width, height) / 2.0
    x = (np.arange(width, dtype=np.float32) - (width - 1) / 2.0) / radius
    y = (np.arange(height, dtype=np.float32) - (height - 1) / 2.0) / radius
    return x[None, :], y[:, None]


def _zoneplate(width, height, _frame_index, _rng, nyquist_frac=1.0):
    radius_px = min(width, height) / 2.0
    x = np.arange(width, dtype=np.float32) - (width - 1) / 2.0
    y = np.arange(height, dtype=np.float32) - (height - 1) / 2.0
    r2 = y[:, None] ** 2 + x[None, :] ** 2
    return 0.5 + 0.5 * np.cos(np.pi * (0.5 * nyquist_frac / radius_px) * r2)


def _grating_sweep(
    width, height, frame_index, _rng, cpp_min=0.005, cpp_max=0.25, hz=1.0, fps=60.0, angle=0.0
):
    x, y = _centred(width, height)
    scale = max(width, height) / 2.0
    u = (np.cos(angle) * x + np.sin(angle) * y) * scale + scale
    chirp = cpp_min * u + 0.5 * (cpp_max - cpp_min) / (2.0 * scale) * u**2
    return 0.5 + 0.5 * np.sin(2.0 * np.pi * (chirp + hz * frame_index / fps))


_BARS75 = np.array(
    [
        [0.75, 0.75, 0.75],
        [0.75, 0.75, 0.0],
        [0.0, 0.75, 0.75],
        [0.0, 0.75, 0.0],
        [0.75, 0.0, 0.75],
        [0.75, 0.0, 0.0],
        [0.0, 0.0, 0.75],
    ],
    dtype=np.float32,
)
_PLUGE = np.array(
    [[0.0, 0.0, 0.2], [1.0, 1.0, 1.0], [0.2, 0.0, 0.2], [0.0, 0.0, 0.0], [0.04, 0.04, 0.04]],
    dtype=np.float32,
)


def _bar_row(width, colors):
    idx = np.minimum((np.arange(width) * len(colors)) // width, len(colors) - 1)
    return colors[idx]


def _smpte_bars(width, height, _frame_index, _rng):
    top = int(height * 2 / 3)
    mid = int(height * 3 / 4)
    frame = np.empty((height, width, 3), dtype=np.float32)
    frame[:top] = _bar_row(width, _BARS75)
    frame[top:mid] = _bar_row(width, _BARS75[::-1] * np.float32([[0.0, 0.0, 1.0]]))
    frame[mid:] = _bar_row(width, _PLUGE)
    return frame


def _luma_ramp(width, height, _frame_index, _rng, steps=0):
    ramp = np.linspace(0.0, 1.0, width, dtype=np.float32)
    if steps:
        ramp = np.floor(ramp * steps) / max(steps - 1, 1)
    return np.repeat(ramp[None, :], height, axis=0)


def _hue_rgb(hue):
    h6 = 6.0 * hue
    return np.stack(
        [
            np.clip(np.abs(h6 - 3.0) - 1.0, 0.0, 1.0),
            np.clip(2.0 - np.abs(h6 - 2.0), 0.0, 1.0),
            np.clip(2.0 - np.abs(h6 - 4.0), 0.0, 1.0),
        ],
        axis=-1,
    )


def _chroma_ramp(width, height, _frame_index, _rng):
    hue = _hue_rgb(np.linspace(0.0, 1.0, width, endpoint=False, dtype=np.float32))
    sat = np.linspace(1.0, 0.0, height, dtype=np.float32)[:, None, None]
    return (1.0 - sat) + sat * hue[None, :, :]


def _dot_lattice(width, height, _frame_index, _rng, pitch=16.0, sigma=1.6, offset=0.0):
    def profile(n):
        p = (np.arange(n, dtype=np.float32) + offset) % pitch - pitch / 2.0
        return np.exp(-0.5 * (p / sigma) ** 2)

    return profile(height)[:, None] * profile(width)[None, :]


def _siemens_star(width, height, _frame_index, _rng, spokes=36):
    x, y = _centred(width, height)
    theta = np.arctan2(np.broadcast_to(y, (height, width)), np.broadcast_to(x, (height, width)))
    return (np.sin(spokes * theta) > 0).astype(np.float32)


def _noise_1f(width, height, _frame_index, rng, beta=1.0):
    white = rng.standard_normal((height, width)).astype(np.float32)
    fy = np.fft.fftfreq(height).astype(np.float32)[:, None]
    fx = np.fft.rfftfreq(width).astype(np.float32)[None, :]
    radial = np.hypot(fy, fx)
    radial[0, 0] = np.inf
    field = np.fft.irfft2(np.fft.rfft2(white) * radial**-beta, s=(height, width))
    field -= field.mean()
    scale = 6.0 * field.std() or 1.0
    return 0.5 + field / scale


def _band_noise(width, height, _frame_index, rng, sigma_frac=BAND_NOISE_SIGMA):
    """Low-pass colour noise: survives misalignment, breaks under displacement.

    A zone plate collapses from 0.999 to 0.013 pointwise fit under one pixel of
    shift and a ramp survives 24 pixels at 0.912; this sits between. Chromatic
    because a grey stimulus reads as the capture card's achromatic splash.
    """
    import cv2  # pylint: disable=import-outside-toplevel,no-member

    blob = max(2.0, sigma_frac * width)
    coarse = rng.random((max(2, int(height / blob)), max(2, int(width / blob)), 3)).astype(np.float32)
    field = cv2.resize(coarse, (width, height), interpolation=cv2.INTER_CUBIC)  # pylint: disable=no-member
    span = float(np.ptp(field)) or 1.0
    return np.clip((field - field.min()) / span, 0.0, 1.0)


def _flat_grey(width, height, _frame_index, _rng, level=0.5):
    return np.full((height, width), np.float32(level))


def _black(width, height, _frame_index, _rng):
    return np.zeros((height, width), dtype=np.float32)


def _motion_ball(width, height, frame_index, _rng, fps=60.0, hz=1.0, radius=0.08, background=0.1):
    t = frame_index / fps
    x, y = _centred(width, height)
    cx = 0.8 * np.sin(2.0 * np.pi * 0.5 * hz * t)
    cy = 0.7 - 1.4 * np.abs(np.sin(np.pi * hz * t))
    dist = np.hypot(x - cx, y - cy)
    edge = 2.0 / max(width, height)
    ball = np.clip((radius - dist) / edge, 0.0, 1.0)
    return background + (1.0 - background) * ball


_GENERATORS = {
    "zoneplate": _zoneplate,
    "grating_sweep": _grating_sweep,
    "smpte_bars": _smpte_bars,
    "luma_ramp": _luma_ramp,
    "chroma_ramp": _chroma_ramp,
    "dot_lattice": _dot_lattice,
    "siemens_star": _siemens_star,
    "noise_1f": _noise_1f,
    "band_noise": _band_noise,
    "flat_grey": _flat_grey,
    "black": _black,
    "motion_ball": _motion_ball,
}


def state_index_capacity(bits=8):
    """Number of distinct state indices a ``bits``-wide strip can carry."""
    return 1 << bits


def _cell_edges(width, bits):
    return np.round(np.linspace(0, width, bits + _GUARD_CELLS + 1)).astype(np.intp)


def _gray_encode(index, bits):
    masked = int(index) & (state_index_capacity(bits) - 1)
    return masked ^ (masked >> 1)


def _gray_decode(code, bits):
    value = int(code)
    shift = 1
    while shift < bits:
        value ^= value >> shift
        shift <<= 1
    return value


def with_state_index(frame, index, *, bits=8, strip_px=8):
    """Burn ``index`` as a gray code into a high-contrast strip on the bottom edge.

    Cell 0 is a white reference and the last cell a black one, so the decoder can
    threshold against the strip's own midpoint under arbitrary gain and pedestal.
    """
    out = np.array(frame, dtype=np.float32, copy=True)
    height, width = out.shape[:2]
    strip_px = min(int(strip_px), height)
    edges = _cell_edges(width, bits)
    code = _gray_encode(index, bits)
    levels = np.empty(bits + _GUARD_CELLS, dtype=np.float32)
    levels[0] = 1.0
    levels[-1] = 0.0
    levels[1:-1] = [(code >> (bits - 1 - b)) & 1 for b in range(bits)]
    for cell, level in enumerate(levels):
        out[height - strip_px :, edges[cell] : edges[cell + 1], :] = level
    return out


def _sample_cells(frame, bits, strip_px):
    height, width = frame.shape[:2]
    strip_px = max(min(int(strip_px), height), 1)
    luma = frame[height - strip_px :].mean(axis=2)
    rows = luma[strip_px // 4 : strip_px - strip_px // 4] if strip_px >= 4 else luma
    n_cells = bits + _GUARD_CELLS
    centres = (np.arange(n_cells, dtype=np.float64) + 0.5) / n_cells
    offsets = np.linspace(-_INNER / 2.0, _INNER / 2.0, _SUBSAMPLES) / n_cells
    cols = np.clip(np.round((centres[:, None] + offsets[None, :]) * width), 0, width - 1).astype(np.intp)
    return np.median(rows[:, cols], axis=(0, 2))


def read_state_index(frame, *, bits=8, strip_px=8, min_contrast=0.15):
    """Recover the gray-coded state index, or ``None`` if no legible strip.

    Bit cells are sampled at their centres in relative coordinates, so horizontal
    scaling is harmless, and thresholded against the guard cells' midpoint, so
    luma gain and black pedestal are too.
    """
    cells = _sample_cells(np.asarray(frame, dtype=np.float32), bits, strip_px)
    high, low = float(cells[0]), float(cells[-1])
    if high - low < min_contrast:
        return None
    code = 0
    for bit in cells[1:-1] > 0.5 * (high + low):
        code = (code << 1) | int(bit)
    return _gray_decode(code, bits)


def crop_strip(frame, *, strip_px=8):
    """Remove the state-index strip before analysis."""
    frame = np.asarray(frame)
    return frame[: frame.shape[0] - int(strip_px)]

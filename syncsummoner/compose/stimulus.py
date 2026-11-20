"""A synthetic moving picture whose motion is known on every frame, made from a seed and nothing stored.

Each shot is a tileable band-limited colour noise texture under drift, spin and zoom, a scrambled Sobol
point over :class:`Coverage`. Each frame's motion is scaled to a TI drawn afresh for it, so the picture's
TI is white about one level and every frame is an independent sample for the follow test; frame ``k``
depends on seed, raster and ``k``.
"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass, field, replace
from functools import lru_cache
from typing import Iterator

import numpy as np
from scipy.stats import qmc

from syncsummoner.progress import track

#: BT.709 luma weights, for the colour each shot's chroma is built around.
KR, KB = 0.2126, 0.0722
#: Sobol dimensions a shot draws: speed, direction, spin, zoom, wavelength, contrast, level, hue, saturation,
#: and how far its TI swings frame to frame.
DIMS = 10
#: Points in one cached block of a seed's Sobol sequence.
BLOCK = 1024


@dataclass(frozen=True)
class Coverage:
    """The ranges shots are drawn over, each ``(low, high)``, scales log-uniform.

    ``speed`` is drift in pixels a frame, ``spin`` radians and ``zoom`` log scale a frame either way:
    together a shot's mix of motion, which each frame scales to its TI. That TI is log-normal about the
    one the median drift gives the median texture (:meth:`ti`), ``swing`` its log deviation.
    ``wavelength`` is the texture's in pixels; the rest are fractions of full scale. ``tint`` weights a
    shot's hue against its chroma noise: at zero every frame spans every hue and the shot's hue is moot.
    """

    speed: tuple[float, float] = (0.25, 48.0)
    spin: float = 0.02
    zoom: float = 0.01
    wavelength: tuple[float, float] = (3.0, 256.0)
    contrast: tuple[float, float] = (0.1, 0.5)
    level: tuple[float, float] = (0.2, 0.8)
    saturation: tuple[float, float] = (0.0, 0.3)
    swing: tuple[float, float] = (0.25, 1.5)
    tint: float = 1.0

    def one_texture(self) -> Coverage:
        """These ranges with the texture's held at their middles: every shot alike but for its motion.

        A survival ratio depends on the picture as well as the setting, so a sweep whose settings each see
        their own texture measures texture as scatter; motion still varies frame to frame.
        """
        middle = math.sqrt(self.wavelength[0] * self.wavelength[1])
        return replace(
            self,
            wavelength=(middle, middle),
            contrast=(sum(self.contrast) / 2,) * 2,
            level=(sum(self.level) / 2,) * 2,
            saturation=(sum(self.saturation) / 2,) * 2,
        )

    def monochrome(self) -> Coverage:
        """:meth:`one_texture` with no chroma: the same luma, black and white."""
        return replace(self.one_texture(), saturation=(0.0, 0.0))

    def full_colour(self) -> Coverage:
        """:meth:`one_texture` with untinted chroma noise as wide as the gamut holds at the mean level.

        With no tint every frame spans every hue, so no setting sees a hue another does not; the saturation
        puts three standard deviations of chroma, the luma's own span, on the BT.709 gamut's nearest edge.
        """
        held = self.one_texture()
        return replace(held, saturation=(gamut_saturation(held.level[0]),) * 2, tint=0.0)

    def ti(self) -> float:
        """The TI frames swing about: the median speed's drift over the median wavelength and contrast."""
        speed, wavelength = math.sqrt(self.speed[0] * self.speed[1]), math.sqrt(math.prod(self.wavelength))
        return _ti_scale(sum(self.contrast) / 2, gradient_rms(wavelength)) * speed


@dataclass(frozen=True)
class Shot:
    """One shot's texture, and its motion: a mix of rates scaled each frame to a TI drawn about ``ti``."""

    velocity: tuple[float, float]
    spin: float
    zoom: float
    wavelength: float
    contrast: float
    level: float
    hue: float
    saturation: float
    swing: float
    ti: float
    seed: int
    tint: float = 1.0


def _log_span(lo_hi: tuple[float, float], x: float) -> float:
    return float(math.exp(math.log(lo_hi[0]) + x * (math.log(lo_hi[1]) - math.log(lo_hi[0]))))


def _span(lo_hi: tuple[float, float], x: float) -> float:
    return float(lo_hi[0] + x * (lo_hi[1] - lo_hi[0]))


@dataclass(frozen=True)
class Stimulus:
    """Shots of ``shot_frames`` frames at ``width`` by ``height``, drawn from ``seed`` over ``coverage``."""

    seed: int
    width: int
    height: int
    shot_frames: int = 90
    texture: int = 1024
    coverage: Coverage = field(default_factory=Coverage)

    def key(self) -> dict:
        """Everything a frame depends on, JSON-ready."""
        return {"stimulus": asdict(self)}

    def shot(self, k: int) -> Shot:
        """Shot ``k``'s parameters; any integer ``k``, negative ones for a lead-in."""
        u, cov = _sobol(self.seed, k), self.coverage
        speed, angle = _log_span(cov.speed, u[0]), 2 * math.pi * u[1]
        return Shot(
            velocity=(speed * math.cos(angle), speed * math.sin(angle)),
            spin=cov.spin * (2 * u[2] - 1),
            zoom=cov.zoom * (2 * u[3] - 1),
            wavelength=_log_span(cov.wavelength, u[4]),
            contrast=_span(cov.contrast, u[5]),
            level=_span(cov.level, u[6]),
            hue=float(u[7]),
            saturation=_span(cov.saturation, u[8]),
            swing=_span(cov.swing, u[9]),
            ti=cov.ti(),
            seed=int(np.random.SeedSequence([self.seed, k % (1 << 32)]).generate_state(1)[0]),
            tint=cov.tint,
        )

    def placed(self, index: int) -> tuple[int, int]:
        """The shot frame ``index`` is in, and how far into it."""
        return index // self.shot_frames, index % self.shot_frames

    def path(self, k: int) -> np.ndarray:
        """Shot ``k``'s motion before each frame: ``(shot_frames, 4)`` of x, y, angle and log zoom."""
        return _path(self.shot(k), self.shot_frames, (self.width, self.height), self.texture)

    def transform(self, index: int) -> np.ndarray:
        """The 2x3 affine taking frame ``index``'s pixel ``(x, y, 1)`` to its point on the shot's texture."""
        k, i = self.placed(index)
        x, y, angle, zoom = self.path(k)[i]
        scale = math.exp(-zoom)
        c, s = math.cos(angle) * scale, math.sin(angle) * scale
        cx, cy = self.width / 2, self.height / 2
        return np.array([[c, -s, cx + x - (c * cx - s * cy)], [s, c, cy + y - (s * cx + c * cy)]])

    def flow(self, index: int) -> np.ndarray:
        """Each pixel's displacement from frame ``index`` to the next, ``(height, width, 2)``, NaN at a cut.

        The texture point a pixel shows on this frame is where the next frame's inverse transform puts it.
        """
        if self.placed(index)[1] + 1 >= self.shot_frames:
            return np.full((self.height, self.width, 2), np.nan)
        y, x = np.indices((self.height, self.width), dtype=np.float64)
        here = np.einsum("ij,jhw->ihw", self.transform(index), np.stack([x, y, np.ones_like(x)]))
        inverse = np.linalg.inv(np.vstack([self.transform(index + 1), [0, 0, 1]]))[:2]
        moved = np.einsum("ij,jhw->ihw", inverse, np.concatenate([here, np.ones((1, *x.shape))]))
        return np.stack([moved[0] - x, moved[1] - y], axis=-1)

    def frame(self, index: int) -> np.ndarray:
        """Frame ``index`` as ``uint8`` RGB at the raster."""
        import cv2  # pylint: disable=import-outside-toplevel,no-member

        shot = self.shot(self.placed(index)[0])
        warped = cv2.warpAffine(  # pylint: disable=no-member
            _texture(self.texture, shot.wavelength, shot.seed),
            self.transform(index),
            (self.width, self.height),
            flags=cv2.INTER_LINEAR | cv2.WARP_INVERSE_MAP,  # pylint: disable=no-member
            borderMode=cv2.BORDER_WRAP,  # pylint: disable=no-member
        )
        return _colour(warped, shot)

    def frames(self, start: int, count: int) -> Iterator[np.ndarray]:
        """Frames ``start`` to ``start + count - 1`` in turn."""
        for index in track(range(start, start + count), desc="stimulus", total=count, unit="frame"):
            yield self.frame(index)


def gamut_saturation(level: float) -> float:
    """The saturation whose untinted chroma's three-sigma circle just meets the BT.709 gamut at ``level``.

    Chroma is ``saturation * noise / 6`` an axis (:func:`_colour`), so three sigmas reach half the
    saturation; R, B and G move along ``2 (1 - KR)``, ``2 (1 - KB)`` and the pair G takes from both.
    """
    kg = 1 - KR - KB
    gains = (2 * (1 - KR), 2 * (1 - KB), math.hypot(2 * KR * (1 - KR) / kg, 2 * KB * (1 - KB) / kg))
    return 2 * min(level, 1 - level) / max(gains)


def _ti_scale(contrast: float, gradient: float) -> float:
    """TI, in luma levels, of a texture-space displacement of one pixel over the whole frame.

    Luma is ``level + contrast * F / 3`` of full scale, so a displacement ``d`` moves it by
    ``255 * contrast / 3 * (grad F . d)``, whose spread over an isotropic ``F`` is ``gradient / sqrt(2)``.
    """
    return 255.0 * contrast / 3.0 * gradient / math.sqrt(2.0)


@lru_cache(maxsize=256)
def gradient_rms(wavelength: float, size: int = 1024) -> float:
    """RMS gradient, per pixel, of the unit-variance noise band :func:`_texture` makes at ``wavelength``."""
    f = np.hypot(*np.meshgrid(np.fft.fftfreq(size), np.fft.fftfreq(size), indexing="ij"))
    power = np.exp(-0.5 * np.log2(np.maximum(f, 1.0 / size) * wavelength) ** 2) ** 2
    power[0, 0] = 0.0
    return float(2 * math.pi * math.sqrt((f**2 * power).sum() / power.sum()))


@lru_cache(maxsize=64)
def _path(shot: Shot, frames: int, raster: tuple[int, int], texture: int) -> np.ndarray:
    """Accumulated drift, spin and zoom before each frame, each frame's step the shot's mix of rates
    scaled to a TI of ``shot.ti * exp(swing * z - swing**2 / 2)``, ``z`` standard normal a frame.

    A step moves texture points by the drift, plus the spin and zoom times their distance from the
    centre in texture pixels, so its RMS over the raster is ``sqrt(|v|^2 + (w^2 + z^2) s^2 r^2)``, ``s`` the
    scale then and ``r^2`` the raster's mean square radius (:func:`_ti_scale` turns it to TI).
    """
    rng = np.random.default_rng(shot.seed)
    target = shot.ti * np.exp(shot.swing * rng.standard_normal(frames) - 0.5 * shot.swing**2)
    per_pixel = _ti_scale(shot.contrast, gradient_rms(shot.wavelength, texture))
    radius2 = (raster[0] ** 2 + raster[1] ** 2) / 12.0
    drift2, turn2 = float(np.hypot(*shot.velocity)) ** 2, shot.spin**2 + shot.zoom**2
    out = np.zeros((frames, 4))
    for t in range(frames - 1):
        scale2 = math.exp(-2.0 * out[t, 3])
        rms = math.sqrt(drift2 + turn2 * scale2 * radius2) * per_pixel
        k = target[t] / rms if rms > 0 else 0.0
        out[t + 1] = out[t] + k * np.array([*shot.velocity, shot.spin, shot.zoom])
    return out


def _sobol(seed: int, k: int) -> np.ndarray:
    """Point ``k`` of the seed's scrambled Sobol sequence, cached a block at a time."""
    return _sobol_block(seed, k // BLOCK)[k % BLOCK]


@lru_cache(maxsize=8)
def _sobol_block(seed: int, block: int) -> np.ndarray:
    """One block of the seed's scrambled Sobol points, each block its own scramble."""
    return qmc.Sobol(DIMS, scramble=True, seed=np.random.default_rng([seed, block % (1 << 32)])).random(BLOCK)


@lru_cache(maxsize=4)
def _texture(size: int, wavelength: float, seed: int) -> np.ndarray:
    """Three tileable unit-variance noise fields, ``(size, size, 3)``, an octave-wide band at ``wavelength``.

    Random phases under a log-normal annulus in frequency: each holds one scale and wraps at its edges.
    """
    rng = np.random.default_rng(seed)
    f = np.hypot(*np.meshgrid(np.fft.fftfreq(size), np.fft.fftfreq(size), indexing="ij"))
    band = np.exp(-0.5 * np.log2(np.maximum(f, 1.0 / size) * wavelength) ** 2)
    band[0, 0] = 0.0
    fields = np.fft.ifft2(band * np.exp(2j * np.pi * rng.random((3, size, size)))).real
    fields /= fields.std(axis=(1, 2), keepdims=True)
    return np.ascontiguousarray(np.moveaxis(fields, 0, -1).astype(np.float32))


def _colour(fields: np.ndarray, shot: Shot) -> np.ndarray:
    """Noise fields as RGB: luma around the shot's level, chroma around its hue, clipped to full scale."""
    y = shot.level + shot.contrast * fields[..., 0] / 3
    hue = 2 * math.pi * shot.hue
    cb = shot.saturation * (shot.tint * math.cos(hue) + fields[..., 1] / 3) / 2
    cr = shot.saturation * (shot.tint * math.sin(hue) + fields[..., 2] / 3) / 2
    r = y + 2 * (1 - KR) * cr
    b = y + 2 * (1 - KB) * cb
    g = (y - KR * r - KB * b) / (1 - KR - KB)
    return (np.clip(np.stack([r, g, b], axis=-1), 0.0, 1.0) * 255 + 0.5).astype(np.uint8)

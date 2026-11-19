"""A synthetic moving picture whose motion is known on every frame, made from a seed and nothing stored.

Each shot is a tileable band-limited colour noise texture under a constant drift, spin and zoom, drawn
as a scrambled Sobol point over :class:`Coverage`; frame ``k`` depends only on seed, raster and ``k``.
"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass, field
from functools import lru_cache
from typing import Iterator

import numpy as np
from scipy.stats import qmc

from syncsummoner.progress import track

#: BT.709 luma weights, for the colour each shot's chroma is built around.
KR, KB = 0.2126, 0.0722
#: Sobol dimensions a shot draws: speed, direction, spin, zoom, wavelength, contrast, level, hue, saturation.
DIMS = 9
#: Points in one cached block of a seed's Sobol sequence.
BLOCK = 1024


@dataclass(frozen=True)
class Coverage:
    """The ranges shots are drawn over, each ``(low, high)``, scales log-uniform.

    ``speed`` is drift in pixels a frame, ``spin`` radians and ``zoom`` log scale a frame either way,
    ``wavelength`` the texture's in pixels; the rest are fractions of full scale.
    """

    speed: tuple[float, float] = (0.25, 48.0)
    spin: float = 0.02
    zoom: float = 0.01
    wavelength: tuple[float, float] = (3.0, 256.0)
    contrast: tuple[float, float] = (0.1, 0.5)
    level: tuple[float, float] = (0.2, 0.8)
    saturation: tuple[float, float] = (0.0, 0.3)


@dataclass(frozen=True)
class Shot:
    """One shot's texture and motion, constant over its frames."""

    velocity: tuple[float, float]
    spin: float
    zoom: float
    wavelength: float
    contrast: float
    level: float
    hue: float
    saturation: float
    seed: int


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
            seed=int(np.random.SeedSequence([self.seed, k % (1 << 32)]).generate_state(1)[0]),
        )

    def placed(self, index: int) -> tuple[int, int]:
        """The shot frame ``index`` is in, and how far into it."""
        return index // self.shot_frames, index % self.shot_frames

    def transform(self, index: int) -> np.ndarray:
        """The 2x3 affine taking frame ``index``'s pixel ``(x, y, 1)`` to its point on the shot's texture."""
        k, i = self.placed(index)
        shot = self.shot(k)
        scale = math.exp(-shot.zoom * i)
        c, s = math.cos(shot.spin * i) * scale, math.sin(shot.spin * i) * scale
        cx, cy = self.width / 2, self.height / 2
        tx = cx + shot.velocity[0] * i - (c * cx - s * cy)
        ty = cy + shot.velocity[1] * i - (s * cx + c * cy)
        return np.array([[c, -s, tx], [s, c, ty]])

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
    cb = shot.saturation * (math.cos(hue) + fields[..., 1] / 3) / 2
    cr = shot.saturation * (math.sin(hue) + fields[..., 2] / 3) / 2
    r = y + 2 * (1 - KR) * cr
    b = y + 2 * (1 - KB) * cb
    g = (y - KR * r - KB * b) / (1 - KR - KB)
    return (np.clip(np.stack([r, g, b], axis=-1), 0.0, 1.0) * 255 + 0.5).astype(np.uint8)

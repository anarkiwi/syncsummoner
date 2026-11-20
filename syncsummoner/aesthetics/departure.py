"""How far a transform's output departs from its input, and whether the output can still be watched.

Luma structure is DSSIM, ``(1 - SSIM) / 2`` (Wang et al. 2004), colour mean CIEDE2000 (CIE 142-2001) over the
lightness span; intensity is the larger. Blank and static are P.910 SI and TI under the live floor
(:func:`survival.live_floor`); flashing is :func:`safety.flash_risk`.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np
from scipy.ndimage import gaussian_filter
from skimage.color import deltaE_ciede2000, rgb2lab  # pylint: disable=no-name-in-module  ; lazily loaded

from syncsummoner.aesthetics.levels import LUMA_709
from syncsummoner.aesthetics.safety import flash_risk
from syncsummoner.aesthetics.survival import (
    live_floor,
    quantisation_si,
    quantisation_ti,
    spatial_information,
    temporal_information,
)

__all__ = ["DELTA_E_SPAN", "Effect", "delta_e", "effect", "ssim"]

#: SSIM's stabilising constants, as fractions of the dynamic range.
K1, K2 = 0.01, 0.03
#: SSIM's Gaussian window: standard deviation and radius in pixels, an 11-tap window.
WINDOW_SIGMA, WINDOW_RADIUS = 1.5, 5
#: CIEDE2000 between black and white: the lightness span of CIELAB.
DELTA_E_SPAN = 100.0


def _unit(frames: np.ndarray) -> np.ndarray:
    """A frame stack in float32 unit range, 8-bit input scaled."""
    arr = np.asarray(frames)
    return arr.astype(np.float32) / np.float32(255.0) if arr.dtype == np.uint8 else arr.astype(np.float32)


def _luma(frames: np.ndarray) -> np.ndarray:
    """Rec.709 luma of an RGB stack, or a luma stack as it is."""
    return frames @ LUMA_709 if frames.ndim == 4 else frames


def _blur(x: np.ndarray) -> np.ndarray:
    """SSIM's window over each frame of a stack, its border cropped so only whole windows remain."""
    out = gaussian_filter(x, sigma=(0.0, WINDOW_SIGMA, WINDOW_SIGMA), truncate=WINDOW_RADIUS / WINDOW_SIGMA)
    r = WINDOW_RADIUS
    return out[:, r:-r, r:-r]


def _ssim_terms(x: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Per frame, SSIM's luminance and contrast-structure maps, ``(T, H', W')``."""
    c1, c2 = K1**2, K2**2
    mx, my = _blur(x), _blur(y)
    vx, vy, cov = _blur(x * x) - mx * mx, _blur(y * y) - my * my, _blur(x * y) - mx * my
    luminance = (2 * mx * my + c1) / (mx * mx + my * my + c1)
    structure = (2 * cov + c2) / (vx + vy + c2)
    return luminance, structure


def ssim(reference: np.ndarray, output: np.ndarray) -> np.ndarray:
    """Mean SSIM of each output frame against its reference, ``(T,)`` in ``[-1, 1]`` (Wang et al. 2004).

    Frames are ``(T, H, W)`` luma or ``(T, H, W, 3)`` RGB, 8-bit or unit range, already at the viewing scale.
    """
    x, y = _luma(_unit(reference)), _luma(_unit(output))
    if x.shape != y.shape or x.ndim != 3:
        raise ValueError(f"need matching (T, H, W[, 3]) stacks, got {x.shape} and {y.shape}")
    if min(x.shape[1:]) <= 2 * WINDOW_RADIUS:
        raise ValueError(f"SSIM needs frames wider than its {2 * WINDOW_RADIUS + 1}-pixel window")
    luminance, structure = _ssim_terms(x, y)
    return (luminance * structure).mean(axis=(1, 2))


def delta_e(reference: np.ndarray, output: np.ndarray) -> np.ndarray:
    """Mean CIEDE2000 of each output frame against its reference, ``(T,)``, from sRGB ``(T, H, W, 3)``."""
    x, y = _unit(reference), _unit(output)
    if x.shape != y.shape or x.ndim != 4:
        raise ValueError(f"need matching (T, H, W, 3) stacks, got {x.shape} and {y.shape}")
    return deltaE_ciede2000(rgb2lab(x, channel_axis=-1), rgb2lab(y, channel_axis=-1)).mean(axis=(1, 2))


@dataclass(frozen=True)
class Effect:
    """What a transform did to one span of consecutive input frames, and whether the output is safe.

    ``structure`` is mean DSSIM, ``colour`` mean CIEDE2000, ``intensity`` the larger of
    ``structure`` and ``colour`` over :data:`DELTA_E_SPAN`. SI and TI are span means.
    """

    frames: int
    structure: float
    colour: float
    intensity: float
    source_si: float
    output_si: float
    source_ti: float
    output_ti: float
    blank: bool
    static: bool
    flash_rate: float
    flash_area: float
    red_rate: float
    red_area: float
    flashing: bool
    red_flashing: bool

    @property
    def safe(self) -> bool:
        """Neither blank, static, nor flashing by either test."""
        return not (self.blank or self.static or self.flashing or self.red_flashing)

    def as_dict(self) -> dict:
        """Plain types, ``safe`` included."""
        return {**asdict(self), "safe": self.safe}


def effect(source: np.ndarray, output: np.ndarray, *, fps: float) -> Effect:
    """Departure and safety of ``output`` against ``source``, consecutive RGB frames ``(T, H, W, 3)``.

    The output is blank when its SI, and static when its TI, falls below the live floor between the
    source's and quantisation noise alone; its flashes are counted at ``fps``.
    """
    x, y = _unit(source), _unit(output)
    structure = float(np.mean((1.0 - ssim(x, y)) / 2.0))
    colour = float(np.mean(delta_e(x, y)))
    lx, ly = _luma(x), _luma(y)
    si = [float(np.mean([spatial_information(f) for f in frames])) for frames in (lx, ly)]
    if x.shape[0] > 1:
        ti = [float(np.mean([temporal_information(a, b) for a, b in zip(f[:-1], f[1:])])) for f in (lx, ly)]
    else:
        ti = [float("nan"), float("nan")]
    risk = flash_risk(y, fps=fps)
    return Effect(
        frames=int(x.shape[0]),
        structure=structure,
        colour=colour,
        intensity=max(structure, colour / DELTA_E_SPAN),
        source_si=si[0],
        output_si=si[1],
        source_ti=ti[0],
        output_ti=ti[1],
        blank=bool(si[1] < live_floor(si[0], quantisation_si())),
        static=bool(ti[1] < live_floor(ti[0], quantisation_ti())),
        flash_rate=risk.flashes_per_s,
        flash_area=risk.area_frac,
        red_rate=risk.red_flashes_per_s,
        red_area=risk.red_area_frac,
        flashing=not risk.general_safe,
        red_flashing=not risk.red_safe,
    )

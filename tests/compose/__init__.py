"""Synthetic fixtures shared by the compose tests: no hardware, no media files, seeded RNG."""

import numpy as np

from syncsummoner.compose.features import Features, analyze_audio
from syncsummoner.device.profile import (
    Axis,
    Cliff,
    LockMap,
    ParamKind,
    ParamSpec,
    ProgramProfile,
    Source,
    Tongue,
)

SR = 22050
SEGMENT_KW = {"kernel_s": 0.5, "min_section_s": 0.4, "prominence_sigma": 0.5}


def click_track(seconds: float = 4.0, bpm: float = 120.0, sr: int = SR, contrast: bool = True):
    """Impulse train at a known tempo, with a spectral change halfway so segmentation has a boundary."""
    n = int(sr * seconds)
    y = np.zeros(n, dtype=np.float32)
    y[:: int(round(sr * 60.0 / bpm))] = 1.0
    tail = np.arange(512)
    imp = (np.exp(-tail / 50.0) * np.sin(2 * np.pi * 180 * tail / sr)).astype(np.float32)
    y = np.convolve(y, imp)[:n].astype(np.float32)
    if contrast:
        t = np.arange(n // 2, n) / sr
        y[n // 2 :] += 0.6 * np.sin(2 * np.pi * 4000 * t).astype(np.float32)
    return y, sr


def _spec(index, axis, kind=ParamKind.CONTINUOUS, cliffs=(), hysteresis=False, dead_zone=None):
    values = np.round(np.linspace(0, 1023, 32)).astype(int)
    return ParamSpec(
        index=index,
        name=f"p{index}",
        native_min=0.0,
        native_max=100.0,
        kind=kind,
        axis=axis,
        values=[int(v) for v in values],
        response=list(np.linspace(0.0, 1.0, 32) ** 1.5),
        sensitivity=0.4 + 0.03 * index,
        monotonic=True,
        dead_zone=dead_zone,
        cliffs=list(cliffs),
        hysteresis=hysteresis,
    )


def make_profile(program: str = "glitch", *, source: Source = Source.HW, rich: bool = True):
    """A fitted profile with a cliff atlas, a lock map, a boolean, a fader and settle times."""
    params = [
        _spec(1, Axis.TEXTURE_SCALE, dead_zone=(0, 96)),
        _spec(2, Axis.MOTION_RATE, cliffs=[Cliff(at=640, jump=0.6, metrics=("clip_frac", "ic"))]),
        _spec(
            3,
            Axis.COLOR_DESTRUCTION,
            hysteresis=True,
            cliffs=[Cliff(at=320, jump=0.3, metrics=("illegal_frac",))],
        ),
        _spec(7, Axis.NOISE, kind=ParamKind.BOOLEAN),
        _spec(12, Axis.UNASSIGNED),
    ]
    if not rich:
        params = params[:1]
    locks = [LockMap(a="time_macro", b=2, tongues=[Tongue(ratio=(1, 1), center=500, width=80, strength=0.9)])]
    return ProgramProfile(
        program=program,
        firmware="1.0.0-rc.37",
        analyzer="aesthetics 0.1.0",
        source=source,
        params=params,
        lock_maps=locks if rich else [],
        settle_frames={1: 2, 2: 3, 3: 2, 7: 1, 12: 1},
    )


def make_features(rng, *, seconds: float = 4.0, bpm: float = 120.0) -> Features:
    """Audio-only features from a synthetic click track."""
    y, sr = click_track(seconds=seconds, bpm=bpm)
    return Features(audio=analyze_audio(y, sr, rng=rng, **SEGMENT_KW), video=None)

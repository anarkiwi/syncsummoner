"""Offline multi-pass rendering: burn timecode, play out, capture, align, crop, composite.

Every pass is a real-time capture. Frames self-identify through a croppable
gray-code edge strip, which is what makes alignment possible without reopening
the capture stream. Program is fixed per pass; loading it blacks the output out.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

import numpy as np

from syncsummoner.device.profile import PARAM_MAX, ProgramProfile
from syncsummoner.compose.planner import plan_automation
from syncsummoner.compose.score import Score
from syncsummoner.compose.vocabulary import Automation


class UnsafeOutputError(RuntimeError):
    """Output tripped a hard safety veto and mitigation was declined."""


SESSION_FORMATS = {"720p60": (1280, 720, 60.0), "ntsc": (720, 480, 29.97)}
EARLY_BIAS_S = 0.015


@dataclass(frozen=True)
class RenderConfig:
    """Session constants and the measured timing budget for one render."""

    width: int = 1280
    height: int = 720
    fps: float = 60.0
    strip_px: int = 8
    bits: int = 16
    latency_s: float = 0.0
    early_bias_s: float = EARLY_BIAS_S
    cc_budget_hz: float = 200.0
    #: ssh target driving playout and the HDMI link; None takes the device layer's default.
    source_host: str | None = None

    @classmethod
    def for_format(cls, name: str, **kw: Any) -> "RenderConfig":
        """Config for a named session format; format is a session constant, never a parameter."""
        width, height, fps = SESSION_FORMATS[name]
        return cls(width=width, height=height, fps=fps, **kw)


@dataclass
class Rig:
    """The real-time endpoints a pass drives, injectable so render stays testable.

    ``link`` is the source HDMI link, held down across every program change; None
    leaves it alone, which only a rig without link control should do.
    """

    session: Any
    capture: Any
    playout: Any
    link: Any = None


def burn_timecode(frame: np.ndarray, index: int, *, bits: int = 16, strip_px: int = 8) -> np.ndarray:
    """Write a gray-coded frame index into a croppable edge strip, behind a constant marker cell."""
    out = np.array(frame, dtype=np.float32, copy=True)
    code = int(index) ^ (int(index) >> 1)
    cells = np.concatenate(([1.0], ((code >> np.arange(bits - 1, -1, -1)) & 1).astype(np.float32)))
    edges = np.round(np.linspace(0, out.shape[1], cells.size + 1)).astype(int)
    out[:strip_px] = np.repeat(cells, np.diff(edges))[None, :, None]
    return out


def read_timecode(
    frame: np.ndarray, *, bits: int = 16, strip_px: int = 8, min_contrast: float = 0.25
) -> int | None:
    """Recover the frame index from the edge strip; ``None`` when the strip is absent or washed out."""
    strip = np.asarray(frame, dtype=np.float64)[:strip_px].mean(axis=(0, 2))
    edges = np.round(np.linspace(0, strip.size, bits + 2)).astype(int)
    cells = np.add.reduceat(strip, edges[:-1]) / np.maximum(np.diff(edges), 1)
    if np.ptp(cells) < min_contrast or cells[0] < cells.mean():
        return None
    binary = np.bitwise_xor.accumulate((cells[1:] > cells.mean()).astype(np.int64))
    return int((binary * (1 << np.arange(bits - 1, -1, -1))).sum())


def crop_strip(frame: np.ndarray, *, strip_px: int = 8) -> np.ndarray:
    """Remove the timecode strip before anything else looks at the frame."""
    return frame[strip_px:]


def schedule(auto: Automation, *, latency_s: float = 0.0, early_bias_s: float = EARLY_BIAS_S) -> Automation:
    """Shift automation earlier by the measured latency plus a 10-20 ms bias.

    Audio leading video is detected at roughly 30-50 ms while the reverse
    tolerates over 100 ms, so the error budget is asymmetric: visuals must never
    run late. Gestures already end on their anchor, so arrival lands on the beat.
    """
    return auto.shift(-(latency_s + early_bias_s))


def align(captured: Mapping[int, np.ndarray], n_frames: int, shape: tuple[int, ...]) -> np.ndarray:
    """Order captured frames by their decoded timecode, holding the last good frame over gaps."""
    out = np.zeros((n_frames,) + tuple(shape), dtype=np.float32)
    have = np.zeros(n_frames, dtype=bool)
    for k, frame in captured.items():
        out[k] = frame
        have[k] = True
    if not have.any():
        return out
    idx = np.maximum.accumulate(np.where(have, np.arange(n_frames), 0))
    idx[: int(np.argmax(have))] = int(np.argmax(have))
    return out[idx]


def play_pass(
    rig: Rig, frames: np.ndarray, auto: Automation, *, program: str, config: RenderConfig
) -> np.ndarray:
    """Run one real-time pass: load the program once, play out, drive CC, capture and align."""
    rig.session.load_program(program, link=rig.link)
    order = np.argsort(auto.times, kind="stable")
    times, indices, values = auto.times[order], auto.indices[order], auto.values[order]
    captured: dict[int, np.ndarray] = {}
    cursor = 0
    shape: tuple[int, ...] = (max(1, frames.shape[1] - config.strip_px),) + tuple(frames.shape[2:])
    for i in range(frames.shape[0]):
        due = int(np.searchsorted(times, i / config.fps, side="right"))
        if due > cursor:
            rig.session.set_params(
                {int(k): float(v) / PARAM_MAX for k, v in zip(indices[cursor:due], values[cursor:due])}
            )
            cursor = due
        rig.playout.show(burn_timecode(frames[i], i, bits=config.bits, strip_px=config.strip_px))
        got = rig.capture.read()
        if got is None:
            continue
        tc = read_timecode(got, bits=config.bits, strip_px=config.strip_px)
        if tc is not None and 0 <= tc < frames.shape[0]:
            captured[tc] = crop_strip(got, strip_px=config.strip_px)
            shape = captured[tc].shape
    return align(captured, frames.shape[0], shape)


def composite(passes: Sequence[np.ndarray], *, mode: str = "screen") -> np.ndarray:
    """Combine per-pass frame stacks offline; layering is where depth beyond one program comes from."""
    shapes = np.array([p.shape[:3] for p in passes]).min(axis=0)
    stack = np.stack([np.asarray(p[: shapes[0], : shapes[1], : shapes[2]], dtype=np.float32) for p in passes])
    if mode == "mean":
        return stack.mean(axis=0)
    if mode == "max":
        return stack.max(axis=0)
    return 1.0 - np.prod(1.0 - np.clip(stack, 0.0, 1.0), axis=0)


def write_video(path: str | Path, frames: np.ndarray, fps: float) -> None:
    """Write an RGB float32 frame stack; OpenCV's BGR convention stops at this boundary."""
    # pylint: disable=no-member
    import cv2

    h, w = frames.shape[1:3]
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"FFV1"), fps, (w, h))
    for frame in frames:
        writer.write(cv2.cvtColor((np.clip(frame, 0, 1) * 255).astype(np.uint8), cv2.COLOR_RGB2BGR))
    writer.release()


def open_rig(config: RenderConfig) -> Rig:
    """Build the real rig from the device layer; imported here so compose never needs hardware."""
    from syncsummoner.device import capture as capture_mod
    from syncsummoner.device import link as link_mod
    from syncsummoner.device import playout as playout_mod
    from syncsummoner.device import session as session_mod
    from syncsummoner.device import transport as transport_mod

    transport = transport_mod.Transport.open()
    host = () if config.source_host is None else (config.source_host,)
    return Rig(
        session=session_mod.Session(transport, cc_budget_hz=config.cc_budget_hz),
        capture=capture_mod.Capture(width=config.width, height=config.height, fps=int(config.fps)),
        playout=playout_mod.Playout(*host, width=config.width, height=config.height),
        link=link_mod.Link(*host),
    )


def _source_frames(source: Any, config: RenderConfig) -> np.ndarray:
    if isinstance(source, np.ndarray):
        return source.astype(np.float32)
    from syncsummoner.compose.features import read_video

    del config
    return read_video(source)[0]


def _passes(
    score: Score,
    frames: np.ndarray,
    profiles: Mapping[str, ProgramProfile],
    rig: Rig,
    config: RenderConfig,
    n_passes: int,
) -> list[np.ndarray]:
    """Run each layer in turn, re-feeding the previous pass so program stays the outer loop."""
    autos = plan_automation(score, profiles, fps=config.fps, cc_budget_hz=config.cc_budget_hz)
    results = []
    current = frames
    for layer in sorted(score.layers, key=lambda item: item.index)[:n_passes]:
        auto = schedule(
            autos.get(layer.index, Automation.empty()),
            latency_s=config.latency_s,
            early_bias_s=config.early_bias_s,
        )
        current = play_pass(rig, current, auto, program=layer.program, config=config)
        results.append(current)
    return results


def enforce_safety(frames: np.ndarray, *, fps: float, mitigate: bool = True) -> np.ndarray:
    """Apply the photosensitive-seizure veto to a finished pass.

    A hard constraint, not a score. Offending windows are temporally low-passed
    rather than discarded, so a take is repaired instead of lost.
    """
    from syncsummoner import aesthetics

    risk = aesthetics.flash_risk(frames, fps=fps)
    if risk.safe:
        return frames
    if not mitigate:
        raise UnsafeOutputError(
            f"{risk.flashes_per_s:.1f} flashes/s over {risk.area_frac:.0%} of frame "
            f"in windows {list(risk.windows)}"
        )
    return aesthetics.mitigate_flashes(frames, risk, fps=fps)


def render(
    score: Score,
    source: Any,
    out: str | Path,
    *,
    passes: int = 1,
    profiles: Mapping[str, ProgramProfile] | None = None,
    rig: Rig | None = None,
    config: RenderConfig | None = None,
    sink: Callable[[str | Path, np.ndarray, float], None] | None = None,
    mode: str = "screen",
) -> None:
    """Render the full score to ``out`` as one real-time capture pass per layer."""
    if profiles is None:
        raise ValueError("render requires the measured profiles the score was planned against")
    config = RenderConfig() if config is None else config
    rig = open_rig(config) if rig is None else rig
    frames = _source_frames(source, config)
    results = _passes(score, frames, profiles, rig, config, passes)
    if not results:
        raise ValueError("score has no layers to render")
    final = composite(results, mode=mode) if len(results) > 1 else results[0]
    (sink or write_video)(out, enforce_safety(final, fps=config.fps), config.fps)


def audition(
    score: Score,
    source: Any,
    *,
    seconds: float = 30.0,
    scale: float = 0.25,
    passes: int = 1,
    profiles: Mapping[str, ProgramProfile] | None = None,
    rig: Rig | None = None,
    config: RenderConfig | None = None,
    mode: str = "screen",
) -> np.ndarray:
    """Render a short downscaled excerpt for planner iteration, returning the frames."""
    if profiles is None:
        raise ValueError("audition requires the measured profiles the score was planned against")
    config = RenderConfig() if config is None else config
    rig = open_rig(config) if rig is None else rig
    frames = _source_frames(source, config)[: max(1, int(round(seconds * config.fps)))]
    step = max(1, int(round(1.0 / max(scale, 1e-6))))
    frames = np.ascontiguousarray(frames[:, ::step, ::step])
    results = _passes(score, frames, profiles, rig, config, passes)
    return composite(results, mode=mode) if len(results) > 1 else results[0]

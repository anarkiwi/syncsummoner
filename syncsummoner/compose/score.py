"""Timeline IR: sections, gesture instances grouped into per-program layers, and YAML round-trip.

Program is a layer attribute, never a per-gesture field: program loading blacks
the output out for seconds, so it is the outermost loop variable everywhere.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import yaml

from syncsummoner.device.profile import CROSSFADER_INDEX, PARAM_MAX, Axis, ProgramProfile
from syncsummoner.compose.features import Section
from syncsummoner.compose.vocabulary import GESTURES, Automation, GestureContext

__all__ = ["Section", "GestureInstance", "Layer", "Score", "control_rate"]

#: The crossfader gates output outright at its extremes, so every gesture is clamped to a blend band.
CROSSFADER_BLEND_FRAC = (0.10, 0.50)


def _clamp_crossfader(auto: Automation) -> Automation:
    """Clip any crossfader (P12) automation into :data:`CROSSFADER_BLEND_FRAC`."""
    mask = auto.indices == CROSSFADER_INDEX
    if not mask.any():
        return auto
    lo, hi = (round(frac * PARAM_MAX) for frac in CROSSFADER_BLEND_FRAC)
    values = np.where(mask, np.clip(auto.values, lo, hi), auto.values).astype(np.int32)
    return replace(auto, values=values)


def control_rate(fps: float) -> float:
    """Modulation Nyquist: the device updates at frame rate, so control tops out at half of it."""
    return fps / 2.0


def program_key(program: str) -> int:
    """Stable integer seed for a program name, so automation does not move when layers renumber."""
    return int.from_bytes(hashlib.blake2b(program.encode(), digest_size=8).digest(), "big")


@dataclass(frozen=True)
class GestureInstance:
    """One placed gesture: which primitive, where its arrival lands, and how it is parameterized."""

    gesture: str
    arrival: float
    duration: float
    axis: str = Axis.UNASSIGNED.value
    intensity: float = 0.5
    targets: tuple[int, ...] = ()
    seed: int = 0

    def to_dict(self) -> dict[str, Any]:
        """Plain-data form for YAML."""
        return {
            "gesture": self.gesture,
            "arrival": float(self.arrival),
            "duration": float(self.duration),
            "axis": self.axis,
            "intensity": float(self.intensity),
            "targets": list(self.targets),
            "seed": int(self.seed),
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "GestureInstance":
        """Rebuild from plain data, filling defaults."""
        return cls(
            gesture=data["gesture"],
            arrival=float(data["arrival"]),
            duration=float(data["duration"]),
            axis=data.get("axis", Axis.UNASSIGNED.value),
            intensity=float(data.get("intensity", 0.5)),
            targets=tuple(int(t) for t in data.get("targets", ())),
            seed=int(data.get("seed", 0)),
        )


@dataclass
class Layer:
    """All gestures rendered through one program, in one capture pass."""

    program: str
    index: int = 0
    gestures: list[GestureInstance] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        """Plain-data form for YAML."""
        return {
            "program": self.program,
            "index": int(self.index),
            "gestures": [g.to_dict() for g in self.gestures],
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "Layer":
        """Rebuild from plain data, filling defaults."""
        return cls(
            program=data["program"],
            index=int(data.get("index", 0)),
            gestures=[GestureInstance.from_dict(g) for g in data.get("gestures", ())],
        )


@dataclass
class Score:
    """The composed timeline: seed, tempo grid, sections, and one layer per program pass."""

    seed: int = 0
    bpm: float = 120.0
    duration: float = 0.0
    fps: float = 60.0
    sections: list[Section] = field(default_factory=list)
    layers: list[Layer] = field(default_factory=list)
    meta: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Plain-data form for YAML; the round-trip through this is lossless."""
        return {
            "seed": int(self.seed),
            "bpm": float(self.bpm),
            "duration": float(self.duration),
            "fps": float(self.fps),
            "sections": [
                {"start": float(s.start), "end": float(s.end), "label": s.label, "destroy": bool(s.destroy)}
                for s in self.sections
            ],
            "layers": [layer.to_dict() for layer in self.layers],
            "meta": dict(self.meta),
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "Score":
        """Rebuild from plain data, filling defaults."""
        return cls(
            seed=int(data.get("seed", 0)),
            bpm=float(data.get("bpm", 120.0)),
            duration=float(data.get("duration", 0.0)),
            fps=float(data.get("fps", 60.0)),
            sections=[
                Section(
                    start=float(s["start"]),
                    end=float(s["end"]),
                    label=s["label"],
                    destroy=bool(s.get("destroy", False)),
                )
                for s in data.get("sections", ())
            ],
            layers=[Layer.from_dict(m) for m in data.get("layers", ())],
            meta=dict(data.get("meta", {})),
        )

    def to_yaml(self) -> str:
        """Serialize to YAML."""
        return yaml.safe_dump(self.to_dict(), sort_keys=False, default_flow_style=False)

    @classmethod
    def from_yaml(cls, text: str) -> "Score":
        """Parse a YAML document."""
        return cls.from_dict(yaml.safe_load(text))

    def save(self, path: str | Path) -> None:
        """Write the score to disk."""
        Path(path).write_text(self.to_yaml(), encoding="utf-8")

    @classmethod
    def load(cls, path: str | Path) -> "Score":
        """Read a score from disk."""
        return cls.from_yaml(Path(path).read_text(encoding="utf-8"))

    def section_at(self, t: float) -> Section | None:
        """Section containing a time, or ``None``."""
        return next((s for s in self.sections if s.start <= t < s.end), None)

    def destroy_mask(self, times: np.ndarray) -> np.ndarray:
        """Boolean mask of the times that fall inside a section marked ``destroy``."""
        mask = np.zeros(times.size, dtype=bool)
        for s in self.sections:
            if s.destroy:
                mask |= (times >= s.start) & (times < s.end)
        return mask

    def with_layers(self, layers: list[Layer]) -> "Score":
        """Copy with a different set of layers."""
        return replace(self, layers=layers)

    def render_layer(self, layer: Layer, profile: ProgramProfile, *, rate: float | None = None) -> Automation:
        """Expand one layer's gesture instances into merged parameter automation."""
        rate = control_rate(self.fps) if rate is None else rate
        parts = []
        for i, inst in enumerate(layer.gestures):
            gesture = GESTURES.get(inst.gesture)
            if gesture is None:
                continue
            ctx = GestureContext(
                arrival=inst.arrival,
                duration=inst.duration,
                rate=rate,
                intensity=inst.intensity,
                axis=Axis(inst.axis),
                rng=np.random.default_rng((self.seed, program_key(layer.program), inst.seed, i)),
                targets=inst.targets,
            )
            parts.append(gesture(profile, ctx))
        return _clamp_crossfader(Automation.concat(parts))

    def automation(
        self, profiles: Mapping[str, ProgramProfile], *, rate: float | None = None
    ) -> dict[int, Automation]:
        """Automation per layer index, keyed so the caller keeps program as the outer loop."""
        return {
            layer.index: self.render_layer(layer, profiles[layer.program], rate=rate)
            for layer in self.layers
            if layer.program in profiles
        }

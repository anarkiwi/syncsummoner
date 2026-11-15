"""Measured device behaviour: parameter specs, program profiles, measurement records.

Axis assignment and the cliff atlas live here, in the profile, not in the planner.
A firmware release with new programs is then absorbed by re-probing rather than by
editing the composer.
"""

from __future__ import annotations

import enum
import re
from dataclasses import dataclass, field, asdict
from typing import Any, Mapping, Sequence

import numpy as np

PARAM_COUNT = 12
PARAM_MAX = 1023
BOOL_THRESHOLD = 512
#: The crossfader; it gates output even where `program info` names it `Null 12`.
CROSSFADER_INDEX = 12


class ParamKind(enum.Enum):
    """Parameter type. Determines how a plan samples it."""

    CONTINUOUS = "continuous"
    BOOLEAN = "boolean"
    QUANTIZED = "quantized"
    UNUSED = "unused"


class Role(enum.Enum):
    """What a parameter does to the program as a whole, which decides what may drive it.

    A bypass returns the input unprocessed, so automating it reads as the effect
    switching off; the wet control blends the effect with its input; a momentary
    control acts on its rising edge and is pulsed, never held.
    """

    EFFECT = "effect"
    BYPASS = "bypass"
    WET = "wet"
    MOMENTARY = "momentary"


#: Words in a program's own parameter label that name a role other than an effect.
ROLE_WORDS = {
    "bypass": Role.BYPASS,
    "reset": Role.MOMENTARY,
    "trigger": Role.MOMENTARY,
    "retrig": Role.MOMENTARY,
    "serve": Role.MOMENTARY,
    "fire": Role.MOMENTARY,
    "impulse": Role.MOMENTARY,
}
#: Labels the crossfader slot carries when it blends the effect with its input.
WET_WORDS = frozenset({"mix", "blend", "fader", "fade", "wet", "dry"})
#: Words of a label that names no function: unassigned slots and unnamed fallbacks.
PLACEHOLDER_WORDS = frozenset({"null", "none", "p"})


def classify_role(index: int, name: str) -> Role:
    """Role of one parameter from its slot and the label its program gives it.

    The crossfader slot is the wet control unless its program labels it as
    something else, since it gates output even where it is left unnamed.
    """
    words = set(re.findall(r"[a-z]+", name.lower()))
    if index == CROSSFADER_INDEX and (words & WET_WORDS or words <= PLACEHOLDER_WORDS):
        return Role.WET
    return next((role for word, role in ROLE_WORDS.items() if word in words), Role.EFFECT)


class Axis(enum.Enum):
    """Canonical axis a program's idiosyncratic parameter maps onto."""

    TEXTURE_SCALE = "texture_scale"
    MOTION_RATE = "motion_rate"
    DISPLACEMENT = "displacement"
    COLOR_DESTRUCTION = "color_destruction"
    KEY_THRESHOLD = "key_threshold"
    FEEDBACK_GAIN = "feedback_gain"
    NOISE = "noise"
    UNASSIGNED = "unassigned"


class ProgramStyle(enum.Enum):
    """Whether a program remaps values in place or re-addresses the picture.

    Analog-style effects are pointwise, so their parameters can only flicker
    when cycled; digital-style effects displace, so cycling them reads as motion.
    """

    ANALOG = "analog"
    DIGITAL = "digital"
    MIXED = "mixed"
    UNKNOWN = "unknown"


class Source(enum.Enum):
    """Where a measurement came from. The planner discounts simulated entries."""

    HW = "hw"
    SIM = "sim"


@dataclass(frozen=True)
class Cliff:
    """A parameter position where a small delta produces a large metric jump.

    These are the glitch seeds: they make glitch targeted rather than random.
    """

    at: int
    jump: float
    metrics: tuple[str, ...]


@dataclass
class ParamSpec:
    """One parameter of one program, as reported by the device and as measured."""

    index: int
    name: str
    native_min: float
    native_max: float
    kind: ParamKind = ParamKind.CONTINUOUS
    axis: Axis = Axis.UNASSIGNED
    steps: int | None = None
    role: Role = Role.EFFECT
    response: list[float] = field(default_factory=list)
    values: list[int] = field(default_factory=list)
    #: Measured value of each proxy metric at each visited setpoint, parallel to ``values``.
    measured: dict[str, list[float]] = field(default_factory=dict)
    #: Effect on the driving metric in units of measurement noise; below 1 the
    #: parameter moves that metric less than the rig's own repeatability.
    sensitivity: float = 0.0
    monotonic: bool = False
    dead_zone: tuple[int, int] | None = None
    cliffs: list[Cliff] = field(default_factory=list)
    hysteresis: bool = False

    @property
    def native_span(self) -> float:
        """Width of the parameter's native unit range."""
        return self.native_max - self.native_min

    def sample_values(self, n: int = 32) -> np.ndarray:
        """Return the raw 0..1023 values a plan should visit, honouring the kind.

        Boolean parameters yield two points, quantized ones yield one point per
        native step; sweeping either at ``n`` points wastes hardware time.
        """
        if self.kind is ParamKind.UNUSED:
            return np.empty(0, dtype=np.int32)
        if self.kind is ParamKind.BOOLEAN:
            return np.array([0, PARAM_MAX], dtype=np.int32)
        if self.kind is ParamKind.QUANTIZED and self.steps:
            return np.round(np.linspace(0, PARAM_MAX, self.steps)).astype(np.int32)
        return np.round(np.linspace(0, PARAM_MAX, n)).astype(np.int32)


@dataclass(frozen=True)
class Tongue:
    """One locked region in an Arnold-tongue map."""

    ratio: tuple[int, int]
    center: int
    width: int
    strength: float


@dataclass
class LockMap:
    """Arnold-tongue structure for one parameter pair.

    Locked regions are widest at simple rational ratios and narrow rapidly for
    complex ones. This is the measured consonance/dissonance control surface.
    """

    a: str
    b: int
    tongues: list[Tongue] = field(default_factory=list)


_PARAM_COLUMNS = tuple(f"p{i + 1}" for i in range(PARAM_COUNT))
_NON_METRIC_COLUMNS = frozenset(
    _PARAM_COLUMNS + ("program", "firmware", "analyzer", "source", "state_index", "stimulus", "settle_frames")
)


@dataclass
class MeasurementRecord:
    """One captured sample: the parameter vector, and what the analyzer saw.

    ``analyzer`` is recorded so captures can be re-scored under a new metrics
    version rather than discarded.
    """

    program: str
    firmware: str
    analyzer: str
    source: Source
    params: tuple[int, ...]
    state_index: int
    stimulus: str
    metrics: dict[str, float]
    settle_frames: int = 0

    def as_row(self) -> dict[str, Any]:
        """Flatten to a Parquet-friendly row."""
        row: dict[str, Any] = {
            "program": self.program,
            "firmware": self.firmware,
            "analyzer": self.analyzer,
            "source": self.source.value,
            "state_index": self.state_index,
            "stimulus": self.stimulus,
            "settle_frames": self.settle_frames,
        }
        row.update({f"p{i + 1}": v for i, v in enumerate(self.params)})
        row.update(self.metrics)
        return row

    @classmethod
    def from_row(cls, row: Mapping[str, Any]) -> "MeasurementRecord":
        """Rebuild from the flat form produced by :meth:`as_row`.

        Columns absent from a heterogeneous union arrive as None and are dropped,
        so a record only carries the metrics it was measured with.
        """
        return cls(
            program=str(row["program"]),
            firmware=str(row["firmware"]),
            analyzer=str(row["analyzer"]),
            source=Source(row["source"]),
            params=tuple(int(row[name]) for name in _PARAM_COLUMNS),
            state_index=int(row["state_index"]),
            stimulus=str(row["stimulus"]),
            metrics={k: float(v) for k, v in row.items() if v is not None and k not in _NON_METRIC_COLUMNS},
            settle_frames=int(row.get("settle_frames") or 0),
        )


@dataclass
class ProgramProfile:
    """Fitted behaviour of one program: what each parameter does, and where it breaks."""

    program: str
    firmware: str
    analyzer: str
    source: Source
    params: list[ParamSpec] = field(default_factory=list)
    interactions: list[tuple[int, int, float]] = field(default_factory=list)
    lock_maps: list[LockMap] = field(default_factory=list)
    stability_by_region: list[dict[str, Any]] = field(default_factory=list)
    settle_frames: dict[int, int] = field(default_factory=dict)
    non_settling: bool = False
    style: ProgramStyle = ProgramStyle.UNKNOWN
    pointwise: float = 0.0
    #: Continuous magnitude of how strongly the program's output still correlates with
    #: its input carrier, not a thresholded flag; 0 means the picture no longer
    #: registers with the source.
    registered: float = 0.0
    binary_hash: str = ""

    def by_role(self, *roles: Role) -> list[ParamSpec]:
        """Parameters holding any of ``roles``."""
        return [p for p in self.params if p.role in roles]

    def by_axis(self, axis: Axis) -> list[ParamSpec]:
        """Parameters assigned to a canonical axis, so gestures stay program-agnostic."""
        return [p for p in self.params if p.axis is axis]

    def cliff_atlas(self) -> list[tuple[int, Cliff]]:
        """Every cliff across every parameter, strongest first."""
        atlas = [(p.index, c) for p in self.params for c in p.cliffs]
        return sorted(atlas, key=lambda t: t[1].jump, reverse=True)

    def to_dict(self) -> dict[str, Any]:
        """Serialize to plain types, enums flattened to their values."""
        return _enums_to_values(asdict(self))

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "ProgramProfile":
        """Rebuild from the plain-type form produced by :meth:`to_dict`."""
        return cls(
            program=data["program"],
            firmware=data["firmware"],
            analyzer=data["analyzer"],
            source=Source(data["source"]),
            params=[_param_from_dict(p) for p in data.get("params", ())],
            interactions=[tuple(i) for i in data.get("interactions", ())],
            lock_maps=[_lockmap_from_dict(m) for m in data.get("lock_maps", ())],
            stability_by_region=list(data.get("stability_by_region", ())),
            settle_frames={int(k): int(v) for k, v in data.get("settle_frames", {}).items()},
            non_settling=bool(data.get("non_settling", False)),
            style=ProgramStyle(data.get("style", "unknown")),
            pointwise=float(data.get("pointwise", 0.0)),
            registered=float(data.get("registered", 0.0)),
            binary_hash=str(data.get("binary_hash", "")),
        )


def _enums_to_values(obj: Any) -> Any:
    if isinstance(obj, enum.Enum):
        return obj.value
    if isinstance(obj, dict):
        return {k: _enums_to_values(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_enums_to_values(v) for v in obj]
    return obj


def _param_from_dict(data: Mapping[str, Any]) -> ParamSpec:
    return ParamSpec(
        index=data["index"],
        name=data["name"],
        native_min=data["native_min"],
        native_max=data["native_max"],
        kind=ParamKind(data.get("kind", "continuous")),
        axis=Axis(data.get("axis", "unassigned")),
        steps=data.get("steps"),
        role=Role(data["role"]) if data.get("role") else classify_role(data["index"], data["name"]),
        response=list(data.get("response", ())),
        values=list(data.get("values", ())),
        measured={k: [float(v) for v in vs] for k, vs in (data.get("measured") or {}).items()},
        sensitivity=data.get("sensitivity", 0.0),
        monotonic=data.get("monotonic", False),
        dead_zone=_as_pair(data.get("dead_zone")),
        cliffs=[
            Cliff(at=c["at"], jump=c["jump"], metrics=tuple(c["metrics"])) for c in data.get("cliffs", ())
        ],
        hysteresis=data.get("hysteresis", False),
    )


def _lockmap_from_dict(data: Mapping[str, Any]) -> LockMap:
    return LockMap(
        a=data["a"],
        b=data["b"],
        tongues=[
            Tongue(
                ratio=(t["ratio"][0], t["ratio"][1]),
                center=t["center"],
                width=t["width"],
                strength=t["strength"],
            )
            for t in data.get("tongues", ())
        ],
    )


def _as_pair(value: Sequence[int] | None) -> tuple[int, int] | None:
    return None if value is None else (int(value[0]), int(value[1]))

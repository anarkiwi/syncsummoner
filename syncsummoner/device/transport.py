"""Videomancer hardware transport: MIDI CC, USB CDC serial verbs, program load.

The only module in the project that imports pyvmancer. Everything above it
speaks ``ParamSpec`` / ``ProgramInfo`` / numpy, never raw shell JSON.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping, Sequence

import numpy as np
import pyvmancer as vm

from .profile import PARAM_COUNT, PARAM_MAX, ParamKind, ParamSpec

#: Native ranges no wider than a default OAT sweep are enumerable, not swept.
DEFAULT_SAMPLE_STEPS = 32
#: ``program info`` names an unassigned slot ``-`` or ``Null <n>``.
UNUSED_NAMES = frozenset({"-", "", "null", "none"})
#: Combined (Manual + Modulation + MIDI) output keys, in order of preference.
COMBINED_KEYS = ("o", "out", "output", "combined")
#: Stored manual value keys.
MANUAL_KEYS = ("m", "manual")
TIMING_KEYS = ("timing", "output_timing", "out_timing", "format")
INPUT_KEYS = ("input", "source", "input_source")
LOCK_KEYS = ("locked", "lock", "input_locked", "detected")


def _is_unused(name: str) -> bool:
    """True when ``program info`` reports the slot as unassigned."""
    text = name.strip().lower()
    return text in UNUSED_NAMES or (text.startswith("null") and text[4:].strip().isdigit())


def _classify(name: str, low: float, high: float, steps: int) -> tuple[ParamKind, int | None]:
    """Derive a parameter kind and step count from its name and native range."""
    if _is_unused(name) or high <= low:
        return ParamKind.UNUSED, None
    if (low, high) == (0.0, 1.0):
        return ParamKind.BOOLEAN, 2
    span = high - low
    if span.is_integer() and low.is_integer() and span + 1 <= steps:
        return ParamKind.QUANTIZED, int(span) + 1
    return ParamKind.CONTINUOUS, None


def _first(data: Mapping[str, Any], keys: Sequence[str], default: Any = None) -> Any:
    """First present key of ``keys`` in ``data``."""
    for key in keys:
        if key in data:
            return data[key]
    return default


def _vector(payload: Any, keys: Sequence[str]) -> np.ndarray | None:
    """Pull a 12-slot integer vector out of a shell JSON payload.

    Accepts either a mapping of parallel arrays (``{"m": [...]}``) or a list of
    per-slot dicts, which is how the two status commands differ.
    """
    if not isinstance(payload, Mapping):
        return None
    for value in (_first(payload, keys),) + tuple(payload.values()):
        if not isinstance(value, (list, tuple)) or len(value) != PARAM_COUNT:
            continue
        if all(isinstance(slot, Mapping) for slot in value):
            slots = [_first(slot, keys) for slot in value]
            if all(slot is not None for slot in slots):
                return np.asarray(slots, dtype=int)
        elif all(isinstance(slot, (int, float)) for slot in value):
            return np.asarray(value, dtype=int)
    return None


@dataclass(frozen=True)
class ProgramInfo:
    """A program's name and its twelve parameter specs, as the device reports them."""

    name: str
    params: list[ParamSpec] = field(default_factory=list)
    program_id: int | None = None

    @classmethod
    def from_json(cls, data: Mapping[str, Any], *, steps: int = DEFAULT_SAMPLE_STEPS) -> "ProgramInfo":
        """Parse a ``program info`` payload into ``ParamSpec`` objects."""
        params = []
        for index, entry in enumerate(data.get("parameters", ()), start=1):
            name = str(entry.get("name", ""))
            low, high = float(entry.get("min", 0)), float(entry.get("max", 0))
            kind, count = _classify(name, low, high, steps)
            params.append(
                ParamSpec(
                    index=index,
                    name=name,
                    native_min=low,
                    native_max=high,
                    kind=kind,
                    steps=count,
                )
            )
        return cls(name=str(data.get("name", "")), params=params, program_id=data.get("id"))

    @property
    def used(self) -> list[ParamSpec]:
        """Parameters this program actually assigns."""
        return [p for p in self.params if p.kind is not ParamKind.UNUSED]


@dataclass(frozen=True)
class VideoStatus:
    """Video input/output state as reported by ``video status``."""

    timing: str | None
    input_source: str | None
    locked: bool
    raw: dict[str, Any] = field(default_factory=dict)

    @property
    def source_locked(self) -> bool:
        """True when the selected input is genuinely carrying a signal.

        The firmware can report the selected source locked while the input's own
        sub-status says otherwise; an unattended sweep must not record that as data.
        """
        sub = self.raw.get(str(self.input_source))
        if isinstance(sub, Mapping) and not sub.get("locked", True):
            return False
        return bool(self.locked)

    @classmethod
    def from_json(cls, data: Mapping[str, Any]) -> "VideoStatus":
        """Parse a ``video status`` payload, tolerating firmware key drift."""
        flat: dict[str, Any] = {}
        for value in data.values():
            if isinstance(value, Mapping):
                flat.update(value)
        flat.update({k: v for k, v in data.items() if not isinstance(v, Mapping)})
        return cls(
            timing=_first(flat, TIMING_KEYS),
            input_source=_first(flat, INPUT_KEYS),
            locked=bool(_first(flat, LOCK_KEYS, False)),
            raw=dict(data),
        )


class Transport:
    """A Videomancer addressed over MIDI CC and the serial shell.

    Parameter indices are 1-based everywhere in this project; the serial
    modulation verbs take 0-based slots and the conversion happens here.
    """

    def __init__(self, device: Any):
        self._device = device
        self._loaded: str | None = None

    @classmethod
    def open(cls, *, serial: str | None = None) -> "Transport":
        """Open both control links to an attached device."""
        return cls(vm.Videomancer.open(serial_number=serial))

    @property
    def device(self) -> Any:
        """The underlying pyvmancer facade."""
        return self._device

    @property
    def _shell(self) -> Any:
        """The serial shell client, raising when the serial link is absent."""
        return self._device.shell

    @property
    def msgs_per_param(self) -> int:
        """CC messages one parameter write puts on the wire."""
        return 2 if self._device.has_midi and self._device.midi.high_resolution else 1

    def firmware(self) -> str:
        """Firmware version string, recorded into every measurement."""
        return self._shell.version()

    def programs(self) -> list[str]:
        """Every installed FPGA program name."""
        return list(self._device.programs())

    def current_program(self) -> str | None:
        """Name of the loaded program."""
        if self._loaded is None:
            self._loaded = self._shell.status().get("program")
        return self._loaded

    def load_program(self, name: str) -> None:
        """Load an FPGA program without waiting; the caller absorbs the blackout."""
        self._device.load_program(name, settle=0)
        self._loaded = name

    def program_info(self, name: str | None = None) -> ProgramInfo:
        """Parameter specs of a program, loading it first when it is not current."""
        if name is not None and name != self.current_program():
            self.load_program(name)
        return ProgramInfo.from_json(self._shell.program_info())

    def set_param(self, index: int, value: float | bool) -> None:
        """Send a MIDI CC for one parameter; the device adds it to the manual value."""
        self._device.set_param(_check_index(index), value)

    def set_manual(self, index: int, value: int) -> None:
        """Set a slot's stored manual value over serial, which is absolute."""
        self._shell.set_modulation(_check_index(index) - 1, int(np.clip(value, 0, PARAM_MAX)))

    def program_state(self) -> np.ndarray:
        """Combined 0..1023 value of every parameter (Manual + Modulation + MIDI)."""
        combined = _vector(self._shell.modulation_status(), COMBINED_KEYS)
        if combined is None:
            combined = _vector(self._shell.program_state(), MANUAL_KEYS)
        if combined is None:
            raise vm.VmancerError("device reported no 12-slot parameter state")
        return np.clip(combined, 0, PARAM_MAX)

    def video_status(self) -> VideoStatus:
        """Video input source, timing and lock state."""
        return VideoStatus.from_json(self._shell.video_status())

    def set_video_timing(self, timing: str) -> None:
        """Force an output timing standard, overriding genlock to the source."""
        self._shell.set_video_timing(timing)

    def transport_play(self) -> None:
        """Start playback; oscillators run from the current timecode."""
        self._device.play()

    def transport_stop(self) -> None:
        """Stop playback, which resets timecode and phase-resets oscillators."""
        self._device.stop()

    def set_bpm(self, bpm: float) -> None:
        """Set internal tempo."""
        self._device.set_bpm(bpm)

    def close(self) -> None:
        """Close both control links."""
        self._device.close()

    def __enter__(self) -> "Transport":
        return self

    def __exit__(self, *exc: Any) -> bool:
        self.close()
        return False


def _check_index(index: int) -> int:
    """Validate a 1-based parameter index."""
    index = int(index)
    if not 1 <= index <= PARAM_COUNT:
        raise ValueError(f"parameter index must be 1..{PARAM_COUNT}, got {index}")
    return index

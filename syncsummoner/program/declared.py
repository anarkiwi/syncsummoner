"""What a declared parameter is: its kind, the register bits it drives and the values it writes.

The control to register map is the SDK ABI (``videomancer_abi.hpp``: pots 1-6 at
addresses 0-5, switches 7-11 at bits 0-4 of address 6, fader 12 at address 7);
``core_top.vhd`` latches SPI address *i* into ``registers_in(i)`` unchanged.
"""

from __future__ import annotations

import enum
from dataclasses import dataclass

import numpy as np

from .vmprog import ControlMode, ParamDecl

PARAM_COUNT = 12
VALUE_MAX = 1023
REGISTER_BITS = 10
REGISTER_COUNT = 32
TIMING_REGISTER = 8
CORE_REGISTERS = (9, 10, 11)
LATCHED_REGISTERS = 12
TOGGLE_REGISTER = 6
FADER_REGISTER = 7
FIRST_TOGGLE = 7
BOOL_THRESHOLD = 512
_STEPS = {ControlMode.STEPS_4 + i: 4 << i for i in range(7)}


class Kind(enum.Enum):
    """How a declared parameter takes values."""

    CONTINUOUS = "continuous"
    QUANTIZED = "quantized"
    BOOLEAN = "boolean"
    UNUSED = "unused"


@dataclass(frozen=True)
class Field:
    """The ``registers_in`` word and bits a control writes."""

    register: int
    bits: tuple[int, ...]

    @property
    def flat(self) -> tuple[int, ...]:
        """Bit positions in the flattened 320-bit ``registers_in`` port (element 0 is the MSB word)."""
        return tuple(flat_bit(self.register, b) for b in self.bits)

    def __str__(self) -> str:
        if len(self.bits) == 1:
            return f"r{self.register}[{self.bits[0]}]"
        return f"r{self.register}[{max(self.bits)}:{min(self.bits)}]"


def flat_bit(register: int, bit: int) -> int:
    """GHDL flattens ``array(0 to 31) of slv(9 downto 0)`` with element 0 in the top word."""
    return (REGISTER_COUNT - 1 - register) * REGISTER_BITS + bit


def field(parameter_id: int) -> Field:
    """Register field of a control by its ``parameter_id`` (1-12)."""
    if not 1 <= parameter_id <= PARAM_COUNT:
        raise ValueError(f"parameter_id {parameter_id} is not a control")
    if parameter_id < FIRST_TOGGLE:
        return Field(parameter_id - 1, tuple(range(REGISTER_BITS)))
    if parameter_id < PARAM_COUNT:
        return Field(TOGGLE_REGISTER, (parameter_id - FIRST_TOGGLE,))
    return Field(FADER_REGISTER, tuple(range(REGISTER_BITS)))


def kind(decl: ParamDecl | None) -> tuple[Kind, int | None]:
    """Kind and declared level count, from control mode and value labels alone."""
    if decl is None:
        return Kind.UNUSED, None
    labels = len(decl.value_labels)
    if decl.control_mode == ControlMode.BOOLEAN or labels == 2:
        return Kind.BOOLEAN, 2
    if labels > 2:
        return Kind.QUANTIZED, labels
    if decl.control_mode in _STEPS:
        return Kind.QUANTIZED, _STEPS[decl.control_mode]
    return Kind.CONTINUOUS, None


def curve(manual: np.ndarray | int, mode: int) -> np.ndarray:
    """``apply_parameter_control_curve`` from the SDK's ``vmprog_parameter_utils.hpp``, vectorised."""
    v = np.asarray(manual, dtype=np.int64)
    if ControlMode.POLAR_DEGS_90 <= mode <= ControlMode.POLAR_DEGS_2880:
        t = np.mod(v, 1024)
        shift = int(mode) - ControlMode.POLAR_DEGS_360
        return (t >> -shift) if shift < 0 else ((t << shift) & VALUE_MAX)
    t = np.clip(v, 0, VALUE_MAX)
    u = VALUE_MAX - t
    sq, usq = (t * t) // VALUE_MAX, (u * u) // VALUE_MAX
    half = t < 512
    table = {
        ControlMode.LINEAR: lambda: t,
        ControlMode.LINEAR_HALF: lambda: t >> 1,
        ControlMode.LINEAR_QUARTER: lambda: t >> 2,
        ControlMode.LINEAR_DOUBLE: lambda: np.minimum(t << 1, VALUE_MAX),
        ControlMode.BOOLEAN: lambda: np.where(t >= BOOL_THRESHOLD, VALUE_MAX, 0),
        ControlMode.STEPS_4: lambda: (t >> 8) * 341,
        ControlMode.STEPS_8: lambda: (t >> 7) * 146,
        ControlMode.STEPS_16: lambda: (t >> 6) * 68,
        ControlMode.STEPS_32: lambda: (t >> 5) * 33,
        ControlMode.STEPS_64: lambda: (t >> 4) << 4,
        ControlMode.STEPS_128: lambda: (t >> 3) << 3,
        ControlMode.STEPS_256: lambda: (t >> 2) << 2,
        ControlMode.QUAD_IN: lambda: sq,
        ControlMode.QUAD_OUT: lambda: VALUE_MAX - usq,
        ControlMode.QUAD_IN_OUT: lambda: np.where(half, (t * t) >> 9, VALUE_MAX - ((u * u) >> 9)),
        ControlMode.SINE_IN: lambda: sq,
        ControlMode.SINE_OUT: lambda: VALUE_MAX - usq,
        ControlMode.SINE_IN_OUT: lambda: np.where(half, (t * t) >> 11, VALUE_MAX - ((u * u) >> 11)),
        ControlMode.CIRC_IN: lambda: sq,
        ControlMode.CIRC_OUT: lambda: VALUE_MAX - usq,
        ControlMode.CIRC_IN_OUT: lambda: np.where(
            half, (sq * t) // VALUE_MAX, VALUE_MAX - (usq * u) // VALUE_MAX
        ),
        ControlMode.QUINT_IN: lambda: ((sq * sq // VALUE_MAX) * t) // VALUE_MAX,
        ControlMode.QUINT_OUT: lambda: VALUE_MAX - ((usq * usq // VALUE_MAX) * u) // VALUE_MAX,
        ControlMode.QUINT_IN_OUT: lambda: np.where(
            half,
            (((sq * sq // VALUE_MAX) * t) << 4) // 1046529,
            VALUE_MAX - ((((usq * usq // VALUE_MAX) * u) << 4) // 1046529),
        ),
        ControlMode.QUART_IN: lambda: (sq * sq) // VALUE_MAX,
        ControlMode.QUART_OUT: lambda: VALUE_MAX - (usq * usq) // VALUE_MAX,
        ControlMode.QUART_IN_OUT: lambda: np.where(
            half, ((sq * sq) << 3) // 1046529, VALUE_MAX - (((usq * usq) << 3) // 1046529)
        ),
        ControlMode.EXPO_IN: lambda: np.where(t == 0, 0, (((sq * t) // VALUE_MAX) * t) // VALUE_MAX),
        ControlMode.EXPO_OUT: lambda: np.where(
            t == VALUE_MAX, VALUE_MAX, VALUE_MAX - (((usq * u) // VALUE_MAX) * u) // VALUE_MAX
        ),
        ControlMode.EXPO_IN_OUT: lambda: np.select(
            [t == 0, t == VALUE_MAX, half],
            [0, VALUE_MAX, (((sq * t) // VALUE_MAX) * t) >> 11],
            VALUE_MAX - ((((usq * u) // VALUE_MAX) * u) >> 11),
        ),
    }
    return table.get(mode, lambda: t)()


def register_value(decl: ParamDecl, manual: np.ndarray | int) -> np.ndarray:
    """The ``registers_in`` field value a manual setting writes.

    Min/max scale only the display: value-labelled knobs declare ``0..N-1`` and their
    programs read the top bits of the raw word (Quilt reads Grid Scale at ``r0[9:8]``).
    """
    out = curve(manual, decl.control_mode)
    if field(decl.parameter_id).register == TOGGLE_REGISTER:
        return (out >= BOOL_THRESHOLD).astype(np.int64)
    return out


def registers(params: tuple[ParamDecl, ...], state: np.ndarray, timing: int | None) -> dict[int, int]:
    """Register words 0-8 for a manual state vector indexed by ``parameter_id - 1``."""
    words = dict.fromkeys(range(TIMING_REGISTER), 0)
    for decl in params:
        f = field(decl.parameter_id)
        value = int(register_value(decl, int(state[decl.parameter_id - 1])))
        words[f.register] |= value << f.bits[0] if len(f.bits) == 1 else value
    if timing is not None:
        words[TIMING_REGISTER] = timing
    return words


def classes(decl: ParamDecl, live_bits: tuple[int, ...]) -> list[np.ndarray]:
    """Manual values grouped by what the program can see of them: one array per distinct read value."""
    manual = np.arange(VALUE_MAX + 1)
    reg = register_value(decl, manual)
    f = field(decl.parameter_id)
    if len(f.bits) == 1:
        seen = reg if live_bits else np.zeros_like(reg)
    else:
        seen = reg & sum(1 << b for b in live_bits)
    _, inverse = np.unique(seen, return_inverse=True)
    return [manual[inverse == k] for k in range(int(inverse.max()) + 1)]

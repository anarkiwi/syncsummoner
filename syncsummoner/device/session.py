"""Measurement discipline: additive parameter addressing, CC budget, settle timing.

``Parameter = Manual + Modulation + MIDI``. A CC is an offset onto the stored
manual value, so absolute addressing only exists relative to a parked reference.
"""

from __future__ import annotations

import time
from typing import Any, Callable, Mapping, Sequence

import numpy as np

from .profile import PARAM_COUNT, PARAM_MAX

#: Parking every manual value at zero makes the additive sum equal the MIDI offset.
PARK_REFERENCE = 0
#: Readback slack, in device units, for park and addressing verification.
DEFAULT_TOLERANCE = 8
#: The crossfader; it gates output even where `program info` names it `Null 12`.
CROSSFADER_INDEX = 12


def default_park() -> np.ndarray:
    """Park reference: zero everywhere, crossfader open so the output stays live.

    Parking the crossfader at zero blacks the output and makes every measurement
    degenerate, which no amount of parameter sweeping recovers from.
    """
    park = np.full(PARAM_COUNT, PARK_REFERENCE, dtype=int)
    park[CROSSFADER_INDEX - 1] = PARAM_MAX
    return park


class DeviceError(RuntimeError):
    """The device did not behave as the measurement discipline requires."""


class ParkError(DeviceError):
    """Parameters did not read back at the parked reference."""


class AddressingError(DeviceError):
    """A parameter write did not land where it was asked to."""


def to_device(value: float | bool | int) -> int:
    """Coerce a user value to device units 0..1023.

    ``bool`` saturates, ``float`` is a normalised 0..1 fraction, ``int`` is
    already in device units.
    """
    if isinstance(value, (bool, np.bool_)):
        return PARAM_MAX if value else 0
    if isinstance(value, (float, np.floating)):
        return int(np.clip(round(float(value) * PARAM_MAX), 0, PARAM_MAX))
    return int(np.clip(int(value), 0, PARAM_MAX))


class Session:
    """A measurement session over one transport: rate limiting, caching, settling.

    ``sleep`` and ``clock`` are injected so timing is exercisable without
    wall-clock delay.
    """

    def __init__(
        self,
        transport: Any,
        *,
        cc_budget_hz: float = 200.0,
        park_values: Sequence[int] | None = None,
        load_blackout_s: float = 4.0,
        tolerance: int = DEFAULT_TOLERANCE,
        verify_settle_s: float = 0.25,
        sleep: Callable[[float], None] = time.sleep,
        clock: Callable[[], float] = time.monotonic,
    ):
        if cc_budget_hz <= 0:
            raise ValueError(f"cc_budget_hz must be positive, got {cc_budget_hz}")
        self.transport = transport
        self.cc_budget_hz = float(cc_budget_hz)
        self.load_blackout_s = float(load_blackout_s)
        self.tolerance = int(tolerance)
        self.verify_settle_s = float(verify_settle_s)
        self._sleep = sleep
        self._clock = clock
        self._park = default_park()
        if park_values is not None:
            self._park = np.clip(np.asarray(park_values, dtype=int), 0, PARAM_MAX)
            if self._park.shape != (PARAM_COUNT,):
                raise ValueError(f"park_values must have {PARAM_COUNT} entries")
        self._sent: dict[int, int] = {}
        self._next_send = clock()

    @property
    def park_values(self) -> np.ndarray:
        """Copy of the parked manual reference."""
        return self._park.copy()

    @property
    def sent(self) -> dict[int, int]:
        """Copy of the cached CC offsets, keyed by 1-based parameter index."""
        return dict(self._sent)

    def _throttle(self, messages: int) -> None:
        """Delay so the aggregate wire rate stays within the CC budget."""
        now = self._clock()
        if self._next_send > now:
            self._sleep(self._next_send - now)
            now = self._next_send
        self._next_send = now + messages / self.cc_budget_hz

    def offsets(self, values: Mapping[int, float | bool]) -> dict[int, int]:
        """CC offsets that land ``values`` absolutely, given the parked reference."""
        out = {}
        for index, value in values.items():
            if not 1 <= int(index) <= PARAM_COUNT:
                raise ValueError(f"parameter index must be 1..{PARAM_COUNT}, got {index}")
            target = to_device(value)
            out[int(index)] = int(np.clip(target - self._park[int(index) - 1], 0, PARAM_MAX))
        return out

    def set_params(self, values: Mapping[int, float | bool]) -> None:
        """Write parameters as additive CC offsets, deduplicated and rate limited."""
        pending = [(i, o) for i, o in self.offsets(values).items() if self._sent.get(i) != o]
        for index, offset in pending:
            self._throttle(self.transport.msgs_per_param)
            self.transport.set_param(index, offset)
            self._sent[index] = offset

    def park(self) -> None:
        """Drive every parameter to the reference and verify it over serial.

        Manual values are written with the absolute serial verb and the MIDI
        contribution is zeroed, so the combined readback must equal the reference.
        Writes are paced: an unpaced burst drops the LSB of 14-bit CC pairs.
        Readback settles first, because serial and MIDI are asynchronous and the
        last parameter written is otherwise sampled before the device applies it.
        """
        for index in range(1, PARAM_COUNT + 1):
            self._throttle(self.transport.msgs_per_param)
            self.transport.set_param(index, 0)
            self.transport.set_manual(index, int(self._park[index - 1]))
            self._sent[index] = 0
        for remaining in (self.verify_settle_s, self.verify_settle_s, None):
            self._sleep(self.verify_settle_s)
            state = np.asarray(self.transport.program_state(), dtype=int)
            drift = np.abs(state - self._park)
            if not np.any(drift > self.tolerance):
                return
            if remaining is None:
                bad = np.flatnonzero(drift > self.tolerance) + 1
                raise ParkError(
                    f"parameters {bad.tolist()} read back {state.tolist()}, want {self._park.tolist()}"
                )

    def verify_addressing(self, index: int = 1, *, target: float | bool = 0.75) -> int:
        """Check empirically that a write lands where asked, and raise when it does not.

        Guards against the additive model silently clipping or offsetting a
        measurement sweep.
        """
        self.park()
        want = to_device(target)
        self.set_params({index: want})
        got = int(np.asarray(self.transport.program_state(), dtype=int)[int(index) - 1])
        if abs(got - want) > self.tolerance:
            raise AddressingError(
                f"P{index} asked for {want}, landed at {got}; "
                f"apparent offset {got - want} from park {int(self._park[int(index) - 1])}"
            )
        return got

    def load_program(self, name: str, *, park: bool = True) -> None:
        """Load a program, absorb the multi-second output blackout, and re-park."""
        self.transport.load_program(name)
        self._sleep(self.load_blackout_s)
        self._sent.clear()
        if park:
            self.park()

    def framediff_floor(
        self, capture: Any, frames: int = 8, *, sigma: float = 3.0, eps: float = 1e-6
    ) -> float:
        """Noise floor of the capture chain, as mean absolute frame difference.

        Measured rather than assumed, so the settle threshold is derived from
        this rig instead of a tuned constant.
        """
        diffs = np.fromiter(_diff_stream(capture, frames), dtype=np.float64)
        if diffs.size == 0:
            return eps
        return float(diffs.mean() + sigma * diffs.std()) + eps

    def settle_frames(
        self,
        capture: Any,
        values: Mapping[int, float | bool],
        *,
        baseline: int = 8,
        max_frames: int = 120,
        hold: int = 4,
        sigma: float = 3.0,
    ) -> int | None:
        """Frames until framediff plateaus after a step, or None when it never does.

        A None result means the program is ``non_settling`` and belongs to the
        dynamics analyzer rather than the static one.
        """
        floor = self.framediff_floor(capture, baseline, sigma=sigma)
        self.set_params(values)
        run = 0
        for index, diff in enumerate(_diff_stream(capture, max_frames), start=1):
            run = run + 1 if diff <= floor else 0
            if run >= hold:
                return index - hold + 1
        return None


def _diff_stream(capture: Any, frames: int):
    """Yield mean absolute frame differences over ``frames`` successive reads."""
    prev = capture.read()
    for _ in range(frames):
        frame = capture.read()
        if frame is None or prev is None:
            return
        yield float(np.abs(frame - prev).mean())
        prev = frame

"""Measurement discipline: additive parameter addressing, write budget, settle timing.

``Parameter = Manual + Modulation + MIDI``. A CC is an offset onto the stored
manual value, so absolute addressing only exists relative to a parked reference.
"""

from __future__ import annotations

import time
from contextlib import ExitStack
from typing import Any, Callable, Iterable, Mapping, Sequence

import numpy as np

from syncsummoner.progress import LOG

from .profile import CROSSFADER_INDEX, PARAM_COUNT, PARAM_MAX, ParamKind, Role

#: Parking every manual value at zero makes the additive sum equal the MIDI offset.
PARK_REFERENCE = 0
#: Readback slack, in device units, for park and addressing verification.
DEFAULT_TOLERANCE = 8
#: Parameter writes per second the host sends; each is ``msgs_per_param`` CC
#: messages on the wire. docs/hardware.md records the measured backlog limit.
CC_BUDGET_HZ = 200.0
#: The operator name that means no modulation.
DISABLED_OPERATOR = "Disabled"

#: A freshly loaded program keeps applying its defaults after the source returns.
LOCK_POLLS = 40
LOCK_POLL_S = 0.25
STABLE_READS = 2


def default_park() -> np.ndarray:
    """Park reference: zero everywhere, so the additive sum equals the CC offset.

    A crossfader parked at max clips every offset for it to zero, since offsets
    are non-negative; parking it at zero like everything else is what lets it
    address its full range instead of sitting pinned open.
    """
    return np.full(PARAM_COUNT, PARK_REFERENCE, dtype=int)


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
        cc_budget_hz: float = CC_BUDGET_HZ,
        park_values: Sequence[int] | None = None,
        tolerance: int = DEFAULT_TOLERANCE,
        verify_settle_s: float = 0.25,
        sleep: Callable[[float], None] = time.sleep,
        clock: Callable[[], float] = time.monotonic,
        journal: Any = None,
    ):
        if cc_budget_hz <= 0:
            raise ValueError(f"cc_budget_hz must be positive, got {cc_budget_hz}")
        self.transport = transport
        self.cc_budget_hz = float(cc_budget_hz)
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
        self.journal = journal
        self._writes = 0
        self._last_load: tuple[str | None, bool] = (None, False)
        self.assign_events: list[dict[str, Any]] = []

    @property
    def park_values(self) -> np.ndarray:
        """Copy of the parked manual reference."""
        return self._park.copy()

    @property
    def sent(self) -> dict[int, int]:
        """Copy of the cached CC offsets, keyed by 1-based parameter index."""
        return dict(self._sent)

    def _throttle(self) -> None:
        """Delay one parameter write so the aggregate write rate stays within the budget."""
        now = self._clock()
        if self._next_send > now:
            self._sleep(self._next_send - now)
            now = self._next_send
        self._next_send = now + 1.0 / self.cc_budget_hz

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
        if pending:
            self._writes += len(pending)
        for index, offset in pending:
            self._throttle()
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
        self.transport.consume_midi_assign()
        for index in range(1, PARAM_COUNT + 1):
            self._throttle()
            self.transport.set_param(index, 0)
            self.transport.set_manual(index, int(self._park[index - 1]))
            self._sent[index] = 0
        for remaining in (self.verify_settle_s, self.verify_settle_s, None):
            self._sleep(self.verify_settle_s)
            state = np.asarray(self.transport.program_state(), dtype=int)
            drift = np.abs(state - self._park)
            if not np.any(drift > self.tolerance):
                self._reassert_cc_map()
                return
            if remaining is None:
                bad = np.flatnonzero(drift > self.tolerance) + 1
                raise ParkError(
                    f"parameters {bad.tolist()} read back {state.tolist()}, want {self._park.tolist()}"
                )

    def _reassert_cc_map(self) -> None:
        """Reassign a MIDI Assign capture of the sink CC, recording it against the last load."""
        assignment = self.transport.reassert_cc_map()
        if assignment is None:
            return
        program, link = self._last_load
        event = {
            "time": time.time(),
            "program": program,
            "params": list(assignment.before.armed),
            "link": link,
        }
        self.assign_events.append(event)
        LOG.warning("%s", assignment.summary())
        self._note("midi_assign", summary=assignment.summary(), **event)

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

    def working_point(self, info: Any, *, exclude: Iterable[int] = ()) -> dict[int, float | bool]:
        """A mid-scale operating point with any bypass held off, excluding driven slots.

        The crossfader is forced in at mid-scale like any other continuous
        effect even where a program calls it unused, because it gates output
        regardless of what the program reports.
        """
        skip = set(exclude)
        values: dict[int, float | bool] = {}
        for spec in info.params:
            if spec.index in skip:
                continue
            if spec.role is Role.BYPASS:
                values[spec.index] = False
            elif spec.index == CROSSFADER_INDEX:
                values[spec.index] = 0.5
            elif spec.kind is ParamKind.UNUSED:
                continue
            elif spec.kind is ParamKind.BOOLEAN:
                values[spec.index] = False
            elif "mix" in spec.name.lower():
                values[spec.index] = 1.0
            else:
                values[spec.index] = 0.5
        return values

    def arm_modulation(
        self,
        info: Any,
        operators: Sequence[str],
        rng: np.random.Generator,
        *,
        exclude: Iterable[int] = (),
        count: int = 5,
        depth: tuple[int, int] = (500, 900),
    ) -> list[str]:
        """Bind on-board operators to unused continuous slots; returns what was bound.

        Operators run on the FPGA at field rate, which no CC stream can match;
        without them a program only moves when a parameter write arrives.
        """
        skip = set(exclude)
        armed = []
        for spec in info.params:
            if len(armed) >= count:
                break
            if spec.kind is not ParamKind.CONTINUOUS or spec.index in skip:
                continue
            operator = operators[int(rng.integers(len(operators)))]
            self.transport.set_modulation_source(spec.index, operator)
            self.transport.set_manual(
                spec.index, PARAM_MAX // 2, time_=int(rng.integers(*depth)), space=512, slope=512
            )
            armed.append(f"{spec.name}<-{operator}")
        return armed

    def disarm_modulation(self) -> None:
        """Return every slot to Disabled so the next program starts clean."""
        for index in range(1, PARAM_COUNT + 1):
            self.transport.set_modulation_source(index, DISABLED_OPERATOR)

    def _note(self, kind: str, **detail: Any) -> None:
        """Record an action, when a journal is attached."""
        if self.journal is not None:
            self.journal.record("call", verb=kind, **detail)

    def health(self, capture: Any = None) -> dict[str, Any]:
        """Probe and journal whether the device is carrying video.

        The parameter-write count rides along, because the interesting question
        is what the device was asked to do between one healthy probe and the next.
        """
        from .journal import probe_health

        state = probe_health(self.transport, capture)
        state["writes"] = self._writes
        if self.journal is not None:
            self.journal.record("health", **state)
        self._writes = 0
        return state

    def ensure_live(
        self, capture: Any, *, timeout_s: float = 20.0, attempts: int = 1, require_motion: bool = True
    ) -> None:
        """Block until the capture is carrying content, resyncing if it is not.

        ``require_motion`` distinguishes the two callers: playing a clip, a frozen
        frame means the output died; probing a still stimulus, no motion is
        correct and only a missing or uniform picture indicates failure. A journalled failure
        reports what was asked of the device since it was last healthy.
        """
        wait = capture.wait_for_content if require_motion else capture.wait_for_lock
        for attempt in range(max(1, attempts) + 1):
            if wait(timeout_s=timeout_s):
                if self.journal is not None:
                    self.journal.record("health", ok=True, via="ensure_live")
                return
            self._note("resync", attempt=attempt)
            if not self.transport.resync():
                break
        trail = self.journal.report() if self.journal is not None else "no journal attached"
        raise DeviceError(f"capture is not carrying content; power-cycle the device\n{trail}")

    def load_program(self, name: str, *, park: bool = True, link: Any = None) -> None:
        """Load a program and re-park.

        An optional source ``link`` is held down across the reconfiguration,
        because the FPGA runs on the clock recovered from the incoming video;
        the transport returns once the device runs the program, waiting on
        liveness rather than a fixed blackout. The drop clears the source
        framebuffer: re-push any stimulus afterwards.
        """
        self._note("load_program", program=name, link=link is not None)
        self._last_load = (name, link is not None)
        with ExitStack() as stack:
            if link is not None:
                stack.enter_context(link.quiet())
            self.transport.load_program(name)
        assignment = self.transport.cc_assignment
        if assignment is not None:
            LOG.info("%s: %s", name, assignment.summary())
            self._note("cc_map", program=name, summary=assignment.summary())
        self._sent.clear()
        if link is not None:
            self.wait_state_settled()
        if park:
            self.park()

    def wait_state_settled(
        self, *, polls: int = LOCK_POLLS, poll_s: float = LOCK_POLL_S, stable: int = STABLE_READS
    ) -> bool:
        """Wait until the device stops applying a freshly loaded program's defaults.

        The source returns before the device has finished initialising, and lock is
        already reported by then, so parameter state settling is the usable signal.
        """
        previous: tuple[int, ...] | None = None
        runs = 0
        for _ in range(int(polls)):
            try:
                state: tuple[int, ...] | None = tuple(int(v) for v in self.transport.program_state())
            except Exception:  # pylint: disable=broad-except
                state = None
            runs = runs + 1 if state is not None and state == previous else 0
            if runs >= int(stable):
                return True
            previous = state
            self._sleep(poll_s)
        return False

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

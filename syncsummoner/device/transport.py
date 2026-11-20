"""Videomancer hardware transport: MIDI CC, USB CDC serial verbs, program load.

The only module in the project that imports pyvmancer. Everything above it
speaks ``ParamSpec`` / ``ProgramInfo`` / numpy, never raw shell JSON.
"""

from __future__ import annotations

import time
from contextlib import contextmanager, nullcontext
from dataclasses import dataclass, field
from typing import Any, Callable, ContextManager, Iterable, Iterator, Mapping, Sequence

import numpy as np
import pyvmancer as vm

from pyvmancer.const import ABSENT_COMMAND_CODES
from pyvmancer.diagnose import diagnose_video
from pyvmancer.route import read_route
from pyvmancer.shell import DEFAULT_AUTHOR
from pyvmancer.video import VideoStatus

from syncsummoner.progress import LOG

from syncsummoner.program.library import Library
from syncsummoner.program.vmprog import ProgramConfig

from .declared import declaration, initial_value, specs
from .profile import PARAM_COUNT, PARAM_MAX, ParamKind, ParamSpec, snap_toggles, toggle_slots

#: Combined (Manual + Modulation + MIDI) output keys, in order of preference.
COMBINED_KEYS = ("o", "out", "output", "combined")
#: Stored manual value keys.
MANUAL_KEYS = ("m", "manual")


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


class HeldLinkError(RuntimeError):
    """A liveness wait ran while the source link was held down, which no device state can outlast."""


def transmitting(status: VideoStatus, timing: str) -> bool | None:
    """Whether the HDMI transmitter drives active video at ``timing``; None when the firmware does not say.

    The capture card exposes no signal state and its only input is this output,
    so the transmitter is the rig's one signal source. pyvmancer's
    ``video.output_active`` (a sink attached, ``hdmi_tx`` with a nonzero VIC and
    data enable) is the driving test; it is skipped when the firmware omits those
    fields, which leaves the answer unknown rather than true. A sink reported
    absent is a definite no.
    """
    found = {finding.check: finding.ok for finding in diagnose_video(status)}
    if found.get("video.output_connected") is False:
        return False
    active = found.get("video.output_active")
    if active is None:
        return None
    return active and str(status.output_timing).lower() == str(timing).lower()


@dataclass(frozen=True)
class ProgramInfo:
    """A program's name and its twelve parameter specs: labels as the device reports, kinds as declared."""

    name: str
    params: list[ParamSpec] = field(default_factory=list)
    program_id: int | None = None
    declared: ProgramConfig | None = None

    @classmethod
    def from_json(cls, data: Mapping[str, Any], declared: ProgramConfig | None = None) -> "ProgramInfo":
        """Parse a ``program info`` payload; every slot ``declared`` leaves undeclared is UNUSED."""
        names = {i: str(entry.get("name", "")) for i, entry in enumerate(data.get("parameters", ()), start=1)}
        return cls(
            name=str(data.get("name", "")),
            params=specs(declared, names),
            program_id=data.get("id"),
            declared=declared,
        )

    @property
    def used(self) -> list[ParamSpec]:
        """Parameters this program actually declares."""
        return [p for p in self.params if p.kind is not ParamKind.UNUSED]

    @property
    def initial(self) -> dict[int, int]:
        """Each declared parameter's initial value in device units, by 1-based index."""
        return {d.parameter_id: initial_value(d) for d in self.declared.params} if self.declared else {}

    def sweepable(self, count: int | None = None, *, exclude: Iterable[int] = ()) -> list[ParamSpec]:
        """Declared continuous or quantized parameters, short of ``exclude``."""
        skip = set(exclude)
        out = [
            p
            for p in self.params
            if p.kind in (ParamKind.CONTINUOUS, ParamKind.QUANTIZED) and p.index not in skip
        ]
        return out if count is None else out[:count]


class Transport:
    """A Videomancer addressed over MIDI CC and the serial shell.

    Parameter indices are 1-based everywhere in this project; the serial
    modulation verbs take 0-based slots and the conversion happens here.
    """

    def __init__(self, device: Any, library: Library | None = None):
        self._device = device
        self.cc_faults: list[dict[str, Any]] = []
        self._library = library
        self._loaded: str | None = None
        self._held: str | None = None
        self._authored: bool | None = None
        self._listed: dict[str, set[str]] | None = None

    @classmethod
    def open(
        cls,
        *,
        serial: str | None = None,
        report: Callable[[str], None] | None = LOG.info,
        stall: float | None = None,
        library: Library | None = None,
    ) -> "Transport":
        """Open both control links to an attached device, declarations read from ``library``.

        Device-side waits last while the device stays on USB, reporting to ``report``;
        ``stall`` is pyvmancer's opt-in guard, failing once nothing observable changes.
        """
        transport = cls(vm.Videomancer.open(serial_number=serial), library)
        if transport.device.has_shell:
            transport.device.shell.report, transport.device.shell.stall = transport.annotate(report), stall
        return transport

    def annotate(self, report: Callable[[str], None] | None) -> Callable[[str], None] | None:
        """``report``, with each progress line naming the link a hold has down."""
        if report is None:
            return None
        return lambda line: report(line if self._held is None else f"{line}, link down")

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

    def watch(self, what: str) -> Any:
        """A pyvmancer liveness ``Watch`` for a wait on device state, with the shell's report and stall.

        Device state follows the source link, so the watch refuses to start or to
        observe while a hold has the link down: such a wait could never end.
        """
        self._refuse_held(what)
        watch = self._shell.watch(what)
        observe = watch.observe

        def guarded(label: str, evidence: Any = None) -> None:
            self._refuse_held(what)
            observe(label, evidence)

        watch.observe = guarded
        return watch

    def _refuse_held(self, what: str) -> None:
        """Raise :class:`HeldLinkError` while a hold has the source link down."""
        if self._held is not None:
            raise HeldLinkError(f"{what}: started while the link is held down for {self._held}")

    @contextmanager
    def _holding(self, name: str, hold: Callable[[], ContextManager[Any]]) -> Iterator[None]:
        """Run the block inside ``hold()``, marking the link down for ``name`` until it is restored."""
        self._held = name
        try:
            with hold():
                yield
        finally:
            self._held = None

    def _has_authors(self) -> bool:
        """Whether the firmware enumerates program authors, asked once per transport.

        Only an unknown-command or usage error means it does not; any other failure raises,
        a malformed reply (``ReplyShapeError``) included.
        """
        if self._authored is None:
            try:
                self._shell.authors()
                self._authored = True
            except vm.ShellError as err:
                if err.code not in ABSENT_COMMAND_CODES:
                    raise
                self._authored = False
        return self._authored

    def authors(self) -> list[str] | None:
        """Program authors the firmware enumerates, or None when it has none."""
        return list(self._shell.authors()) if self._has_authors() else None

    def route(self) -> int | None:
        """Front panel Vid Route Mode as its settings value; None when the firmware has no such key."""
        return read_route(self._shell)

    def firmware(self) -> str:
        """Firmware version string, recorded into every measurement."""
        return self._shell.version()

    def programs(self) -> list[str]:
        """Every installed FPGA program name, kept per author for :meth:`_needs_enumerate`."""
        names = list(self._device.programs())
        listed: dict[str, set[str]] = {}
        for name in names:
            author, slash, plain = name.partition("/")
            listed.setdefault(author if slash else DEFAULT_AUTHOR, set()).add(
                _match_key(plain if slash else name)
            )
        self._listed = listed
        return names

    def _needs_enumerate(self, author: str, plain: str) -> bool:
        """Whether ``program load plain`` could load another author's program unless ``author`` is enumerated.

        The firmware matches a name in the active author's index first and otherwise searches every
        author's folder and switches to the one holding it, so an active ``author``, or a name this
        session listed under ``author`` alone, loads the right program without the index rebuild.
        """
        if self._shell.active_author() == author:
            return False
        if self._listed is None or author not in self._listed:
            return True
        key = _match_key(plain)
        return key not in self._listed[author] or any(
            key in names for other, names in self._listed.items() if other != author
        )

    def current_program(self) -> str | None:
        """Name of the loaded program: as loaded here, else the device's plain name."""
        if self._loaded is None:
            self._loaded = self._shell.status().get("program")
        return self._loaded

    def prepare(self, name: str) -> None:
        """Make ``name``'s author the active one now, so loading ``name`` later needs no index rebuild.

        The firmware rebuilds an author's index when a load switches author, and a load of a name whose
        author is not active searches every author's folder instead; either can outlast a lead-in.
        """
        author, slash, plain = name.partition("/")
        if not slash:
            author, plain = DEFAULT_AUTHOR, name
        if (slash or self._has_authors()) and self._needs_enumerate(author, plain):
            self._shell.enumerate_author(author, self._shell.watch(f"enumerating {author}"))

    def load_program(self, name: str, hold: Callable[[], ContextManager[Any]] | None = None) -> None:
        """Load ``Name`` or ``author/Name``, returning once the device runs it with its CC map assigned.

        ``hold`` brackets only the ``program load`` command up to its reply, which
        commits the load. The program runs on the clock recovered from the source
        link a hold takes down, so the run, ``program info`` and the CC map are
        awaited on liveness after ``hold`` has restored it, never inside it.
        """
        author, slash, plain = name.partition("/")
        if not slash:
            author, plain = DEFAULT_AUTHOR, name
        shell, device = self._shell, self._device
        reply = shell.watch(f"loading {name}")
        if (slash or self._has_authors()) and self._needs_enumerate(author, plain):
            shell.enumerate_author(author, reply)
        self._loaded = None
        with self._holding(name, hold) if hold is not None else nullcontext():
            answer = shell.command("program", "load", plain, watch=reply)
        if not answer.ok:
            raise vm.VmancerError(f"expected ok from 'program load {plain}', got {answer.payload!r}")
        shell.wait_loaded(plain, watch=self.watch(f"loading {name}"))
        device.program = name
        device.parameters(refresh=True)
        self.check_cc_map(f"after loading {name}")
        self._loaded = name

    def check_cc_map(self, when: str) -> Any:
        """Read the device's CC map back; a slot off the CCs this host sends is recorded and raised.

        Nothing is sent to sink a MIDI Assign, and the map is never rewritten: on this firmware a
        CC119 is a MIDI reset, and the map lives in persistent settings.
        """
        try:
            return self._device.verify_cc_map()
        except vm.CcMapError as err:
            self.cc_faults.append({"time": time.time(), "when": when, "error": str(err)})
            LOG.error("CC map fault %s: %s", when, err)
            raise

    def verify_cc_map(self) -> Any:
        """Read the CC map back at the end of a take, raising ``pyvmancer.CcMapError`` on any difference."""
        return self.check_cc_map("at the end of a take")

    def program_info(self, name: str | None = None) -> ProgramInfo:
        """Parameter specs of a program, loading it first when it is not current.

        The device reports a plain name that does not say which author it came
        from, so only a program this transport loaded counts as current.
        """
        return self.describe(self.program_payload(name))

    def describe(self, payload: Mapping[str, Any]) -> ProgramInfo:
        """A ``program info`` payload of the loaded program, typed by that program's own declaration.

        Raises :class:`~syncsummoner.device.declared.UndeclaredProgramError` when no
        ``.vmprog`` or firmware image declares it.
        """
        loaded = (
            self._loaded if self._loaded and self._loaded.rpartition("/")[2] == payload.get("name") else None
        )
        return ProgramInfo.from_json(
            payload, declaration(loaded or str(payload.get("name", "")), self._library)
        )

    def program_payload(self, name: str | None = None) -> dict[str, Any]:
        """The device's own ``program info`` reply, unparsed, so it can be archived as is."""
        if name is not None and name != self._loaded:
            self.load_program(name)
        return dict(self._shell.program_info())

    def set_param(self, index: int, value: float | bool) -> None:
        """Send a MIDI CC for one parameter; the device adds it to the manual value.

        A toggle slot's value is sent as the firmware reads it (:func:`~.profile.snap_toggles`).
        """
        index = _check_index(index)
        if index in toggle_slots():
            value = int(snap_toggles(index, PARAM_MAX if value is True else int(value)))
        self._device.set_param(index, value)

    def set_manual(self, index: int, value: int) -> None:
        """Set a slot's manual value; the serial verb is absolute, unlike a CC, which is an offset onto it."""
        self._shell.set_modulation(_check_index(index) - 1, int(np.clip(value, 0, PARAM_MAX)))

    def program_manifest(self) -> Any:
        """Installed program library, or None when the device carries no manifest.

        Coverage is partial: firmware built-ins never appear, so a caller must
        fall back to measurement for anything the manifest does not describe.
        """
        try:
            return self._shell.program_manifest()
        except Exception:
            return None

    def file_hash(self, path: str) -> str:
        """Device-computed digest of a file, the cache key for derived measurements."""
        return str(self._shell.hash_file(path))

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
        return self._shell.video_state()

    def resync(self, **kwargs: Any) -> bool:
        """Re-initialise the output raster; see pyvmancer for the timing bounce."""
        return bool(self._shell.resync(**kwargs))

    def set_video_timing(self, timing: str) -> None:
        """Force an output timing standard, overriding genlock to the source."""
        self._shell.set_video_timing(timing)

    def set_video_input(self, source: str) -> None:
        """Select the input the processor reads from."""
        self._shell.set_video_input(source)

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


def _match_key(name: str) -> str:
    """A program name as ``program load`` matches it: case-insensitive, underscores as spaces."""
    return name.replace("_", " ").casefold()

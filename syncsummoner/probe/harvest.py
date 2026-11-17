"""Archive every program's native capture against a parameter sweep, in one recording.

ffmpeg records the card for the whole of a program's sweep while the host only
writes setpoints and notes when each was held. Frames are attributed to setpoints
afterwards from the card's own capture times, so nothing paces or samples a loop.
"""

from __future__ import annotations

import hashlib
import json
import time
from contextlib import ExitStack
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Iterable, Sequence

import numpy as np
from pyvmancer.const import ROUTE_HDMI_IN
from pyvmancer.errors import DeviceGoneError, PortBusyError, ProgramLoadError, ShellError

from syncsummoner.device.playout import DEFAULT_LOOP_FPS
from syncsummoner.device.profile import ParamKind
from syncsummoner.device.recorder import (
    BlankTakeError,
    RouteError,
    TakeReport,
    inspect_take,
    require_input,
    require_route,
    require_signal,
    rig_state,
    settle,
)
from syncsummoner.device.session import Session
from syncsummoner.probe import codeframes, plans
from syncsummoner.probe.archive import GAP, FrameRow
from syncsummoner.probe.runner import raw_params
from syncsummoner.probe.store import KeyKind, ProgramKey, atomic_write

__all__ = [
    "HarvestConfig",
    "HarvestError",
    "HarvestReport",
    "NoPictureError",
    "ProgramResult",
    "Window",
    "attribute",
    "carries_stimulus",
    "control",
    "discard_dark",
    "harvest",
    "harvest_program",
    "prepare_footage",
    "prove_passthrough",
    "rig_fault",
    "route_fault",
    "run_stimulus",
    "stimulus_loop",
    "sweep",
    "sweep_vectors",
    "upload_stimulus",
    "holds_program",
    "wait_healthy",
]

#: Loaded to ask whether the device still carries the source.
WEDGE_PROBE = "Passthru"
#: The two faults that stop a run, each on positive evidence only: same remedy, never conflated.
GONE_NOTE = "the device left USB: power cycle it, then rerun to resume"
BLACKED_NOTE = "black output from a live source: the device needs a power cycle, then rerun"
ROUTE_NOTE = "Vid Route Mode is wrong: set it on the front panel, then rerun"
#: Notes that are positive evidence the rig, not a program, is at fault.
FAULT_NOTES = (GONE_NOTE, BLACKED_NOTE, ROUTE_NOTE)


class HarvestError(RuntimeError):
    """The rig never reached a state an archive run could start from."""


class NoPictureError(BlankTakeError):
    """A passthrough that loaded and is transmitting the session timing shows no picture."""


@dataclass(frozen=True)
class HarvestConfig:
    """Geometry, sweep size and the settle timings an archive run is paced by."""

    width: int = 1920
    height: int = 1080
    capture_fps: int = 30
    #: The output timing the device must drive, as it names it; the raster above is its size.
    timing: str = "1080p30"
    #: A whole number of vblanks per loop frame, as the locked player requires.
    loop_fps: float = DEFAULT_LOOP_FPS
    loop_frames: int = codeframes.DEFAULT_COUNT
    setpoints: int = 32
    dwell_s: float = 1.0
    settle_s: float = 0.35
    #: The card's budget to deliver a probe once the transmitter drives the session timing.
    probe_timeout_s: float = 40.0
    probe_s: float = 2.0
    probe_path: str = "/tmp/syncsummoner-probe.mkv"
    #: The input the session feeds the device, verified after every load; None skips the check.
    source: str | None = "hdmi"
    #: Vid Route Mode's settings value, verified before the run and after every load; None skips it.
    route: int | None = ROUTE_HDMI_IN
    startup_s: float = 2.0
    health_timeout_s: float = 3600.0
    health_poll_s: float = 10.0
    seed: int = 11
    loop_seed: int = 7
    #: A prepared timecoded clip played in place of the codeframe loop; None plays the loop.
    footage: str | None = None
    #: Per program, parameters a render's evidence left undetermined and the steps each needs.
    focus: tuple[tuple[str, int, int], ...] = ()


@dataclass(frozen=True)
class Window:
    """One setpoint, its raw parameter vector, and the span it was held over."""

    setpoint: int
    params: tuple[int, ...]
    start: float
    end: float


@dataclass(frozen=True)
class ProgramResult:
    """What one program cost and yielded; ``error`` is set when it did not run."""

    program: str
    frames: int = 0
    measured: int = 0
    seconds: float = 0.0
    report: TakeReport | None = None
    error: str = ""

    @property
    def cached(self) -> bool:
        """True when the program was already archived under the same key."""
        return not self.frames and not self.error

    @property
    def dark(self) -> bool:
        """True when frames were archived but none ever showed a picture.

        The archive's standard is :attr:`TakeReport.ever_structured`, not a
        render's :attr:`TakeReport.usable`: a sweep's flat setpoints and a held
        picture are still data.
        """
        return self.report is not None and not self.report.ever_structured

    def __str__(self) -> str:
        if self.error:
            return f"{self.program}: FAILED {self.error}"
        if self.cached:
            return f"{self.program}: cached"
        return (
            f"{self.program}: {self.measured}/{self.frames} frames measured "
            f"in {self.seconds:.0f}s, {self.report}"
        )


@dataclass(frozen=True)
class HarvestReport:
    """Every program's outcome, and whether a rig fault, named by ``fault``, ended the run early."""

    results: list[ProgramResult] = field(default_factory=list)
    gone: bool = False
    blacked: bool = False
    seconds: float = 0.0
    fault: str = ""

    @property
    def stopped(self) -> bool:
        """Whether a rig fault ended the run before the last program."""
        return self.gone or self.blacked or bool(self.fault)

    @property
    def frames(self) -> int:
        """Frames archived across the whole run."""
        return sum(r.frames for r in self.results)

    @property
    def failures(self) -> list[ProgramResult]:
        """Programs that raised rather than archiving."""
        return [r for r in self.results if r.error]

    def summary(self) -> list[str]:
        """Closing lines: the totals, every program that failed with its error, and any fault."""
        failed = [r for r in self.failures if r.report is None]
        lines = [
            f"{self.frames} frames from {len(self.results)} programs in {self.seconds / 60:.1f} min, "
            f"{len(failed)} failed"
        ]
        lines += [f"  {r.program}: {r.error}" for r in failed]
        return lines + ([f"stopped: {self.fault}"] if self.fault else [])


def sweep_vectors(info: Any, config: HarvestConfig, rng: np.random.Generator) -> list[dict]:
    """Sobol vectors over every declared parameter, each sampled as its kind."""
    if not info.used:
        return [{}]
    return list(plans.sobol(info.params, n=config.setpoints, rng=rng))


class ReadbackError(RuntimeError):
    """The device read back parameter values other than the setpoint it was sent."""


def verify_readback(session: Any, setpoint: int, want: Sequence[int]) -> None:
    """Raise :class:`ReadbackError` unless the device's combined values are exactly ``want``."""
    got = [int(v) for v in session.transport.program_state()]
    wrong = [i + 1 for i, (a, b) in enumerate(zip(want, got)) if a != b]
    if wrong:
        raise ReadbackError(
            f"setpoint {setpoint}: P{', P'.join(map(str, wrong))} read back "
            f"{[got[i - 1] for i in wrong]}, sent {[int(want[i - 1]) for i in wrong]}"
        )


def focus_vectors(program: str, info: Any, config: HarvestConfig) -> list[dict]:
    """One-at-a-time steps across each parameter ``config.focus`` names for ``program``, evenly spaced.

    These are the parameters a render's evidence could not decide; their steps follow the sweep.
    """
    out = []
    for name, param, steps in config.focus:
        spec = next((p for p in info.params if p.index == param), None)
        if name != program or spec is None or steps < 1:
            continue
        values = (
            [bool(k % 2) for k in range(steps)]
            if spec.kind is ParamKind.BOOLEAN
            else np.linspace(0.0, 1.0, steps)
        )
        out.extend({param: v if isinstance(v, bool) else float(v)} for v in values)
    return out


def read_focus(path: str | Path) -> tuple[tuple[str, int, int], ...]:
    """A render's ``schedule.json`` as ``(program, param, steps)`` triples."""
    return tuple(
        (e["program"], int(e["param"]), int(e["steps"]))
        for e in json.loads(Path(path).read_text(encoding="utf-8"))
    )


def sweep(
    session: Any,
    base: dict,
    vectors: Sequence[dict],
    *,
    config: HarvestConfig,
    sleep: Callable[[float], None] = time.sleep,
    clock: Callable[[], float] = time.monotonic,
) -> list[Window]:
    """Write every setpoint in turn, returning the span each was held over.

    A window opens only after the parameters have settled and the device has read back the
    setpoint exactly, so the frames it claims were captured under that vector and no other.
    """
    windows = []
    for setpoint, vector in enumerate(vectors):
        session.set_params(vector)
        sleep(config.settle_s)
        verify_readback(session, setpoint, raw_params({**base, **vector}))
        start = clock()
        sleep(config.dwell_s)
        windows.append(Window(setpoint, raw_params({**base, **vector}), start, clock()))
    return windows


def attribute(
    times: np.ndarray, windows: Sequence[Window], program: str, *, base: Sequence[int]
) -> list[FrameRow]:
    """Assign each captured frame to the setpoint that was being held when it arrived.

    Frames between windows are kept as :data:`GAP` rather than dropped, so the
    sidecar stays one row per frame and the archive needs no re-encode.
    """
    times = np.asarray(times, dtype=np.float64)
    if not windows:
        return []
    starts = np.array([w.start for w in windows])
    ends = np.array([w.end for w in windows])
    order = np.argsort(starts, kind="stable")
    starts, ends = starts[order], ends[order]
    slot = np.searchsorted(starts, times, side="right") - 1
    held = np.clip(slot, 0, len(windows) - 1)
    inside = (slot >= 0) & (times <= ends[held])
    fallback = tuple(int(v) for v in base)
    rows = []
    for index, (captured, hit, at) in enumerate(zip(times, inside, held)):
        window = windows[int(order[at])]
        rows.append(
            FrameRow(
                frame=index,
                program=program,
                params=window.params if hit else fallback,
                setpoint=window.setpoint if hit else GAP,
                captured=float(captured),
            )
        )
    return rows


def holds_program(transport: Any, *, program: str = WEDGE_PROBE) -> bool:
    """Whether the device loads ``program`` and runs it.

    The load waits on liveness, so it returns only once the program runs, the
    device refuses it with an error reply or a load that settles elsewhere,
    or the device leaves USB; the last propagates as ``DeviceGoneError``.
    """
    try:
        transport.load_program(program)
        return True
    except (ShellError, ProgramLoadError):
        return False


def prove_passthrough(
    session: Any,
    recorder: Any,
    transport: Any,
    *,
    config: HarvestConfig,
    program: str = WEDGE_PROBE,
    link: Any = None,
    player: Any = None,
    sleep: Callable[[float], None] = time.sleep,
    clock: Callable[[], float] = time.monotonic,
) -> TakeReport:
    """Load a passthrough and wait for its picture, raising :class:`NoPictureError` without one.

    Once it drives the session timing its capture must be :attr:`TakeReport.mostly_picture`.
    Vid Route Mode and the input are verified first, so a wrong route or input fails as
    :class:`~syncsummoner.device.recorder.InputRouteError` rather than as black.
    ``config`` needs ``timing``, ``source``, ``route``, the ``probe_*`` budget, and ``loop_fps`` for a player.
    """
    session.load_program(program, park=True, link=link)
    require_route(transport, program=program, route=config.route)
    if player is not None and not player.is_running():
        player.start(fps=config.loop_fps)
    session.set_params(session.working_point(transport.program_info()))
    report = settle(
        recorder,
        transport,
        program=program,
        timing=config.timing,
        probe_timeout_s=config.probe_timeout_s,
        probe_path=config.probe_path,
        probe_s=config.probe_s,
        clock=clock,
        sleep=sleep,
    )
    require_input(transport, program=program, source=config.source)
    if not report.mostly_picture:
        raise NoPictureError(
            f"{program}: the passthrough shows no picture ({report}; {rig_state(transport)})"
        )
    return report


def carries_stimulus(session: Any, recorder: Any, transport: Any, **live: Any) -> bool:
    """Does a passthrough still carry the source? Proof the rig, not the program, is dark.

    One program going dark is legitimate, so only a failing passthrough shows the
    device has stopped emitting while still answering and reporting source lock.
    """
    try:
        prove_passthrough(session, recorder, transport, **live)
        return True
    except NoPictureError:
        return False


def control(session: Any, recorder: Any, transport: Any, *, config: Any, **live: Any) -> str:
    """Run the passthrough control on the rig as it stands; empty only when it shows a picture.

    Otherwise it says why: a :data:`FAULT_NOTES` note on positive evidence (the device left USB,
    a wrong Vid Route Mode, a transmitting passthrough with no picture), else what the check raised.
    """
    try:
        prove_passthrough(session, recorder, transport, config=config, **live)
        return ""
    except DeviceGoneError as exc:
        return f"{GONE_NOTE} ({exc})"
    except RouteError as exc:
        return f"{ROUTE_NOTE} ({exc})"
    except NoPictureError as exc:
        return f"{BLACKED_NOTE} ({exc})"
    except Exception as exc:
        return f"{WEDGE_PROBE} check raised {type(exc).__name__}: {exc}"


def rig_fault(
    session: Any,
    recorder: Any,
    transport: Any,
    *,
    config: HarvestConfig,
    log: Callable[[str], None] = lambda _message: None,
    **live: Any,
) -> str:
    """Why the rig, not a program, must stop the run; empty unless :func:`control` has positive evidence.

    Anything else the control raises, a refused load or a capture that never delivered
    included, is logged and the run goes on, since none of it shows the rig is at fault.
    """
    said = control(session, recorder, transport, config=config, **live)
    if said.startswith(FAULT_NOTES):
        return said
    if said:
        log(said)
    return ""


def route_fault(transport: Any, *, config: Any) -> str:
    """:data:`ROUTE_NOTE` with the reading when Vid Route Mode is not ``config.route``; else empty."""
    try:
        require_route(transport, program="before the run", route=config.route)
        return ""
    except RouteError as exc:
        return f"{ROUTE_NOTE} ({exc})"


def wait_healthy(
    open_transport: Callable[[], Any],
    *,
    config: HarvestConfig,
    sleep: Callable[[float], None] = time.sleep,
    clock: Callable[[], float] = time.monotonic,
) -> Any:
    """Block until an attached device will hold a program, reopening across a power cycle.

    The deadline is on the device appearing and taking a load at all; each
    load itself waits on liveness. A port another process holds raises at once.
    """
    deadline = clock() + config.health_timeout_s
    while clock() < deadline:
        try:
            transport = open_transport()
            if holds_program(transport):
                return transport
            transport.close()
        except PortBusyError:
            raise
        except Exception:
            pass
        sleep(config.health_poll_s)
    raise HarvestError(f"no device would hold a program within {config.health_timeout_s}s")


def stimulus_loop(config: HarvestConfig, rng: np.random.Generator | None = None) -> codeframes.CodeLoop:
    """The loop a run plays out, rebuilt from the run's own seed at playout geometry.

    The seed is the whole stimulus, so an archived run can be re-read against the
    picture it was shown without keeping a copy of it.
    """
    generator = np.random.default_rng(config.loop_seed) if rng is None else rng
    return codeframes.build_loop(
        width=config.width, height=config.height, rng=generator, count=config.loop_frames
    )


def upload_stimulus(player: Any, config: HarvestConfig, rng: np.random.Generator | None = None) -> int:
    """Push the footage clip, else the codeframe loop, into the player's tmpfs; returns bytes sent.

    The stimulus is invariant across programs and setpoints, so it is paid for
    once per run rather than once per pass.
    """
    if config.footage:
        return player.upload(config.footage)
    return player.upload(stimulus_loop(config, rng).frames())


def _sha256(path: Path) -> str:
    """Hex digest of a file, read in chunks."""
    with path.open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def prepare_footage(
    source: Any,
    directory: str | Path,
    *,
    render: Any,
    write: Callable[..., int],
    start: float = 0.0,
    seconds: float | None = None,
) -> dict[str, Any]:
    """Write ``source`` with ``write`` as a timecoded clip at ``render``'s geometry, and its record.

    The clip goes to ``stimulus.mkv`` and the record, written last as the commit
    marker, to ``stimulus.json``; a record for the same excerpt whose clip still
    hashes the same is reused, so a resumed run plays the picture it began with.
    """
    folder = Path(directory)
    clip, marker = folder / "stimulus.mkv", folder / "stimulus.json"
    wanted = {"source": str(source), "start": start, "seconds": seconds, "fps": render.fps}
    try:
        held = json.loads(marker.read_text(encoding="utf-8"))
        if all(held.get(k) == v for k, v in wanted.items()) and held.get("sha256") == _sha256(clip):
            return held
    except (OSError, ValueError):
        pass
    folder.mkdir(parents=True, exist_ok=True)
    frames = write(source, clip, config=render, seconds=seconds, start=start)
    record = {
        "kind": "footage",
        "clip": clip.name,
        "sha256": _sha256(clip),
        "frames": int(frames),
        "bits": render.bits,
        "strip_px": render.strip_px,
        **wanted,
    }
    atomic_write(marker, lambda tmp: tmp.write_text(json.dumps(record, indent=2), encoding="utf-8"))
    return record


def run_stimulus(config: HarvestConfig) -> dict[str, Any]:
    """The record of what a run plays: the footage clip's own, else the codeframe loop's seed."""
    if config.footage:
        return json.loads(Path(config.footage).with_name("stimulus.json").read_text(encoding="utf-8"))
    return {"kind": "codeframes", "loop_seed": config.loop_seed, "loop_frames": config.loop_frames}


def discard_dark(archive: Any, program: str, key: ProgramKey, result: ProgramResult) -> ProgramResult:
    """Remove a black program's archive, keeping the level that condemned it.

    A resume must not skip it as archived, and the level is what the dark verdict
    is later keyed and reported on.
    """
    del key
    for path in archive.paths(program):
        path.unlink(missing_ok=True)
    return ProgramResult(program, error="discarded: archived only uniform frames", report=result.report)


def harvest_program(
    session: Any,
    recorder: Any,
    archive: Any,
    transport: Any,
    program: str,
    key: ProgramKey,
    *,
    config: HarvestConfig,
    rng: np.random.Generator,
    link: Any = None,
    player: Any = None,
    stimulus: dict[str, Any] | None = None,
    sleep: Callable[[float], None] = time.sleep,
    clock: Callable[[], float] = time.monotonic,
) -> ProgramResult:
    """Sweep one program into a single recording, then commit it as its archive.

    The load drops the source link and blacks the output, so the transmitter is
    waited for before the recording starts rather than measured through it, and
    confirmed again at both ends of the recording. ``stimulus`` defaults to :func:`run_stimulus`.
    """
    start = clock()
    session.load_program(program, park=True, link=link)
    require_route(transport, program=program, route=config.route)
    if player is not None and not player.is_running():
        player.start(fps=config.loop_fps)
    payload = transport.program_payload()
    info = transport.describe(payload)
    base = session.working_point(info)
    session.set_params(base)
    settle(
        recorder,
        transport,
        program=program,
        timing=config.timing,
        probe_timeout_s=config.probe_timeout_s,
        probe_path=config.probe_path,
        probe_s=config.probe_s,
        clock=clock,
        sleep=sleep,
    )
    require_input(transport, program=program, source=config.source)
    video = archive.scratch(program)
    try:
        require_signal(transport, program=program, timing=config.timing, when="as the sweep began")
        with recorder.recording(video):
            windows = sweep(
                session,
                base,
                sweep_vectors(info, config, rng) + focus_vectors(program, info, config),
                config=config,
                sleep=sleep,
                clock=clock,
            )
        transport.verify_cc_map()
        require_route(transport, program=program, route=config.route)
        require_signal(transport, program=program, timing=config.timing, when="as the sweep ended")
        rows = attribute(recorder.timestamps(video), windows, program, base=raw_params(base))
        report = inspect_take(video, ffmpeg=recorder.ffmpeg)
        archive.commit(
            program,
            key,
            video,
            rows,
            width=config.width,
            height=config.height,
            fps=config.capture_fps,
            program_info=payload,
            stimulus=run_stimulus(config) if stimulus is None else stimulus,
        )
    finally:
        video.unlink(missing_ok=True)
    return ProgramResult(
        program,
        frames=len(rows),
        measured=sum(1 for row in rows if row.setpoint != GAP),
        seconds=clock() - start,
        report=report,
    )


def harvest(
    archive: Any,
    *,
    open_transport: Callable[[], Any],
    recorder: Any,
    player: Any = None,
    link: Any = None,
    programs: Iterable[str] | None = None,
    config: HarvestConfig | None = None,
    rng: np.random.Generator | None = None,
    session_factory: Callable[..., Any] = Session,
    log: Callable[[str], None] | None = None,
    retry_dark: bool = False,
    sleep: Callable[[float], None] = time.sleep,
    clock: Callable[[], float] = time.monotonic,
) -> HarvestReport:
    """Archive every program, resuming what is stored and stopping only on a rig fault.

    A program that raises is recorded as failed, with no dark verdict so a resume
    retries it, and the run goes on unless :func:`rig_fault` blames the rig.

    Archives are keyed on name and firmware rather than on the program binary,
    because hashing a binary runs over the serial shell at wire speed and this
    device wedges under exactly that load. A resume skips programs marked dark
    unless ``retry_dark``, which re-judges them under the current criterion.
    """
    config = HarvestConfig() if config is None else config
    rng = np.random.default_rng(config.seed) if rng is None else rng
    note = log if log is not None else lambda _message: None
    archive.directory.mkdir(parents=True, exist_ok=True)
    stimulus = run_stimulus(config)
    if player is not None:
        upload_stimulus(player, config)
    note("waiting for a device that will hold a program")
    transport = wait_healthy(open_transport, config=config, sleep=sleep, clock=clock)
    session = session_factory(transport)
    firmware = str(transport.firmware())
    names: Sequence[str] = list(programs) if programs else sorted(transport.programs())
    results, fault, start = [], "", clock()
    live = {"link": link, "player": player, "sleep": sleep, "clock": clock}
    try:
        fault = route_fault(transport, config=config)
        if fault:
            note(fault)
        with ExitStack() as stack:
            if player is not None:
                stack.enter_context(player.playing(fps=config.loop_fps))
                sleep(config.startup_s)
            for name in () if fault else names:
                key = ProgramKey(name, firmware, KeyKind.NAME_FIRMWARE)
                if archive.has(name, key, stimulus=stimulus) or (not retry_dark and archive.dark(name, key)):
                    results.append(ProgramResult(name))
                    note(str(results[-1]))
                    continue
                try:
                    results.append(
                        harvest_program(
                            session,
                            recorder,
                            archive,
                            transport,
                            name,
                            key,
                            config=config,
                            rng=rng,
                            stimulus=stimulus,
                            **live,
                        )
                    )
                except Exception as exc:
                    archive.clear_dark(name)
                    results.append(ProgramResult(name, error=f"{type(exc).__name__}: {exc}"))
                    note(str(results[-1]))
                    fault = rig_fault(session, recorder, transport, config=config, log=note, **live)
                else:
                    note(str(results[-1]))
                    if not results[-1].dark:
                        continue
                    luma = results[-1].report.luma if results[-1].report else 0.0
                    results[-1] = discard_dark(archive, name, key, results[-1])
                    fault = rig_fault(session, recorder, transport, config=config, log=note, **live)
                    if not fault:
                        archive.mark_dark(name, key, luma)
                        note(f"{name} discarded: dark, but the rig still carries the source")
                if fault:
                    note(fault)
                    break
    finally:
        transport.close()
    return HarvestReport(
        results=results,
        gone=fault.startswith(GONE_NOTE),
        blacked=fault.startswith(BLACKED_NOTE),
        seconds=clock() - start,
        fault=fault,
    )

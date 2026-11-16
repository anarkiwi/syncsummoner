"""Archive every program's native capture against a parameter sweep, in one recording.

ffmpeg records the card for the whole of a program's sweep while the host only
writes setpoints and notes when each was held. Frames are attributed to setpoints
afterwards from the card's own capture times, so nothing paces or samples a loop.
"""

from __future__ import annotations

import time
from contextlib import ExitStack, suppress
from dataclasses import dataclass, field, replace
from typing import Any, Callable, Iterable, Sequence

import numpy as np
from pyvmancer.errors import ShellError

from syncsummoner.device.profile import CROSSFADER_INDEX, ParamKind
from syncsummoner.device.recorder import BlankTakeError, TakeReport, inspect_take, settle
from syncsummoner.device.session import Session
from syncsummoner.device.transport import ProgramInfo
from syncsummoner.probe import codeframes, plans
from syncsummoner.probe.archive import GAP, FrameRow
from syncsummoner.probe.runner import raw_params
from syncsummoner.probe.store import KeyKind, ProgramKey

__all__ = [
    "HarvestConfig",
    "HarvestError",
    "HarvestReport",
    "ProgramResult",
    "Window",
    "attribute",
    "carries_stimulus",
    "discard_dark",
    "harvest",
    "harvest_program",
    "prove_passthrough",
    "rig_fault",
    "stimulus_loop",
    "sweep",
    "sweep_vectors",
    "upload_stimulus",
    "wait_healthy",
    "wedged",
]

#: Loaded to ask whether the device will still hold any program at all.
WEDGE_PROBE = "Passthru"
#: The two faults that stop a run: same remedy, different diagnosis, never conflated.
WEDGED_NOTE = "no program will load: the device needs a power cycle, then rerun to resume"
BLACKED_NOTE = "black output from a live source: the device needs a power cycle, then rerun"


class HarvestError(RuntimeError):
    """The rig never reached a state an archive run could start from."""


@dataclass(frozen=True)
class HarvestConfig:
    """Geometry, sweep size and the settle timings an archive run is paced by."""

    width: int = 1920
    height: int = 1080
    capture_fps: int = 30
    loop_fps: float = 12.0
    loop_frames: int = codeframes.DEFAULT_COUNT
    setpoints: int = 32
    dwell_s: float = 1.0
    settle_s: float = 0.35
    #: Relock after a load was measured at up to 19.1s, so anything shorter gives up early.
    live_timeout_s: float = 40.0
    probe_s: float = 2.0
    probe_path: str = "/tmp/syncsummoner-probe.mkv"
    load_blackout_s: float = 4.5
    startup_s: float = 2.0
    health_timeout_s: float = 3600.0
    health_poll_s: float = 10.0
    wedge_settle_s: float = 2.0
    seed: int = 11
    loop_seed: int = 7


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
    wedged: bool = False
    blacked: bool = False
    seconds: float = 0.0
    fault: str = ""

    @property
    def stopped(self) -> bool:
        """Whether a device fault ended the run before the last program."""
        return self.wedged or self.blacked

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
    """Sobol vectors over every used parameter, the crossfader swept like any other.

    Pinning the crossfader open produced false blank takes (Bitcullis, Corollas);
    it now sweeps 0..100% same as any other continuous effect.
    """
    spec = [replace(p, kind=ParamKind.CONTINUOUS) if p.index == CROSSFADER_INDEX else p for p in info.params]
    if not any(p.kind is not ParamKind.UNUSED for p in spec):
        return [{}]
    return list(plans.sobol(spec, n=config.setpoints, rng=rng))


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

    A window opens only after the parameters have settled, so the frames it
    claims were captured under that vector and no other.
    """
    windows = []
    for setpoint, vector in enumerate(vectors):
        session.set_params(vector)
        sleep(config.settle_s)
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


def wedged(
    transport: Any,
    *,
    program: str = WEDGE_PROBE,
    settle_s: float = 2.0,
    sleep: Callable[[float], None] = time.sleep,
) -> bool:
    """True when the device answers but will no longer hold any program.

    Measured: after roughly fifteen loads ``program info`` returns "no program
    loaded" and only a power cycle clears it; reload, resync and a timing change
    all fail, so there is nothing to retry. That read, after the settle, is the
    verdict: a shell error from the load itself is not, since the load can be
    acknowledged and read back before it completes.
    """
    try:
        with suppress(ShellError):
            transport.load_program(program)
        sleep(settle_s)
        transport.program_info()
        return False
    except Exception:
        return True


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
    """Load a passthrough and wait for its picture, raising :class:`BlankTakeError` without one.

    The passthrough carries the known chromatic stimulus, so it is held to the
    load's :attr:`TakeReport.mostly_picture`, where a splash is evidence of a dead link.
    """
    session.load_program(program, park=True, link=link)
    if player is not None and not player.is_running():
        player.start(fps=config.loop_fps)
    session.set_params(session.working_point(transport.program_info()))
    return settle(
        recorder,
        program=program,
        timeout_s=config.live_timeout_s,
        probe_path=config.probe_path,
        probe_s=config.probe_s,
        clock=clock,
        sleep=sleep,
    )


def carries_stimulus(session: Any, recorder: Any, transport: Any, **live: Any) -> bool:
    """Does a passthrough still carry the source? Proof the rig, not the program, is dark.

    One program going dark is legitimate, so only a failing passthrough shows the
    device has stopped emitting while still answering and reporting source lock.
    """
    try:
        prove_passthrough(session, recorder, transport, **live)
        return True
    except BlankTakeError:
        return False


def rig_fault(
    session: Any,
    recorder: Any,
    transport: Any,
    *,
    config: HarvestConfig,
    wedge_first: bool = True,
    log: Callable[[str], None] = lambda _message: None,
    **live: Any,
) -> str:
    """Why the rig, not a program, must stop the run; empty while it still carries the source.

    A device that will hold no program, or one that has gone, is :data:`WEDGED_NOTE`;
    a passthrough without a picture, the capture included, is :data:`BLACKED_NOTE`.
    A passthrough that raises proves only that its load failed, so the device is
    then asked whether it holds a program at all.
    """
    sleep = live.get("sleep", time.sleep)
    if wedge_first and wedged(transport, settle_s=config.wedge_settle_s, sleep=sleep):
        return WEDGED_NOTE
    try:
        prove_passthrough(session, recorder, transport, config=config, **live)
        return ""
    except BlankTakeError as exc:
        return f"{BLACKED_NOTE} ({exc})"
    except Exception as exc:
        log(f"{WEDGE_PROBE} check raised {type(exc).__name__}: {exc}")
        return WEDGED_NOTE if wedged(transport, settle_s=config.wedge_settle_s, sleep=sleep) else ""


def wait_healthy(
    open_transport: Callable[[], Any],
    *,
    config: HarvestConfig,
    sleep: Callable[[float], None] = time.sleep,
    clock: Callable[[], float] = time.monotonic,
) -> Any:
    """Block until a device will hold a program, reopening across a power cycle."""
    deadline = clock() + config.health_timeout_s
    while clock() < deadline:
        try:
            transport = open_transport()
            if not wedged(transport, settle_s=config.wedge_settle_s, sleep=sleep):
                return transport
            transport.close()
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
    """Push the codeframe loop into the player's tmpfs; returns bytes sent.

    The loop is invariant across programs and setpoints, so it is paid for once
    per run rather than once per pass.
    """
    return player.upload(stimulus_loop(config, rng).frames())


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
    sleep: Callable[[float], None] = time.sleep,
    clock: Callable[[], float] = time.monotonic,
) -> ProgramResult:
    """Sweep one program into a single recording, then commit it as its archive.

    The load drops the source link and blacks the output, so the picture is
    waited for before the recording starts rather than measured through it.
    """
    start = clock()
    session.load_program(program, park=True, link=link)
    if player is not None and not player.is_running():
        player.start(fps=config.loop_fps)
    payload = transport.program_payload()
    info = ProgramInfo.from_json(payload)
    base = session.working_point(info)
    session.set_params(base)
    settle(
        recorder,
        program=program,
        timeout_s=config.live_timeout_s,
        probe_path=config.probe_path,
        probe_s=config.probe_s,
        clock=clock,
        sleep=sleep,
    )
    video = archive.scratch(program)
    try:
        with recorder.recording(video):
            windows = sweep(
                session,
                base,
                sweep_vectors(info, config, rng),
                config=config,
                sleep=sleep,
                clock=clock,
            )
        transport.verify_cc_map()
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
    if player is not None:
        upload_stimulus(player, config)
    note("waiting for a device that will hold a program")
    transport = wait_healthy(open_transport, config=config, sleep=sleep, clock=clock)
    session = session_factory(transport, load_blackout_s=config.load_blackout_s)
    firmware = str(transport.firmware())
    names: Sequence[str] = list(programs) if programs else sorted(transport.programs())
    results, fault, start = [], "", clock()
    live = {"link": link, "player": player, "sleep": sleep, "clock": clock}
    try:
        with ExitStack() as stack:
            if player is not None:
                stack.enter_context(player.playing(fps=config.loop_fps))
                sleep(config.startup_s)
            for name in names:
                key = ProgramKey(name, firmware, KeyKind.NAME_FIRMWARE)
                if archive.has(name, key) or (not retry_dark and archive.dark(name, key)):
                    results.append(ProgramResult(name))
                    note(str(results[-1]))
                    continue
                try:
                    results.append(
                        harvest_program(
                            session, recorder, archive, transport, name, key, config=config, rng=rng, **live
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
                    fault = rig_fault(
                        session, recorder, transport, config=config, wedge_first=False, log=note, **live
                    )
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
        wedged=fault.startswith(WEDGED_NOTE),
        blacked=fault.startswith(BLACKED_NOTE),
        seconds=clock() - start,
        fault=fault,
    )

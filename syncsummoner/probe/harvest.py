"""Archive every program's native capture frames against a parameter sweep.

Probing costs hours of rig time and the device wedges partway through, so frames
are stored once with the vector that produced them and metrics are recomputed
offline. Every endpoint is injected, so the driver runs with no hardware.
"""

from __future__ import annotations

import time
from contextlib import ExitStack
from dataclasses import dataclass, field
from typing import Any, Callable, Iterable, Sequence

import numpy as np

from syncsummoner.device.profile import CROSSFADER_INDEX, ParamKind
from syncsummoner.device.session import Session
from syncsummoner.probe import codeframes, plans
from syncsummoner.probe.runner import raw_params
from syncsummoner.probe.store import KeyKind, ProgramKey

__all__ = [
    "HarvestConfig",
    "HarvestError",
    "HarvestReport",
    "ProgramResult",
    "harvest",
    "carries_stimulus",
    "harvest_program",
    "native_burst",
    "sweep_vectors",
    "upload_stimulus",
    "wait_healthy",
    "wedged",
]

#: Mean luma below which an archived program is reported as black rather than content.
DARK_LUMA = 20.0
#: Loaded to ask whether the device will still hold any program at all.
WEDGE_PROBE = "Passthru"


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
    frames_per_point: int = 30
    burst_settle: int = 4
    burst_timeout_s: float = 20.0
    settle_s: float = 0.35
    content_timeout_s: float = 10.0
    load_blackout_s: float = 4.5
    startup_s: float = 2.0
    health_timeout_s: float = 3600.0
    health_poll_s: float = 10.0
    wedge_settle_s: float = 2.0
    canary_frames: int = 8
    seed: int = 11
    loop_seed: int = 7


@dataclass(frozen=True)
class ProgramResult:
    """What one program cost and yielded; ``error`` is set when it did not run."""

    program: str
    frames: int = 0
    seconds: float = 0.0
    luma: float = 0.0
    error: str = ""

    @property
    def cached(self) -> bool:
        """True when the program was already archived under the same key."""
        return not self.frames and not self.error

    @property
    def dark(self) -> bool:
        """True when frames were archived but carry no picture."""
        return bool(self.frames) and self.luma < DARK_LUMA

    def __str__(self) -> str:
        if self.error:
            return f"{self.program}: FAILED {self.error}"
        if self.cached:
            return f"{self.program}: cached"
        dark = " DARK" if self.dark else ""
        return f"{self.program}: {self.frames} frames in {self.seconds:.0f}s Y={self.luma:.1f}{dark}"


@dataclass(frozen=True)
class HarvestReport:
    """Every program's outcome, and whether the device wedged before the end."""

    results: list[ProgramResult] = field(default_factory=list)
    wedged: bool = False
    blacked: bool = False
    seconds: float = 0.0

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


def native_burst(
    capture: Any,
    count: int,
    *,
    settle: int = 4,
    timeout_s: float = 20.0,
    clock: Callable[[], float] = time.monotonic,
) -> list[np.ndarray]:
    """Grab ``count`` native 4:2:2 frames, discarding a short settle run."""
    deadline = clock() + timeout_s
    got: list[np.ndarray] = []
    seen = 0
    while len(got) < count and clock() < deadline:
        frame = capture.read_native()
        if frame is None:
            continue
        seen += 1
        if seen > settle:
            got.append(frame)
    return got


def sweep_vectors(info: Any, config: HarvestConfig, rng: np.random.Generator) -> list[dict]:
    """Sobol vectors over every used parameter but the crossfader.

    The crossfader only gates output and a ``set_params`` offset is non-negative,
    so it cannot be closed; ``ProgramInfo.sweepable`` is not used because it also
    drops booleans, which do change the picture.
    """
    spec = [p for p in info.params if p.index != CROSSFADER_INDEX]
    if not any(p.kind is not ParamKind.UNUSED for p in spec):
        return [{}]
    return list(plans.sobol(spec, n=config.setpoints, rng=rng))


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
    all fail, so there is nothing to retry.
    """
    try:
        transport.load_program(program)
        sleep(settle_s)
        transport.program_info()
        return False
    except Exception:
        return True


def carries_stimulus(
    session: Any,
    capture: Any,
    transport: Any,
    *,
    config: HarvestConfig,
    program: str = WEDGE_PROBE,
    link: Any = None,
    player: Any = None,
    clock: Callable[[], float] = time.monotonic,
) -> bool:
    """Does a passthrough still carry the source? Proof the rig, not the program, is dark.

    One program going dark is legitimate, so only a failing passthrough shows the
    device has stopped emitting while still answering and reporting source lock.
    """
    session.load_program(program, park=True, link=link)
    if player is not None and not player.is_running():
        player.start(fps=config.loop_fps)
    session.set_params(session.working_point(transport.program_info()))
    capture.wait_for_content(timeout_s=config.content_timeout_s)
    burst = native_burst(capture, config.canary_frames, timeout_s=config.burst_timeout_s, clock=clock)
    return bool(burst) and float(np.mean([f[..., 0].mean() for f in burst])) >= DARK_LUMA


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


def upload_stimulus(player: Any, config: HarvestConfig, rng: np.random.Generator | None = None) -> int:
    """Push the codeframe loop into the player's tmpfs; returns bytes sent.

    The loop is invariant across programs and setpoints, so it is paid for once
    per run rather than once per pass.
    """
    generator = np.random.default_rng(config.loop_seed) if rng is None else rng
    loop = codeframes.build_loop(
        width=config.width, height=config.height, rng=generator, count=config.loop_frames
    )
    return player.upload(loop.frames())


def harvest_program(
    session: Any,
    capture: Any,
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
    """Sweep one program and archive the native frames each setpoint produced.

    The load drops the source link, and ``wait_for_content`` afterwards is what
    keeps the archive off the black frames a fresh load emits for seconds.
    """
    start = clock()
    session.load_program(program, park=True, link=link)
    if player is not None and not player.is_running():
        player.start(fps=config.loop_fps)
    info = transport.program_info()
    session.set_params(session.working_point(info))
    capture.wait_for_content(timeout_s=config.content_timeout_s)
    written, levels = 0, []
    with archive.writer(program, key, width=config.width, height=config.height) as out:
        for setpoint, vector in enumerate(sweep_vectors(info, config, rng)):
            session.set_params(vector)
            sleep(config.settle_s)
            burst = native_burst(
                capture,
                config.frames_per_point,
                settle=config.burst_settle,
                timeout_s=config.burst_timeout_s,
                clock=clock,
            )
            for frame in burst:
                out.write(frame, params=raw_params(vector), setpoint=setpoint)
                levels.append(float(frame[..., 0].mean()))
                written += 1
    return ProgramResult(program, written, clock() - start, float(np.mean(levels)) if levels else 0.0)


def harvest(
    archive: Any,
    *,
    open_transport: Callable[[], Any],
    open_capture: Callable[[], Any],
    player: Any = None,
    link: Any = None,
    programs: Iterable[str] | None = None,
    config: HarvestConfig | None = None,
    rng: np.random.Generator | None = None,
    session_factory: Callable[..., Any] = Session,
    log: Callable[[str], None] | None = None,
    sleep: Callable[[float], None] = time.sleep,
    clock: Callable[[], float] = time.monotonic,
) -> HarvestReport:
    """Archive every program, resuming what is stored and stopping when the rig wedges.

    Archives are keyed on name and firmware rather than on the program binary,
    because hashing a binary runs over the serial shell at wire speed and this
    device wedges under exactly that load.
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
    results, stopped, blacked, start = [], False, False, clock()
    try:
        with ExitStack() as stack:
            if player is not None:
                stack.enter_context(player.playing(fps=config.loop_fps))
                sleep(config.startup_s)
            capture = stack.enter_context(open_capture())
            for name in names:
                key = ProgramKey(name, firmware, KeyKind.NAME_FIRMWARE)
                if archive.has(name, key):
                    results.append(ProgramResult(name))
                    note(str(results[-1]))
                    continue
                try:
                    results.append(
                        harvest_program(
                            session,
                            capture,
                            archive,
                            transport,
                            name,
                            key,
                            config=config,
                            rng=rng,
                            link=link,
                            player=player,
                            sleep=sleep,
                            clock=clock,
                        )
                    )
                except Exception as exc:
                    results.append(ProgramResult(name, error=f"{type(exc).__name__}: {exc}"))
                    note(str(results[-1]))
                    if wedged(transport, settle_s=config.wedge_settle_s, sleep=sleep):
                        stopped = True
                        note("no program will load: the device needs a power cycle, then rerun to resume")
                        break
                    continue
                note(str(results[-1]))
                if not results[-1].dark:
                    continue
                for path in archive.paths(name):
                    path.unlink(missing_ok=True)
                results[-1] = ProgramResult(name, error="discarded: archived only black frames")
                if carries_stimulus(
                    session, capture, transport, config=config, link=link, player=player, clock=clock
                ):
                    note(f"{name} discarded: dark, but the rig still carries the source")
                    continue
                blacked = True
                note("black output from a live source: the device needs a power cycle, then rerun")
                break
    finally:
        transport.close()
    return HarvestReport(results=results, wedged=stopped, blacked=blacked, seconds=clock() - start)

"""A locked take whose every CC is sent for a known clip frame, with the record to prove it landed.

The program is loaded before playout and held dry for the lead-in, so the strip anchors the
take; toggling the dry control times the device's tick and the CC-to-picture latency. Every
write is then sent midway between the ticks around its frame's. See docs/evidence.md.
"""

from __future__ import annotations

import json
import subprocess
from contextlib import contextmanager
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Iterator, Mapping

import numpy as np

from syncsummoner.compose import evidence as ev
from syncsummoner.compose.lock import live_attribution, read_live, save_lock
from syncsummoner.device.profile import CROSSFADER_INDEX, PARAM_MAX, Role
from syncsummoner.device.wire import READ, Poller, Sender, save_wire
from syncsummoner.progress import LOG, meter

#: Settings key the device counts its boots in.
BOOT_COUNT_KEY = 9
#: Size the lead-in's pictures are compared at to see where a calibration write landed.
CALIBRATION_SIZE = (96, 54)
#: Plan rows: target clip frame, 1-based parameter, device value.
PLAN = np.dtype([("target", "<i8"), ("param", "<i8"), ("value", "<i8")])
#: Where a take directory keeps the latest CC-to-picture latency a dry take measured.
LATENCY_FILE = "cclatency.json"


class RebootError(RuntimeError):
    """The device's boot count rose across a take, so it rebooted while the take ran."""


class LeadInTimeError(RuntimeError):
    """The lead-in ended before the take's clock and latency were known."""


class NoLatencyError(LeadInTimeError):
    """A program with no dry control needs a CC-to-picture latency no dry take has measured yet."""


@dataclass(frozen=True)
class LeadIn:
    """How the lead-in is spent, in seconds from playout start and in calibration writes."""

    toggles: int = 8
    toggle_s: tuple[float, float] = (0.3, 0.5)
    settle_s: float = 1.0
    wet_s: float = 2.0
    poll_s: float = 0.005
    probe_s: float = 0.1
    seed: int = 0


def dry_state(info: Any) -> dict[int, int] | None:
    """The write that returns a program's input unprocessed: its bypass on, or a crossfader wet control off.

    None when the program names neither, so its lead-in cannot carry the strip.
    """
    for spec in info.params:
        if spec.role is Role.BYPASS:
            return {spec.index: PARAM_MAX}
    for spec in info.params:
        if spec.role is Role.WET and spec.index == CROSSFADER_INDEX:
            return {spec.index: 0}
    return None


def frame_plan(auto: Any, *, fps: float, early_bias_s: float) -> np.ndarray:
    """Each automation write's target frame: the first frame starting at or after its time less the bias."""
    times = np.asarray(auto.times, dtype=np.float64)
    target = np.ceil(np.round((times - early_bias_s) * fps, 6)).astype(np.int64)
    plan = np.zeros(times.size, dtype=PLAN)
    plan["target"], plan["param"], plan["value"] = np.maximum(target, 0), auto.indices, auto.values
    return plan[np.argsort(plan["target"], kind="stable")]


def initial_state(working: Mapping[int, Any], plan: np.ndarray) -> dict[int, int]:
    """The parameters at clip frame 0: the working point, then whatever frame 0's writes set."""
    state = {int(k): to_units(v) for k, v in working.items()}
    for row in plan[plan["target"] <= 0]:
        state[int(row["param"])] = int(row["value"])
    return state


def to_units(value: Any) -> int:
    """A working-point value in device units."""
    if isinstance(value, (bool, np.bool_)):
        return PARAM_MAX if value else 0
    if isinstance(value, (float, np.floating)):
        return int(np.clip(round(float(value) * PARAM_MAX), 0, PARAM_MAX))
    return int(np.clip(int(value), 0, PARAM_MAX))


def boot_count(shell: Any) -> int:
    """The device's boot count."""
    return int(json.loads(shell.get_setting(BOOT_COUNT_KEY))["value"])


def small_frames(
    path: str | Path, *, size: tuple[int, int], strip_px: int, ffmpeg: str = "ffmpeg"
) -> np.ndarray:
    """Every stored frame so far, strip cropped, area-scaled to ``size`` in planar YUV."""
    width, height = size
    argv = [ffmpeg, "-loglevel", "quiet", "-i", str(path), "-fps_mode", "passthrough", "-vf"]
    argv += [f"crop=iw:ih-{strip_px}:0:0,scale={width}:{height}:flags=area", "-f", "rawvideo"]
    argv += ["-pix_fmt", "yuv444p", "pipe:1"]
    data = subprocess.run(argv, check=False, capture_output=True).stdout
    stride = 3 * width * height
    count = len(data) // stride
    return np.frombuffer(data, np.uint8, count * stride).reshape(count, 3, height, width)


def frame_starts(size: int, runs: Any, per_frame: int) -> np.ndarray:
    """Whether each capture is the first of its frame, by its run's cadence."""
    out = np.zeros(size, dtype=bool)
    for run in runs:
        k = np.arange(run.first, min(run.last + 1, size))
        out[k] = (run.slot + k - run.first) % per_frame == 0
    return out


@dataclass
class Clock:
    """The take's clock as measured so far: frame times, the device's tick, and the tick-to-capture delay."""

    times: ev.FrameTimes
    ticks: ev.Ticks
    delta: float
    run: Any = None
    per_frame: int = 2
    shown: np.ndarray = field(default_factory=lambda: np.zeros(0, dtype=np.int64))

    def when(self, target: int) -> int:
        """Host time to send a write that should land on ``target``."""
        return int(ev.send_time(np.array([target]), self.ticks, self.times, self.delta)[0])

    def extend(self, t_ns: np.ndarray, sequence: np.ndarray) -> None:
        """Carry the frame clock on through captures the anchored run has gained since."""
        run = self.run
        if run is None:
            return
        tail = np.flatnonzero(np.diff(sequence[run.last :]) != 1)
        last = run.last + (int(tail[0]) if tail.size else t_ns.size - 1 - run.last)
        k = np.arange(run.first, last + 1)
        start = k[(run.slot + k - run.first) % self.per_frame == 0]
        frame = (run.slot + start - run.first) // self.per_frame - self.times.lead
        recent = start[-min(start.size, 600) :], frame[-min(start.size, 600) :]
        if recent[0].size > 2:
            slope, icept = np.polyfit(recent[1], t_ns[recent[0]].astype(np.float64), 1)
            self.times = ev.FrameTimes(self.times.ns, self.times.lead, float(icept), float(slope))
            self.ticks = ev.Ticks(self.ticks.lo, self.ticks.hi, float(slope), self.ticks.count)


@dataclass
class Record:
    """What a take put on the wire and read back, with the clock it was sent by."""

    plan: np.ndarray
    initial: np.ndarray
    sends: np.ndarray
    arrays: dict
    delta: float
    dry: dict | None
    loaded: dict
    cc_map: np.ndarray


def measure_latency(
    images: np.ndarray,
    frames: np.ndarray,
    found: Any,
    toggles: np.ndarray,
    moves: np.ndarray,
    times: ev.FrameTimes,
    ticks: ev.Ticks,
    until: float | None = None,
) -> ev.Latency:
    """The tick-to-capture delay from where each calibration write showed in the lead-in's picture."""
    size = min(images.shape[0], frames.size, found.shown.size)
    starts = frame_starts(size, found.runs, found.per_frame)
    landed = ev.toggle_landings(
        images[:size], frames["t_ns"][:size], starts, found.shown[:size], toggles, until
    )
    n = ev.observed_ticks(toggles, moves, ticks)
    return ev.latency(np.where((landed >= -times.lead) & (n >= 0), landed, ev.CALIBRATION), n, ticks, times)


def write_latency(directory: str | Path, measured: ev.Latency, program: str) -> None:
    """Add a dry take's measured latency to the ones a take without a dry control draws on."""
    path = Path(directory, LATENCY_FILE)
    takes = json.loads(path.read_text(encoding="utf-8")).get("takes", []) if path.is_file() else []
    takes.append({"program": program, **measured.as_dict()})
    path.write_text(json.dumps({"takes": takes}, indent=1), encoding="utf-8")


def read_latency(directory: str | Path) -> float:
    """The tick-to-capture delay for a take whose card phase no picture measured, in ns.

    The delay is a constant plus the card's capture phase, which two captures a frame keep
    within half a frame; the midpoint of the dry takes' extremes is the value whose worst
    error is least. Takes spread by half a frame or more contradict that.
    """
    path = Path(directory, LATENCY_FILE)
    if not path.is_file():
        raise NoLatencyError(
            f"no {LATENCY_FILE} in {directory}: a program without a dry control needs a dry take's "
            "measured CC-to-picture latency; render a program with a bypass first"
        )
    takes = json.loads(path.read_text(encoding="utf-8"))["takes"]
    deltas = np.array([t["delta"] for t in takes], dtype=np.float64)
    half = 0.5 * float(np.median([t["period"] for t in takes]))
    if np.ptp(deltas) >= half:
        raise LeadInTimeError(
            f"dry takes in {path} put the tick-to-capture delay at {np.round(deltas / 1e6, 2).tolist()} ms, "
            "half a frame or more apart, so no card phase explains them"
        )
    centre = 0.5 * float(deltas.min() + deltas.max())
    LOG.info(
        "tick-to-capture delay %.2f ms from %d dry takes, wrong by at most %.2f ms",
        centre / 1e6,
        deltas.size,
        (half - 0.5 * float(np.ptp(deltas))) / 1e6,
    )
    return centre


def _sleep_until(
    t_ns: int, *, clock: Callable[[], int] = time.monotonic_ns, sleep: Callable[[float], None] = time.sleep
) -> None:
    while (left := t_ns - clock()) > 0:
        sleep(min(left / 1e9, 0.5))


def calibrate(sender: Sender, dry: Mapping[int, int], initial: Mapping[int, int], lead_in: LeadIn) -> None:
    """Toggle the dry control between dry and the take's wet value at random intervals, ending dry."""
    rng = np.random.default_rng(lead_in.seed)
    ((slot, dry_value),) = dry.items()
    wet = int(initial.get(slot, 0))
    for index in range(2 * lead_in.toggles):
        time.sleep(float(rng.uniform(*lead_in.toggle_s)))
        sender.send(slot, wet if index % 2 == 0 else dry_value, int(ev.CALIBRATION))


def probe_ticks(sender: Sender, initial: Mapping[int, int], lead_in: LeadIn) -> None:
    """Step every slot off its initial value and back, so the readback shows the device's tick."""
    for slot, value in sorted(initial.items()):
        sender.send(slot, value + 1 if value < PARAM_MAX else value - 1, int(ev.CALIBRATION))
        time.sleep(lead_in.probe_s)
        sender.send(slot, value, int(ev.CALIBRATION))
        time.sleep(lead_in.probe_s)


def locked_take(
    rig: Any,
    recorder: Any,
    out: str | Path,
    *,
    program: str,
    seconds: float,
    config: Any,
    plan: np.ndarray,
    working: Callable[[Any], Mapping[int, Any]],
    load: Callable[[str], Any],
    lead_in: LeadIn = LeadIn(),
) -> Record:
    """Record ``program`` playing the clip with every planned write sent for its frame.

    ``load(name)`` loads a program and checks the rig, returning its program info, and
    ``working(info)`` gives its working point. A rise in the boot count is a :class:`RebootError`.
    """
    shell = rig.transport.device.shell
    midi = rig.transport.device.midi
    fps = float(config.fps)
    lead, count = int(round(config.lead_in_s * fps)), int(round(seconds * fps))
    booted = boot_count(shell)
    info = load(program)
    dry = dry_state(info)
    initial = initial_state(working(info), plan)
    park = rig.session.park_values

    def write(param: int, value: int) -> int:
        offset = int(np.clip(value - park[param - 1], 0, PARAM_MAX))
        rig.transport.set_param(param, offset)
        return int(midi.get_param(param)) + int(park[param - 1])

    if dry is None:
        read_latency(Path(out).parent)
        load("Passthru")
    start_state = {**initial, **(dry or {})}
    start_vector = [start_state.get(i, int(park[i - 1])) for i in range(1, 13)]
    sender, poller = Sender(write), Poller(shell, interval_s=lead_in.poll_s)
    held: dict[str, Any] = {"rows": plan, "delta": float("nan"), "loaded": None}
    try:
        with (
            recorder.recording(out),
            rig.playout.playing(fps=fps) as player,
            _monitored(shell, poller, sender),
        ):
            if dry is None:
                load(program)
            _play(
                rig,
                out,
                sender=sender,
                poller=poller,
                shell=shell,
                player=player,
                held=held,
                program=program,
                config=config,
                lead=lead,
                count=count,
                dry=dry,
                initial=initial,
                start_state=start_state,
                lead_in=lead_in,
            )
            hold = int(player.status().get("hold", 1))
        save_lock(out, rig.playout.flips(), hold=hold, lead=lead, count=count)
    finally:
        asked = held["loaded"]
        save_wire(
            out,
            plan=held["rows"],
            initial=np.asarray(park, dtype=np.int64),
            start=np.array(start_vector, dtype=np.int64),
            sends=sender.array(),
            delta=np.float64(held["delta"]),
            cc_map=np.asarray(midi.cc_map, dtype=np.int64),
            loaded=np.array(
                json.dumps(
                    {}
                    if asked is None
                    else {
                        "fpga": asked.payload,
                        "error": asked.error,
                        "t0_ns": asked.t0_ns,
                        "t1_ns": asked.t1_ns,
                    }
                )
            ),
            dry=np.array(json.dumps(dry)),
            program=np.array(program),
            names=np.array(json.dumps({int(spec.index): spec.name for spec in info.params})),
            **poller.arrays(),
        )
        rebooted = boot_count(shell)
        if rebooted != booted:
            raise RebootError(
                f"{program}: the device's boot count went from {booted} to {rebooted} during the take"
            )
    return Record(
        plan=held["rows"],
        initial=np.asarray(park, dtype=np.int64),
        sends=sender.array(),
        arrays=poller.arrays(),
        delta=held["delta"],
        dry=dry,
        loaded={} if held["loaded"] is None else {"fpga": held["loaded"].payload},
        cc_map=np.asarray(midi.cc_map, dtype=np.int64),
    )


def _play(
    rig: Any,
    out: str | Path,
    *,
    sender: Sender,
    poller: Poller,
    shell: Any,
    player: Any,
    held: dict,
    program: str,
    config: Any,
    lead: int,
    count: int,
    dry: Mapping[int, int] | None,
    initial: Mapping[int, int],
    start_state: Mapping[int, int],
    lead_in: LeadIn,
) -> None:
    """The take from playout start: lead-in writes, calibration, the live clock, then every planned write."""
    began = time.monotonic_ns()
    fps = float(config.fps)
    shell.command("midi", "monitor", "on")
    del shell.log_lines[:]
    poller.start()
    for slot, value in sorted(start_state.items()):
        sender.send(slot, value, ev.START)
    time.sleep(lead_in.settle_s)
    if dry is not None:
        calibrate(sender, dry, initial, lead_in)
    else:
        probe_ticks(sender, initial, lead_in)
    hold = int(player.status().get("hold", 1))
    ffmpeg = getattr(config, "ffmpeg", "ffmpeg")
    codes, images = read_live(
        out, strip_px=config.strip_px, bits=config.bits, size=CALIBRATION_SIZE, ffmpeg=ffmpeg
    )
    found, frames = live_attribution(
        out, rig.playout.flips(), codes, config=config, hold=hold, lead=lead, count=count
    )
    times = ev.frame_times(frames["t_ns"], found.shown, found.runs, found.per_frame, lead)
    moves = ev.transitions(np.array([(a, b, tuple(o)) for a, b, o in list(poller.reads)], dtype=READ))
    ticks = ev.tick_bracket(moves["a"], moves["b"], times.period)
    calibration = sender.array()
    calibration = calibration[calibration["target"] == ev.CALIBRATION]
    if dry is not None:
        measured = measure_latency(images, frames, found, calibration, moves, times, ticks)
        held["delta"] = measured.delta
        write_latency(Path(out).parent, measured, program)
        LOG.info(
            "%s: CC-to-picture %.3f frames tick to capture over %d writes",
            program,
            measured.frames,
            measured.samples,
        )
    else:
        held["delta"] = read_latency(Path(out).parent)
    clock = Clock(
        times=times, ticks=ticks, delta=held["delta"], run=found.runs[-1], per_frame=found.per_frame
    )
    rows = plan_rows(held["rows"], dry, initial, wet_frame=-int(round(lead_in.wet_s * fps)))
    held["rows"] = rows
    first_due = clock.when(int(rows["target"][0])) if rows.size else clock.when(0)
    if time.monotonic_ns() >= first_due:
        raise LeadInTimeError(
            f"{program}: the clock was measured {(time.monotonic_ns() - first_due) / 1e9:.2f}s after the "
            "first write was due; lengthen the lead-in"
        )
    sender.when = clock.when
    sender.plan([(int(r["target"]), int(r["param"]), int(r["value"])) for r in rows])
    sender.start()
    _sleep_until(clock.when(-1) - int(clock.ticks.period))
    held["loaded"] = poller.ask("fpga", "status", timeout_s=5.0)
    end = int(clock.times.at(count)) + int(config.tail_s * 1e9)
    with meter((end - began) / 1e9, desc="take") as progress:
        while (now := time.monotonic_ns()) < end:
            time.sleep(min(1.0, (end - now) / 1e9))
            log = _read_sidecar(out)
            clock.extend(log["t_ns"], log["sequence"])
            progress.to((time.monotonic_ns() - began) / 1e9)
    sender.drain(timeout_s=1.0)


def plan_rows(
    plan: np.ndarray, dry: Mapping[int, int] | None, initial: Mapping[int, int], *, wet_frame: int
) -> np.ndarray:
    """The take's writes: the dry control back to its take value at ``wet_frame``, then the plan."""
    if dry is None:
        return plan
    ((slot, _),) = dry.items()
    return np.r_[np.array([(wet_frame, slot, initial.get(slot, 0))], dtype=PLAN), plan]


@contextmanager
def _monitored(shell: Any, poller: Poller, sender: Sender) -> Iterator[None]:
    """Stop the sender and poller and turn the monitor off however the take ends."""
    try:
        yield
    finally:
        sender.stop()
        poller.stop()
        try:
            shell.command("midi", "monitor", "off")
        except Exception as err:  # pylint: disable=broad-exception-caught
            LOG.warning("midi monitor left on: %s", err)


def _read_sidecar(take: str | Path) -> np.ndarray:
    from syncsummoner.device.v4l2rec import read_log  # pylint: disable=import-outside-toplevel

    return read_log(take)


def state_at(initial: np.ndarray, plan: np.ndarray, frame: int) -> np.ndarray:
    """The parameter vector the plan holds from ``frame`` on, starting from ``initial``."""
    state = np.asarray(initial, dtype=np.int64).copy()
    for row in plan[plan["target"] <= frame]:
        state[int(row["param"]) - 1] = int(row["value"])
    return state


def loaded_before(
    program: str, wire: Mapping[str, Any], ticks: ev.Ticks, times: ev.FrameTimes, delta: float
) -> dict:
    """Whether the device ran ``program`` and read back frame 0's parameters before frame 0's tick."""
    said = json.loads(str(wire["loaded"]))
    try:
        fpga = json.loads(said.get("fpga") or "{}")
    except json.JSONDecodeError:
        fpga = {}
    running = (
        fpga.get("state") == "running"
        and str(fpga.get("program", "")).casefold() == program.rpartition("/")[2].casefold()
    )
    want = state_at(wire["start"], wire["plan"], 0)
    reads = wire["reads"]
    tick0 = np.rint((times.at(0) - delta - ticks.phase) / ticks.period)
    cutoff = ticks.lo + tick0 * ticks.period
    before = np.flatnonzero(reads["t1_ns"] < cutoff)
    last = reads["o"][before[-1]] if before.size else None
    agree = last is not None and bool(np.array_equal(last, want))
    early = said.get("t1_ns", 0) < cutoff
    summary = (
        f"fpga {'runs' if running else 'does not run'} {program} "
        f"({'before' if early else 'after'} frame 0's tick); last readback before it "
        + ("matches" if agree else f"{None if last is None else last.tolist()} against {want.tolist()}")
    )
    return {"ok": bool(running and agree and early), "summary": summary, "want": want.tolist()}


def take_evidence(
    take: str | Path, *, program: str, clip: str | Path, config: Any, names: Mapping[int, str] | None = None
) -> ev.Evidence:
    """The three links for a recorded take, from its wire record, its attribution and its picture."""
    from syncsummoner.compose.lock import attribute_take  # pylint: disable=import-outside-toplevel
    from syncsummoner.device.v4l2rec import read_log  # pylint: disable=import-outside-toplevel
    from syncsummoner.device.wire import load_wire  # pylint: disable=import-outside-toplevel

    wire = load_wire(take)
    if names is None:
        names = {int(k): v for k, v in json.loads(str(wire["names"])).items()} if "names" in wire else {}
    found = attribute_take(take, config=config)
    frames = read_log(take)
    times = ev.frame_times(frames["t_ns"], found.shown, found.runs, found.per_frame, found.lead)
    moves = ev.transitions(wire["reads"])
    ticks = ev.tick_bracket(moves["a"], moves["b"], times.period)
    sends = wire["sends"]
    calibration = sends[sends["target"] == ev.CALIBRATION]
    ffmpeg = getattr(config, "ffmpeg", "ffmpeg")
    images = small_frames(take, size=CALIBRATION_SIZE, strip_px=config.strip_px, ffmpeg=ffmpeg)
    delta = float(wire["delta"])
    measured = {"delta": delta, "frames": delta / ticks.period, "spread": float("nan"), "samples": 0}
    if json.loads(str(wire["dry"])) is not None and calibration.size:
        later = sends["t0_ns"][(sends["target"] > ev.START) & (sends["t0_ns"] > calibration["t1_ns"].max())]
        until = float(later.min()) if later.size else None
        lat = measure_latency(images, frames, found, calibration, moves, times, ticks, until)
        delta, measured = lat.delta, lat.as_dict()
    plan = wire["plan"]
    seen = ev.observed_ticks(sends, moves, ticks)
    landed_rows = ev.landed(plan, sends, ticks, times, delta, seen)
    applied_rows = ev.applied(wire["reads"], sends, wire["initial"], ticks, times, delta, seen)
    receipts = wire.get("receipts")
    got = None
    if receipts is not None and receipts.size:
        got = ev.received(ev.expected_messages(sends, wire["cc_map"]), receipts)
    count = found.picked.size
    out = images[found.picked].astype(np.float32)
    src = small_frames(clip, size=CALIBRATION_SIZE, strip_px=config.strip_px, ffmpeg=ffmpeg)
    src = src[found.lead : found.lead + count].astype(np.float32)
    luma_range = np.ptp(out[:, 0].reshape(count, -1), axis=1)
    rows = landed_rows.rows
    sent = rows["sent"] >= 0
    moved = ev.acted_upon(out, src, luma_range, rows["landed"][sent], rows["param"][sent], names)
    return ev.Evidence(
        program=program,
        latency=measured,
        ticks={
            "width_ms": ticks.width / 1e6,
            "count": ticks.count,
            "period_ns": ticks.period,
            "deadline_ms": [v / 1e6 for v in ev.deadline(sends, seen, ticks)],
        },
        landed=landed_rows,
        applied=applied_rows,
        received=got,
        moved=moved,
        loaded=loaded_before(program, wire, ticks, times, delta),
    )


def evidence_path(take: str | Path) -> Path:
    """Where a take's evidence report sits."""
    return Path(f"{take}.evidence.json")

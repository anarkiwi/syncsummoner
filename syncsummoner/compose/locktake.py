"""A locked take whose every CC is sent for a known clip frame, with the record to prove it landed.

The program is loaded before playout and held dry for the lead-in, so the strip anchors the
take; toggling the dry control times the device's tick and the CC-to-picture latency. Every
write is then sent midway between the ticks around its frame's. See docs/evidence.md.
"""

from __future__ import annotations

import json
import subprocess
import sys
from contextlib import contextmanager
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Iterator, Mapping

import numpy as np

from syncsummoner.compose import evidence as ev
from syncsummoner.compose.lock import live_attribution, read_live, save_lock
from syncsummoner.device.profile import CROSSFADER_INDEX, PARAM_MAX, Role
from syncsummoner.device.wire import Poller, Sender, save_wire
from syncsummoner.progress import LOG, meter

#: Settings key the device counts its boots in.
BOOT_COUNT_KEY = 9
#: Size the lead-in's pictures are compared at to see where a calibration write landed.
CALIBRATION_SIZE = (96, 54)
#: Plan rows: target clip frame, 1-based parameter, device value.
PLAN = np.dtype([("target", "<i8"), ("param", "<i8"), ("value", "<i8")])
#: Interpreter thread switch interval during a take, well inside the tick a late wake-up must make.
SWITCH_S = 0.0005
#: Where a take directory keeps the latest CC-to-picture latency a dry take measured.
LATENCY_FILE = "ccpicture.json"


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
    """The take's clock as measured so far: frame times and the send-to-capture latency."""

    times: ev.FrameTimes
    lat: ev.Latency
    run: Any = None
    per_frame: int = 2

    @property
    def ticks(self) -> ev.Ticks:
        """The instants a write must precede to land on each frame."""
        return ev.grid(self.times, self.lat)

    def when(self, target: int) -> int:
        """Host time to send a write that should land on ``target``."""
        return int(ev.send_time(np.array([target]), self.ticks, self.times, self.lat.delta)[0])

    def extend(self, t_ns: np.ndarray, sequence: np.ndarray) -> None:
        """Carry the frame line on over every frame start the anchored run has gained."""
        run = self.run
        if run is None:
            return
        tail = np.flatnonzero(np.diff(sequence[run.last :]) != 1)
        last = run.last + (int(tail[0]) if tail.size else t_ns.size - 1 - run.last)
        k = np.arange(run.first, last + 1)
        start = k[(run.slot + k - run.first) % self.per_frame == 0]
        frame = (run.slot + start - run.first) // self.per_frame - self.times.lead
        if start.size > 2:
            slope, icept = np.polyfit(frame, t_ns[start].astype(np.float64), 1)
            self.times = ev.FrameTimes(self.times.ns, self.times.lead, float(icept), float(slope))


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
    times: ev.FrameTimes,
    until: float | None = None,
) -> ev.Latency:
    """The send-to-capture latency from where each calibration write showed in the lead-in's picture."""
    size = min(images.shape[0], frames.size, found.shown.size)
    starts = frame_starts(size, found.runs, found.per_frame)
    landed = ev.toggle_landings(
        images[:size], frames["t_ns"][:size], starts, found.shown[:size], toggles, until
    )
    return ev.latency(toggles["t1_ns"], landed, times)


def write_latency(directory: str | Path, measured: ev.Latency, program: str) -> None:
    """Add a dry take's measured latency to the ones a take without a dry control draws on."""
    path = Path(directory, LATENCY_FILE)
    takes = json.loads(path.read_text(encoding="utf-8")).get("takes", []) if path.is_file() else []
    takes.append({"program": program, **measured.as_dict()})
    path.write_text(json.dumps({"takes": takes}, indent=1), encoding="utf-8")


def read_latency(directory: str | Path) -> ev.Latency:
    """The send-to-capture latency for a take whose card phase no picture measured.

    It is a constant plus the card's capture phase, which two captures a frame keep within half a
    frame; the band runs from the dry takes' least bound less the unseen part of that half frame
    to their greatest plus it. A band of half a frame or more cannot name a landing frame.
    """
    path = Path(directory, LATENCY_FILE)
    if not path.is_file():
        raise NoLatencyError(
            f"no {LATENCY_FILE} in {directory}: a program without a dry control needs a dry take's "
            "measured CC-to-picture latency; render a program with a bypass first"
        )
    takes = json.loads(path.read_text(encoding="utf-8"))["takes"]
    lo = min(t["lo"] for t in takes)
    hi = max(t["hi"] for t in takes)
    period = float(np.median([t["period"] for t in takes]))
    if hi - lo >= period / 2:
        raise LeadInTimeError(
            f"dry takes in {path} put the CC-to-picture latency over {lo / 1e6:.2f}..{hi / 1e6:.2f} ms, "
            "half a frame or more, too wide to say which frame a write lands on"
        )
    unseen = 0.5 * (period / 2 - (hi - lo))
    found = ev.Latency(lo=lo - unseen, hi=hi + unseen, samples=0, period=period)
    LOG.info(
        "CC-to-picture %.2f ms from %d dry takes, within %.2f ms",
        found.delta / 1e6,
        len(takes),
        found.spread / 1e6,
    )
    return found


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
            latency=np.array(json.dumps(held.get("latency", {}))),
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
    switch = sys.getswitchinterval()
    sys.setswitchinterval(SWITCH_S)
    try:
        _run(
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
            began=began,
            fps=fps,
        )
    finally:
        sys.setswitchinterval(switch)


def _run(
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
    began: int,
    fps: float,
) -> None:
    """The take's body, run with the interpreter switching threads often enough for the sender."""
    shell.command("midi", "monitor", "on")
    del shell.log_lines[:]
    poller.start()
    for slot, value in sorted(start_state.items()):
        sender.send(slot, value, ev.START)
    time.sleep(lead_in.settle_s)
    if dry is not None:
        calibrate(sender, dry, initial, lead_in)
    hold = int(player.status().get("hold", 1))
    ffmpeg = getattr(config, "ffmpeg", "ffmpeg")
    codes, images = read_live(
        out, strip_px=config.strip_px, bits=config.bits, size=CALIBRATION_SIZE, ffmpeg=ffmpeg
    )
    found, frames = live_attribution(
        out, rig.playout.flips(), codes, config=config, hold=hold, lead=lead, count=count
    )
    times = ev.frame_times(frames["t_ns"], found.shown, found.runs, found.per_frame, lead)
    calibration = sender.array()
    calibration = calibration[calibration["target"] == ev.CALIBRATION]
    if dry is not None:
        lat = measure_latency(images, frames, found, calibration, times)
        write_latency(Path(out).parent, lat, program)
        LOG.info("%s: CC-to-picture %.3f frames over %d writes", program, lat.frames, lat.samples)
    else:
        lat = read_latency(Path(out).parent)
    held["delta"], held["latency"] = lat.delta, lat.as_dict()
    clock = Clock(times=times, lat=lat, run=found.runs[-1], per_frame=found.per_frame)
    rows = plan_rows(held["rows"], dry, initial, wet_frame=-int(round(lead_in.wet_s * fps)))
    held["rows"] = rows
    first_due = clock.when(int(rows["target"][0])) if rows.size else clock.when(0)
    if time.monotonic_ns() >= first_due:
        raise LeadInTimeError(
            f"{program}: the clock was measured {(time.monotonic_ns() - first_due) / 1e9:.2f}s after the "
            "first write was due; lengthen the lead-in"
        )
    log = _read_sidecar(out)
    clock.extend(log["t_ns"], log["sequence"])
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
    sends = wire["sends"]
    calibration = sends[sends["target"] == ev.CALIBRATION]
    ffmpeg = getattr(config, "ffmpeg", "ffmpeg")
    images = small_frames(take, size=CALIBRATION_SIZE, strip_px=config.strip_px, ffmpeg=ffmpeg)
    used = json.loads(str(wire["latency"])) if "latency" in wire else {}
    if "lo" in used:
        lat = ev.Latency(lo=used["lo"], hi=used["hi"], samples=used.get("samples", 0), period=times.period)
    else:
        lat = read_latency(Path(take).parent) if Path(take).parent.joinpath(LATENCY_FILE).is_file() else None
    if json.loads(str(wire["dry"])) is not None and calibration.size:
        later = sends["t0_ns"][(sends["target"] > ev.START) & (sends["t0_ns"] > calibration["t1_ns"].max())]
        until = float(later.min()) if later.size else None
        try:
            lat = measure_latency(images, frames, found, calibration, times, until)
        except ev.EvidenceError as err:
            if lat is None:
                raise
            LOG.warning("%s: %s; using the band measured live in the lead-in", take, err)
    if lat is None:
        raise ev.EvidenceError(
            f"{take}: no CC-to-picture latency, measured or on file, to land its writes by"
        )
    ticks, delta = ev.grid(times, lat), lat.delta
    plan = wire["plan"]
    landed_rows = ev.landed(plan, sends, ticks, times, delta)
    applied_rows = ev.applied(wire["reads"], sends, wire["initial"], ticks, times, delta)
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
        latency=lat.as_dict(),
        ticks={"width_ms": ticks.width / 1e6, "count": ticks.count, "period_ns": ticks.period},
        landed=landed_rows,
        applied=applied_rows,
        received=got,
        moved=moved,
        loaded=loaded_before(program, wire, ticks, times, delta),
    )


def add_schedule(directory: str | Path, entries: list[dict]) -> Path:
    """Merge a take's undetermined parameters into the directory's ``schedule.json``, by program."""
    path = Path(directory, "schedule.json")
    kept = json.loads(path.read_text(encoding="utf-8")) if path.is_file() else []
    programs = {e["program"] for e in entries}
    path.write_text(
        json.dumps([e for e in kept if e["program"] not in programs] + entries, indent=1), encoding="utf-8"
    )
    return path


def evidence_path(take: str | Path) -> Path:
    """Where a take's evidence report sits."""
    return Path(f"{take}.evidence.json")

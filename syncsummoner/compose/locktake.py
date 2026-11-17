"""A locked take whose every CC is sent for a known clip frame, with the record to prove it landed.

The program is loaded before playout and held dry for the lead-in, so the strip anchors the
take; toggling the dry control times the device's tick and the CC-to-picture latency. Every
write is then sent midway between the ticks around its frame's. See docs/evidence.md.
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, Callable, Mapping

import numpy as np

from syncsummoner.compose import evidence as ev
from syncsummoner.compose import frameline
from syncsummoner.compose.lock import capture_times, changes
from syncsummoner.device.profile import PARAM_MAX
from syncsummoner.device import wire as W
from syncsummoner.device.wire import Sender
from syncsummoner.probe import decoded
from syncsummoner.progress import LOG

#: Settings key the device counts its boots in.
BOOT_COUNT_KEY = 9
#: Plan rows: target clip frame, 1-based parameter, device value.
PLAN = np.dtype([("target", "<i8"), ("param", "<i8"), ("value", "<i8")])
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


def program_bypass(info: Any) -> Any:
    """The control setting the program's HDL shows returns its input with every other control dead.

    A :class:`~syncsummoner.program.model.Bypass`; None when the program has no declaration, no such
    setting, or the model cannot be built.
    """
    from syncsummoner.program import model  # pylint: disable=import-outside-toplevel

    declared = getattr(info, "declared", None)
    if declared is None:
        return None
    try:
        return model(declared.program_id).bypass()
    except (KeyError, RuntimeError, OSError) as err:
        LOG.warning("%s: no bypass from the program model: %s", info.name, err)
        return None


def declared_bypass(program: str) -> Any:
    """:func:`program_bypass` from the program's declaration, before it is loaded."""
    from syncsummoner.device.declared import (  # pylint: disable=import-outside-toplevel
        UndeclaredProgramError,
        declaration,
    )

    try:
        declared = declaration(program)
    except UndeclaredProgramError as err:
        LOG.warning("%s: no bypass from a declaration: %s", program, err)
        return None
    return program_bypass(type("Declared", (), {"name": program, "declared": declared})())


def dry_state(subject: Any, bypass: Callable[[Any], Any] = program_bypass) -> dict[int, int] | None:
    """The write that returns a program's input unprocessed: its bypass control set to its bypass value.

    ``bypass(subject)`` finds it; None when there is none, so the lead-in cannot carry the strip.
    """
    pick = bypass(subject)
    return None if pick is None else {pick.parameter_id: pick.value}


def frame_plan(auto: Any, *, fps: float, early_bias_s: float) -> np.ndarray:
    """Each automation write's target frame: the first frame starting at or after its time less the bias.

    The device applies a slot's value once a frame, so only the last write per slot and frame is kept,
    and one that repeats the slot's previous value is dropped.
    """
    times = np.asarray(auto.times, dtype=np.float64)
    target = np.ceil(np.round((times - early_bias_s) * fps, 6)).astype(np.int64)
    plan = np.zeros(times.size, dtype=PLAN)
    plan["target"], plan["param"], plan["value"] = np.maximum(target, 0), auto.indices, auto.values
    return schedule(plan[np.argsort(plan["target"], kind="stable")])


def schedule(plan: np.ndarray, state: np.ndarray | None = None) -> np.ndarray:
    """One write per slot and frame, the last asked for, none repeating the value its slot holds.

    ``plan`` is in send order; ``state`` is the 12 slot values before it, unknown when not given.
    The result is in target order, slots in their send order within a frame.
    """
    if not plan.size:
        return plan
    order = np.lexsort((np.arange(plan.size), plan["target"], plan["param"]))
    rows = plan[order]
    same_slot = rows["param"][1:] == rows["param"][:-1]
    last = np.r_[~(same_slot & (rows["target"][1:] == rows["target"][:-1])), True]
    rows, order = rows[last], order[last]
    first = np.r_[True, rows["param"][1:] != rows["param"][:-1]]
    held = np.r_[-1, rows["value"][:-1]]
    if state is not None:
        held[first] = np.asarray(state, dtype=np.int64)[rows["param"][first] - 1]
    else:
        held[first] = -1
    keep = rows["value"] != held
    rows, order = rows[keep], order[keep]
    return rows[np.lexsort((order, rows["target"]))]


def wet_at(lead_in: LeadIn, fps: float) -> int:
    """The lead-in frame a take's dry control returns to its take value on; later frames are wet."""
    return -int(round(lead_in.wet_s * fps))


def frame_load(rows: np.ndarray, per_write: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Each frame that carries writes, and the messages it carries."""
    if not rows.size:
        return np.zeros(0, dtype=np.int64), np.zeros(0, dtype=np.int64)
    frames, inverse = np.unique(rows["target"], return_inverse=True)
    return frames, np.bincount(inverse, weights=np.asarray(per_write)[rows["param"] - 1]).astype(np.int64)


def describe_load(auto: Any, plan: np.ndarray, rows: np.ndarray, per_write: np.ndarray) -> str:
    """Automation writes, frame-plan rows, the rows a take sends, and the messages its frames carry."""
    frames, messages = frame_load(rows, per_write)
    said = f"{len(auto)} automation writes, {plan.size} frame-plan rows, {rows.size} sent; "
    if not messages.size:
        return said + "nothing to send"
    busiest = int(np.argmax(messages))
    return said + (
        f"{frames.size} frames carry writes, {int(messages.sum())} messages, "
        f"busiest frame {int(frames[busiest])} with {int(messages[busiest])}, "
        f"median {float(np.median(messages)):g} a frame"
    )


def clip_plan(auto: Any, seconds: float, config: Any) -> np.ndarray:
    """A layer's automation as frame-targeted writes inside a clip of ``seconds``."""
    plan = frame_plan(auto, fps=config.fps, early_bias_s=config.early_bias_s)
    return plan[plan["target"] < int(round(seconds * config.fps))]


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


@dataclass
class Clock:
    """The take's clock as measured so far: the frame line, and the send-to-frame-start latency.

    ``first`` is the first capture of the run the clip plays in and ``seen`` the captures its line
    has looked at; the line is carried on as the card's log grows (:meth:`extend`).
    """

    times: ev.FrameTimes
    lat: ev.Latency
    line: Any = None
    first: int = 0
    seen: int = 0
    jitter: float = 0.0
    resolution: float = 0.0

    @classmethod
    def of(cls, found: Any, lat: ev.Latency) -> "Clock":
        """The clock an attribution of the take so far gives."""
        return cls(
            times=ev.frame_times(found),
            lat=lat,
            line=found.line,
            first=int(found.span[0]),
            seen=int(found.span[1]) + 1,
            jitter=float(found.jitter_ns),
            resolution=float(found.resolution),
        )

    @property
    def ticks(self) -> ev.Ticks:
        """The instants a write must precede to land on each frame."""
        return ev.grid(self.times, self.lat)

    def when(self, target: int) -> int:
        """Host time to send a write that should land on ``target``."""
        return int(ev.send_time(np.array([target]), self.ticks, self.times, self.lat.delta)[0])

    def extend(self, log: np.ndarray) -> None:
        """Carry the frame line on over the frame starts the clip's run has gained, by its bytes.

        A capture's bytes are only known to start a frame once the two after it are logged.
        """
        if self.line is None:
            return
        part = log[self.first :]
        run = part[frameline.segments(part["t_ns"], part["sequence"]) == 0]
        done = self.first + run.size - 2
        if done <= self.seen:
            return
        seg = np.zeros(run.size, dtype=np.int64)
        capture, _ = frameline.capture_line(run["t_ns"], run["sequence"], seg)
        tau = capture_times(run["t_ns"], run["sequence"], seg, capture)
        k = np.flatnonzero(changes(run["digest"]))
        k = k[(k + self.first >= self.seen) & (k + self.first < done)]
        self.line = frameline.extend(self.line, tau[k - 1], tau[k], self.jitter, self.resolution)
        self.seen = done
        lead = self.times.lead
        self.times = ev.FrameTimes(lead, self.line.origin + lead * self.line.period, self.line.period)


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
    resent: np.ndarray


def measure_latency(
    images: np.ndarray,
    frames: np.ndarray,
    found: Any,
    toggles: np.ndarray,
    times: ev.FrameTimes,
    until: float | None = None,
    taken: np.ndarray | None = None,
) -> ev.Latency:
    """The latency from when the device took each calibration write to where it showed in the picture.

    ``taken`` brackets each toggle's taking (:func:`wire.echoed`), else its send interval does.
    """
    size = min(images.shape[0], frames.size, found.shown.size)
    landed = ev.toggle_landings(
        images[:size], frames["t_ns"][:size], found.starts[:size], found.shown[:size], toggles, until
    )
    lat = ev.latency(ev.taken(toggles, taken), landed, times)
    shown = np.unique(np.r_[landed, landed - 1][np.r_[landed, landed] >= -times.lead])
    least, most = found.line.polygon.at(shown + times.lead)
    line = times.at(shown)
    return replace(lat, line_lo=float(np.min(least - line)), line_hi=float(np.max(most - line)))


def write_latency(directory: str | Path, measured: ev.Latency, program: str) -> None:
    """Keep a dry take's measured latency for the takes without a dry control, one a program and basis."""
    path = Path(directory, LATENCY_FILE)
    takes = json.loads(path.read_text(encoding="utf-8")).get("takes", []) if path.is_file() else []
    entry = {"program": program, **measured.as_dict()}
    takes = [t for t in takes if (t.get("program"), t.get("basis")) != (program, entry["basis"])] + [entry]
    path.write_text(json.dumps({"takes": takes}, indent=1), encoding="utf-8")


def read_latency(directory: str | Path) -> ev.Latency:
    """The send-to-frame-start latency for a take with no dry control: the band every dry take admits.

    Each dry take bounds the device's latency whichever line its captures admit
    (:attr:`~syncsummoner.compose.evidence.Latency.bounds`), so it lies where they all meet. Only takes
    measured to the frame line count. However wide the band, it only widens each write's landing window
    (:func:`~syncsummoner.compose.evidence.window`); bands that do not meet contradict each other.
    """
    path = Path(directory, LATENCY_FILE)
    held = json.loads(path.read_text(encoding="utf-8"))["takes"] if path.is_file() else []
    takes = [t for t in held if t.get("basis") == ev.BASIS]
    if not takes:
        raise NoLatencyError(
            f"no {ev.BASIS}-basis latency in {path}: a program without a dry control needs a dry take's "
            "measured CC-to-picture latency; render a program with a bypass first"
        )
    lo = max(t["lo"] + t.get("line_lo", 0.0) for t in takes)
    hi = min(t["hi"] + t.get("line_hi", 0.0) for t in takes)
    period = float(np.median([t["period"] for t in takes]))
    if hi < lo:
        raise LeadInTimeError(
            f"dry takes in {path} put the CC-to-picture latency over {lo / 1e6:.2f}..{hi / 1e6:.2f} ms, "
            "an empty band, so the dry takes contradict each other"
        )
    found = ev.Latency(lo=lo, hi=hi, samples=0, period=period)
    LOG.info(
        "CC-to-picture %.2f ms from %d dry takes, within %.2f ms",
        found.delta / 1e6,
        len(takes),
        found.spread / 1e6,
    )
    return found


def calibrate(sender: Sender, dry: Mapping[int, int], initial: Mapping[int, int], lead_in: LeadIn) -> None:
    """Toggle the dry control between dry and the take's wet value at random intervals, ending dry."""
    rng = np.random.default_rng(lead_in.seed)
    ((slot, dry_value),) = dry.items()
    wet = int(initial.get(slot, 0))
    for index in range(2 * lead_in.toggles):
        time.sleep(float(rng.uniform(*lead_in.toggle_s)))
        sender.send(slot, wet if index % 2 == 0 else dry_value, int(ev.CALIBRATION))


def plan_rows(
    plan: np.ndarray,
    dry: Mapping[int, int] | None,
    initial: Mapping[int, int],
    *,
    wet_frame: int,
    dry_frame: int | None = None,
) -> np.ndarray:
    """The take's writes: dry control to the take value at ``wet_frame``, the plan, dry at ``dry_frame``."""
    if dry is None:
        return plan
    ((slot, value),) = dry.items()
    rows = [np.array([(wet_frame, slot, initial.get(slot, 0))], dtype=PLAN), plan]
    if dry_frame is not None:
        rows.append(np.array([(dry_frame, slot, value)], dtype=PLAN))
    return np.concatenate(rows)


def state_at(initial: np.ndarray, plan: np.ndarray, frame: int) -> np.ndarray:
    """The parameter vector the plan holds from ``frame`` on, starting from ``initial``."""
    state = np.asarray(initial, dtype=np.int64).copy()
    for row in plan[plan["target"] <= frame]:
        state[int(row["param"]) - 1] = int(row["value"])
    return state


def loaded_before(
    program: str, wire: Mapping[str, Any], ticks: ev.Ticks, times: ev.FrameTimes, delta: float
) -> dict:
    """Whether the device ran ``program`` and read back frame 0's parameters before frame 0's tick.

    A later take of a session reads back the state the take before left; its frame 0 lands on the tick.
    """
    said = json.loads(str(wire["loaded"]))
    try:
        fpga = json.loads(said.get("fpga") or "{}")
    except json.JSONDecodeError:
        fpga = {}
    running = (
        fpga.get("state") == "running"
        and str(fpga.get("program", "")).casefold() == program.rpartition("/")[2].casefold()
    )
    later = "session" in wire and json.loads(str(wire["session"])).get("index", 0) > 0
    want = state_at(wire["start"], wire["plan"], -1 if later else 0)
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


def judged_steps(
    program: str, start: np.ndarray, rows: np.ndarray, count: int
) -> tuple[np.ndarray, ev.Regime] | None:
    """The program model's verdict on each landed write's step, and the switch regime of ``count`` frames.

    None when the model cannot be built, so the take is judged without it.
    """
    from syncsummoner.program import model  # pylint: disable=import-outside-toplevel
    from syncsummoner.program.predict import classify, regimes  # pylint: disable=import-outside-toplevel

    try:
        built = model(program)
        return classify(built, start, rows), ev.Regime(*regimes(built, start, rows, count))
    except (KeyError, RuntimeError, OSError) as err:
        LOG.warning("%s: no step verdicts from the program model: %s", program, err)
        return None


def step_counts(rows: np.ndarray, kinds: np.ndarray | None) -> dict[int, dict[str, int]]:
    """Per parameter, how many of its sent writes the model judged each way."""
    if kinds is None:
        return {}
    sent = rows["sent"] >= 0
    out: dict[int, dict[str, int]] = {}
    for param, kind in zip(rows["param"][sent], kinds[sent]):
        mine = out.setdefault(int(param), {})
        mine[str(kind)] = mine.get(str(kind), 0) + 1
    return out


def take_latency(
    take: str | Path, wire: Mapping[str, Any], seen: tuple, echo: np.ndarray | None
) -> ev.Latency:
    """A take's CC-to-picture latency: its calibration's when its picture shows it, else the one it used.

    ``seen`` is the take's ``(images, frames, attribution, frame times)``; ``echo`` brackets its sends.
    """
    images, frames, found, times = seen
    sends = wire["sends"]
    toggled = sends["target"] == ev.CALIBRATION
    calibration = sends[toggled]
    used = json.loads(str(wire["latency"])) if "latency" in wire else {}
    lat = None
    if used.get("basis") == ev.BASIS:
        lat = ev.Latency(lo=used["lo"], hi=used["hi"], samples=used.get("samples", 0), period=times.period)
    if json.loads(str(wire["dry"])) is not None and calibration.size:
        later = sends["t0_ns"][(sends["target"] > ev.START) & (sends["t0_ns"] > calibration["t1_ns"].max())]
        until = float(later.min()) if later.size else None
        try:
            taken = None if echo is None else echo[toggled]
            lat = measure_latency(images, frames, found, calibration, times, until, taken)
            write_latency(Path(take).parent, lat, str(wire.get("program", Path(take).stem)))
        except ev.EvidenceError as err:
            if lat is None:
                raise
            LOG.warning("%s: %s; using the band measured live in the lead-in", take, err)
    if lat is None:
        try:
            lat = read_latency(Path(take).parent)
        except NoLatencyError:
            lat = None
        except LeadInTimeError as err:
            raise ev.EvidenceError(f"{take}: {err}") from err
    if lat is None:
        raise ev.EvidenceError(
            f"{take}: no CC-to-picture latency, measured or on file, to land its writes by"
        )
    return lat


def take_evidence(
    take: str | Path,
    *,
    program: str,
    clip: str | Path,
    config: Any,
    names: Mapping[int, str] | None = None,
    predict: bool = True,
    found: Any = None,
    features: decoded.Features | None = None,
    source: decoded.Features | None = None,
) -> ev.Evidence:
    """The three links for a recorded take, from its wire record, its attribution and its picture.

    ``found`` is the take's attribution and ``features`` and ``source`` the take's and the clip's
    decodes, each made here when not given. With ``predict``, the program model judges each step from
    the state it was taken in, and only the steps it leaves live test a parameter.
    """
    from syncsummoner.compose.lock import attribute_take  # pylint: disable=import-outside-toplevel
    from syncsummoner.device.v4l2rec import read_log  # pylint: disable=import-outside-toplevel
    from syncsummoner.device.wire import load_wire  # pylint: disable=import-outside-toplevel

    ffmpeg = getattr(config, "ffmpeg", "ffmpeg")
    spec = decoded.Spec.of(config)
    features = decoded.features(take, spec, ffmpeg=ffmpeg) if features is None else features
    source = decoded.features(clip, spec, ffmpeg=ffmpeg) if source is None else source
    whole = load_wire(take)
    wire = own_wire(whole)
    if names is None:
        names = {int(k): v for k, v in json.loads(str(wire["names"])).items()} if "names" in wire else {}
    found = attribute_take(take, config=config, features=features) if found is None else found
    frames = read_log(take)
    times = ev.frame_times(found)
    sends = wire["sends"]
    echo, every = take_echo(wire), take_echo(whole)
    resent = wire.get("resent")
    images = features.small
    lat = take_latency(take, wire, (images, frames, found, times), echo)
    ticks, delta = ev.grid(times, lat), lat.delta
    plan = wire["plan"]
    landed_rows = ev.landed(
        plan, sends, ticks, times, delta, echo=echo, resent=resent, spread=line_spread(found)
    )
    reads = wire["reads"]
    reads = reads[(reads["t0_ns"] >= frames["t_ns"][0]) & (reads["t0_ns"] <= frames["t_ns"][-1])]
    applied_rows = ev.applied(
        reads, whole["sends"], wire["initial"], ticks, times, delta, echo=every, resent=whole.get("resent")
    )
    got = None
    if echo is not None:
        got = own_received(whole)
    count = found.picked.size
    out = images[found.picked].astype(np.float32)
    src = source.small[found.source_offset : found.source_offset + count].astype(np.float32)
    luma_range = np.ptp(out[:, 0].reshape(count, -1), axis=1)
    rows = landed_rows.rows
    sent = (rows["sent"] >= 0) & (rows["landed"] != ev.CALIBRATION)
    kinds, regime = (judged_steps(program, wire["start"], rows, count) if predict else None) or (None, None)
    moved = ev.acted_upon(
        out,
        src,
        luma_range,
        rows["last"][sent],
        rows["param"][sent],
        names,
        kinds=None if kinds is None else kinds[sent],
        regime=regime,
        spans=rows["last"][sent] - rows["first"][sent] + 1,
    )
    return ev.Evidence(
        program=program,
        latency=lat.as_dict(),
        ticks={"width_ms": ticks.width / 1e6, "count": ticks.count, "period_ns": ticks.period},
        landed=landed_rows,
        applied=applied_rows,
        received=got,
        moved=moved,
        loaded=loaded_before(program, wire, ticks, times, delta),
        predicted=step_counts(rows, kinds),
    )


def line_spread(found: Any) -> tuple[float, float]:
    """How far before and after its frame line the take's captures admit its clip frames' starts."""
    n = np.arange(found.lead, found.lead + found.picked.size)
    least, most = found.line.polygon.at(n)
    line = found.line.origin + n * found.line.period
    return float(np.min(least - line, initial=0.0)), float(np.max(most - line, initial=0.0))


def own_wire(wire: Mapping[str, Any]) -> dict:
    """A wire record cut to the sends its take owns, resent copies renumbered.

    A take recorded alone owns every send.
    """
    own = np.asarray(wire.get("own", np.ones(wire["sends"].size, dtype=bool)), dtype=bool)
    resent = np.asarray(wire.get("resent", np.zeros(0, np.int64)), dtype=np.int64)
    kept = np.searchsorted(np.flatnonzero(own), resent[own[resent]]) if resent.size else resent
    return {**wire, "sends": wire["sends"][own], "resent": kept}


def own_received(wire: Mapping[str, Any]) -> W.Received:
    """The receipts of the messages a take's own sends carried, out of every send of its session."""
    sends = wire["sends"]
    own = np.asarray(wire.get("own", np.ones(sends.size, dtype=bool)), dtype=bool)
    expected = W.expected_messages(sends, wire["cc_map"])
    whole = W.received(expected, wire["receipts"], sends)
    lost = own[whole.missing["send"]]
    mine = int(np.count_nonzero(own[expected["send"]]))
    matched = mine - int(lost.sum())
    others = whole.matched - matched
    return replace(
        whole, sent=mine, received=whole.received - others, matched=matched, missing=whole.missing[lost]
    )


def take_echo(wire: Mapping[str, Any]) -> np.ndarray | None:
    """When the device took each of a take's sends, from its receipts; None for a take without any."""
    receipts = wire.get("receipts")
    if receipts is None or not receipts.size:
        return None
    return W.echoed(
        wire["sends"],
        receipts,
        wire["reads"],
        wire["cc_map"],
        read_errors=wire.get("read_errors"),
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

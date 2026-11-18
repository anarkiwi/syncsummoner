"""A session: one program loaded once and its takes played back to back while the card holds lock.

The Pi plays the lead-in, each take's clip frames and a lead-out as one stream (:meth:`Layout.spans`) and
the recorder rolls between take files on the capture clock, so every take shares one frame line, anchored
by the lead-in's and lead-out's unprocessed strips. See docs/evidence.md.
"""

from __future__ import annotations

import gc
import json
import math
import sys
import time
import uuid
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Iterator, Mapping, Sequence

import numpy as np

from syncsummoner.compose import evidence as ev
from syncsummoner.compose.lock import (
    ANCHOR_FRAMES,
    live_attribution,
    read_live,
    save_lock,
    session_log,
    session_record,
)
from syncsummoner.compose.lock import changes
from syncsummoner.compose.locktake import (
    PLAN,
    Clock,
    LeadIn,
    LeadInTimeError,
    Record,
    RebootError,
    boot_count,
    calibrate,
    declared_bypass,
    dry_state,
    initial_state,
    measure_latency,
    plan_rows,
    read_latency,
    schedule,
    state_at,
    wet_at,
    write_latency,
)
from syncsummoner.compose.timecode import ClipLayout, LayoutError
from syncsummoner.device import loadtimes
from syncsummoner.device import wire as W
from syncsummoner.device.profile import PARAM_COUNT, PARAM_MAX, snap_toggles
from syncsummoner.device.wire import Echoes, Poller, Sender, encoder, save_wire
from syncsummoner.probe.decoded import CALIBRATION_SIZE
from syncsummoner.progress import LOG, meter

#: The program whose picture is its input, carrying the strips for a program with no dry control.
PASSTHRU = "Passthru"
#: Longest a session sleeps between carrying its frame line on over the captures since.
WAKE_S = 1.0
#: Interpreter thread switch interval during a session, well inside the tick a late wake-up must make.
SWITCH_S = 0.0005


class NoLoadTimeError(RuntimeError):
    """A lead-in must hold a program load, and no load has been timed to size it by."""


@dataclass(frozen=True)
class Layout:
    """A session's frames: ``lead`` lead-in, each take's ``counts`` clip frames from 0, ``tail`` lead-out.

    ``source_lead`` is the clip file's own lead-in. Session clip frame ``f`` is take ``k``'s frame
    ``f - offsets[k]`` and line frame ``lead + f``.
    """

    lead: int
    counts: tuple[int, ...]
    tail: int
    source_lead: int

    @property
    def offsets(self) -> np.ndarray:
        """Each take's clip frame 0 as a session clip frame."""
        return np.r_[0, np.cumsum(self.counts)[:-1]].astype(np.int64)

    @property
    def count(self) -> int:
        """Session clip frames, every take's."""
        return int(sum(self.counts))

    def spans(self) -> list[tuple[int, int]]:
        """The clip file's frames the Pi plays, ``a..b-1`` a span, in turn and with no seam."""
        last = len(self.counts) - 1
        return [
            (
                self.source_lead - (self.lead if k == 0 else 0),
                self.source_lead + c + (self.tail if k == last else 0),
            )
            for k, c in enumerate(self.counts)
        ]

    def check(self, clip: ClipLayout) -> "Layout":
        """This layout once ``clip`` holds every frame it plays, else a :class:`LayoutError` saying why."""
        if self.lead > clip.lead or self.source_lead != clip.lead:
            raise LayoutError(
                f"the session's lead-in needs {self.lead} frames and the clip's holds {clip.lead}: "
                f"timecode it with a lead-in of at least {self.lead}"
            )
        if max(self.counts) > clip.frames:
            raise LayoutError(f"a take plays {max(self.counts)} clip frames and the clip has {clip.frames}")
        if self.counts[-1] + self.tail > clip.frames + clip.tail:
            raise LayoutError(
                f"the session's lead-out needs {self.tail} frames after clip frame {self.counts[-1]} and the "
                f"clip holds {clip.frames + clip.tail - self.counts[-1]}: timecode it with a longer lead-out"
            )
        return self

    def owner(self, targets: np.ndarray) -> np.ndarray:
        """The take recording each session clip frame: the first the lead-in's, the last the lead-out's."""
        return np.clip(
            np.searchsorted(self.offsets, np.asarray(targets), side="right") - 1, 0, len(self.counts) - 1
        )


def frames_of(seconds: float, fps: float) -> int:
    """Whole frames covering ``seconds``."""
    return int(math.ceil(round(seconds * fps, 6)))


def lead_in_frames(lead_in: LeadIn, fps: float, *, dry: bool, load_s: float | None) -> int:
    """The frames a lead-in must hold: what comes before the clock, the clock, then any wet stretch.

    Dry: settle, :data:`~.lock.ANCHOR_FRAMES` and the toggles at their longest. Otherwise: the anchor under
    Passthru, the longest load timed and settle. The host decodes faster than real time, so the clock takes
    at most as long again.
    """
    if dry:
        before = frames_of(lead_in.settle_s + 2 * lead_in.toggles * lead_in.toggle_s[1], fps) + ANCHOR_FRAMES
        return 2 * before + frames_of(lead_in.wet_s, fps)
    if load_s is None:
        raise NoLoadTimeError("a program with no dry control loads inside the lead-in, and no load is timed")
    return 2 * (ANCHOR_FRAMES + frames_of(load_s + lead_in.settle_s, fps))


def lead_out_frames(lead_in: LeadIn, fps: float, *, dry: bool, load_s: float | None, latency_s: float) -> int:
    """The frames a lead-out must hold after the last clip frame, ending in :data:`~.lock.ANCHOR_FRAMES`.

    Dry: held dry as long as the lead-in held the take value before clip frame 0. Otherwise: the longest
    load timed, then the frame ``latency_s`` lands Passthru's first picture on.
    """
    if dry:
        return frames_of(lead_in.wet_s, fps) + ANCHOR_FRAMES
    if load_s is None:
        raise NoLoadTimeError(
            "a program with no dry control reloads Passthru in its lead-out, and no load is timed"
        )
    return frames_of(load_s + latency_s, fps) + 1 + ANCHOR_FRAMES


def session_frames(
    lead_in: LeadIn, fps: float, *, dry: bool, directory: str | Path, band: ev.Latency | None = None
) -> tuple[int, int]:
    """A session's lead-in and lead-out frames, from the rig's loads and, with no dry control, the band on file."""
    loads = loadtimes.longest()
    if not dry:
        band = read_latency(directory) if band is None else band
    after = 0.0 if band is None else (band.hi + band.period) / 1e9
    return (
        lead_in_frames(lead_in, fps, dry=dry, load_s=loads),
        lead_out_frames(lead_in, fps, dry=dry, load_s=loads, latency_s=after),
    )


@dataclass(frozen=True)
class Take:
    """One take of a session: its file, its writes targeting its own clip frames, and how many it plays."""

    out: Path
    plan: np.ndarray
    count: int


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
    clip: ClipLayout,
    lead_in: LeadIn = LeadIn(),
    bypass: Callable[[str], Any] = declared_bypass,
) -> Record:
    """Record ``program`` playing ``seconds`` of the clip with every planned write sent for its frame.

    A session of one take (:func:`locked_session`).
    """
    take = Take(Path(out), plan, int(round(seconds * float(config.fps))))
    (record,) = locked_session(
        rig,
        recorder,
        [take],
        program=program,
        config=config,
        working=working,
        load=load,
        clip=clip,
        lead_in=lead_in,
        bypass=bypass,
    )
    return record


def locked_session(
    rig: Any,
    recorder: Any,
    takes: Sequence[Take],
    *,
    program: str,
    config: Any,
    working: Callable[[Any], Mapping[int, Any]],
    load: Callable[[str], Any],
    clip: ClipLayout,
    lead_in: LeadIn = LeadIn(),
    bypass: Callable[[str], Any] = declared_bypass,
) -> list[Record]:
    """Record ``program`` loaded once playing every take back to back, each write sent for its frame.

    ``load(name)`` loads a program and checks the rig, returning its info; ``working(info)`` is its working
    point, ``bypass(name)`` its dry setting, else Passthru carries the ends. ``clip`` is the uploaded clip's
    :class:`~.timecode.ClipLayout`. A rise in the boot count is a :class:`RebootError`.
    """
    shell, midi = rig.transport.device.shell, rig.transport.device.midi
    fps = float(config.fps)
    directory = Path(takes[0].out).parent
    booted = boot_count(shell)
    dry = dry_state(program, bypass)
    band = None if dry is not None else read_latency(directory)
    park = np.asarray(rig.session.park_values, dtype=np.int64)
    played: list[dict] = []
    firmware = _firmware(rig)

    def timed(name: str) -> Any:
        began = time.monotonic_ns()
        said = load(name)
        ended = time.monotonic_ns()
        loadtimes.record(name, began, ended, firmware=firmware)
        if held["recording"]:
            played.append({"program": name, "t0_ns": began, "t1_ns": ended})
        return said

    held: dict[str, Any] = {
        "recording": False,
        "delta": float("nan"),
        "loaded": None,
        "codes": [],
        "dry": dry,
    }
    held["info"] = None
    if dry is None:
        timed(PASSTHRU)
    else:
        held["info"] = timed(program)
    layout = _laid_out(
        program,
        takes,
        clip,
        session_frames(lead_in, fps, dry=dry is not None, directory=directory, band=band),
    )
    plan = np.concatenate([np.zeros(0, PLAN)] + [_shifted(t.plan, o) for t, o in zip(takes, layout.offsets)])

    def prepare(info: Any) -> None:
        initial = initial_state(working(info), plan)
        start_state = {**initial, **(dry or {})}
        slots = np.arange(1, PARAM_COUNT + 1)
        start = snap_toggles(slots, [start_state.get(i, int(park[i - 1])) for i in slots])
        asked = plan_rows(plan, dry, initial, wet_frame=wet_at(lead_in, fps), dry_frame=layout.count).copy()
        asked["value"] = snap_toggles(asked["param"], asked["value"])
        held.update(info=info, initial=initial, start=start, rows=schedule(asked, start))

    held.update(rows=plan[:0], start=park.copy())
    if held["info"] is not None:
        prepare(held["info"])
    echoes = Echoes()
    encode = encoder(midi.cc_map, high_resolution=getattr(midi, "high_resolution", True))

    sender = Sender(_writer(rig, park), echoes=echoes, encode=encode)
    poller = Poller(shell, interval_s=lead_in.poll_s, echoes=echoes)
    session = {"id": uuid.uuid4().hex, "offsets": layout.offsets.tolist(), "counts": list(layout.counts)}
    try:
        with (
            recorder.recording(takes[0].out) as proc,
            rig.playout.playing(fps=fps, spans=layout.spans()) as player,
            _monitored(shell, poller, sender),
        ):
            held["recording"] = True
            _play(
                rig,
                Live(recorder, proc, takes, layout),
                sender=sender,
                poller=poller,
                shell=shell,
                player=player,
                held=held,
                program=program,
                config=config,
                dry=dry,
                park=park,
                lead_in=lead_in,
                load=timed,
                prepare=prepare,
            )
            hold = int(player.status().get("hold", 1))
        wire = _session_wire(held, sender, poller, played, dry)
        _save_locks(takes, layout, session, rig.playout.flips(), hold, wire, held, config)
    finally:
        wire = _session_wire(held, sender, poller, played, dry)
        for index, take in enumerate(takes):
            _save_take_wire(take, index, layout, wire, held, program=program, park=park, session=session)
        rebooted = boot_count(shell)
        if rebooted != booted:
            raise RebootError(
                f"{program}: the device's boot count went from {booted} to {rebooted} during the session"
            )
    return [_take_record(index, layout, wire, held, park) for index in range(len(takes))]


def _firmware(rig: Any) -> str | None:
    """The firmware the rig's device runs, to key its loads by; None when the transport cannot say."""
    said = getattr(rig.transport, "firmware", None)
    return None if said is None else str(said())


def _writer(rig: Any, park: np.ndarray) -> Callable[[int, int], int]:
    """A write of a slot's device value as an offset from its park, returning what it reads back."""
    midi = rig.transport.device.midi

    def write(param: int, value: int) -> int:
        rig.transport.set_param(param, int(np.clip(value - park[param - 1], 0, PARAM_MAX)))
        return int(midi.get_param(param)) + int(park[param - 1])

    return write


def _laid_out(program: str, takes: Sequence[Take], clip: ClipLayout, ends: tuple[int, int]) -> Layout:
    """The session's layout on ``clip``, its lead-in and lead-out ``ends``, checked to fit it."""
    layout = Layout(ends[0], tuple(int(t.count) for t in takes), ends[1], int(clip.lead)).check(clip)
    LOG.info(
        "%s: %d takes of %s clip frames between a %d-frame lead-in and a %d-frame lead-out, loaded once",
        program,
        len(takes),
        list(layout.counts),
        layout.lead,
        layout.tail,
    )
    return layout


@dataclass(frozen=True)
class Live:
    """A session as it records: the recorder and its process, the takes, and their layout."""

    recorder: Any
    proc: Any
    takes: Sequence[Take]
    layout: Layout

    @property
    def first(self) -> Path:
        """The first take's file, the one the lead-in is recorded in."""
        return Path(self.takes[0].out)

    def log(self) -> np.ndarray:
        """Every capture of the session so far, once each, from the takes it has reached."""
        logs = []
        for take in self.takes:
            try:
                logs.append(_read_sidecar(take.out))
            except FileNotFoundError:
                break
        return session_log(logs)


def _shifted(plan: np.ndarray, offset: int) -> np.ndarray:
    """``plan`` with its targets moved ``offset`` frames on."""
    moved = np.array(plan, dtype=PLAN, copy=True)
    moved["target"] += int(offset)
    return moved


def _session_wire(held: dict, sender: Sender, poller: Poller, played: list[dict], dry: Any) -> dict:
    """The session's wire record so far, in session clip frames."""
    return {
        "sends": sender.array(),
        "resent": sender.resent(),
        "latency": np.array(json.dumps(held.get("latency", {}))),
        "dry": np.array(json.dumps(dry)),
        "loads": np.array(json.dumps(played)),
        **poller.arrays(),
    }


def _owned(sends: np.ndarray, layout: Layout, index: int) -> np.ndarray:
    """Which of a session's sends a take owns: those targeting the frames its file records."""
    framed = sends["target"] > ev.START
    owner = np.where(framed, layout.owner(np.where(framed, sends["target"], 0)), 0)
    return owner == index


def _take_rows(held: dict, layout: Layout, index: int) -> tuple[np.ndarray, np.ndarray]:
    """A take's share of the session's writes, in its own clip frames, and the state standing before them."""
    rows, offset = held["rows"], int(layout.offsets[index])
    mine = rows[layout.owner(rows["target"]) == index] if rows.size else rows
    start = held["start"] if not index else state_at(held["start"], rows, offset - 1)
    return _shifted(mine, -offset), np.asarray(start, dtype=np.int64)


def _take_sends(wire: dict, layout: Layout, index: int) -> np.ndarray:
    """Every session send, its target in the take's own clip frames."""
    sends = np.array(wire["sends"], copy=True)
    framed = sends["target"] > ev.START
    sends["target"][framed] -= int(layout.offsets[index])
    return sends


def _save_take_wire(
    take: Take,
    index: int,
    layout: Layout,
    wire: dict,
    held: dict,
    *,
    program: str,
    park: np.ndarray,
    session: dict,
) -> None:
    """A take's wire record: every send of the session, marked with those the take owns."""
    plan, start = _take_rows(held, layout, index)
    asked, info = held["loaded"], held.get("info")
    save_wire(
        take.out,
        plan=plan,
        initial=np.asarray(park, dtype=np.int64),
        start=start,
        sends=_take_sends(wire, layout, index),
        own=_owned(wire["sends"], layout, index),
        delta=np.float64(held["delta"]),
        cc_map=np.asarray(held.get("cc_map", []), dtype=np.int64),
        loaded=np.array(
            json.dumps(
                {}
                if asked is None
                else {"fpga": asked.payload, "error": asked.error, "t0_ns": asked.t0_ns, "t1_ns": asked.t1_ns}
            )
        ),
        program=np.array(program),
        names=np.array(
            json.dumps({} if info is None else {int(spec.index): spec.name for spec in info.params})
        ),
        session=np.array(json.dumps({**session, "index": index})),
        **{k: v for k, v in wire.items() if k != "sends"},
    )


def _take_record(index: int, layout: Layout, wire: dict, held: dict, park: np.ndarray) -> Record:
    """What one take of a session put on the wire, its own sends only."""
    own = _owned(wire["sends"], layout, index)
    resent = np.asarray(wire["resent"], dtype=np.int64)
    kept = resent[own[resent]] if resent.size else resent
    return Record(
        plan=_take_rows(held, layout, index)[0],
        initial=np.asarray(park, dtype=np.int64),
        sends=_take_sends(wire, layout, index)[own],
        arrays={k: v for k, v in wire.items() if k not in ("sends", "resent", "latency", "dry", "loads")},
        delta=held["delta"],
        dry=held["dry"],
        loaded={} if held["loaded"] is None else {"fpga": held["loaded"].payload},
        cc_map=np.asarray(held.get("cc_map", []), dtype=np.int64),
        resent=np.searchsorted(np.flatnonzero(own), kept),
    )


def _save_locks(
    takes: Sequence[Take],
    layout: Layout,
    session: dict,
    flips: np.ndarray,
    hold: int,
    wire: dict,
    held: dict,
    config: Any,
) -> None:
    """Each take's lock record: the session's frames, its capture log, the strips read, when processed."""
    from syncsummoner.compose.lock import _written_span  # pylint: disable=import-outside-toplevel

    log = session_log([_read_sidecar(t.out) for t in takes])
    codes = np.full(log.size, -1, dtype=np.int64)
    live = np.asarray(held["codes"], dtype=np.int64)
    codes[: min(live.size, log.size)] = live[: log.size]
    if len(takes) > 1:
        last, _ = read_live(
            takes[-1].out,
            strip_px=config.strip_px,
            bits=config.bits,
            size=CALIBRATION_SIZE,
            ffmpeg=getattr(config, "ffmpeg", "ffmpeg"),
        )
        seq = _read_sidecar(takes[-1].out)["sequence"][: last.size]
        codes[np.searchsorted(log["sequence"], seq)] = last[: seq.size]
    span = _written_span({**wire, "cc_map": np.asarray(held.get("cc_map", []))}, layout.count)
    for index, take in enumerate(takes):
        said = {**session, "index": index, "source_lead": layout.source_lead}
        save_lock(
            take.out,
            flips,
            hold=hold,
            lead=layout.lead,
            count=layout.count,
            tail=layout.tail,
            wet=True,
            **session_record(said, log, codes, span),
        )


def _await_starts(live: Live, since_ns: int, frames: int, period_ns: float) -> None:
    """Wait until the session's captures show ``frames`` frames starting after ``since_ns``."""
    while True:
        log = live.log()
        after = log[log["t_ns"] > since_ns]
        if np.count_nonzero(changes(after["digest"])) >= frames:
            return
        time.sleep(period_ns / 1e9)


def _play(rig: Any, live: Live, **kw: Any) -> None:
    """The session from playout start, with the interpreter switching threads often enough for the sender.

    The cyclic collector is off meanwhile: a full collection holds the interpreter lock for longer
    than the half frame a send has to make its tick, stalling the sender and the poller together.
    """
    switch = sys.getswitchinterval()
    sys.setswitchinterval(SWITCH_S)
    collecting = gc.isenabled()
    gc.disable()
    try:
        _run(rig, live, began=time.monotonic_ns(), **kw)
    finally:
        sys.setswitchinterval(switch)
        if collecting:
            gc.enable()


def _run(
    rig: Any,
    live: Live,
    *,
    sender: Sender,
    poller: Poller,
    shell: Any,
    player: Any,
    held: dict,
    program: str,
    config: Any,
    dry: Mapping[int, int] | None,
    park: np.ndarray,
    lead_in: LeadIn,
    load: Callable[[str], Any],
    prepare: Callable[[Any], None],
    began: int,
) -> None:
    """The session's body: anchor, start state, calibration, the live clock, every take's plan, the lead-out.

    Without a dry control the program is loaded once the lead-in's anchor has shown under Passthru, and
    Passthru again once the last take has played, the sender, poller and monitor stopped first so each
    load is the shell's only client. Every write is acknowledged before the next goes.
    """
    layout, period = live.layout, 1e9 / float(config.fps)
    held["cc_map"] = rig.transport.device.midi.cc_map
    shell.command("midi", "monitor", "on")
    del shell.log_lines[:]
    poller.start()
    if dry is None:
        _await_starts(live, began, ANCHOR_FRAMES, period)
        prepare(load(program))
    start = held["start"]
    for slot in np.flatnonzero(start != park) + 1:
        sender.send(int(slot), int(start[slot - 1]), ev.START)
    time.sleep(lead_in.settle_s)
    if dry is not None:
        _await_starts(live, time.monotonic_ns(), ANCHOR_FRAMES, period)
        calibrate(sender, dry, held["initial"], lead_in)
    hold = int(player.status().get("hold", 1))
    ffmpeg = getattr(config, "ffmpeg", "ffmpeg")
    codes, images = read_live(
        live.first, strip_px=config.strip_px, bits=config.bits, size=CALIBRATION_SIZE, ffmpeg=ffmpeg
    )
    held["codes"] = codes
    found, frames = live_attribution(
        live.first, rig.playout.flips(), codes, config=config, hold=hold, lead=layout.lead, count=layout.count
    )
    lat = _live_latency(rig, live, sender, poller, (images, frames, found), dry=dry, program=program)
    held["delta"], held["latency"] = lat.delta, lat.as_dict()
    clock = Clock.of(found, lat)
    rows = held["rows"]
    first_due = clock.when(int(rows["target"][0])) if rows.size else clock.when(0)
    if time.monotonic_ns() >= first_due:
        raise LeadInTimeError(
            f"{program}: the clock was measured {(time.monotonic_ns() - first_due) / 1e9:.2f}s after the "
            "first write was due, past the lead-in this session derived"
        )
    clock.extend(live.log())
    sender.when = clock.when
    sender.plan([(int(r["target"]), int(r["param"]), int(r["value"])) for r in rows])
    sender.start()
    _sleep_until(clock.when(-1) - int(clock.ticks.period))
    held["loaded"] = poller.ask("fpga", "status", timeout_s=5.0)
    end = int(clock.times.at(layout.count + (0 if dry is None else layout.tail))) + int(config.tail_s * 1e9)
    rolls = list(range(1, len(live.takes)))
    with meter((end - began) / 1e9, desc="session") as progress:
        while (now := time.monotonic_ns()) < end:
            rolls = _roll_due(live, clock, rolls, now, period)
            time.sleep(min(WAKE_S, (end - now) / 1e9))
            clock.extend(live.log())
            progress.to((time.monotonic_ns() - began) / 1e9)
    sender.drain(timeout_s=1.0)
    if dry is None:
        _reload_quiet(sender, poller, shell, load)
        _sleep_until(int(clock.times.at(layout.count + layout.tail)) + int(config.tail_s * 1e9))


def _live_latency(
    rig: Any, live: Live, sender: Sender, poller: Poller, seen: tuple, *, dry: Any, program: str
) -> ev.Latency:
    """The CC-to-picture latency the lead-in's calibration measured, else the band on file."""
    images, frames, found = seen
    directory = live.first.parent
    if dry is None:
        return read_latency(directory)
    sends, wire = sender.array(), poller.arrays()
    toggled = sends["target"] == ev.CALIBRATION
    echo = W.echoed(sends, wire["receipts"], wire["reads"], rig.transport.device.midi.cc_map)
    try:
        lat = measure_latency(
            images, frames, found, sends[toggled], ev.frame_times(found), taken=echo[toggled]
        )
    except ev.EvidenceError as err:
        LOG.warning("%s: %s; landing by the band dry takes measured", program, err)
        return read_latency(directory)
    write_latency(directory, lat, program)
    LOG.info("%s: CC-to-picture %.3f frames over %d writes", program, lat.frames, lat.samples)
    return lat


def _roll_due(live: Live, clock: Clock, rolls: list[int], now: int, period: float) -> list[int]:
    """Roll the recording on to each take whose first frame is near: sent at least a wake before the
    captures it shares with the take before, those every line the clock admits puts within a frame of it.
    """
    left = []
    for k in rolls:
        lo, hi = clock.line.polygon.at(np.array([live.layout.lead + int(live.layout.offsets[k])]))
        opens, closes = int(lo[0] - period), int(hi[0] + period)
        if now + 2 * WAKE_S * 1e9 < opens:
            left.append(k)
            continue
        live.recorder.roll(live.proc, live.takes[k].out, opens, closes)
    return left


def _reload_quiet(sender: Sender, poller: Poller, shell: Any, reload: Callable[[str], Any]) -> None:
    """Load Passthru for the lead-out with the sender, poller and monitor stopped: the shell's only client."""
    sender.stop()
    poller.stop()
    shell.command("midi", "monitor", "off")
    reload(PASSTHRU)


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


def _sleep_until(t_ns: int) -> None:
    """Sleep to ``t_ns`` on the host clock."""
    while (left := t_ns - time.monotonic_ns()) > 0:
        time.sleep(min(left / 1e9, 0.5))

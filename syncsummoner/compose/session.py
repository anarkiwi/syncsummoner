"""A session: one program loaded once and its takes played back to back while the card holds lock.

The Pi plays the lead-in, each take's clip frames and a lead-out as one stream (:meth:`Layout.spans`) and
the recorder rolls between take files as the Pi reports each take's first frame near, so every take shares
one frame line, anchored by the lead-in's and lead-out's unprocessed strips. See docs/evidence.md.
"""

from __future__ import annotations

import gc
import json
import math
import sys
import threading
import time
import uuid
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Iterator, Mapping, Sequence

import numpy as np

from syncsummoner.compose import evidence as ev
from syncsummoner.compose import relocks
from syncsummoner.compose.lock import (
    ANCHOR_FRAMES,
    read_live,
    save_lock,
    session_log,
    session_record,
)
from syncsummoner.compose.lock import changes
from syncsummoner.compose.locktake import (
    PLAN,
    LeadIn,
    LeadInTimeError,
    Record,
    RebootError,
    boot_count,
    declared_bypass,
    dry_state,
    initial_state,
    plan_rows,
    schedule,
    state_at,
    wet_at,
)
from syncsummoner.compose.timecode import ClipLayout, LayoutError
from syncsummoner.device import loadtimes
from syncsummoner.device import wire as W
from syncsummoner.device.profile import PARAM_COUNT, PARAM_MAX, snap_toggles
from syncsummoner.device.playout import ASKED, PlayoutError
from syncsummoner.device.playout import SHOWN as PLAYED
from syncsummoner.device.wire import Echoes, Poller, Sender, encoder, save_wire
from syncsummoner.probe.decoded import CALIBRATION_SIZE
from syncsummoner.progress import LOG, meter

#: The program whose picture is its input, carrying the strips for a program with no dry control.
PASSTHRU = "Passthru"
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
    """The frames a lead-in must hold: the strips' anchor, the start state settling, then any wet stretch.

    Dry: settle, the start state's writes and the offset's bounds (``bound_s``), then the wet stretch,
    the frames before it the strips' anchor. Otherwise: the anchor under Passthru, the longest load
    timed, then the same (docs/evidence.md).
    """
    if dry:
        return frames_of(lead_in.settle_s + lead_in.bound_s, fps) + frames_of(lead_in.wet_s, fps)
    if load_s is None:
        raise NoLoadTimeError("a program with no dry control loads inside the lead-in, and no load is timed")
    return ANCHOR_FRAMES + frames_of(load_s + lead_in.settle_s + lead_in.bound_s, fps)


def lead_out_frames(lead_in: LeadIn, fps: float, *, dry: bool, load_s: float | None) -> int:
    """The frames a lead-out must hold after the last clip frame, ending in :data:`~.lock.ANCHOR_FRAMES`.

    Dry: held dry as long as the lead-in held the take value before clip frame 0. Otherwise: the longest
    load timed, then the frame the device latches Passthru's first picture on.
    """
    if dry:
        return frames_of(lead_in.wet_s, fps) + ANCHOR_FRAMES
    if load_s is None:
        raise NoLoadTimeError(
            "a program with no dry control reloads Passthru in its lead-out, and no load is timed"
        )
    return frames_of(load_s, fps) + 1 + ANCHOR_FRAMES


def session_frames(
    lead_in: LeadIn,
    fps: float,
    *,
    dry: bool,
    load_s: float | None = None,
    reload_s: float | None = None,
) -> tuple[int, int]:
    """A session's lead-in and lead-out frames: the lead-in for a load of ``load_s``, else the rig's longest
    timed, and the lead-out for a Passthru reload of ``reload_s``, else the same load."""
    loads = loadtimes.longest() if load_s is None else load_s
    return (
        lead_in_frames(lead_in, fps, dry=dry, load_s=loads),
        lead_out_frames(lead_in, fps, dry=dry, load_s=loads if reload_s is None else reload_s),
    )


def clip_frames(lead_in: LeadIn, fps: float, clip: ClipLayout, *, dry: bool) -> tuple[int, int]:
    """A session's lead-in and lead-out frames on ``clip``.

    The lead-out holds the longest Passthru reload timed, the only load it holds. Without a dry control the
    lead-in holds the longest load timed of any program, but no more than ``clip``'s lead-in holds: a load
    that outlasts it is a :class:`LeadInTimeError` as the session runs, so one slow load on record never
    sizes every later session past its clip.
    """
    lead, tail = session_frames(lead_in, fps, dry=dry, reload_s=loadtimes.longest(program=PASSTHRU))
    least = lead if dry else lead_in_frames(lead_in, fps, dry=False, load_s=0.0)
    held = max(min(lead, int(clip.lead)), least)
    if held < lead:
        LOG.warning(
            "the longest load timed needs a %d-frame lead-in and the clip holds %d: a load outlasting "
            "%d frames will fault the session",
            lead,
            held,
            held,
        )
    return held, tail


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
    booted = boot_count(shell)
    dry = dry_state(program, bypass)
    lead_in = relocks.lead_in(program, lead_in)
    park = np.asarray(rig.session.park_values, dtype=np.int64)
    played: list[dict] = []

    def timed(name: str) -> Any:
        began = time.monotonic_ns()
        said = load(name)
        ended = time.monotonic_ns()
        loadtimes.record(name, began, ended, firmware=_firmware(rig))
        if held["recording"]:
            played.append({"program": name, "t0_ns": began, "t1_ns": ended})
        return said

    held: dict[str, Any] = {"recording": False, "loaded": None, "dry": dry, "info": None}
    if dry is None:
        timed(PASSTHRU)
    else:
        held["info"] = timed(program)
    layout = _laid_out(
        program,
        takes,
        clip,
        clip_frames(lead_in, fps, clip, dry=dry is not None),
    )
    held["lead"] = layout.lead
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
            rig.playout.following() as shown,
            _monitored(shell, poller, sender),
        ):
            held["recording"] = True
            if dry is not None:
                _arm(shell, poller, sender, held, park, shown)
            with rig.playout.playing(
                fps=fps, spans=layout.spans(), heard=lambda: shown.latest() != W.NO_FLOOR
            ) as player:
                _play(
                    rig,
                    Live(recorder, proc, takes, layout),
                    sender=sender,
                    poller=poller,
                    shown=shown,
                    shell=shell,
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
        _keep_relock(
            program,
            dry,
            _save_locks(takes, layout, session, rig.playout.flips(), hold, wire, held, config),
            wire["sends"],
            _firmware(rig),
        )
    finally:
        wire = _session_wire(held, sender, poller, played, dry)
        for index, take in enumerate(takes):
            _save_take_wire(take, index, layout, wire, held, program=program, park=park, session=session)
        _leave(shell, held, booted, program)
    return [_take_record(index, layout, wire, held, park) for index in range(len(takes))]


def _keep_relock(program: str, dry: Any, log: np.ndarray, sends: np.ndarray, firmware: str | None) -> None:
    """Keep the card's relock after a dry session's wet write on the rig's record, saying a long one."""
    found = None if dry is None else relocks.measure(log, sends)
    if found is None:
        return
    relocks.record(program, found[0], found[1], firmware=firmware)
    if found[1] > LeadIn().wet_s:
        LOG.warning(
            "%s: the card relocked %.3f s after the wet write; its sessions wait it out", program, found[1]
        )


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


def _session_frame(frame: int, lead: int) -> int:
    """A player stream frame as a session clip frame; :data:`~.wire.NO_FLOOR` stays as it is."""
    return frame if frame == W.NO_FLOOR else frame - lead


def _session_wire(held: dict, sender: Sender, poller: Poller, played: list[dict], dry: Any) -> dict:
    """The session's wire record so far, in session clip frames: the Pi's reports and replies among it."""
    shown = held.get("shown")
    reports = np.zeros(0, PLAYED) if shown is None else shown.log()
    replies = np.zeros(0, ASKED) if shown is None else shown.asked()
    reports["frame"] -= held.get("lead", 0)
    replies["frame"] -= held.get("lead", 0)
    return {
        "sends": sender.array(),
        "resent": sender.resent(),
        "dry": np.array(json.dumps(dry)),
        "loads": np.array(json.dumps(played)),
        "shown": reports,
        "asked": replies,
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
    sends["floor"] = [_session_frame(int(f), int(layout.offsets[index])) for f in sends["floor"]]
    return sends


def _take_frames(records: np.ndarray, layout: Layout, index: int) -> np.ndarray:
    """The Pi's reports or replies, their frames in the take's own clip frames."""
    out = np.array(records, copy=True)
    out["frame"] -= int(layout.offsets[index])
    return out


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
        **{k: v for k, v in wire.items() if k not in ("sends", "shown", "asked")},
        shown=_take_frames(wire["shown"], layout, index),
        asked=_take_frames(wire["asked"], layout, index),
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
        arrays={k: v for k, v in wire.items() if k not in ("sends", "resent", "dry", "loads")},
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
) -> np.ndarray:
    """Each take's lock record: the session's frames, its capture log, the strips read, when processed.

    Returns the session's capture log.
    """
    from syncsummoner.compose.lock import _written_span  # pylint: disable=import-outside-toplevel

    log = session_log([_read_sidecar(t.out) for t in takes])
    codes = np.full(log.size, -1, dtype=np.int64)
    for take in takes[:: max(len(takes) - 1, 1)]:
        read, _ = read_live(
            take.out,
            strip_px=config.strip_px,
            bits=config.bits,
            size=CALIBRATION_SIZE,
            ffmpeg=getattr(config, "ffmpeg", "ffmpeg"),
        )
        seq = _read_sidecar(take.out)["sequence"][: read.size]
        codes[np.searchsorted(log["sequence"], seq)] = read[: seq.size]
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
    return log


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
    shown: Any,
    shell: Any,
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
    """The session's body: anchor, start state, the device-to-clip offset, every take's plan, the lead-out.

    Without a dry control the program is loaded once the lead-in's anchor has shown under Passthru, before
    the poller starts and the monitor is on, and Passthru again once the last take has played, the sender,
    poller and monitor stopped first, so each load is the shell's only client. The Pi is asked for its
    last flip after every frame tick read, and each write goes once the device has ticked for the clip
    frame before its target, the offset the lead-in's events bound (docs/evidence.md); every write is
    acknowledged before the next goes.
    """
    layout, period = live.layout, 1e9 / float(config.fps)
    stall = loadtimes.stall_guard(getattr(config, "stall_s", None))
    held["cc_map"], held["shown"] = rig.transport.device.midi.cc_map, shown
    sender.floor = lambda: _session_frame(shown.latest(), layout.lead)
    if dry is None:
        _await_starts(live, began, ANCHOR_FRAMES, period)
        prepare(load(program))
        _arm(shell, poller, sender, held, park, shown)
    shift = _offset(poller, held, sender, dry, layout.lead, period)
    held["shift"] = shift
    LOG.info("%s: device frames lead clip frames by %d..%d", program, shift.lo, shift.hi)
    rows = held["rows"]
    first = int(rows["target"][0]) if rows.size else 0
    settle = 0 if dry is not None else frames_of(lead_in.settle_s, float(config.fps))
    now = json.loads(poller.ask("transport", "status", timeout_s=5.0).payload or "{}").get("field_count")
    if now is None or int(now) >= first + shift.lo - 1 - settle:
        raise LeadInTimeError(
            f"{program}: the start state was written and the device-to-clip offset known after the device "
            f"ticked for clip frame {first - 1 - settle}, leaving it less than its {settle}-frame settle "
            "before the first write: past the lead-in this session derived"
        )
    sender.due = lambda target, stop: poller.wait_count(target + shift.lo - 1, stop)
    sender.plan([(int(r["target"]), int(r["param"]), int(r["value"])) for r in rows])
    sender.start()
    poller.wait_count(shift.lo - 2, threading.Event())
    held["loaded"] = poller.ask("fpga", "status", timeout_s=5.0)
    last = layout.lead + layout.count + (0 if dry is None else layout.tail - 1)
    with meter((last - layout.lead) / float(config.fps), desc="session") as progress:
        for k in range(1, len(live.takes)):
            _roll(live, shown, k, period, stall)
            progress.to((layout.offsets[k]) / float(config.fps))
        _heard(shown, last, period, stall)
        progress.to((last - layout.lead) / float(config.fps))
    time.sleep(config.tail_s)
    sender.drain(timeout_s=1.0)
    if dry is None:
        _reload_quiet(sender, poller, shell, load)
        _heard(shown, layout.lead + layout.count + layout.tail - 1, period, stall)
        time.sleep(config.tail_s)


def _leave(shell: Any, held: dict, booted: int, program: str) -> None:
    """Leave the transport as the session found it; a rise in the boot count is a :class:`RebootError`."""
    if held.get("transport") not in (None, "playing"):
        shell.command("transport", "stop")
    rebooted = boot_count(shell)
    if rebooted != booted:
        raise RebootError(
            f"{program}: the device's boot count went from {booted} to {rebooted} during the session"
        )


def _arm(shell: Any, poller: Poller, sender: Sender, held: dict, park: np.ndarray, shown: Any) -> None:
    """Play the transport, turn the monitor on, start polling, and write the start state.

    Each frame tick read asks the Pi for its last flip once one has been heard. With a bypass this is
    done before playout, so the start state settles before the lead-in begins.
    """
    held["transport"] = _transport_playing(shell)
    shell.command("midi", "monitor", "on")
    del shell.log_lines[:]
    poller.on_tick = lambda: shown.ask() if shown.latest() != W.NO_FLOOR else None
    poller.start()
    start = held["start"]
    for slot in np.flatnonzero(start != park) + 1:
        sender.send(int(slot), int(start[slot - 1]), ev.START)


def _transport_playing(shell: Any) -> str:
    """Put the device's transport in play, so its ``field_count`` counts frames; returns its state before."""
    was = str(json.loads(shell.command("transport", "status").payload).get("state"))
    if was != "playing":
        shell.command("transport", "play")
    return was


def _offset(poller: Poller, held: dict, sender: Sender, dry: Any, frames: int, period_ns: float) -> ev.Offset:
    """The device-to-clip offset the session's events bound so far, waiting a frame tick while too few do.

    Waiting past ``frames`` ticks, the lead-in's length, is a :class:`LeadInTimeError`.
    """
    began = None
    while True:
        wire = _session_wire(held, sender, poller, [], dry)
        try:
            return ev.offset(
                np.zeros((0, 2), np.int64),
                np.zeros(0, np.int64),
                wire["asked"],
                wire["shown"],
                wire["ticks"],
                period_ns,
            )
        except ev.UnboundError as err:
            count = int(wire["ticks"]["field"][-1]) if wire["ticks"].size else 0
            began = count if began is None else began
            if count - began > frames:
                raise LeadInTimeError(f"{err}, after the lead-in's {frames} frames") from err
            if poller.wait_count(count + 1, threading.Event()) is None:
                raise


def _heard(shown: Any, frame: int, period: float, stall_s: float) -> int:
    """When the Pi was heard showing ``frame``, waiting as long as the frames before it take at ``period``
    nanoseconds a frame, and ``stall_s`` more; a :class:`PlayoutError` once that has passed unheard."""
    latest = shown.latest()
    timeout_s = max(frame - max(latest, 0), 0) * period / 1e9 + stall_s
    heard = shown.wait(frame, timeout_s=timeout_s)
    if heard is None:
        raise PlayoutError(
            f"the Pi was not heard showing frame {frame} within {timeout_s:.1f} s of frame {latest}: "
            "its player stopped reporting or its stream ended"
        )
    return int(heard)


def _roll(live: Live, shown: Any, k: int, period: float, stall_s: float) -> None:
    """Roll the recording on to take ``k`` once the Pi is heard showing the frame two before its first.

    Both files keep the captures from that report to two frames past the boundary, so the boundary's
    captures are in each whichever card delay holds.
    """
    heard = _heard(shown, live.layout.lead + int(live.layout.offsets[k]) - 2, period, stall_s)
    live.recorder.roll(live.proc, live.takes[k].out, heard, int(heard + 4 * period))


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

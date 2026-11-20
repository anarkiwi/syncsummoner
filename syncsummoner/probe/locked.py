"""An archive sweep as frame-locked takes: one level a hold, every frame attributed to the clip.

The cover's setpoints are targeted one hold apart and cut into
takes no longer than the clip. Each take's evidence lands every write on a frame; the frames each
setpoint settled over become its archive rows, and a program's takes are linked into one archive.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, Sequence

import numpy as np

from syncsummoner.compose.locktake import PLAN
from syncsummoner.device.profile import PARAM_COUNT, snap_toggles
from syncsummoner.probe.archive import GAP, MJPEG_PIX_FMT, FrameRow
from syncsummoner.probe.fit import load_measurements, save_measurements
from syncsummoner.probe.store import atomic_write
from syncsummoner.progress import LOG
from syncsummoner.program.gates import tour

__all__ = [
    "Chunk",
    "SweepError",
    "capture",
    "carried",
    "collect",
    "commit_sweep",
    "disagreeing",
    "joint",
    "levels",
    "owners",
    "plan_sweep",
    "read_plan",
    "number_runs",
    "realized",
    "recorded",
    "rows_path",
    "save_rows",
    "settled_rows",
    "write_plan",
]


class SweepError(RuntimeError):
    """A sweep cannot be planned, its takes did not send what was planned, or they cannot be joined."""


@dataclass(frozen=True)
class Chunk:
    """One take's share of a sweep: its writes, the setpoint each write moves to, and those setpoints.

    ``states[k]`` is setpoint ``first + k``'s full state and ``owner[i]`` the ``k`` plan row ``i``
    moves to; setpoint ``first`` is the take's start state, written at frame 0.
    """

    first: int
    states: np.ndarray
    plan: np.ndarray
    owner: np.ndarray
    frames: int
    settle: int
    hold: int

    def as_dict(self) -> dict:
        """JSON-ready."""
        return {
            "first": self.first,
            "states": self.states.tolist(),
            "plan": self.plan.tolist(),
            "owner": self.owner.tolist(),
            "frames": self.frames,
            "settle": self.settle,
            "hold": self.hold,
        }

    @classmethod
    def from_dict(cls, d: dict) -> "Chunk":
        """Inverse of :meth:`as_dict`."""
        return cls(
            first=int(d["first"]),
            states=np.asarray(d["states"], dtype=np.int64).reshape(-1, PARAM_COUNT),
            plan=np.array([tuple(r) for r in d["plan"]], dtype=PLAN),
            owner=np.asarray(d["owner"], dtype=np.int64),
            frames=int(d["frames"]),
            settle=int(d["settle"]),
            hold=int(d["hold"]),
        )


def levels(found: Any) -> np.ndarray:
    """Every setpoint of the cover's sweep in order, as full states: the first base state, then each move."""
    return np.array([after for _, after in found.vectors()], dtype=np.int64).reshape(-1, PARAM_COUNT)


def recorded(record: dict, values: dict[int, np.ndarray]) -> np.ndarray:
    """The sweep of a cover as ``program cover --json`` records it, in :meth:`Cover.vectors` order.

    Each chosen base state, then each parameter stepped there walked through ``values[p]`` by
    :func:`gates.tour` and back to the base.
    """
    states = np.asarray(record["states"], dtype=np.int64)
    out: list[np.ndarray] = []
    for i in record["chosen"]:
        base = states[i]
        out.append(base)
        for p in record["stepped"].get(str(i), []):
            for v in tour(values[p], int(base[p - 1])):
                out.append(out[-1].copy())
                out[-1][p - 1] = v
            if not np.array_equal(out[-1], base):
                out.append(base)
    return np.array(out, dtype=np.int64).reshape(-1, PARAM_COUNT)


def joint(states: np.ndarray, selector: int, mover: int, values: dict[int, np.ndarray]) -> np.ndarray:
    """``states`` then ``selector`` held at each of its settings, a ``mover`` move under each.

    For a control that acts only on what another's moves write, such as a page select: the move
    goes to the mover's setting farthest from its base and back, and the selector returns last.
    """
    base = np.asarray(states[-1], dtype=np.int64)
    far = values[mover][np.argmax(np.abs(values[mover] - base[mover - 1]))]
    out = [np.asarray(s, dtype=np.int64) for s in states]
    for v in np.r_[base[selector - 1], tour(values[selector], int(base[selector - 1]))]:
        held = base.copy()
        held[selector - 1] = v
        moved = held.copy()
        moved[mover - 1] = far
        out += [held, moved, held]
    out.append(base)
    keep = np.r_[True, np.any(np.diff(np.array(out), axis=0) != 0, axis=1)]
    return np.array(out, dtype=np.int64)[keep]


def carried(states: np.ndarray, toggles: Sequence[int]) -> np.ndarray:
    """``states`` as the link carries them, with any setpoint that then repeats the one before dropped.

    A toggle slot is sent as off or on, as the firmware reads it (:func:`~.profile.snap_toggles`).
    """
    states = np.asarray(states, dtype=np.int64)
    out = snap_toggles(np.arange(1, PARAM_COUNT + 1)[None, :], states, toggles)
    return out[np.r_[True, np.any(np.diff(out, axis=0) != 0, axis=1)]]


def staggered(at: int, state: np.ndarray, before: np.ndarray) -> list[tuple[int, int, int]]:
    """A setpoint's writes from frame ``at``, one changed slot a frame, so each lands on its own frame.

    The device takes writes one at a time, each acknowledged before the next, so a frame holds only
    as many as its tick interval lets through; one a frame always fits.
    """
    moved = np.flatnonzero(np.asarray(state) != np.asarray(before)) + 1
    return [(int(at) + i, int(s), int(state[s - 1])) for i, s in enumerate(moved)]


def restaggered(chunk: Chunk) -> Chunk:
    """``chunk`` with each setpoint's writes after frame 0 spread one a frame from where they start."""
    plan = np.array(chunk.plan, copy=True)
    body = np.flatnonzero(plan["target"] > 0)
    for owner in np.unique(chunk.owner[body]):
        mine = body[chunk.owner[body] == owner]
        plan["target"][mine] = plan["target"][mine].min() + np.arange(mine.size)
    return replace(chunk, plan=plan)


def plan_sweep(states: np.ndarray, *, settle: int, hold: int, frames: int) -> list[Chunk]:
    """Cut a sweep into takes of at most ``frames`` clip frames, each setpoint held ``hold`` after its writes.

    A setpoint's changed slots are written one a frame (:func:`staggered`); when they land is what the
    take realizes, and :func:`settled_rows` keeps only the frames after the last. A take opens on its
    setpoint as the start state.
    """
    states = np.asarray(states, dtype=np.int64)
    if hold > frames:
        raise SweepError(f"a hold of {hold} frames does not fit a take of {frames}")
    out, k = [], 0
    while k < len(states):
        rows = [(0, p + 1, int(v)) for p, v in enumerate(states[k])]
        owner, j, at = [0] * PARAM_COUNT, k + 1, hold
        while j < len(states):
            mine = staggered(at, states[j], states[j - 1])
            if at + len(mine) + hold > frames + 1:
                break
            rows += mine
            owner += [j - k] * len(mine)
            at += max(len(mine) - 1, 0) + hold
            j += 1
        out.append(
            Chunk(
                first=k,
                states=states[k:j],
                plan=np.array(rows, dtype=PLAN),
                owner=np.asarray(owner, dtype=np.int64),
                frames=at,
                settle=int(settle),
                hold=int(hold),
            )
        )
        k = j
    return out


def owners(chunk: Chunk, landed: np.ndarray) -> np.ndarray:
    """The setpoint offset each of a take's landed writes moves to, matched slot by slot in order.

    The take may have moved writes to other frames to pace the link, so the k-th write of a slot
    answers the k-th planned one. :class:`SweepError` if the slots' value sequences differ.
    """
    body = chunk.plan["target"] > 0
    plan, owner = chunk.plan[body], chunk.owner[body]
    ip = np.lexsort((plan["target"], plan["param"]))
    il = np.lexsort((landed["target"], landed["param"]))
    if not (
        np.array_equal(plan["param"][ip], landed["param"][il])
        and np.array_equal(plan["value"][ip], landed["value"][il])
    ):
        raise SweepError(f"the take's {landed.size} writes are not the {plan.size} its chunk planned")
    out = np.empty(landed.size, dtype=np.int64)
    out[il] = owner[ip]
    return out


def realized(start: np.ndarray, rows: np.ndarray, count: int) -> np.ndarray:
    """The state in effect on each of ``count`` frames: ``start``, then each sent write from its landing."""
    state = np.repeat(np.asarray(start, dtype=np.int64)[None, :], count, axis=0)
    sent = rows[rows["sent"] >= 0]
    for i in np.argsort(sent["landed"], kind="stable"):
        state[max(int(sent["landed"][i]), 0) :, sent["param"][i] - 1] = sent["wire"][i]
    return state


def windows(chunk: Chunk, landed: np.ndarray, count: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """A take's in-clip writes; per setpoint, the first frame its writes may land on and its window's opening.

    A setpoint's window opens ``chunk.settle`` frames after the end of its writes' latest landing window.
    The dry control's writes around the clip, in the lead-in and from the lead-out on, own no setpoint;
    a setpoint no write moves to, repeating the one before, starts at ``count``, never.
    """
    rows = landed[(landed["target"] > 0) & (landed["target"] < chunk.frames)]
    owner = owners(chunk, rows)
    n = len(chunk.states)
    sent = rows["sent"] >= 0
    first, last = np.full(n, count, dtype=np.int64), np.zeros(n, dtype=np.int64)
    first[0] = 0
    np.minimum.at(first, owner[sent], rows["first"][sent])
    np.maximum.at(last, owner[sent], rows["last"][sent])
    return rows, first, last + chunk.settle


@dataclass(frozen=True)
class Verdict:
    """Whether a sweep take measured what it planned: every write taken and applied before its window opened.

    ``untaken`` are writes never sent, never echoed, duplicated or out of order; ``unopened`` setpoints
    whose window does not open before the next setpoint's writes may land; ``misapplied`` readbacks that
    disagree with the landed writes inside an open window. ``late`` writes took effect after their
    nominal frame but are only reported: a setting's window waits for its last write.
    """

    untaken: int
    unopened: int
    misapplied: int
    late: int

    @property
    def ok(self) -> bool:
        """No fault that could put a measured frame under a state other than its setpoint's."""
        return not (self.untaken or self.unopened or self.misapplied)

    def as_dict(self) -> dict:
        """JSON-ready, with the verdict."""
        return {
            "ok": self.ok,
            "untaken": self.untaken,
            "unopened": self.unopened,
            "misapplied": self.misapplied,
            "late": self.late,
        }


def verdict(chunk: Chunk, landed: Any, disagree: np.ndarray, count: int) -> Verdict:
    """A sweep take's :class:`Verdict` from its landings (:class:`evidence.Landed`) and disagreeing frames."""
    _, first, opens = windows(chunk, landed.rows, count)
    moved = first < count
    starts = np.sort(first[moved])
    ends = np.r_[starts[1:], count]
    window_end = ends[np.searchsorted(starts, first[moved])]
    frames = np.asarray(disagree, dtype=np.int64)
    inside = np.zeros(frames.size, dtype=bool)
    for a, b in zip(opens[moved], window_end):
        inside |= (frames >= a) & (frames < b)
    return Verdict(
        untaken=int(landed.missing + landed.unechoed + landed.duplicated + landed.out_of_order),
        unopened=int(np.count_nonzero(opens[moved] >= window_end)),
        misapplied=int(inside.sum()),
        late=int(landed.late),
    )


def settled_rows(
    chunk: Chunk,
    landed: np.ndarray,
    *,
    picked: np.ndarray,
    shown: np.ndarray,
    captured: np.ndarray,
    lead: int,
    program: str,
    bad_frames: Sequence[int] = (),
    ambiguity: np.ndarray | None = None,
) -> list[FrameRow]:
    """One archive row per capture, under the state the take realized: its settled setpoint, else :data:`GAP`.

    A frame belongs to the setpoint whose writes may have started landing last by then, once
    ``chunk.settle`` frames have passed since the end of its writes' latest landing window, and only
    where the realized state is that setpoint's (:func:`windows`). Frames in ``bad_frames`` keep no setpoint;
    ``ambiguity`` is, per capture, how many frames earlier the take's clock also admits it shows.
    """
    count, n = picked.size, len(chunk.states)
    rows, first, opens = windows(chunk, landed, count)
    state = realized(chunk.states[0], rows, count)
    order = np.argsort(first, kind="stable")
    frame = np.arange(count)
    k = order[np.clip(np.searchsorted(first[order], frame, side="right") - 1, 0, n - 1)]
    ok = (frame >= opens[k]) & np.all(state == chunk.states[k], axis=1)
    setpoint = np.where(ok, chunk.first + k, GAP)
    bad = np.asarray(bad_frames, dtype=np.int64)
    setpoint[bad[(bad >= 0) & (bad < count)]] = GAP
    held, at = np.full(shown.size, GAP, dtype=np.int64), np.zeros(shown.size, dtype=np.int64)
    seen = picked >= 0
    held[picked[seen]], at[picked[seen]] = setpoint[seen], np.flatnonzero(seen)
    in_take = np.clip(np.where(shown >= 0, shown, at), 0, count - 1)
    return [
        FrameRow(
            frame=c,
            program=program,
            params=tuple(int(v) for v in state[in_take[c]]),
            setpoint=int(held[c]),
            captured=float(captured[c]),
            shown=int(shown[c]) + lead if shown[c] >= 0 else GAP,
            ambiguity=0 if ambiguity is None else int(ambiguity[c]),
        )
        for c in range(shown.size)
    ]


def plan_path(directory: str | Path, slug_: str) -> Path:
    """Where a program's sweep plan is kept in a takes directory."""
    return Path(directory, f"sweep-{slug_}.json")


def write_plan(path: str | Path, chunks: Sequence[Chunk], **extra: Any) -> Path:
    """Keep a program's chunks, with ``extra`` fields such as its response and program name."""
    payload = json.dumps({**extra, "chunks": [c.as_dict() for c in chunks]}, indent=1)
    atomic_write(Path(path), lambda tmp: tmp.write_text(payload, encoding="utf-8"))
    return Path(path)


def read_plan(path: str | Path) -> tuple[list[Chunk], dict]:
    """A program's chunks and the fields kept with them."""
    record = json.loads(Path(path).read_text(encoding="utf-8"))
    return [Chunk.from_dict(c) for c in record.pop("chunks")], record


def rows_path(take: str | Path) -> Path:
    """Where a judged sweep take's archive rows are kept."""
    return Path(f"{take}.rows.parquet")


def commit_sweep(
    archive: Any,
    program: str,
    key: Any,
    takes: Sequence[str | Path],
    chunks: Sequence[Chunk],
    *,
    width: int,
    height: int,
    fps: float,
    program_info: dict | None,
    stimulus: dict,
) -> dict:
    """Link a program's judged takes, in chunk order, into its archive: one listing, one row per capture."""
    rows: list[FrameRow] = []
    for take in takes:
        offset = len(rows)
        rows += [
            FrameRow.from_row({**r, "frame": int(r["frame"]) + offset})
            for r in load_measurements(rows_path(take))
        ]
    states = np.concatenate([c.states for c in chunks])
    return archive.commit(
        program,
        key,
        list(takes),
        rows,
        width=width,
        height=height,
        fps=fps,
        program_info=program_info,
        stimulus=stimulus,
        schedule=[
            {"setpoint": k, "from": states[max(k - 1, 0)].tolist(), "to": states[k].tolist()}
            for k in range(len(states))
        ],
        pix_fmt=MJPEG_PIX_FMT,
    )


def save_rows(take: str | Path, rows: Sequence[FrameRow]) -> Path:
    """Keep a judged take's archive rows beside it."""
    path = rows_path(take)
    atomic_write(path, lambda tmp: save_measurements(rows, tmp))
    return path


def capture(
    config: Any, chunks: Sequence[Chunk], outs: Sequence[str | Path], *, program: str, clip: str | Path
) -> list[dict]:  # pragma: no cover - needs the rig
    """Record the chunks as one session of ``program`` on the clip; per take, its device state and any fault.

    The program is loaded once for them all; a fault anywhere is every take's, since they share one session.
    """
    from pyvmancer.ccmap import cc_map_report  # pylint: disable=import-outside-toplevel

    from syncsummoner.compose import locktake, render, session  # pylint: disable=import-outside-toplevel
    from syncsummoner.compose.timecode import clip_layout  # pylint: disable=import-outside-toplevel
    from syncsummoner.device.recorder import (  # pylint: disable=import-outside-toplevel
        Recorder,
        await_transmitting,
        require_input,
        require_route,
        require_signal,
    )
    from syncsummoner.device.v4l2rec import read_log, sidecar  # pylint: disable=import-outside-toplevel

    outs = [Path(out) for out in outs]
    rig = render.open_rig(config, capture=False)
    shell = rig.transport.device.shell

    def state() -> dict:
        report = cc_map_report(shell)
        return {
            "firmware": rig.transport.firmware(),
            "boot_count": locktake.boot_count(shell),
            "cc_map_ok": None if report is None else bool(report.ok),
        }

    def load(name: str) -> Any:
        rig.session.load_program(name, link=rig.link)
        require_route(rig.transport, program=name, route=config.route)
        await_transmitting(rig.transport, program=name, timing=config.timing)
        require_input(rig.transport, program=name, source=config.source)
        if name == program:
            record["program_info"] = rig.transport.program_payload()
        return rig.transport.program_info()

    record: dict = {"program": program}
    try:
        record["before"] = state()
        try:
            require_route(rig.transport, program=program, route=config.route)
            rig.playout.upload(str(clip))
            session.locked_session(
                rig,
                Recorder(width=config.width, height=config.height, fps=config.fps),
                [session.Take(out, chunk.plan, chunk.frames) for out, chunk in zip(outs, chunks)],
                program=program,
                config=config,
                working=lambda _info: {},
                load=load,
                clip=clip_layout(clip),
            )
            rig.transport.verify_cc_map()
            require_signal(rig.transport, when="as the sweep ended", program=program, timing=config.timing)
        except Exception as err:  # pylint: disable=broad-exception-caught
            record["fault"] = f"{type(err).__name__}: {err}"
        record["after"] = state()
        if record["after"]["boot_count"] != record["before"]["boot_count"]:
            record.setdefault("fault", "RebootError: the boot count rose across the session")
        if not record["after"]["cc_map_ok"]:
            record.setdefault("fault", "CcMapError: the CC map changed across the session")
    finally:
        rig.transport.close()
    said = []
    for out, chunk in zip(outs, chunks):
        mine = {**record, "chunk": chunk.first, "writes": int(chunk.plan.size), "session_takes": len(outs)}
        seq = read_log(out)["sequence"].astype(np.int64) if sidecar(out).is_file() else np.zeros(0, np.int64)
        lost = int(np.clip(np.diff(seq) - 1, 0, None).sum())
        mine["frames_lost_by_driver"] = lost
        if lost:
            mine.setdefault("fault", f"TakeFaultError: the capture driver lost {lost} frames")
        if "fault" not in mine:
            mine.update(chroma_record(out, clip, config))
        said.append(mine)
    return said


def chroma_record(take: str | Path, clip: str | Path, config: Any) -> dict:
    """A captured take's dry lead-in channel order, and a fault when its chroma pair came back exchanged.

    The fault sends the take back for another session: a program that drops its output's active-video flag
    pairs its chroma afresh on every load, so the pair can come back exchanged for a whole session.
    """
    from syncsummoner.compose import chroma as lead_in  # pylint: disable=import-outside-toplevel
    from syncsummoner.compose.timecode import clip_layout  # pylint: disable=import-outside-toplevel
    from syncsummoner.probe import decoded  # pylint: disable=import-outside-toplevel

    ffmpeg = getattr(config, "ffmpeg", "ffmpeg")
    try:
        found = lead_in.lead_in_order(
            take,
            clip,
            spec=decoded.Spec.of(config),
            fps=config.fps,
            width=config.width,
            lead=clip_layout(clip, ffprobe=decoded.ffprobe_for(ffmpeg)).lead,
            ffmpeg=ffmpeg,
        )
    except lead_in.UNREADABLE as err:
        LOG.warning("%s: no channel order from the lead-in: %s", take, err)
        return {"chroma": {"error": f"{type(err).__name__}: {err}"}}
    said: dict = {"chroma": found.as_dict()}
    if found.judged and not found.ok:
        said["fault"] = f"ChromaOrderError: the dry lead-in came back {found.order}"
    return said


def collect(
    take: str | Path,
    chunk: Chunk,
    *,
    clip: str | Path,
    config: Any,
    workers: int | None = None,
    settle: dict | None = None,
) -> dict:
    """Judge a sweep take, keeping its evidence, and its archive rows beside it; a summary of both.

    ``settle`` says which settle the rows waited and where it was measured; it is kept in the evidence.
    """
    from syncsummoner.compose.judge import judge_take  # pylint: disable=import-outside-toplevel
    from syncsummoner.device.v4l2rec import read_log  # pylint: disable=import-outside-toplevel
    from syncsummoner.device.wire import load_wire  # pylint: disable=import-outside-toplevel

    program = str(load_wire(take)["program"])
    seen = judge_take(take, clip=clip, config=config, program=program, workers=workers)
    rows = settled_rows(
        chunk,
        seen.proof.landed.rows,
        picked=seen.found.picked,
        shown=seen.found.shown,
        captured=read_log(take)["t_ns"].astype(np.float64) / 1e9,
        lead=seen.found.source_offset,
        program=program,
        bad_frames=disagreeing(seen.proof.applied),
        ambiguity=getattr(seen.found, "ambiguity", None),
    )
    from syncsummoner.compose import frameline  # pylint: disable=import-outside-toplevel
    from syncsummoner.compose.lock import save_clock  # pylint: disable=import-outside-toplevel
    from syncsummoner.probe import decoded  # pylint: disable=import-outside-toplevel
    from syncsummoner.probe.footage import source_series  # pylint: disable=import-outside-toplevel

    log = read_log(take)
    clock = [dict(said) for said in getattr(seen.found, "clock", ())]
    if clock:
        source = decoded.features(clip, decoded.Spec.of(config), workers=workers)
        series = source_series(source, strip_px=config.strip_px)
        rows = number_runs(
            rows,
            clock,
            frameline.segments(log["t_ns"], log["sequence"]),
            seen.features.luma,
            series,
            lead=seen.found.source_offset,
        )
        save_clock(take, clock)
    save_rows(take, rows)
    _note_settle(take, settle or {"frames": int(chunk.settle), "source": "plan"})
    kept = {r.setpoint for r in rows if r.setpoint != GAP}
    judged = verdict(chunk, seen.proof.landed, disagreeing(seen.proof.applied), seen.found.picked.size)
    return {
        "take": str(take),
        "program": program,
        "evidence_ok": judged.ok,
        "verdict": judged.as_dict(),
        "settle": settle or {"frames": int(chunk.settle), "source": "plan"},
        "landed": seen.proof.landed.summary(),
        "applied": seen.proof.applied.summary(),
        "received": None if seen.proof.received is None else seen.proof.received.summary(),
        "setpoints": len(chunk.states),
        "setpoints_kept": len(kept),
        "rows_measured": sum(1 for r in rows if r.setpoint != GAP),
        "numbering": [{k: c.get(k) for k in ("run", "candidates", "selection")} for c in clock],
    }


def number_runs(
    rows: list[FrameRow],
    clock: list[dict],
    seg: np.ndarray,
    luma: np.ndarray,
    series: tuple,
    *,
    lead: int = 0,
) -> list:
    """Each clock-numbered run's one numbering, chosen jointly over its settings where their follow says.

    A run the clock leaves several candidates is numbered by the highest; every shift back to the lowest is
    scored over the run's settings at that shift's own leads and pooled (:func:`replay.pooled_follow`). A
    shift that beats every other (:func:`replay.select_shift`) renumbers the run's rows, ambiguity none;
    otherwise they keep it. Each run's ``clock`` record gains its ``selection``. ``series`` is the clip's
    :func:`footage.source_series`, and rows' played frames reach it through :func:`footage.source_frame`.
    """
    from syncsummoner.probe.footage import source_frame  # pylint: disable=import-outside-toplevel
    from syncsummoner.probe.replay import (  # pylint: disable=import-outside-toplevel
        _measured,
        pooled_follow,
        select_shift,
    )

    setpoint = np.array([r.setpoint for r in rows], dtype=np.int64)
    shown = np.array([r.shown for r in rows], dtype=np.int64)
    frames = list(luma)
    for said in clock:
        found = said["candidates"]
        span = max(found) - min(found)
        if not span:
            said["selection"] = {"shift": 0, "numbering": max(found)}
            continue
        mine = (seg[: len(rows)] == said["run"]) & (setpoint != GAP) & (shown >= 0)
        measured = [
            _measured(frames, members, source_frame(shown[members], lead), series, len(series[0]))
            for k in np.unique(setpoint[mine])
            for members in [np.flatnonzero(mine & (setpoint == k))]
        ]
        chosen = select_shift(pooled_follow(measured, span))
        said["selection"] = {
            **chosen,
            "numbering": None if chosen["shift"] is None else max(found) - chosen["shift"],
        }
        if chosen["shift"] is not None:
            at = np.flatnonzero(seg[: len(rows)] == said["run"])
            for c in at:
                moved = rows[c].shown - chosen["shift"] if rows[c].shown >= 0 else rows[c].shown
                rows[c] = replace(rows[c], shown=moved, ambiguity=0)
    return rows


def _note_settle(take: str | Path, settle: dict) -> None:
    """Add the settle a take's rows waited, and its source, to the take's evidence report."""
    from syncsummoner.compose.locktake import evidence_path  # pylint: disable=import-outside-toplevel

    path = evidence_path(take)
    report = json.loads(path.read_text(encoding="utf-8")) if path.is_file() else {}
    path.write_text(json.dumps({**report, "settle": settle}, indent=1), encoding="utf-8")


def disagreeing(applied: Any) -> np.ndarray:
    """Clip frames at which a readback disagreed with the writes landed by then."""
    return np.unique(np.asarray(applied.disagree["frame"], dtype=np.int64))

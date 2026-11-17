"""An archive sweep as frame-locked takes: one level a hold, every frame attributed to the clip.

The cover's setpoints are targeted one hold apart and cut into
takes no longer than the clip. Each take's evidence lands every write on a frame; the frames each
setpoint settled over become its archive rows, and a program's takes join into one archive.
"""

from __future__ import annotations

import json
import subprocess
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Sequence

import numpy as np

from syncsummoner.compose.locktake import PLAN
from syncsummoner.device.profile import PARAM_COUNT, snap_toggles
from syncsummoner.probe.archive import GAP, MJPEG_PIX_FMT, FrameRow
from syncsummoner.probe.fit import load_measurements, save_measurements
from syncsummoner.probe.store import atomic_write
from syncsummoner.program.gates import tour

__all__ = [
    "Chunk",
    "SweepError",
    "capture",
    "carried",
    "collect",
    "commit_sweep",
    "concat",
    "disagreeing",
    "joint",
    "levels",
    "owners",
    "plan_sweep",
    "read_plan",
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


def plan_sweep(states: np.ndarray, *, settle: int, hold: int, frames: int) -> list[Chunk]:
    """Cut a sweep into takes of at most ``frames`` clip frames, each setpoint one ``hold`` after the last.

    A setpoint's changed slots all target one frame; when they land is what the take realizes, and
    :func:`settled_rows` keeps only the frames after it. A take opens on its setpoint as the start state.
    """
    states = np.asarray(states, dtype=np.int64)
    if hold > frames:
        raise SweepError(f"a hold of {hold} frames does not fit a take of {frames}")
    out, k = [], 0
    while k < len(states):
        rows = [(0, p + 1, int(v)) for p, v in enumerate(states[k])]
        owner, j = [0] * PARAM_COUNT, k + 1
        while j < len(states) and (j - k + 1) * hold <= frames:
            slots = np.flatnonzero(states[j] != states[j - 1]) + 1
            rows += [((j - k) * hold, int(s), int(states[j][s - 1])) for s in slots]
            owner += [j - k] * slots.size
            j += 1
        out.append(
            Chunk(
                first=k,
                states=states[k:j],
                plan=np.array(rows, dtype=PLAN),
                owner=np.asarray(owner, dtype=np.int64),
                frames=(j - k) * hold,
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
) -> list[FrameRow]:
    """One archive row per capture, under the state the take realized: its settled setpoint, else :data:`GAP`.

    A frame belongs to the setpoint whose writes may have started landing last by then, once
    ``chunk.settle`` frames have passed since the end of its writes' latest landing window, and only
    where the realized state is that setpoint's. Frames in ``bad_frames`` keep no setpoint. The
    dry control's writes around the clip, in the lead-in and from the lead-out on, own no setpoint.
    """
    rows = landed[(landed["target"] > 0) & (landed["target"] < chunk.frames)]
    owner = owners(chunk, rows)
    count, n = picked.size, len(chunk.states)
    state = realized(chunk.states[0], rows, count)
    sent = rows["sent"] >= 0
    first, last = np.full(n, count, dtype=np.int64), np.zeros(n, dtype=np.int64)
    first[0] = 0
    np.minimum.at(first, owner[sent], rows["first"][sent])
    np.maximum.at(last, owner[sent], rows["last"][sent])
    order = np.argsort(first, kind="stable")
    frame = np.arange(count)
    k = order[np.clip(np.searchsorted(first[order], frame, side="right") - 1, 0, n - 1)]
    ok = (frame >= last[k] + chunk.settle) & np.all(state == chunk.states[k], axis=1)
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


def concat(
    videos: Sequence[str | Path],
    out: str | Path,
    *,
    ffmpeg: str = "ffmpeg",
    run: Callable[..., Any] = subprocess.run,
) -> None:
    """Join ``videos`` into ``out`` packet for packet, with no re-encode."""
    with tempfile.NamedTemporaryFile("w", suffix=".txt", dir=Path(out).parent, delete=False) as listing:
        listing.write("".join(f"file '{Path(v).resolve()}'\n" for v in videos))
    try:
        argv = [ffmpeg, "-v", "error", "-y", "-f", "concat", "-safe", "0", "-i", listing.name]
        result = run(argv + ["-map", "0:v", "-c", "copy", str(out)], capture_output=True, check=False)
    finally:
        Path(listing.name).unlink(missing_ok=True)
    if result.returncode:
        raise SweepError(f"joining {len(videos)} takes failed: {result.stderr!r}")


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
    ffmpeg: str = "ffmpeg",
    run: Callable[..., Any] = subprocess.run,
) -> dict:
    """Join a program's judged takes, in chunk order, into its archive: one video, one row per capture."""
    rows: list[FrameRow] = []
    for take in takes:
        offset = len(rows)
        rows += [
            FrameRow.from_row({**r, "frame": int(r["frame"]) + offset})
            for r in load_measurements(rows_path(take))
        ]
    states = np.concatenate([c.states for c in chunks])
    video = archive.scratch(program)
    concat(takes, video, ffmpeg=ffmpeg, run=run)
    return archive.commit(
        program,
        key,
        video,
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
    config: Any, chunk: Chunk, out: str | Path, *, program: str, clip: str | Path
) -> dict:  # pragma: no cover - needs the rig
    """Record one chunk as a locked take of ``program`` on the clip; its device state and any fault."""
    from pyvmancer.ccmap import cc_map_report  # pylint: disable=import-outside-toplevel

    from syncsummoner.compose import locktake, render  # pylint: disable=import-outside-toplevel
    from syncsummoner.device.recorder import (  # pylint: disable=import-outside-toplevel
        Recorder,
        await_transmitting,
        require_input,
        require_route,
        require_signal,
    )
    from syncsummoner.device.v4l2rec import read_log  # pylint: disable=import-outside-toplevel

    out = Path(out)
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

    record: dict = {"program": program, "chunk": chunk.first, "writes": int(chunk.plan.size)}
    try:
        record["before"] = state()
        try:
            require_route(rig.transport, program=program, route=config.route)
            rig.playout.upload(str(clip))
            locktake.locked_take(
                rig,
                Recorder(width=config.width, height=config.height, fps=config.fps),
                out,
                program=program,
                seconds=chunk.frames / config.fps,
                config=config,
                plan=chunk.plan,
                working=lambda _info: {},
                load=load,
            )
            rig.transport.verify_cc_map()
            require_signal(rig.transport, when="as the sweep ended", program=program, timing=config.timing)
        except Exception as err:  # pylint: disable=broad-exception-caught
            record["fault"] = f"{type(err).__name__}: {err}"
        lost = int(np.clip(np.diff(read_log(out)["sequence"].astype(np.int64)) - 1, 0, None).sum())
        record["frames_lost_by_driver"] = lost
        record["after"] = state()
        if lost:
            record.setdefault("fault", f"TakeFaultError: the capture driver lost {lost} frames")
        if record["after"]["boot_count"] != record["before"]["boot_count"]:
            record.setdefault("fault", "RebootError: the boot count rose across the take")
        if not record["after"]["cc_map_ok"]:
            record.setdefault("fault", "CcMapError: the CC map changed across the take")
    finally:
        rig.transport.close()
    return record


def collect(
    take: str | Path, chunk: Chunk, *, clip: str | Path, config: Any, workers: int | None = None
) -> dict:
    """Judge a sweep take, keeping its evidence, and its archive rows beside it; a summary of both."""
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
        lead=seen.found.lead,
        program=program,
        bad_frames=disagreeing(seen.proof.applied),
    )
    save_rows(take, rows)
    kept = {r.setpoint for r in rows if r.setpoint != GAP}
    return {
        "take": str(take),
        "program": program,
        "evidence_ok": bool(seen.proof.ok),
        "landed": seen.proof.landed.summary(),
        "applied": seen.proof.applied.summary(),
        "received": None if seen.proof.received is None else seen.proof.received.summary(),
        "setpoints": len(chunk.states),
        "setpoints_kept": len(kept),
        "rows_measured": sum(1 for r in rows if r.setpoint != GAP),
    }


def disagreeing(applied: Any) -> np.ndarray:
    """Clip frames at which a readback disagreed with the writes landed by then."""
    return np.unique(np.asarray(applied.disagree["frame"], dtype=np.int64))

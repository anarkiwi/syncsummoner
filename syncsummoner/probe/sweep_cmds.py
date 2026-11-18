"""``probe hold``, ``probe timecode`` and ``probe sweep``: an archive sweep as frame-locked takes."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from syncsummoner.progress import LOG


def _hold_cmd(args: argparse.Namespace) -> int:
    """Each program's settle and hold from finished locked takes, written as ``hold-<program>.json``."""
    from syncsummoner.compose.render import RenderConfig
    from syncsummoner.compose.response import of_takes
    from syncsummoner.probe.store import slug

    config = RenderConfig.for_format(args.format, strip_px=args.strip_px)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    found = of_takes(args.takes, clip=args.source, config=config, workers=args.workers)
    for program, said in sorted(found.items()):
        said.save(out / f"hold-{slug(program)}.json")
        print(said.table())
    return 0 if all(r.bounded for r in found.values()) else 1


def _timecode_cmd(args: argparse.Namespace) -> int:
    """Timecode footage for sweep sessions, its lead-in and lead-out each holding the longest load a device
    wait allows (the stall guard), so no load the rig lets through outgrows the clip."""
    from syncsummoner.compose.locktake import LeadIn
    from syncsummoner.compose.render import RenderConfig
    from syncsummoner.compose.session import session_frames
    from syncsummoner.compose.timecode import write_timecoded
    from syncsummoner.device.loadtimes import stall_guard

    config = RenderConfig.for_format(args.format, strip_px=args.strip_px)
    load_s = stall_guard(args.stall)
    lead, tail = session_frames(LeadIn(), config.fps, dry=False, load_s=load_s)
    frames = write_timecoded(args.source, args.out, config=config, lead=lead, tail=tail)
    print(f"{args.out}: {frames} frames, {lead} lead-in and {tail} lead-out frames, for a {load_s} s load")
    return 0


def _sweep_plan_cmd(args: argparse.Namespace) -> int:
    """Cut a program's covering sweep into locked takes one hold a level, from its measured hold."""
    from syncsummoner.compose.timecode import clip_layout
    from pyvmancer.midi import ParamKind, param_kind

    from syncsummoner.device.profile import PARAM_COUNT
    from syncsummoner.probe.locked import carried, joint, levels, plan_path, plan_sweep, recorded, write_plan
    from syncsummoner.probe.store import slug
    from syncsummoner.program import model
    from syncsummoner.program.gates import Gating, cover, representatives

    held = held_record(args.hold, args.program)
    frames = clip_layout(args.source).frames
    m = model(args.program)
    toggles = [p for p in range(1, PARAM_COUNT + 1) if param_kind(p) is ParamKind.TOGGLE]
    values = (
        {p: representatives(m, p, bits) for p, bits in m.resolution().items() if bits}
        if args.cover or args.joint
        else {}
    )
    if args.cover:
        states = recorded(json.loads(Path(args.cover).read_text(encoding="utf-8")), values)
    else:
        states = levels(cover(m, gating=Gating(m)))
    for selector, mover in (tuple(int(x) for x in pair.split(":")) for pair in args.joint):
        states = joint(states, selector, mover, values)
    states = carried(states, toggles)
    chunks = plan_sweep(states, settle=int(held["settle"]), hold=int(held["hold"]), frames=frames)
    path = write_plan(
        plan_path(args.takes, slug(args.program)),
        chunks,
        program=args.program,
        response=held,
        source=args.source,
    )
    print(
        f"{args.program}: {len(states)} setpoints at a hold of {held['hold']} frames, "
        f"{len(chunks)} takes of {[c.frames for c in chunks]} frames; plan in {path}"
    )
    return 0


def clip_ti(luma: Any, lead: int, tail: int) -> Any:
    """TI of each clip frame against the frame played before it, the lead-in's last for the first."""
    from syncsummoner.aesthetics.survival import information

    return information(luma[lead - 1 : len(luma) - tail])[1][1:]


def downbeat_frames(audio: str, fps: float) -> Any:
    """The track's downbeats as clip frames, the clip played from the track's start."""
    import numpy as np

    from syncsummoner.compose.features import beat_track, downbeat_times, load_audio, onset_strength

    y, sr = load_audio(audio)
    hop = 512
    env = onset_strength(y, sr, hop=hop)
    beats, _ = beat_track(env, sr / hop)
    return np.rint(downbeat_times(beats, env, sr / hop) * fps).astype(np.int64)


def _sweep_regions_cmd(args: argparse.Namespace) -> int:
    """Plan a program's region sweep: Sobol settings in each regime, each held as the follow test needs."""
    import numpy as np
    from pyvmancer.midi import ParamKind, param_kind

    from syncsummoner.compose.locktake import LeadIn
    from syncsummoner.compose.session import session_frames
    from syncsummoner.compose.render import RenderConfig
    from syncsummoner.compose.timecode import clip_layout
    from syncsummoner.device.profile import PARAM_COUNT
    from syncsummoner.probe import decoded
    from syncsummoner.probe.locked import joint, plan_path, write_plan
    from syncsummoner.probe.regions import (
        allot,
        chunks_of,
        design,
        distinct,
        guard_frames,
        region_dims,
        second_order,
        slots,
        without_bypass,
    )
    from syncsummoner.probe.store import slug
    from syncsummoner.program import model
    from syncsummoner.program.gates import representatives

    config = RenderConfig.for_format(args.format, strip_px=args.strip_px)
    held = (
        held_record(args.hold, args.program)
        if args.hold
        else {"program": args.program, "settle": 0, "measured": False}
    )
    record = json.loads(Path(args.cover).read_text(encoding="utf-8"))
    clip = decoded.features(args.source, decoded.Spec.of(config), workers=args.workers)
    layout = clip_layout(args.source)
    ti = clip_ti(clip.luma, layout.lead, layout.tail)
    edges = downbeat_frames(args.audio, config.fps)
    guard = guard_frames(int(held["settle"]))
    m = model(args.program)
    ends = session_frames(LeadIn(), config.fps, dry=m.bypass() is not None)
    overhead_s = sum(ends) / config.fps + config.tail_s
    values = {p: representatives(m, p, bits) for p, bits in m.resolution().items() if bits}
    states = np.asarray(record["states"], dtype=np.int64)
    chosen = [int(i) for i in record["chosen"]]
    kept = without_bypass(values, m.bypass())
    dims = region_dims(record, kept)
    extra = np.zeros((0, PARAM_COUNT), dtype=np.int64)
    for selector, mover in (tuple(int(x) for x in pair.split(":")) for pair in args.joint):
        extra = np.r_[extra, joint(states[chosen[:1]], selector, mover, values)[1:]]
    if args.budget_s is None:
        counts = [min(second_order(len(d)), distinct(d, kept)) for d in dims]
        placed = slots(
            ti, edges, guard=guard, overhead_s=overhead_s, fps=config.fps, limit=sum(counts) + len(extra)
        )
    else:
        placed = slots(ti, edges, guard=guard, budget_s=args.budget_s, overhead_s=overhead_s, fps=config.fps)
        counts = [
            min(n, distinct(d, kept))
            for n, d in zip(allot(max(len(placed) - len(extra), 0), [len(d) for d in dims]), dims)
        ]
    sampled = [
        design(states[i], d, kept, n, args.seed + k) for k, (i, d, n) in enumerate(zip(chosen, dims, counts))
    ]
    every = np.concatenate([*sampled, extra]) if sampled else extra
    capacity, placed = len(placed), placed[: len(every)]
    toggles = [p for p in range(1, PARAM_COUNT + 1) if param_kind(p) is ParamKind.TOGGLE]
    chunks = chunks_of(every, placed, settle=int(held["settle"]), toggles=toggles)
    hold = np.array([s.hold for s in placed])
    regions = {
        "settings": int(len(every)),
        "capacity": capacity,
        "per_regime": dict(zip((str(i) for i in chosen), counts)),
        "dims": dict(zip((str(i) for i in chosen), dims)),
        "joint": int(len(extra)),
        "guard_frames": guard,
        "hold_frames": {"min": int(hold.min()), "median": float(np.median(hold)), "max": int(hold.max())},
        "budget_s": args.budget_s,
        "rig_s": float(sum(c.frames for c in chunks) / config.fps + overhead_s),
        "seed": args.seed,
    }
    path = write_plan(
        plan_path(args.takes, slug(args.program)),
        chunks,
        program=args.program,
        response=held,
        source=args.source,
        regions=regions,
    )
    print(
        f"{args.program}: {regions['settings']} settings ({regions['per_regime']} by regime, "
        f"{len(extra)} joint) of {capacity} the budget holds, "
        f"hold {regions['hold_frames']} frames after a guard of {guard}, {len(chunks)} takes, "
        f"{regions['rig_s']:.0f} s of rig time; plan in {path}"
    )
    return 0


def _sweep_take(directory: str, program: str, chunk: int) -> Path:
    from syncsummoner.probe.store import slug

    return Path(directory, f"take-{slug(program)}-{chunk:02d}.mkv")


def _sweep_cmd(args: argparse.Namespace) -> int:
    """Capture a plan's chunks back to back as one session, or one chunk alone; 2 on any take's fault."""
    from syncsummoner.compose.render import RenderConfig
    from syncsummoner.probe.locked import capture, read_plan

    chunks, said = read_plan(args.plan)
    config = RenderConfig.for_format(
        args.format, strip_px=args.strip_px, source_host=args.source_host, stall_s=args.stall
    )
    picked = range(len(chunks)) if args.chunk is None else [args.chunk]
    outs = [_sweep_take(Path(args.plan).parent, said["program"], k) for k in picked]
    records = capture(config, [chunks[k] for k in picked], outs, program=said["program"], clip=said["source"])
    for k, out, record in zip(picked, outs, records):
        meta = out.with_name(out.name + ".meta.json")
        kept = json.loads(meta.read_text(encoding="utf-8")) if meta.is_file() else {}
        kept[f"attempt-{args.attempt}"] = record
        meta.write_text(json.dumps(kept, indent=1, default=str), encoding="utf-8")
        LOG.info("%s chunk %d: %s", said["program"], k, record.get("fault", "captured"))
    return 2 if any("fault" in r for r in records) else 0


def _smoke_cmd(args: argparse.Namespace) -> int:
    """A few frames toggling one parameter, captured as a sweep is, then landed from its wire alone."""
    import numpy as np

    from syncsummoner.compose.evidence import START
    from syncsummoner.compose.locktake import landing
    from syncsummoner.device.wire import load_wire
    from syncsummoner.probe.locked import plan_sweep, read_plan, write_plan
    from syncsummoner.probe.store import slug

    chunks, said = read_plan(args.plan)
    states = np.tile(np.asarray(chunks[0].states[0], dtype=np.int64), (args.states, 1))
    states[0::2, args.param - 1], states[1::2, args.param - 1] = args.values
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    plan = write_plan(
        out / f"sweep-{slug(said['program'])}.json",
        plan_sweep(states, settle=0, hold=args.hold, frames=args.states * args.hold),
        **{k: v for k, v in said.items() if k != "regions"},
    )
    code = _sweep_cmd(argparse.Namespace(**{**vars(args), "plan": str(plan), "chunk": 0, "attempt": 1}))
    wire = load_wire(_sweep_take(str(out), said["program"], 0))
    shift, latch, landed = landing(wire, 1e9 / 30.0)
    latch = latch[wire["sends"]["target"] > START] if "sends" in wire else latch
    print(f"{said['program']}: device frames lead clip frames by {shift.lo}..{shift.hi}")
    print(f"{said['program']}: {landed.summary()}")
    for row in landed.rows:
        window = latch[row["sent"]].tolist() if row["sent"] >= 0 else None
        print(
            f"  target {row['target']:4d} P{row['param']}={row['value']:4d}: "
            f"device {window}, clip {row['first']}..{row['last']}"
        )
    return code or (0 if landed.ok and shift.exact else 3)


def held_record(hold: str | Path, program: str) -> dict:
    """A ``probe hold`` record, refused when it measured another program."""
    held = json.loads(Path(hold).read_text(encoding="utf-8"))
    if held["program"] != program:
        raise ValueError(f"{hold} measured {held['program']}, not {program}")
    return held


def settled_by(
    chunk: Any, program: str, hold: str | None, plan: str, *, measured: bool = True
) -> tuple[Any, dict]:
    """``chunk`` with the settle a ``probe hold`` record measured, else its plan's; where it came from, and
    whether it was measured: a plan made with no hold record (``measured`` false) waited none."""
    from dataclasses import replace  # pylint: disable=import-outside-toplevel

    if hold is None:
        return chunk, {"frames": int(chunk.settle), "source": str(plan), "measured": bool(measured)}
    held = held_record(hold, program)
    settle = int(held["settle"])
    return replace(chunk, settle=settle), {"frames": settle, "source": str(hold), "measured": True}


def _sweep_collect_cmd(args: argparse.Namespace) -> int:
    """Judge one captured sweep take and keep its archive rows beside it."""
    from syncsummoner.compose.render import RenderConfig
    from syncsummoner.probe.locked import collect, read_plan

    chunks, said = read_plan(args.plan)
    config = RenderConfig.for_format(args.format, strip_px=args.strip_px)
    take = _sweep_take(Path(args.plan).parent, said["program"], args.chunk)
    measured = said.get("response", {}).get("measured", True)
    chunk, settle = settled_by(chunks[args.chunk], said["program"], args.hold, args.plan, measured=measured)
    summary = collect(take, chunk, clip=said["source"], config=config, workers=args.workers, settle=settle)
    print(json.dumps(summary, indent=1))
    return 0 if summary["evidence_ok"] else 1


def settle_measured(take: str | Path) -> bool:
    """Whether a collected take's rows waited a measured settle, as its evidence report records."""
    from syncsummoner.compose.locktake import evidence_path  # pylint: disable=import-outside-toplevel

    path = evidence_path(take)
    report = json.loads(path.read_text(encoding="utf-8")) if path.is_file() else {}
    return bool(report.get("settle", {}).get("measured", True))


def _sweep_commit_cmd(args: argparse.Namespace) -> int:
    """Join a program's collected sweep takes into its archive, keyed on name and firmware."""
    import hashlib

    from syncsummoner.compose.render import RenderConfig
    from syncsummoner.compose.timecode import clip_layout
    from syncsummoner.probe import decoded
    from syncsummoner.probe.archive import FrameArchive
    from syncsummoner.probe.locked import commit_sweep, read_plan
    from syncsummoner.probe.store import KeyKind, ProgramKey

    chunks, said = read_plan(args.plan)
    config = RenderConfig.for_format(args.format, strip_px=args.strip_px)
    program = said["program"]
    takes = [_sweep_take(Path(args.plan).parent, program, k) for k in range(len(chunks))]
    unmeasured = [str(t) for t in takes if not settle_measured(t)]
    if unmeasured:
        raise ValueError(
            f"{program}'s rows in {unmeasured} waited no measured settle: measure the program's hold with "
            "`probe hold` and collect them again with --hold"
        )
    metas = [json.loads(t.with_name(t.name + ".meta.json").read_text(encoding="utf-8")) for t in takes]
    good = [next(a for _, a in sorted(m.items()) if "fault" not in a) for m in metas]
    firmware = {str(a["before"]["firmware"]) for a in good}
    if len(firmware) != 1:
        raise ValueError(f"{program}'s takes ran on firmware {sorted(firmware)}")
    archive = FrameArchive(args.archive)
    archive.directory.mkdir(parents=True, exist_ok=True)
    clip = Path(said["source"])
    shown = archive.directory / clip.name
    if not shown.exists():
        shown.symlink_to(clip.resolve())
    with clip.open("rb") as handle:
        digest = hashlib.file_digest(handle, "sha256").hexdigest()
    total = int(decoded.features(clip, decoded.Spec.of(config)).small.shape[0])
    stimulus = {
        "kind": "footage",
        "clip": clip.name,
        "sha256": digest,
        "frames": total,
        "fps": config.fps,
        "bits": config.bits,
        "strip_px": config.strip_px,
        "lead": clip_layout(clip).lead,
        "source": str(clip),
    }
    meta = commit_sweep(
        archive,
        program,
        ProgramKey(program, firmware.pop(), KeyKind.NAME_FIRMWARE),
        takes,
        chunks,
        width=config.width,
        height=config.height,
        fps=config.fps,
        program_info=good[0].get("program_info"),
        stimulus=stimulus,
    )
    print(
        json.dumps({k: meta[k] for k in ("program", "frames", "measured", "key_material", "video")}, indent=1)
    )
    return 0


def _add_timecode(probe_sub: Any) -> Any:
    stamp = probe_sub.add_parser(
        "timecode", help="timecode footage for sweep sessions, sized by the stall guard"
    )
    stamp.add_argument("source", help="the footage")
    stamp.add_argument("--out", required=True, help="the timecoded clip to write")
    stamp.set_defaults(func=_timecode_cmd)
    return stamp


def _add_smoke(sweep_sub: Any) -> Any:
    smoke = sweep_sub.add_parser(
        "smoke", help="capture a few seconds toggling one parameter and land every write by command order"
    )
    smoke.add_argument("plan", help="a sweep plan of the program, for its start state, clip and program")
    smoke.add_argument("--param", type=int, required=True, help="the 1-based parameter toggled")
    smoke.add_argument(
        "--values", type=int, nargs=2, required=True, help="the two device values toggled between"
    )
    smoke.add_argument("--states", type=int, default=8, help="settings, each held --hold frames")
    smoke.add_argument("--hold", type=int, default=8)
    smoke.add_argument("--out", required=True, help="where the plan and take go")
    smoke.add_argument("--source-host", help="ssh target driving playout and the HDMI link")
    smoke.set_defaults(func=_smoke_cmd)
    return smoke


def add_parsers(probe_sub: Any) -> None:
    """The ``probe hold``, ``probe timecode`` and ``probe sweep`` commands."""
    hold = probe_sub.add_parser("hold", help="each program's settle and archive hold from locked takes")
    hold.add_argument("takes", nargs="+")
    hold.add_argument("--source", required=True, help="the timecoded clip the takes played")
    hold.add_argument("--out", default=".", help="where hold-<program>.json goes")
    stamp = _add_timecode(probe_sub)
    sweep = probe_sub.add_parser(
        "sweep", help="an archive sweep as locked takes: plan, capture, collect, commit"
    )
    sweep_sub = sweep.add_subparsers(dest="sweep_cmd", required=True)
    plan = sweep_sub.add_parser("plan", help="cut a program's covering sweep into takes")
    plan.add_argument("program")
    plan.add_argument("--hold", required=True, help="the program's hold-<program>.json")
    plan.add_argument("--source", required=True, help="the timecoded clip the takes will play")
    plan.add_argument("--takes", required=True, help="the sweep's takes directory")
    plan.add_argument(
        "--structural", action="store_true", help="accepted for scripts: every cover is by observability"
    )
    plan.add_argument("--cover", help="a cover as `program cover --json` wrote it, in place of searching one")
    plan.add_argument(
        "--joint",
        action="append",
        default=[],
        metavar="SELECTOR:MOVER",
        help="also hold SELECTOR at each setting with a MOVER move under each (repeatable)",
    )
    plan.set_defaults(func=_sweep_plan_cmd)
    regions = sweep_sub.add_parser(
        "regions", help="plan a region sweep: Sobol settings per regime, held to judge motion"
    )
    regions.add_argument("program")
    regions.add_argument("--cover", required=True, help="a cover as `program cover --json` wrote it")
    regions.add_argument(
        "--hold",
        help="the program's hold-<program>.json; without one each setting's guard is its landing window "
        "alone, and its takes' rows must be collected again with a measured hold before they are committed",
    )
    regions.add_argument("--source", required=True, help="the timecoded clip the takes will play")
    regions.add_argument("--audio", required=True, help="the track the clip is scored to, for its bars")
    regions.add_argument("--takes", required=True, help="the sweep's takes directory")
    regions.add_argument(
        "--budget-s",
        type=float,
        help="rig seconds the program's takes may use; without it each regime gets the settings a "
        "second-order model of its parameters needs, at most their distinct settings",
    )
    regions.add_argument("--seed", type=int, default=0, help="seed of the scrambled Sobol design")
    regions.add_argument("--joint", action="append", default=[], metavar="SELECTOR:MOVER", help="as for plan")
    regions.set_defaults(func=_sweep_regions_cmd)
    smoke = _add_smoke(sweep_sub)
    run = sweep_sub.add_parser(
        "capture", help="capture a plan's takes back to back on the rig, its program loaded once"
    )
    run.add_argument("plan")
    run.add_argument("--chunk", type=int, help="capture only this take, alone, as for a recapture")
    run.add_argument("--attempt", type=int, default=1)
    run.add_argument("--source-host", help="ssh target driving playout and the HDMI link")
    run.set_defaults(func=_sweep_cmd)
    gather = sweep_sub.add_parser("collect", help="judge one captured take and keep its archive rows")
    gather.add_argument("plan")
    gather.add_argument("--chunk", type=int, required=True)
    gather.add_argument("--hold", help="a hold-<program>.json whose measured settle overrides the plan's")
    gather.set_defaults(func=_sweep_collect_cmd)
    commit = sweep_sub.add_parser("commit", help="join a program's collected takes into its archive")
    commit.add_argument("plan")
    commit.add_argument("--archive", required=True)
    commit.set_defaults(func=_sweep_commit_cmd)
    for node in (hold, stamp, plan, regions, run, smoke, gather, commit):
        node.add_argument("--format", default="1080p30", help="the session format")
        node.add_argument("--strip-px", type=int, default=8, help="rows of timecode strip along the bottom")
    for node in (hold, plan, regions, run, gather, commit):
        node.add_argument("--workers", type=int, help="decoders at once")
    hold.set_defaults(func=_hold_cmd)

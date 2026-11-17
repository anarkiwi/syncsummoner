"""``probe hold`` and ``probe sweep``: an archive sweep as frame-locked takes, from plan to archive."""

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


def _sweep_plan_cmd(args: argparse.Namespace) -> int:
    """Cut a program's covering sweep into locked takes one hold a level, from its measured hold."""
    from syncsummoner.compose.render import RenderConfig, lead_frames, tail_frames
    from syncsummoner.probe import decoded
    from pyvmancer.midi import ParamKind, param_kind

    from syncsummoner.device.profile import PARAM_COUNT
    from syncsummoner.probe.locked import carried, joint, levels, plan_path, plan_sweep, recorded, write_plan
    from syncsummoner.probe.store import slug
    from syncsummoner.program import model
    from syncsummoner.program.gates import Gating, cover, representatives

    config = RenderConfig.for_format(args.format, strip_px=args.strip_px)
    held = json.loads(Path(args.hold).read_text(encoding="utf-8"))
    if held["program"] != args.program:
        raise ValueError(f"{args.hold} measured {held['program']}, not {args.program}")
    clip = decoded.features(args.source, decoded.Spec.of(config), workers=args.workers)
    frames = int(clip.small.shape[0]) - lead_frames(config) - tail_frames(config)
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

    from syncsummoner.compose.locktake import read_latency
    from syncsummoner.compose.render import RenderConfig, lead_frames, tail_frames
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
        slots,
    )
    from syncsummoner.probe.store import slug
    from syncsummoner.program import model
    from syncsummoner.program.gates import representatives

    config = RenderConfig.for_format(args.format, strip_px=args.strip_px)
    held = json.loads(Path(args.hold).read_text(encoding="utf-8"))
    if held["program"] != args.program:
        raise ValueError(f"{args.hold} measured {held['program']}, not {args.program}")
    record = json.loads(Path(args.cover).read_text(encoding="utf-8"))
    clip = decoded.features(args.source, decoded.Spec.of(config), workers=args.workers)
    ti = clip_ti(clip.luma, lead_frames(config), tail_frames(config))
    edges = downbeat_frames(args.audio, config.fps)
    lat = read_latency(args.latency or args.takes)
    guard = guard_frames(int(held["settle"]), lat.hi - lat.lo, 1e9 / config.fps)
    placed = slots(
        ti,
        edges,
        guard=guard,
        budget_s=args.budget_s,
        overhead_s=2 * config.lead_in_s + config.tail_s,
        fps=config.fps,
    )
    m = model(args.program)
    values = {p: representatives(m, p, bits) for p, bits in m.resolution().items() if bits}
    states = np.asarray(record["states"], dtype=np.int64)
    chosen = [int(i) for i in record["chosen"]]
    dims = region_dims(record, values, m.bypass())
    extra = np.zeros((0, PARAM_COUNT), dtype=np.int64)
    for selector, mover in (tuple(int(x) for x in pair.split(":")) for pair in args.joint):
        extra = np.r_[extra, joint(states[chosen[:1]], selector, mover, values)[1:]]
    counts = [
        min(n, distinct(d, values))
        for n, d in zip(allot(max(len(placed) - len(extra), 0), [len(d) for d in dims]), dims)
    ]
    sampled = [
        design(states[i], d, values, n, args.seed + k)
        for k, (i, d, n) in enumerate(zip(chosen, dims, counts))
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
        "rig_s": float(
            sum(c.frames for c in chunks) / config.fps + len(chunks) * (2 * config.lead_in_s + config.tail_s)
        ),
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


def _sweep_cmd(args: argparse.Namespace) -> int:  # pragma: no cover - needs the rig
    """Capture one chunk of a sweep plan as a locked take; 2 on a take fault."""
    from syncsummoner.compose.render import RenderConfig
    from syncsummoner.probe.locked import capture, read_plan

    chunks, said = read_plan(args.plan)
    config = RenderConfig.for_format(args.format, strip_px=args.strip_px, source_host=args.source_host)
    out = _sweep_take(Path(args.plan).parent, said["program"], args.chunk)
    record = capture(config, chunks[args.chunk], out, program=said["program"], clip=said["source"])
    meta = out.with_name(out.name + ".meta.json")
    kept = json.loads(meta.read_text(encoding="utf-8")) if meta.is_file() else {}
    kept[f"attempt-{args.attempt}"] = record
    meta.write_text(json.dumps(kept, indent=1, default=str), encoding="utf-8")
    LOG.info("%s chunk %d: %s", said["program"], args.chunk, record.get("fault", "captured"))
    return 2 if "fault" in record else 0


def settled_by(chunk: Any, program: str, hold: str | None, plan: str) -> tuple[Any, dict]:
    """``chunk`` with the settle a ``probe hold`` record measured, else its plan's; and where it came from."""
    from dataclasses import replace  # pylint: disable=import-outside-toplevel

    if hold is None:
        return chunk, {"frames": int(chunk.settle), "source": str(plan)}
    held = json.loads(Path(hold).read_text(encoding="utf-8"))
    if held["program"] != program:
        raise ValueError(f"{hold} measured {held['program']}, not {program}")
    return replace(chunk, settle=int(held["settle"])), {"frames": int(held["settle"]), "source": str(hold)}


def _sweep_collect_cmd(args: argparse.Namespace) -> int:
    """Judge one captured sweep take and keep its archive rows beside it."""
    from syncsummoner.compose.render import RenderConfig
    from syncsummoner.probe.locked import collect, read_plan

    chunks, said = read_plan(args.plan)
    config = RenderConfig.for_format(args.format, strip_px=args.strip_px)
    take = _sweep_take(Path(args.plan).parent, said["program"], args.chunk)
    chunk, settle = settled_by(chunks[args.chunk], said["program"], args.hold, args.plan)
    summary = collect(take, chunk, clip=said["source"], config=config, workers=args.workers, settle=settle)
    print(json.dumps(summary, indent=1))
    return 0 if summary["evidence_ok"] else 1


def _sweep_commit_cmd(args: argparse.Namespace) -> int:
    """Join a program's collected sweep takes into its archive, keyed on name and firmware."""
    import hashlib

    from syncsummoner.compose.render import RenderConfig, lead_frames
    from syncsummoner.probe import decoded
    from syncsummoner.probe.archive import FrameArchive
    from syncsummoner.probe.locked import commit_sweep, read_plan
    from syncsummoner.probe.store import KeyKind, ProgramKey

    chunks, said = read_plan(args.plan)
    config = RenderConfig.for_format(args.format, strip_px=args.strip_px)
    program = said["program"]
    takes = [_sweep_take(Path(args.plan).parent, program, k) for k in range(len(chunks))]
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
        "lead": lead_frames(config),
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


def add_parsers(probe_sub: Any) -> None:
    """The ``probe hold`` and ``probe sweep`` commands."""
    hold = probe_sub.add_parser("hold", help="each program's settle and archive hold from locked takes")
    hold.add_argument("takes", nargs="+")
    hold.add_argument("--source", required=True, help="the timecoded clip the takes played")
    hold.add_argument("--out", default=".", help="where hold-<program>.json goes")
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
    regions.add_argument("--hold", required=True, help="the program's hold-<program>.json")
    regions.add_argument("--source", required=True, help="the timecoded clip the takes will play")
    regions.add_argument("--audio", required=True, help="the track the clip is scored to, for its bars")
    regions.add_argument("--takes", required=True, help="the sweep's takes directory")
    regions.add_argument(
        "--latency", help="where the dry takes' ccpicture.json is; the takes directory by default"
    )
    regions.add_argument(
        "--budget-s", type=float, required=True, help="rig seconds the program's takes may use"
    )
    regions.add_argument("--seed", type=int, default=0, help="seed of the scrambled Sobol design")
    regions.add_argument("--joint", action="append", default=[], metavar="SELECTOR:MOVER", help="as for plan")
    regions.set_defaults(func=_sweep_regions_cmd)
    run = sweep_sub.add_parser("capture", help="capture one take of a plan on the rig")
    run.add_argument("plan")
    run.add_argument("--chunk", type=int, required=True)
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
    for node in (hold, plan, regions, run, gather, commit):
        node.add_argument("--format", default="1080p30", help="the session format")
        node.add_argument("--strip-px", type=int, default=8, help="rows of timecode strip along the bottom")
        node.add_argument("--workers", type=int, help="decoders at once")
    hold.set_defaults(func=_hold_cmd)

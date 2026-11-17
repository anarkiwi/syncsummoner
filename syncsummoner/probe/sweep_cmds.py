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
        states = levels(cover(m, gating=Gating(m, steps=not args.structural)))
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


def _sweep_collect_cmd(args: argparse.Namespace) -> int:
    """Judge one captured sweep take and keep its archive rows beside it."""
    from syncsummoner.compose.render import RenderConfig
    from syncsummoner.probe.locked import collect, read_plan

    chunks, said = read_plan(args.plan)
    config = RenderConfig.for_format(args.format, strip_px=args.strip_px)
    take = _sweep_take(Path(args.plan).parent, said["program"], args.chunk)
    summary = collect(take, chunks[args.chunk], clip=said["source"], config=config, workers=args.workers)
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
        "--structural", action="store_true", help="cover by the folded cone alone, as program cover"
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
    run = sweep_sub.add_parser("capture", help="capture one take of a plan on the rig")
    run.add_argument("plan")
    run.add_argument("--chunk", type=int, required=True)
    run.add_argument("--attempt", type=int, default=1)
    run.add_argument("--source-host", help="ssh target driving playout and the HDMI link")
    run.set_defaults(func=_sweep_cmd)
    gather = sweep_sub.add_parser("collect", help="judge one captured take and keep its archive rows")
    gather.add_argument("plan")
    gather.add_argument("--chunk", type=int, required=True)
    gather.set_defaults(func=_sweep_collect_cmd)
    commit = sweep_sub.add_parser("commit", help="join a program's collected takes into its archive")
    commit.add_argument("plan")
    commit.add_argument("--archive", required=True)
    commit.set_defaults(func=_sweep_commit_cmd)
    for node in (hold, plan, run, gather, commit):
        node.add_argument("--format", default="1080p30", help="the session format")
        node.add_argument("--strip-px", type=int, default=8, help="rows of timecode strip along the bottom")
        node.add_argument("--workers", type=int, help="decoders at once")
    hold.set_defaults(func=_hold_cmd)

"""Command line entry point.

Subcommand modules are imported lazily so a partial install, or a missing
optional dependency, only breaks the subcommands that need it.
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np


def _device_cmd(args: argparse.Namespace) -> int:
    from pyvmancer.discovery import find_devices

    from syncsummoner.device.transport import Transport

    if args.device_cmd == "list":
        print(json.dumps([_as_mapping(d) for d in find_devices()], indent=2, default=str))
        return 0

    dev = Transport.open(serial=args.serial)
    try:
        if args.device_cmd == "ping":
            status = dev.video_status()
            print(json.dumps({"firmware": dev.firmware(), "video": status.raw}, indent=2))
        else:
            print(json.dumps(dev.programs(), indent=2))
    finally:
        dev.close()
    return 0


def _as_mapping(obj) -> dict:
    """Dict view of a record that may be a dataclass, a slots class, or neither."""
    if dataclasses.is_dataclass(obj):
        return {f.name: getattr(obj, f.name) for f in dataclasses.fields(obj)}
    slots = getattr(type(obj), "__slots__", None)
    if slots:
        return {name: getattr(obj, name, None) for name in slots}
    return dict(vars(obj))


def _profiles_in(directory: Path) -> dict:
    from syncsummoner.probe.fit import load_profile

    profiles = {}
    for path in sorted(Path(directory).glob("*.yaml")):
        profile = load_profile(path)
        profiles[profile.program] = profile
    return profiles


def _link(args: argparse.Namespace):
    """Source HDMI link for a run, or None when the caller opted out.

    Every program change drops the link, because the device runs on the clock
    recovered from the incoming video and reports lock before it has settled.
    """
    from syncsummoner.device.link import Link

    if getattr(args, "no_link", False):
        return None
    host = getattr(args, "source_host", None)
    return Link(host) if host else Link()


def _measure(session, capture, args, *, program, specs, firmware, rng, writer):
    """Records from every configured plan for one program, archiving frames if asked."""
    from syncsummoner import aesthetics
    from syncsummoner.probe import plans, runner

    records = []
    for plan_name in args.plan.split(","):
        records += runner.run_plan(
            session,
            capture,
            _make_plan(plans, plan_name, specs, rng),
            program=program,
            analyzer=aesthetics,
            firmware=firmware,
            allow_untagged=args.allow_untagged,
            archive=writer,
        )
    return records


def _probe_hardware(args: argparse.Namespace, rng, specs_by_program: dict) -> tuple[list, Any]:
    """Measure every requested program on the rig, resuming whatever is stored."""
    from contextlib import ExitStack

    from syncsummoner.device.capture import Capture, PixelMode
    from syncsummoner.device.session import Session
    from syncsummoner.device.transport import Transport
    from syncsummoner.probe.archive import FrameArchive
    from syncsummoner.probe.store import ResultStore, program_key

    dev = Transport.open(serial=args.serial)
    store = ResultStore(args.store or Path(args.out))
    frames = FrameArchive(args.archive) if args.archive else None
    link, records = _link(args), []
    try:
        names = dev.programs() if args.program == "all" else args.program.split(",")
        firmware, manifest = dev.firmware(), dev.program_manifest()
        session = Session(dev)
        mode = PixelMode.YVYU if frames is not None else PixelMode.BGR
        with Capture(device=args.capture, mode=mode) as capture:
            for name in names:
                key = program_key(dev, name, firmware=firmware, manifest=manifest)
                done = store.get(name, key)
                if done is not None:
                    records += done
                    continue
                session.load_program(name, link=link)
                session.ensure_live(capture, require_motion=False)
                specs_by_program[name] = dev.program_info().params
                with ExitStack() as stack:
                    writer = (
                        None
                        if frames is None
                        else stack.enter_context(
                            frames.writer(name, key, width=capture.width, height=capture.height)
                        )
                    )
                    measured = _measure(
                        session,
                        capture,
                        args,
                        program=name,
                        specs=specs_by_program[name],
                        firmware=firmware,
                        rng=rng,
                        writer=writer,
                    )
                if measured:
                    store.put(name, key, measured)
                records += measured
    finally:
        dev.close()
    return records, store


def _probe_cmd(args: argparse.Namespace) -> int:
    from syncsummoner import aesthetics
    from syncsummoner.probe import plans, sim
    from syncsummoner.probe.fit import fit_profile, save_measurements, save_profile

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(args.seed)

    specs_by_program: dict[str, list] = {}
    if args.probe_cmd == "sim":
        store = None
        records = sim.run_plan_sim(plans.oat([]), program=args.program, analyzer=aesthetics, rng=rng)
    else:
        records, store = _probe_hardware(args, rng, specs_by_program)

    written = []
    for program in sorted({r.program for r in records}):
        subset = [r for r in records if r.program == program]
        path = out / f"{program}.yaml"
        save_profile(fit_profile(subset, specs=specs_by_program.get(program)), path)
        if store is None:
            save_measurements(subset, out / f"{program}.parquet")
        written.append(str(path))
    print(json.dumps(written, indent=2))
    return 0


def _harvest_cmd(args: argparse.Namespace) -> int:
    """Drive a whole-library native frame archive run against the rig."""
    from syncsummoner.device.capture import Capture, PixelMode
    from syncsummoner.device.playout import LoopPlayer
    from syncsummoner.device.transport import Transport
    from syncsummoner.probe.archive import FrameArchive
    from syncsummoner.probe.harvest import HarvestConfig, harvest

    config = HarvestConfig(
        width=args.width,
        height=args.height,
        capture_fps=args.capture_fps,
        setpoints=args.setpoints,
        frames_per_point=args.frames_per_point,
        seed=args.seed,
    )
    host = {} if args.source_host is None else {"host": args.source_host}
    report = harvest(
        FrameArchive(args.out),
        open_transport=lambda: Transport.open(serial=args.serial),
        open_capture=lambda: Capture(
            device=args.capture,
            width=config.width,
            height=config.height,
            fps=config.capture_fps,
            mode=PixelMode.YVYU,
        ),
        player=LoopPlayer(width=config.width, height=config.height, **host),
        link=_link(args),
        programs=None if args.program == "all" else args.program.split(","),
        config=config,
        log=lambda message: print(message, flush=True),
    )
    print(f"{report.frames} frames from {len(report.results)} programs in {report.seconds / 60:.1f} min")
    return 1 if report.wedged else 0


def _make_plan(plans, name: str, specs, rng):
    """Build one named sweep plan over the program's parameter specs."""
    if name == "oat":
        return plans.oat(specs)
    if name == "sobol":
        return plans.sobol(specs, n=64, rng=rng)
    if name == "tongue":
        return plans.tongue_raster(specs, (1, 2), n=8)
    if name == "hysteresis":
        return plans.hysteresis(specs, 1, n=16)
    raise ValueError(f"unknown plan {name!r}")


def _profile_cmd(args: argparse.Namespace) -> int:
    profiles = _profiles_in(Path(args.profiles))
    if args.profile_cmd == "show":
        for name, profile in sorted(profiles.items()):
            used = [p for p in profile.params if p.kind.value != "unused"]
            print(f"{name}: {len(used)} params, source={profile.source.value}, analyzer={profile.analyzer}")
    else:
        for index, cliff in profiles[args.program].cliff_atlas():
            print(f"P{index:<3} at={cliff.at:<5} jump={cliff.jump:.3f} metrics={','.join(cliff.metrics)}")
    return 0


def _analyze_cmd(args: argparse.Namespace) -> int:
    from syncsummoner.compose.features import analyze

    rng = np.random.default_rng(args.seed)
    result = analyze(args.clip, args.audio, rng=rng)
    payload = json.dumps(dataclasses.asdict(result), indent=2, default=_jsonable)
    if args.output:
        Path(args.output).write_text(payload, encoding="utf-8")
    else:
        print(payload)
    return 0


def _jsonable(obj):
    """Serialize numpy values the feature dataclasses carry."""
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, (np.integer, np.floating)):
        return obj.item()
    return str(obj)


def _compose_cmd(args: argparse.Namespace) -> int:
    from syncsummoner.compose.features import analyze
    from syncsummoner.compose.planner import search

    rng = np.random.default_rng(args.seed)
    profiles = _profiles_in(Path(args.profiles))
    if not profiles:
        raise ValueError(f"no profiles in {args.profiles}; run `syncsummoner probe run` first")
    features = analyze(args.clip, args.audio, rng=rng)
    score = search(profiles, features, style=args.style, rng=rng, budget=args.budget)
    score.save(Path(args.output))
    print(f"{args.output}: {len(score.layers)} layers over {score.duration:.1f}s")
    return 0


def _render_cmd(args: argparse.Namespace) -> int:
    from syncsummoner.compose import render as render_mod
    from syncsummoner.compose.score import Score

    score = Score.load(Path(args.score))
    profiles = _profiles_in(Path(args.profiles))
    config = render_mod.RenderConfig(source_host=args.source_host)
    if args.render_cmd == "audition":
        frames = render_mod.audition(
            score, args.source, seconds=args.seconds, passes=args.passes, profiles=profiles, config=config
        )
        render_mod.write_video(args.output, frames, score.fps * args.seconds / max(len(frames), 1))
    else:
        render_mod.render(
            score, args.source, args.output, passes=args.passes, profiles=profiles, config=config
        )
    print(args.output)
    return 0


def _add_link_args(parser: argparse.ArgumentParser) -> None:
    """Source-host options for any subcommand that changes program."""
    parser.add_argument("--source-host", help="ssh target driving the stimulus and the HDMI link")
    parser.add_argument("--no-link", action="store_true", help="do not drop the link across a load")


def build_parser() -> argparse.ArgumentParser:  # pylint: disable=too-many-statements
    """Construct the argument parser for every subcommand."""
    parser = argparse.ArgumentParser(prog="syncsummoner", description=__doc__)
    parser.add_argument("--serial", help="select a specific Videomancer by serial")
    sub = parser.add_subparsers(dest="cmd", required=True)

    device = sub.add_parser("device", help="discovery and status")
    device.add_argument("device_cmd", choices=["list", "ping", "programs"])
    device.set_defaults(func=_device_cmd)

    probe = sub.add_parser("probe", help="measure device behaviour")
    probe_sub = probe.add_subparsers(dest="probe_cmd", required=True)
    run = probe_sub.add_parser("run")
    run.add_argument("--program", default="all")
    run.add_argument("--plan", default="oat,sobol")
    run.add_argument("--capture", default="/dev/video0")
    run.add_argument("--allow-untagged", action="store_true", help="dwell instead of state-index match")
    run.add_argument("--seed", type=int, default=0)
    run.add_argument("--out", default="profiles/")
    run.add_argument("--store", help="resumable per-program result store (default: --out)")
    run.add_argument("--archive", help="also store the native capture frames under this directory")
    _add_link_args(run)
    simulate = probe_sub.add_parser("sim")
    simulate.add_argument("--program", required=True)
    simulate.add_argument("--seed", type=int, default=0)
    simulate.add_argument("--out", default="profiles/")
    probe.set_defaults(func=_probe_cmd)

    collect = probe_sub.add_parser("archive", help="archive native frames for every program")
    collect.add_argument("--program", default="all")
    collect.add_argument("--capture", default="/dev/video0")
    collect.add_argument("--out", default="archive/")
    collect.add_argument("--width", type=int, default=1920)
    collect.add_argument("--height", type=int, default=1080)
    collect.add_argument("--capture-fps", type=int, default=30)
    collect.add_argument("--setpoints", type=int, default=32)
    collect.add_argument("--frames-per-point", type=int, default=30)
    collect.add_argument("--seed", type=int, default=11)
    _add_link_args(collect)
    collect.set_defaults(func=_harvest_cmd)

    profile = sub.add_parser("profile", help="inspect fitted profiles")
    profile_sub = profile.add_subparsers(dest="profile_cmd", required=True)
    for name in ("show", "atlas"):
        node = profile_sub.add_parser(name)
        node.add_argument("--profiles", default="profiles/")
        if name == "atlas":
            node.add_argument("--program", required=True)
    profile.set_defaults(func=_profile_cmd)

    analyze = sub.add_parser("analyze", help="extract audio and video features")
    analyze.add_argument("clip")
    analyze.add_argument("--audio")
    analyze.add_argument("--seed", type=int, default=0)
    analyze.add_argument("-o", "--output")
    analyze.set_defaults(func=_analyze_cmd)

    compose = sub.add_parser("compose", help="search for a score")
    compose.add_argument("clip")
    compose.add_argument("--audio")
    compose.add_argument("--profiles", default="profiles/")
    compose.add_argument("--style", default="default")
    compose.add_argument("--seed", type=int, default=0)
    compose.add_argument("--budget", type=int, default=48)
    compose.add_argument("-o", "--output", default="score.yaml")
    compose.set_defaults(func=_compose_cmd)

    audition = sub.add_parser("audition", help="render a short low-resolution proxy")
    audition.add_argument("score")
    audition.add_argument("--source", required=True)
    audition.add_argument("--profiles", default="profiles/")
    audition.add_argument("--seconds", type=float, default=30.0)
    audition.add_argument("--passes", type=int, default=1)
    audition.add_argument("--source-host", help="ssh target driving playout and the HDMI link")
    audition.add_argument("-o", "--output", default="audition.mkv")
    audition.set_defaults(func=_render_cmd, render_cmd="audition")

    full = sub.add_parser("render", help="render the full pass")
    full.add_argument("score")
    full.add_argument("--source", required=True)
    full.add_argument("--profiles", default="profiles/")
    full.add_argument("--passes", type=int, default=1)
    full.add_argument("--source-host", help="ssh target driving playout and the HDMI link")
    full.add_argument("-o", "--output", default="out.mkv")
    full.set_defaults(func=_render_cmd, render_cmd="render")

    return parser


def main(argv: list[str] | None = None) -> int:
    """Parse arguments and dispatch."""
    args = build_parser().parse_args(argv)
    try:
        return args.func(args)
    except Exception as err:
        print(f"error: {err}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())

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


def _probe_cmd(args: argparse.Namespace) -> int:
    from syncsummoner import aesthetics
    from syncsummoner.device.capture import Capture
    from syncsummoner.device.session import Session
    from syncsummoner.device.transport import Transport
    from syncsummoner.probe import plans, runner, sim
    from syncsummoner.probe.fit import fit_profile, save_measurements, save_profile

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(args.seed)

    if args.probe_cmd == "sim":
        specs = []
        records = sim.run_plan_sim(
            plans.oat(specs), program=args.program, analyzer=aesthetics.__version__, rng=rng
        )
    else:
        dev = Transport.open(serial=args.serial)
        try:
            names = dev.programs() if args.program == "all" else args.program.split(",")
            records = []
            session = Session(dev)
            with Capture(device=args.capture) as capture:
                capture.wait_for_lock()
                for name in names:
                    session.load_program(name)
                    specs = dev.program_info().params
                    for plan_name in args.plan.split(","):
                        plan = _make_plan(plans, plan_name, specs, rng)
                        records += runner.run_plan(
                            session,
                            capture,
                            plan,
                            program=name,
                            analyzer=aesthetics.__version__,
                            firmware=dev.firmware(),
                            allow_untagged=args.allow_untagged,
                        )
        finally:
            dev.close()

    written = []
    for program in sorted({r.program for r in records}):
        subset = [r for r in records if r.program == program]
        path = out / f"{program}.yaml"
        save_profile(fit_profile(subset), path)
        save_measurements(subset, out / f"{program}.parquet")
        written.append(str(path))
    print(json.dumps(written, indent=2))
    return 0


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
    if args.render_cmd == "audition":
        frames = render_mod.audition(
            score, args.source, seconds=args.seconds, passes=args.passes, profiles=profiles
        )
        render_mod.write_video(args.output, frames, score.fps * args.seconds / max(len(frames), 1))
    else:
        render_mod.render(score, args.source, args.output, passes=args.passes, profiles=profiles)
    print(args.output)
    return 0


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
    simulate = probe_sub.add_parser("sim")
    simulate.add_argument("--program", required=True)
    simulate.add_argument("--seed", type=int, default=0)
    simulate.add_argument("--out", default="profiles/")
    probe.set_defaults(func=_probe_cmd)

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
    audition.add_argument("-o", "--output", default="audition.mkv")
    audition.set_defaults(func=_render_cmd, render_cmd="audition")

    full = sub.add_parser("render", help="render the full pass")
    full.add_argument("score")
    full.add_argument("--source", required=True)
    full.add_argument("--profiles", default="profiles/")
    full.add_argument("--passes", type=int, default=1)
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

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

from syncsummoner.progress import LOG, configure


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


def _measure(session, capture, args, *, program, specs, firmware, rng):
    """Records from every configured plan for one program."""
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
        )
    return records


def _probe_hardware(args: argparse.Namespace, rng, specs_by_program: dict) -> tuple[list, Any]:
    """Measure every requested program on the rig, resuming whatever is stored."""
    from syncsummoner.device.capture import Capture
    from syncsummoner.device.session import Session
    from syncsummoner.device.transport import Transport
    from syncsummoner.probe.store import ResultStore, program_key

    dev = Transport.open(serial=args.serial)
    store = ResultStore(args.store or Path(args.out))
    link, records = _link(args), []
    try:
        names = dev.programs() if args.program == "all" else args.program.split(",")
        firmware, manifest = dev.firmware(), dev.program_manifest()
        session = Session(dev)
        with Capture(device=args.capture) as capture:
            for name in names:
                key = program_key(dev, name, firmware=firmware, manifest=manifest)
                done = store.get(name, key)
                if done is not None:
                    records += done
                    continue
                session.load_program(name, link=link)
                session.ensure_live(capture, require_motion=False)
                specs_by_program[name] = dev.program_info().params
                measured = _measure(
                    session,
                    capture,
                    args,
                    program=name,
                    specs=specs_by_program[name],
                    firmware=firmware,
                    rng=rng,
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


def _refit_cmd(args: argparse.Namespace) -> int:
    """Fit profiles from an archived run, with no device and no rig time."""
    from syncsummoner import aesthetics
    from syncsummoner.device.recorder import require_ffmpeg
    from syncsummoner.probe.archive import FrameArchive
    from syncsummoner.probe.behaviour import behaviours
    from syncsummoner.probe.fit import fit_profile, save_measurements, save_profile
    from syncsummoner.probe.replay import replay
    from syncsummoner.probe.store import slug
    from syncsummoner.probe.style import measured_styles

    require_ffmpeg(args.ffmpeg)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    programs = None if args.program == "all" else args.program.split(",")
    archive = FrameArchive(args.archive, ffmpeg=args.ffmpeg)
    measured = replay(
        archive,
        analyzer=aesthetics,
        programs=programs,
        jobs=args.jobs,
        log=LOG.info,
    )
    found = behaviours(archive, sorted(measured), jobs=args.jobs, log=LOG.warning)
    # A non-monotone value map kills the carrier as displacement does; only the fit tells them apart.
    styles, band = measured_styles({name: b.pointwise for name, b in found.items()}, {})
    LOG.info("analog above %.2f, digital below %.2f: no labels here, so the band is assumed", *band)
    written = []
    for program, records in sorted(measured.items()):
        path = out / f"{slug(program)}.yaml"
        profile = fit_profile(records)
        profile.registered, profile.pointwise = found[program]
        profile.style = styles[program]
        save_profile(profile, path)
        save_measurements(records, out / f"{slug(program)}.parquet")
        written.append(str(path))
    LOG.info("%d profiles in %s", len(written), out)
    return 0 if written else 1


def _harvest_cmd(args: argparse.Namespace) -> int:
    """Drive a whole-library native frame archive run against the rig."""
    from syncsummoner.device.playout import LoopPlayer
    from syncsummoner.device.recorder import FFV1, Recorder, require_ffmpeg
    from syncsummoner.device.transport import Transport
    from syncsummoner.probe.archive import FrameArchive
    from syncsummoner.probe.harvest import HarvestConfig, harvest

    require_ffmpeg()
    config = HarvestConfig(
        width=args.width,
        height=args.height,
        capture_fps=args.capture_fps,
        setpoints=args.setpoints,
        dwell_s=args.dwell,
        seed=args.seed,
    )
    host = {} if args.source_host is None else {"host": args.source_host}
    report = harvest(
        FrameArchive(args.out),
        open_transport=lambda: Transport.open(serial=args.serial),
        recorder=Recorder(
            device=args.capture,
            width=config.width,
            height=config.height,
            fps=config.capture_fps,
            mode=FFV1,
            copyts=True,
        ),
        player=LoopPlayer(width=config.width, height=config.height, **host),
        link=_link(args),
        programs=None if args.program == "all" else args.program.split(","),
        config=config,
        log=LOG.info,
    )
    LOG.info(
        "%d frames from %d programs in %.1f min", report.frames, len(report.results), report.seconds / 60
    )
    return 1 if report.stopped else 0


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

    from syncsummoner.compose.render import SESSION_FORMATS

    rng = np.random.default_rng(args.seed)
    profiles = _profiles_in(Path(args.profiles))
    if not profiles:
        raise ValueError(f"no profiles in {args.profiles}; run `syncsummoner probe run` first")
    fps = SESSION_FORMATS[args.format][2] if args.format else 60.0
    LOG.info("%d profiles from %s", len(profiles), args.profiles)
    features = analyze(args.clip, args.audio, rng=rng)
    score = search(
        profiles,
        features,
        style=args.style,
        rng=rng,
        budget=args.budget,
        density=args.density,
        n_passes=args.passes,
        fps=fps,
    )
    score.save(Path(args.output))
    LOG.info("%s: %d layers over %.1fs", args.output, len(score.layers), score.duration)
    print(args.output)
    return 0


def _take_path(args: argparse.Namespace) -> str:
    """Where the raw capture goes; ``--output`` names the finished clip, not the pass."""
    if args.take:
        return args.take
    out = Path(args.output)
    return str(out.with_name(f"{out.stem}.take{out.suffix or '.mkv'}"))


def _make_room(*paths: str | Path) -> None:
    """Create the directories a run writes into, so a pass fails on the rig and not on a path."""
    for path in paths:
        Path(path).parent.mkdir(parents=True, exist_ok=True)


def _fade(args: argparse.Namespace, edge: str) -> float:
    """Fade for one edge: its own override, else ``--fade``, else the library default."""
    from syncsummoner.compose.master import DEFAULT_FADE_S

    value = getattr(args, edge)
    value = args.fade if value is None else value
    return DEFAULT_FADE_S if value is None else float(value)


def _render_cmd(args: argparse.Namespace) -> int:
    from syncsummoner.compose import master as master_mod
    from syncsummoner.compose import render as render_mod
    from syncsummoner.compose.score import Score
    from syncsummoner.device.recorder import require_ffmpeg

    require_ffmpeg()
    score = Score.load(Path(args.score))
    if args.render_cmd == "audition":
        score = score.window(args.start, args.start + args.seconds)
        LOG.info("audition: %.1fs from %.1fs", score.duration, args.start)
    profiles = _profiles_in(Path(args.profiles))
    config = (
        render_mod.RenderConfig.for_format(args.format, source_host=args.source_host)
        if args.format
        else render_mod.RenderConfig(source_host=args.source_host)
    )
    take, start, lead = _take_path(args), getattr(args, "start", 0.0), 0.0
    _make_room(take, args.output, args.scratch)
    if args.cut_programs:
        plan = render_mod.render_cuts(
            score,
            args.source,
            take,
            profiles=profiles,
            programs=[p.strip() for p in args.cut_programs.split(",") if p.strip()],
            config=config,
            scratch=args.scratch,
            prepared=args.prepared,
            start=start,
            takes=args.takes,
        )
        for cut in plan:
            LOG.info("  %7.2f-%7.2fs  %s", cut.start, cut.end, cut.program)
        LOG.info("%s: %d cuts", take, len(plan))
    else:
        report = render_mod.render_played(
            score,
            args.source,
            take,
            profiles=profiles,
            config=config,
            scratch=args.scratch,
            prepared=args.prepared,
            start=start,
        )
        LOG.info("%s: %s", take, report)
        if not report.usable:
            return 1
        lead = render_mod.picture_start(take, config=config)
        LOG.info("picture starts %.2fs into the take", lead)
    if args.no_master:
        return 0
    seconds = master_mod.master(
        take,
        args.audio,
        args.output,
        seconds=score.duration if args.render_cmd == "audition" else None,
        fade_in=_fade(args, "fade_in"),
        fade_out=_fade(args, "fade_out"),
        audio_start=start,
        video_start=lead,
    )
    LOG.info("%s: %.1fs mastered from %s", args.output, seconds, take)
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
    parser.add_argument("-v", "--verbose", action="count", default=0, help="log every stage in detail")
    parser.add_argument("-q", "--quiet", action="store_true", help="warnings and errors only")
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
    _add_link_args(run)
    simulate = probe_sub.add_parser("sim")
    simulate.add_argument("--program", required=True)
    simulate.add_argument("--seed", type=int, default=0)
    simulate.add_argument("--out", default="profiles/")
    probe.set_defaults(func=_probe_cmd)

    refit = probe_sub.add_parser("refit", help="fit profiles from an archived run, with no device")
    refit.add_argument("--program", default="all")
    refit.add_argument("--archive", default="archive/")
    refit.add_argument("--out", default="profiles/")
    refit.add_argument("--ffmpeg", default="ffmpeg", help="decoder to read the archive with")
    refit.add_argument("--jobs", type=int, default=1, help="programs to read at once")
    refit.set_defaults(func=_refit_cmd)

    collect = probe_sub.add_parser("archive", help="archive native frames for every program")
    collect.add_argument("--program", default="all")
    collect.add_argument("--capture", default="/dev/video0")
    collect.add_argument("--out", default="archive/")
    collect.add_argument("--width", type=int, default=1920)
    collect.add_argument("--height", type=int, default=1080)
    collect.add_argument("--capture-fps", type=int, default=30)
    collect.add_argument("--setpoints", type=int, default=32)
    collect.add_argument("--dwell", type=float, default=1.0, help="seconds held per setpoint")
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
    compose.add_argument("--density", type=float, default=0.5, help="gestures per section, 0 to 1")
    compose.add_argument("--passes", type=int, default=1, help="layers to keep, one capture pass each")
    compose.add_argument("--format", help="session format the score is written for, e.g. 1080p30")
    compose.add_argument("-o", "--output", default="score.yaml")
    compose.set_defaults(func=_compose_cmd)

    for name, help_text in (("render", "render the full pass"), ("audition", "render a short excerpt")):
        node = sub.add_parser(name, help=help_text)
        node.add_argument("score")
        node.add_argument("--source", required=True)
        node.add_argument("--audio", help="track to mux and fade; omitted leaves the clip silent")
        node.add_argument("--profiles", default="profiles/")
        node.add_argument("--source-host", help="ssh target driving playout and the HDMI link")
        node.add_argument("-o", "--output", default=f"{name}.mp4", help="the finished clip")
        node.add_argument("--take", help="raw capture path (default: alongside --output)")
        node.add_argument("--format", help="session format the rig runs at, e.g. 1080p30")
        node.add_argument(
            "--played", action="store_true", help="play the source from the rig, at rate, for one pass"
        )
        node.add_argument("--scratch", default="timecoded.mkv", help="where the timecoded source is built")
        node.add_argument(
            "--prepared", action="store_true", help="take --scratch as an already timecoded clip"
        )
        node.add_argument(
            "--cut-programs", help="cut between these programs on the score's sections, one pass each"
        )
        node.add_argument("--takes", default=".", help="where the per-program passes are written")
        node.add_argument("--fade", type=float, help="seconds of fade at each edge (default: 1)")
        node.add_argument("--fade-in", type=float, help="override the fade in")
        node.add_argument("--fade-out", type=float, help="override the fade out")
        node.add_argument("--no-master", action="store_true", help="stop at the raw take")
        node.set_defaults(func=_render_cmd, render_cmd=name)
        if name == "audition":
            node.add_argument("--seconds", type=float, default=30.0, help="excerpt length")
            node.add_argument("--start", type=float, default=0.0, help="where the excerpt begins")

    return parser


def main(argv: list[str] | None = None) -> int:
    """Parse arguments and dispatch."""
    args = build_parser().parse_args(argv)
    configure(-1 if args.quiet else args.verbose)
    try:
        return args.func(args)
    except Exception as err:
        print(f"error: {err}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())

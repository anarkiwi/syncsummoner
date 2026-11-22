"""The ``program`` command: each program's declared parameters and which are live, from its HDL."""

from __future__ import annotations

import argparse
import json

from . import crosscheck, predict


def add_parser(sub) -> None:
    """The ``program`` command and its subcommands."""
    program = sub.add_parser(
        "program", help="each program's declared parameters and which are live, from its HDL"
    )
    program_sub = program.add_subparsers(dest="program_cmd", required=True)
    program_sub.add_parser("list", help="programs found on the card copy and firmware images")
    for name, text in (
        ("show", "register bits, levels, liveness and gates"),
        ("step", "can one step change it"),
        ("cover", "what makes each parameter live, and the fewest base states that step every one live"),
        ("bypass", "the control setting proven to return the input with every other control dead"),
    ):
        node = program_sub.add_parser(name, help=text)
        node.add_argument("name", help="device name (author/Name), bare built-in name, or program id")
        if name == "step":
            node.add_argument("param", type=int, help="parameter_id 1..12")
            node.add_argument("a", type=int, help="manual value before")
            node.add_argument("b", type=int, help="manual value after")
        node.add_argument("--state", help="initial (default), a preset name, or 12 comma-separated values")
        node.add_argument("--set", action="append", default=[], metavar="P=V", help="override one control")
        node.add_argument("--timing", type=int, help="video timing id held in register 8 (default: free)")
        node.add_argument("--config", default="hd_hdmi", help="IO configuration (default: hd_hdmi)")
        node.add_argument(
            "--source", choices=("rtl", "bitstream"), help="netlist source (default: rtl if published)"
        )
        node.add_argument(
            "--method",
            choices=("bdd", "sat", "ternary"),
            help="answer liveness by this method alone (default: BDDs, then SAT once they outgrow memory)",
        )
        node.add_argument("--json", action="store_true")
        node.add_argument(
            "--structural", action="store_true", help="accepted for scripts: every cover is by observability"
        )
    program.set_defaults(func=_program_cmd)
    predict.add_parser(program_sub)
    crosscheck.add_parser(program_sub)


def _program_cmd(args: argparse.Namespace) -> int:
    """List programs, or show or step one program's parameter model."""
    # pylint: disable-next=import-outside-toplevel
    from syncsummoner.program import Library, gates, model, parse_state
    from syncsummoner.program.report import rows, table  # pylint: disable=import-outside-toplevel

    library = Library.default()
    if args.program_cmd == "list":
        for prog in library.programs:
            src = "rtl" if prog.sources else "bitstream"
            print(f"{prog.name:<40} {prog.package.config.program_id:<40} {src:<9} {prog.origin}")
        return 0
    m = model(args.name, library, config=args.config, source=args.source, method=args.method)
    sets = dict(tuple(map(int, s.split("=", 1))) for s in args.set)
    state = parse_state(m, args.state, sets)
    if args.program_cmd == "step":
        step = m.step_matters(state, args.param, args.a, args.b, args.timing)
        print(
            json.dumps(step.__dict__)
            if args.json
            else f"P{step.parameter_id} {step.a}->{step.b}: "
            f"{ {True: 'changes', False: 'no change', None: 'undetermined'}[step.matters]} ({step.how})"
        )
        return 0
    if args.program_cmd == "bypass":
        found = m.bypass(state, args.timing)
        if args.json:
            print(json.dumps(None if found is None else found.__dict__))
        elif found is None:
            print(f"{m.program.name}: no setting is proven to pass the input with every other control dead")
        else:
            decl = m.decl(found.parameter_id)
            print(
                f"{m.program.name}: P{found.parameter_id} {decl.name} at {found.value}, proven at "
                f"{len(found.proven)} values {min(found.proven)}..{max(found.proven)}"
            )
        return 0
    if args.program_cmd == "cover":
        print(gates.report(m, state, args.timing, as_json=args.json))
        return 0
    print(json.dumps(rows(m, state, args.timing), indent=1) if args.json else table(m, state, args.timing))
    return 0

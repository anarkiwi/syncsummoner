"""Tables of a program's parameters: register bits, levels, liveness and gates at one state."""

from __future__ import annotations

from typing import Any

import numpy as np

from .declared import Kind, field
from .model import ProgramModel


def rows(m: ProgramModel, state: np.ndarray, timing: int | None = None) -> list[dict[str, Any]]:
    """One row per declared parameter at ``state``."""
    live = m.live(state, timing)
    read = m.resolution()
    out = []
    for decl in m.params:
        pid = decl.parameter_id
        k, declared = m.kind(pid)
        bits = live[pid].live_bits
        f = field(pid)
        out.append(
            {
                "param": pid,
                "name": decl.name,
                "kind": k.value,
                "declared_levels": declared,
                "register": str(f),
                "reads": _span(f.register, read[pid]),
                "value": int(state[pid - 1]),
                "live_bits": _span(f.register, bits),
                "levels": m.levels(pid, bits),
                "live": live[pid].live,
                "gated_by": list(live[pid].gated_by),
            }
        )
    return out


def _span(register: int, bits) -> str:
    if not bits:
        return "-"
    bits = np.sort(np.asarray(bits))
    runs = np.split(bits, np.nonzero(np.diff(bits) != 1)[0] + 1)
    text = [f"{r[-1]}:{r[0]}" if len(r) > 1 else f"{r[0]}" for r in reversed(runs)]
    return f"r{register}[{','.join(text)}]"


def table(m: ProgramModel, state: np.ndarray, timing: int | None = None) -> str:
    """Plain-text table for the CLI."""
    cfg = m.program.package.config
    head = (
        f"{m.program.name}  {cfg.program_id} {cfg.version_text}  netlist: {m.source} ({m.config})"
        f"  from {m.program.origin}\n"
        f"{'P':>3} {'name':<16} {'kind':<11} {'value':>5} {'drives':<10} {'reads':<12} "
        f"{'live bits':<12} {'levels':>6}  verdict"
    )
    lines = [head]
    for r in rows(m, state, timing):
        kind = r["kind"] + (f"/{r['declared_levels']}" if r["declared_levels"] else "")
        verdict = "live" if r["live"] else ("dead" if r["kind"] != Kind.UNUSED.value else "unused")
        if not r["live"]:
            verdict += (
                f", gated by {', '.join(r['gated_by'])}"
                if r["gated_by"]
                else ", no single control revives it"
            )
        lines.append(
            f"{r['param']:>3} {r['name'][:16]:<16} {kind:<11} {r['value']:>5} {r['register']:<10} {r['reads']:<12} "
            f"{r['live_bits']:<12} {r['levels']:>6}  {verdict}"
        )
    return "\n".join(lines)

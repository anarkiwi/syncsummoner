"""Netlists of a program, constant-folded per state with Yosys, and what still reaches the picture.

A program's netlist comes from its VHDL (GHDL through the Yosys plugin) or, with no
source, from its iCE40 bitstream (IceStorm). Either way ``registers_in`` is a 320-bit
input port and the video outputs are the output ports, so one fold serves both.
"""

from __future__ import annotations

import copy
import json
import os
import re
import shutil
import subprocess
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Mapping

import numpy as np
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import breadth_first_order

from .declared import LATCHED_REGISTERS, REGISTER_BITS, REGISTER_COUNT, flat_bit

PORT = "registers_in"
ABC_GATES = "AND,NAND,OR,NOR,XOR,XNOR,ANDNOT,ORNOT,MUX"
FOLD = f"opt -full; techmap; opt -full; abc -g {ABC_GATES}; opt -full -purge"
_FF = re.compile(r"^\$_(DFFE?|SDFFC?E?)_([PN]+)([01]?)([PN]?)_$")
MAX_ROUNDS = 16


class ToolError(RuntimeError):
    """An HDL tool is missing or failed."""


class Tools:
    """Locates and runs Yosys, GHDL, IceStorm and Icarus from one directory or ``PATH``."""

    def __init__(self, bin_dir: str | os.PathLike | None = None):
        self.bin_dir = bin_dir or os.environ.get("SYNCSUMMONER_HDL_BIN")

    def path(self, name: str) -> str:
        """Absolute path of a tool, raising when it is not installed."""
        found = shutil.which(name, path=str(self.bin_dir) if self.bin_dir else None)
        if not found:
            raise ToolError(f"{name} not found; install the OSS CAD Suite (Dockerfile target hdl)")
        return found

    @property
    def share(self) -> Path:
        """Yosys's data directory."""
        return Path(self.path("yosys")).resolve().parent.parent / "share" / "yosys"

    def available(self) -> bool:
        """True when every tool the analysis uses is installed."""
        try:
            for name in ("yosys", "ghdl", "iceunpack", "icebox_vlog", "iverilog", "vvp"):
                self.path(name)
        except ToolError:
            return False
        return True

    def run(self, args: list[str], cwd: Path, timeout: float | None = None) -> str:
        """Run a tool, returning stdout; failure raises with the tail of its output."""
        proc = subprocess.run(
            [self.path(args[0]), *args[1:]],
            cwd=cwd,
            capture_output=True,
            text=True,
            timeout=timeout,
            check=False,
        )
        if proc.returncode:
            raise ToolError(f"{args[0]} failed: {(proc.stdout + proc.stderr)[-2000:]}")
        return proc.stdout

    def yosys(self, script: str, cwd: Path, *, ghdl: bool = False, timeout: float | None = None) -> str:
        """Run a Yosys script; its log is returned."""
        (cwd / "run.ys").write_text(script)
        args = ["yosys", "-q", "-l", "run.log", *(["-m", "ghdl"] if ghdl else []), "-s", "run.ys"]
        self.run(args, cwd, timeout)
        return (cwd / "run.log").read_text()


@dataclass
class Netlist:
    """A flattened Yosys JSON design whose top has a ``registers_in`` port and video outputs."""

    design: dict
    top: str

    @property
    def module(self) -> dict:
        """The top module."""
        return self.design["modules"][self.top]

    def ports(self, direction: str) -> dict[str, list]:
        """Port bit lists of one direction."""
        return {n: p["bits"] for n, p in self.module["ports"].items() if p["direction"] == direction}

    def save(self, path: Path) -> Path:
        """Write as Yosys JSON."""
        path.write_text(json.dumps(self.design))
        return path

    @classmethod
    def load(cls, path: Path, top: str) -> "Netlist":
        """Read Yosys JSON."""
        return cls(json.loads(Path(path).read_text()), top)


def _scratch(work: Path | None) -> Path:
    if work is not None:
        Path(work).mkdir(parents=True, exist_ok=True)
    return Path(tempfile.mkdtemp(prefix="hdl-", dir=work))


def elaborate(
    tools: Tools, sdk: Path, core: str, config: str, sources: Iterable[Path], work: Path
) -> Netlist:
    """``program_top`` from VHDL against the SDK core's entity and common RTL."""
    work.mkdir(parents=True, exist_ok=True)
    rtl = sdk / "fpga" / "common" / "rtl"
    files = [
        sdk / "fpga" / "core" / core / "rtl" / "core_pkg.vhd",
        sdk / "fpga" / "core" / core / "rtl" / "program_top.vhd",
    ]
    files += sorted(p for p in rtl.glob("*/*.vhd") if p.parent.name != "core_config")
    files += [rtl / "core_config" / f"{config}_pkg.vhd", *map(Path, sources)]
    lib = work / "lib"
    shutil.rmtree(lib, ignore_errors=True)
    lib.mkdir()
    opts = ["--std=08", "-fsynopsys", f"--workdir={lib}"]
    tools.run(["ghdl", "-i", *opts, *map(str, files)], work)
    tools.run(["ghdl", "-m", *opts, "program_top"], work)
    tools.yosys(
        f"ghdl {' '.join(opts)} program_top; hierarchy -top program_top; proc; flatten; memory -nomap;"
        " opt_clean; write_json base.json",
        work,
        ghdl=True,
    )
    return Netlist.load(work / "base.json", "program_top")


def _bits(module: dict) -> Iterable[list]:
    for cell in module["cells"].values():
        yield from cell["connections"].values()
    for net in module["netnames"].values():
        yield net["bits"]
    for port in module["ports"].values():
        yield port["bits"]


def _substitute(module: dict, table: Mapping[int, object]) -> None:
    for bits in _bits(module):
        bits[:] = [table.get(b, b) if isinstance(b, int) else b for b in bits]
    for net in module["netnames"].values():
        init = net.get("attributes", {}).get("init")
        if isinstance(init, str) and len(init) == len(net["bits"]):
            net["attributes"]["init"] = "".join(
                "x" if not isinstance(b, int) else c for c, b in zip(init, reversed(net["bits"]))
            )


def _fresh(module: dict) -> int:
    return 1 + max((b for bits in _bits(module) for b in bits if isinstance(b, int)), default=1)


def tie(net: Netlist, ties: Mapping[int, int], free: Mapping[str, Iterable[int]]) -> Netlist:
    """Drive ``registers_in`` bits with constants or named free inputs; undefined bits become free too.

    ``ties`` maps flat bit positions to 0/1; ``free`` names one 1-bit input per listed position.
    Positions in neither stay free under the name ``registers_in_<k>``.
    """
    out = Netlist(copy.deepcopy(net.design), net.top)
    module = out.module
    port = module["ports"].pop(PORT)["bits"]
    named = {k: name for name, ks in free.items() for k in ks}
    _substitute(module, {port[k]: str(v) for k, v in ties.items() if isinstance(port[k], int)})
    for k, bit in enumerate(port):
        if k not in ties and isinstance(bit, int):
            name = named.get(k, f"{PORT}_{k}")
            module["ports"].setdefault(name, {"direction": "input", "bits": []})["bits"].append(bit)
    nxt = _fresh(module)
    undef = []
    for bits in _bits(module):
        for i, b in enumerate(bits):
            if b in ("x", "z"):
                bits[i] = nxt
                undef.append(nxt)
                nxt += 1
    if undef:
        module["ports"]["undefined"] = {"direction": "input", "bits": undef}
    return out


def _settled(cell: dict) -> str | None:
    """The constant a gate-level FF holds once its enables have fired, if its next value is fixed."""
    match = _FF.match(cell["type"])
    d = cell["connections"].get("D", [None])[0]
    if not match or d not in ("0", "1"):
        return None
    reset = match.group(3)
    return d if not reset or reset == d else None


def fold(tools: Tools, net: Netlist, work: Path | None = None) -> Netlist:
    """Constant-propagate to a fixpoint: Yosys ``opt``/``abc``, then FFs whose next value is a constant."""
    tmp = _scratch(work)
    try:
        current = net
        for _ in range(MAX_ROUNDS):
            current.save(tmp / "in.json")
            tools.yosys(f"read_json in.json; hierarchy -top {net.top}; {FOLD}; write_json out.json", tmp)
            current = Netlist.load(tmp / "out.json", net.top)
            cells = current.module["cells"]
            settled = {n: v for n, c in cells.items() if (v := _settled(c)) is not None}
            if not settled:
                return current
            table = {}
            for name, value in settled.items():
                table[cells.pop(name)["connections"]["Q"][0]] = value
            _substitute(current.module, table)
            _rename(current.module)
        raise ToolError(f"fold did not converge in {MAX_ROUNDS} rounds")
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def _rename(module: dict) -> None:
    """Number cells afresh and drop hidden uninitialised wires, so a new pass's names cannot collide."""
    module["cells"] = {f"$fold${i}": c for i, c in enumerate(module["cells"].values())}
    module["netnames"] = {
        n: w
        for n, w in module["netnames"].items()
        if not w.get("hide_name") or "init" in w.get("attributes", {})
    }


def _directions(design: dict, cell: dict) -> dict[str, str]:
    if "port_directions" in cell:
        return cell["port_directions"]
    lib = design["modules"].get(cell["type"], {}).get("ports", {})
    return {p: lib[p]["direction"] for p in cell["connections"] if p in lib}


def reaching(net: Netlist, outputs: Iterable[str] | None = None) -> set[str]:
    """Input ports with any bit in the structural fan-in cone of the outputs."""
    module = net.module
    outs = net.ports("output") if outputs is None else {n: net.ports("output")[n] for n in outputs}
    src, dst = [], []
    for cell in module["cells"].values():
        dirs = _directions(net.design, cell)
        ins = [
            b for p, d in dirs.items() if d != "output" for b in cell["connections"][p] if isinstance(b, int)
        ]
        for p, d in dirs.items():
            if d in ("output", "inout"):
                for o in cell["connections"][p]:
                    if isinstance(o, int):
                        src += [o] * len(ins)
                        dst += ins
    root = _fresh(module)
    targets = [b for bits in outs.values() for b in bits if isinstance(b, int)]
    src += [root] * len(targets)
    dst += targets
    graph = coo_matrix((np.ones(len(src), dtype=bool), (src, dst)), shape=(root + 1, root + 1)).tocsr()
    seen = np.zeros(root + 1, dtype=bool)
    seen[breadth_first_order(graph, root, directed=True, return_predecessors=False)] = True
    return {
        name for name, bits in net.ports("input").items() if any(isinstance(b, int) and seen[b] for b in bits)
    }


@dataclass(frozen=True)
class Verdict:
    """Whether two folded designs can produce different video, and how that was decided."""

    differs: bool | None
    how: str


def _alone(net: Netlist, name: str) -> dict:
    """The design with its top renamed and library cells dropped, for a second netlist in one session."""
    return {"modules": {name: net.module}}


_COSIM = """`timescale 1ns/1ps
module tb;
{decls}
gold g({gconns});
gate h({hconns});
integer c, f;
initial begin
  for (c = 0; c < {cycles}; c = c + 1) begin
{drive}
    #5 {clock} = 1; #5 {clock} = 0;
    if ({differ}) begin $display("DIFFER %0d", c); $finish; end
  end
  $display("SAME"); $finish;
end
endmodule
"""


def cosimulate(
    tools: Tools, gold: Netlist, gate: Netlist, clock: str, *, cycles: int, seed: int = 1
) -> int | None:
    """First cycle at which the two designs' outputs differ under one random stimulus, from zeroed state."""
    tmp = _scratch(None)
    try:
        (tmp / "a.json").write_text(json.dumps(_alone(gold, "gold")))
        (tmp / "b.json").write_text(json.dumps(_alone(gate, "gate")))
        lib = json.dumps({"modules": {k: v for k, v in gold.design["modules"].items() if k != gold.top}})
        (tmp / "lib.json").write_text(lib)
        tools.yosys(
            "read_json lib.json; read_json a.json; read_json b.json; setundef -zero -undriven -init -params gold gate;"
            " select gold gate; write_verilog -noattr -selected designs.v",
            tmp,
        )
        ins = {n: len(b) for n, b in gold.ports("input").items()}
        outs = {n: len(b) for n, b in gold.ports("output").items() if n in gate.ports("output")}
        name = {n: f"w{i}" for i, n in enumerate([*ins, *outs])}
        decls = [f"reg [{w - 1}:0] {name[n]} = 0;" for n, w in ins.items()]
        decls += [f"wire [{w - 1}:0] {name[n]}_g, {name[n]}_h;" for n, w in outs.items()]
        gate_ins = gate.ports("input")

        def conns(side: str, has: Mapping) -> str:
            pins = [f".{_escape(n)}({name[n]})" for n in ins if n in has]
            return ", ".join(pins + [f".{_escape(n)}({name[n]}_{side})" for n in outs])

        drive = "\n".join(
            f"    {name[n]} = {{{', '.join(['$random(f)'] * -(-w // 32))}}};"
            for n, w in ins.items()
            if n != clock
        )
        differ = " || ".join(f"{name[n]}_g !== {name[n]}_h" for n in outs) or "0"
        (tmp / "tb.v").write_text(
            _COSIM.format(
                decls="\n".join(decls),
                gconns=conns("g", ins),
                hconns=conns("h", gate_ins),
                cycles=cycles,
                drive=f"    f = {seed} + c;\n" + drive if drive else "",
                clock=name[clock],
                differ=differ,
            )
        )
        cells = tools.share / "ice40" / "cells_sim.v"
        tools.run(["iverilog", "-g2012", "-o", "sim.vvp", "tb.v", "designs.v", str(cells)], tmp)
        out = tools.run(["vvp", "-n", "sim.vvp"], tmp)
        hit = re.search(r"DIFFER (\d+)", out)
        return int(hit.group(1)) if hit else None
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def _graph(net: Netlist) -> tuple[list, list, dict]:
    """Cells as (label, input sources) with sources a driving (cell, port, bit), a named input bit or a constant."""
    cells = list(net.module["cells"].values())
    driver, labels, sources = {}, [], []
    for i, cell in enumerate(cells):
        for port, d in _directions(net.design, cell).items():
            if d != "input":
                for k, b in enumerate(cell["connections"][port]):
                    driver[b] = (i, port, k)
    named = {b: ("in", n, k) for n, bits in net.ports("input").items() for k, b in enumerate(bits)}
    anchors = {}
    for n, bits in net.ports("output").items():
        for k, b in enumerate(bits):
            if b in driver:
                anchors.setdefault(driver[b][0], []).append((n, k, driver[b][1], driver[b][2]))
    init = {}
    for net_ in net.module["netnames"].values():
        value = net_.get("attributes", {}).get("init")
        if isinstance(value, str) and len(value) == len(net_["bits"]):
            init.update(zip(net_["bits"], reversed(value)))
    for i, cell in enumerate(cells):
        params = tuple(sorted((k, str(v)) for k, v in cell.get("parameters", {}).items()))
        q = tuple(init.get(b, "x") for b in cell["connections"].get("Q", ()))
        labels.append((cell["type"], params, q, tuple(sorted(anchors.get(i, ())))))
        dirs = _directions(net.design, cell)
        sources.append(
            [
                (p, k, driver.get(b) or named.get(b) or ("const", b))
                for p in sorted(cell["connections"])
                if dirs.get(p) != "output"
                for k, b in enumerate(cell["connections"][p])
            ]
        )
    outs = {
        (n, k): driver.get(b) or named.get(b) or ("const", b)
        for n, bits in net.ports("output").items()
        for k, b in enumerate(bits)
    }
    return labels, sources, outs


def _refine(labels: list, sources: list) -> list[int]:
    users: list[list] = [[] for _ in labels]
    for j, src in enumerate(sources):
        for p, k, s in src:
            if isinstance(s[0], int):
                users[s[0]].append((j, p, k, s[1], s[2]))
    colors = [hash(lab) for lab in labels]
    distinct = len(set(colors))
    for _ in range(256):
        colors = [
            hash(
                (
                    colors[i],
                    tuple(
                        (p, k, (colors[s[0]], s[1], s[2]) if isinstance(s[0], int) else s) for p, k, s in src
                    ),
                    tuple(sorted((colors[j], p, k, op, ok) for j, p, k, op, ok in users[i])),
                )
            )
            for i, src in enumerate(sources)
        ]
        if len(set(colors)) == distinct:
            break
        distinct = len(set(colors))
    return colors


def isomorphic(gold: Netlist, gate: Netlist) -> bool:
    """True only when an explicit cell mapping preserving every type, parameter and connection is found."""
    (la, sa, oa), (lb, sb, ob) = _graph(gold), _graph(gate)
    if len(la) != len(lb) or set(oa) != set(ob):
        return False
    ca, cb = _refine(la, sa), _refine(lb, sb)
    if sorted(ca) != sorted(cb):
        return False
    image = [0] * len(ca)
    for i, j in zip(sorted(range(len(ca)), key=ca.__getitem__), sorted(range(len(cb)), key=cb.__getitem__)):
        image[i] = j

    def mapped(src):
        return (image[src[0]], src[1], src[2]) if isinstance(src[0], int) else src

    return all(
        la[i] == lb[image[i]] and [(p, k, mapped(x)) for p, k, x in sa[i]] == sb[image[i]]
        for i in range(len(la))
    ) and all(mapped(v) == ob[key] for key, v in oa.items())


def equivalent(tools: Tools, gold: Netlist, gate: Netlist, *, seq: int = 4, timeout: float = 600) -> bool:
    """True when Yosys proves every matched signal equivalent (``equiv_simple``/``equiv_induct``)."""
    tmp = _scratch(None)
    try:
        (tmp / "a.json").write_text(json.dumps({"modules": {**gold.design["modules"], "gold": gold.module}}))
        (tmp / "b.json").write_text(json.dumps(_alone(gate, "gate")))
        script = (
            f"read_json a.json; delete {gold.top}; read_json b.json; equiv_make gold gate equiv; hierarchy -top equiv;"
            f" equiv_struct; equiv_simple -seq {seq}"
        )
        memories = any(c["type"].startswith("$mem") for c in gold.module["cells"].values())
        tail = "; equiv_status -assert" if memories else f"; equiv_induct -seq {seq}; equiv_status -assert"
        tools.yosys(script + tail, tmp, timeout=timeout)
        return True
    except (ToolError, subprocess.TimeoutExpired):
        return False
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def compare(tools: Tools, gold: Netlist, gate: Netlist, clock: str, *, cycles: int = 200_000) -> Verdict:
    """Whether two folded designs can drive different video: proof, concrete witness, or undecided."""
    if isomorphic(gold, gate):
        return Verdict(False, "the two folded designs are the same circuit")
    if equivalent(tools, gold, gate):
        return Verdict(False, "Yosys proves the two folded designs equivalent")
    hit = cosimulate(tools, gold, gate, clock, cycles=cycles)
    if hit is not None:
        return Verdict(True, f"outputs differ at cycle {hit} of a shared random stimulus from power-up")
    return Verdict(
        None, f"not proven equivalent, and {cycles} cycles of shared random stimulus show no difference"
    )


ICE40_PINS = {"clk": "ADV7611_CLK", "sck": "RP2040_SPI0_SCK", "sdi": "RP2040_SPI0_SDI"}
ICE40_PINS |= {"cs": "RP2040_SPI0_CS_N", "hs": "ADV7611_HSYNC", "vs": "ADV7611_VSYNC"}

_TB = """`timescale 1ns/1ps
module tb;
reg clk=0, sck=0, sdi=0, cs=1, hs=1, vs=1;
chip dut({conns});
always #5 clk=~clk;
integer f, t, i, k;
reg [9:0] pat [0:{n}];
task tick(input integer c); repeat(c) @(posedge clk); endtask
task bit_(input b); begin sdi=b; tick(4); sck=1; tick(4); sck=0; end endtask
task spi(input [4:0] a, input [9:0] d); begin
  cs=0; tick(6); for(k=4;k>=0;k=k-1) bit_(a[k]); bit_(1); for(k=9;k>=0;k=k-1) bit_(d[k]); tick(6); cs=1; tick(6);
end endtask
task dump; $fdisplay(f, "%b", {{{regs}}}); endtask
initial begin
  $readmemh("pats.hex", pat); f=$fopen("dump.txt","w"); tick(20);
  for(t=0;t<{trials};t=t+1) begin
    for(i=0;i<{latched};i=i+1) spi(i, pat[t*{latched}+i]);
    tick(10); dump; hs=0; tick(40); hs=1; tick(40); dump;
  end
  $fclose(f); $finish;
end
endmodule
"""


def _escape(name: str) -> str:
    return name if re.fullmatch(r"[A-Za-z_][A-Za-z0-9_$]*", name) else f"\\{name} "


def shadow_registers(
    tools: Tools, work: Path, trials: int = 24, seed: int = 0
) -> dict[tuple[int, int], list[str]]:
    """FFs holding each ``registers_in`` bit, found by writing SPI frames per the ABI and latching on HSYNC.

    A bit's FFs show its previous pattern before the HSYNC edge and the new one after it,
    across random patterns; the SPI RAM FFs, which already show the new one, are excluded.
    """
    module = Netlist.load(work / "chip.json", "chip").module
    pins = {v: k for k, v in ICE40_PINS.items()}
    conns = [
        f".{_escape(name)}({pins.get(name, '0')})"
        for name, p in module["ports"].items()
        if p["direction"] == "input"
    ]
    regs = [
        n for n, net in module["netnames"].items() if not net.get("hide_name") and _driven_by_ff(module, net)
    ]
    pats = np.random.default_rng(seed).integers(0, 1 << REGISTER_BITS, size=(trials, LATCHED_REGISTERS))
    (work / "pats.hex").write_text("\n".join(f"{v:03x}" for v in pats.ravel()))
    (work / "tb.v").write_text(
        _TB.format(
            conns=", ".join(conns),
            n=pats.size - 1,
            regs=", ".join(f"dut.{_escape(r)}" for r in regs),
            trials=trials,
            latched=LATCHED_REGISTERS,
        )
    )
    tools.run(
        ["iverilog", "-g2012", "-o", "sim.vvp", "tb.v", "chip.v", str(tools.share / "ice40" / "cells_sim.v")],
        work,
    )
    tools.run(["vvp", "-n", "sim.vvp"], work)
    widths = [len(module["netnames"][r]["bits"]) for r in regs]
    rows = np.array(
        [[c == "1" for c in line.strip()] for line in (work / "dump.txt").read_text().split()], bool
    )
    before, after = rows[0::2], rows[1::2]
    offsets = np.cumsum([0, *widths])
    want = ((pats[:, :, None] >> np.arange(REGISTER_BITS)) & 1).astype(bool).reshape(trials, -1)
    prev = np.vstack([np.zeros_like(want[:1]), want[:-1]])
    match = (after[:, :, None] == want[:, None, :]).all(0) & (before[1:, :, None] == prev[1:, None, :]).all(0)
    found: dict[tuple[int, int], list[str]] = {}
    for col, k in zip(*np.nonzero(match)):
        reg = int(np.searchsorted(offsets, col, side="right") - 1)
        msb_first = offsets[reg + 1] - 1 - col
        found.setdefault(divmod(int(k), REGISTER_BITS), []).append(f"{regs[reg]}:{msb_first}")
    return found


def _driven_by_ff(module: dict, net: dict) -> bool:
    bits = set(b for b in net["bits"] if isinstance(b, int))
    return bool(bits) and any(
        c["type"].startswith(("$dff", "$adff", "$sdff", "$dffe", "$adffe", "$sdffe", "$sdffce", "$aldff"))
        and bits & set(c["connections"].get("Q", []))
        for c in module["cells"].values()
    )


def decompile(tools: Tools, bitstream: bytes, pcf: Path, package: str, work: Path) -> tuple[Netlist, dict]:
    """The chip netlist of an iCE40 bitstream with its ``registers_in`` FFs cut out into a port."""
    work.mkdir(parents=True, exist_ok=True)
    (work / "chip.bin").write_bytes(bitstream)
    tools.run(["iceunpack", "chip.bin", "chip.asc"], work)
    (work / "raw.v").write_text(
        tools.run(["icebox_vlog", "-d", package, "-s", "-p", str(pcf), "chip.asc"], work)
    )
    tools.yosys(
        "read_verilog -lib +/ice40/cells_sim.v; read_verilog raw.v; hierarchy -top chip -purge_lib; proc; opt_clean;"
        " write_verilog -noattr chip.v; write_json chip.json",
        work,
    )
    found = shadow_registers(tools, work)
    net = Netlist.load(work / "chip.json", "chip")
    module = net.module
    driver = {
        c["connections"]["Q"][0]: name
        for name, c in module["cells"].items()
        if "Q" in c["connections"] and len(c["connections"]["Q"]) == 1
    }
    nxt = _fresh(module)
    port, alias = [], {}
    for k in range(REGISTER_COUNT * REGISTER_BITS):
        reg, bit = divmod(k, REGISTER_BITS)
        reg = REGISTER_COUNT - 1 - reg
        ids = list(dict.fromkeys(_netbit(module, ref) for ref in found.get((reg, bit), ())))
        if not ids:
            port.append(nxt)
            nxt += 1
            continue
        for b in ids:
            module["cells"].pop(driver[b])
            alias[b] = ids[0]
        port.append(ids[0])
    _substitute(module, alias)
    module["ports"][PORT] = {"direction": "input", "bits": port}
    mapping = {f"r{r}[{b}]": refs for (r, b), refs in sorted(found.items())}
    return net, mapping


def _netbit(module: dict, ref: str) -> int:
    name, idx = ref.rsplit(":", 1)
    return module["netnames"][name]["bits"][int(idx)]

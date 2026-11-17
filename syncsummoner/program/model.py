"""Per-program parameter model: the declarations, the bits each drives, and which bits a state leaves live.

Results are cached per program id, version, package digest, netlist source and
query under ``SYNCSUMMONER_PROGRAM_CACHE`` (default ``.program-cache``).
"""

from __future__ import annotations

import hashlib
import json
import os
import threading
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable, Mapping, Sequence

import numpy as np

from syncsummoner.progress import track

from . import hdl
from .declared import (
    CORE_REGISTERS,
    LATCHED_REGISTERS,
    PARAM_COUNT,
    REGISTER_BITS,
    REGISTER_COUNT,
    TIMING_REGISTER,
    Field,
    Kind,
    classes,
    field,
    flat_bit,
    kind,
    register_value,
    registers,
)
from .library import Library, Program
from .vmprog import ParamDecl, SectionType

DEFAULT_CONFIG = "hd_hdmi"
PACKAGE = "tq144:4k"
PCF = "fpga/hardware/rev_b/constraints/pin_map.pcf"
TIMING = "timing"


@dataclass(frozen=True)
class Liveness:
    """One parameter in one state: the bits still reaching the video, and what gates it if none do."""

    parameter_id: int
    live_bits: tuple[int, ...]
    gated_by: tuple[str, ...] = ()

    @property
    def live(self) -> bool:
        """True when any bit the parameter writes can still change the output."""
        return bool(self.live_bits)


@dataclass(frozen=True)
class Step:
    """Whether moving one parameter from ``a`` to ``b`` can change the video; ``None`` when undecided."""

    parameter_id: int
    a: int
    b: int
    matters: bool | None
    how: str


def _hash(obj) -> str:
    return hashlib.sha256(json.dumps(obj, sort_keys=True).encode()).hexdigest()[:20]


class ProgramModel:
    """Exact parameter liveness for one program, from its VHDL when published, else its bitstream."""

    def __init__(
        self,
        program: Program,
        library: Library,
        *,
        config: str = DEFAULT_CONFIG,
        tools: hdl.Tools | None = None,
        cache: Path | None = None,
        source: str | None = None,
    ):
        self.program = program
        self.library = library
        self.config = config
        self.tools = tools or hdl.Tools()
        self.source = source or ("rtl" if program.sources else "bitstream")
        root = Path(cache or os.environ.get("SYNCSUMMONER_PROGRAM_CACHE") or ".program-cache").resolve()
        self.cache = root / program.key / f"{self.source}-{config}"
        self._net: hdl.Netlist | None = None
        self._lock = threading.Lock()

    @property
    def params(self) -> tuple[ParamDecl, ...]:
        """The program's declarations, exactly as its package states them."""
        return self.program.package.config.params

    def decl(self, parameter_id: int) -> ParamDecl | None:
        """Declaration of one control."""
        return self.program.package.config.param(parameter_id)

    def kind(self, parameter_id: int) -> tuple[Kind, int | None]:
        """Declared kind and level count."""
        return kind(self.decl(parameter_id))

    @staticmethod
    def field(parameter_id: int) -> Field:
        """The ``registers_in`` word and bits a control writes."""
        return field(parameter_id)

    def initial(self) -> np.ndarray:
        """Manual state at load: each declaration's initial value at its raw 0..1023 scale."""
        state = np.zeros(PARAM_COUNT, dtype=np.int64)
        for d in self.params:
            span = d.max_value - d.min_value
            raw = (
                d.initial_value
                if span >= 1023
                else round((d.initial_value - d.min_value) * 1023 / max(span, 1))
            )
            state[d.parameter_id - 1] = raw
        return state

    def preset(self, name: str) -> np.ndarray:
        """Manual state of a factory preset."""
        return np.asarray(self.program.package.config.preset(name).values, dtype=np.int64)

    def _cached(self, kind_: str, key, compute):
        path = self.cache / f"{kind_}-{_hash(key)}.json"
        if path.exists():
            return json.loads(path.read_text())
        value = compute()
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(".tmp")
        tmp.write_text(json.dumps(value))
        tmp.replace(path)
        return value

    @property
    def netlist(self) -> hdl.Netlist:
        """``program_top`` from source, or the whole chip from the bitstream, with ``registers_in`` as a port."""
        with self._lock:
            if self._net is None:
                self._net = self._load()
        return self._net

    def _load(self) -> hdl.Netlist:
        path = self.cache / "base.json"
        if not path.exists():
            net = self._build(self.cache / "build")
            path.parent.mkdir(parents=True, exist_ok=True)
            net.save(path.with_suffix(".tmp")).replace(path)
        return hdl.Netlist.load(path, "program_top" if self.source == "rtl" else "chip")

    def _build(self, work: Path) -> hdl.Netlist:
        cfg = self.program.package.config
        if self.source == "rtl":
            return hdl.elaborate(
                self.tools, self.library.sdk, cfg.core, self.config, self.program.sources, work
            )
        net, mapping = hdl.decompile(self.tools, self.bitstream(), self.library.sdk / PCF, PACKAGE, work)
        (self.cache / "register-map.json").parent.mkdir(parents=True, exist_ok=True)
        (self.cache / "register-map.json").write_text(json.dumps(mapping, indent=1))
        return net

    def bitstream(self) -> bytes:
        """The package's bitstream for this IO configuration."""
        return self.program.package.bitstreams()[SectionType[self.config.upper()]]

    def _ties(self, state: np.ndarray, timing: int | None, free: Iterable[int]) -> tuple[dict, dict]:
        words = registers(self.params, state, timing)
        loose = {name for name in free}
        ties, named = {}, {}
        for pid in loose:
            if pid == TIMING:
                continue
            f = field(pid)
            named[f"p{pid}"] = list(f.flat)
        for reg in range(REGISTER_COUNT):
            if reg in CORE_REGISTERS or (reg == TIMING_REGISTER and (timing is None or TIMING in loose)):
                continue
            for bit in range(REGISTER_BITS):
                k = flat_bit(reg, bit)
                if not any(k in ks for ks in named.values()):
                    ties[k] = (words.get(reg, 0) >> bit) & 1 if reg < LATCHED_REGISTERS else 0
        return ties, named

    def _reach(self, state: np.ndarray, timing: int | None, free: Sequence) -> dict[int, list[int]]:
        """Live bits of each freed control, with everything else tied to ``state``."""
        ties, named = self._ties(state, timing, free)
        key = {"ties": sorted(ties.items()), "named": sorted((k, v) for k, v in named.items())}

        def compute():
            net = hdl.tie(
                self.netlist, ties, {f"{n}_b{i}": [k] for n, ks in named.items() for i, k in enumerate(ks)}
            )
            reached = hdl.reaching(hdl.fold(self.tools, net, self.cache))
            out = {}
            for name, ks in named.items():
                pid = int(name[1:])
                bits = field(pid).bits
                out[pid] = [bits[i] for i in range(len(ks)) if f"{name}_b{i}" in reached]
            return out

        return {int(k): v for k, v in self._cached("reach", key, compute).items()}

    def resolution(self) -> dict[int, tuple[int, ...]]:
        """Bits of each control the program reads at all, every register free."""
        free = [d.parameter_id for d in self.params] + [TIMING]
        reach = self._reach(np.zeros(PARAM_COUNT, dtype=np.int64), None, free)
        return {d.parameter_id: tuple(reach.get(d.parameter_id, ())) for d in self.params}

    def live(self, state: np.ndarray | None = None, timing: int | None = None, *, gates: bool = True) -> dict:
        """Each declared control's live bits with every other control held at ``state``.

        A control with no live bits lists the controls (or ``timing``) whose freeing revives it.
        """
        state = self.initial() if state is None else np.asarray(state, dtype=np.int64)
        pids = [d.parameter_id for d in self.params]
        with ThreadPoolExecutor(max_workers=os.cpu_count()) as pool:
            reach = dict(
                zip(
                    pids,
                    track(
                        pool.map(lambda p: self._reach(state, timing, [p])[p], pids),
                        total=len(pids),
                        desc=f"{self.program.name} live",
                        unit="param",
                    ),
                )
            )
            out = {p: Liveness(p, tuple(bits)) for p, bits in reach.items()}
            if not gates:
                return out
            dead = [p for p, bits in reach.items() if not bits]
            others = [*pids, *([TIMING] if timing is not None else [])]
            jobs = [(p, q) for p in dead for q in others if q != p]
            hits = list(
                track(
                    pool.map(lambda pq: bool(self._reach(state, timing, [pq[0], pq[1]])[pq[0]]), jobs),
                    total=len(jobs),
                    desc=f"{self.program.name} gates",
                    unit="pair",
                )
            )
        for p in dead:
            names = tuple(
                f"P{q}" if q != TIMING else TIMING for (pp, q), hit in zip(jobs, hits) if pp == p and hit
            )
            out[p] = Liveness(p, (), names)
        return out

    def _folded(self, state: np.ndarray, timing: int | None) -> hdl.Netlist:
        ties, _ = self._ties(state, timing, [])
        return hdl.fold(self.tools, hdl.tie(self.netlist, ties, {}), self.cache)

    def step_matters(
        self, state: np.ndarray | None, parameter_id: int, a: int, b: int, timing: int | None = None
    ) -> Step:
        """Whether changing one control from ``a`` to ``b`` can change the video, others held at ``state``."""
        state = self.initial() if state is None else np.asarray(state, dtype=np.int64)
        decl = self.decl(parameter_id)
        if decl is None:
            return Step(parameter_id, a, b, False, "the program declares no such parameter")
        live = self._reach(state, timing, [parameter_id])[parameter_id]
        mask = sum(1 << bit for bit in live) if len(field(parameter_id).bits) > 1 else int(bool(live))
        ra, rb = int(register_value(decl, a)), int(register_value(decl, b))
        if not (ra ^ rb) & mask:
            return Step(parameter_id, a, b, False, "the values differ only in bits no output depends on here")
        sa, sb = state.copy(), state.copy()
        sa[parameter_id - 1], sb[parameter_id - 1] = a, b
        key = {"a": sa.tolist(), "b": sb.tolist(), "timing": timing}

        def compute():
            clock = "clk" if self.source == "rtl" else hdl.ICE40_PINS["clk"]
            verdict = hdl.compare(self.tools, self._folded(sa, timing), self._folded(sb, timing), clock)
            return asdict(verdict)

        verdict = self._cached("step", key, compute)
        return Step(parameter_id, a, b, verdict["differs"], verdict["how"])

    def schedule(self, state: np.ndarray | None = None, timing: int | None = None) -> dict[int, np.ndarray]:
        """For each live control, one manual value per distinct setting the program can see in ``state``."""
        out = {}
        for pid, liveness in self.live(state, timing, gates=False).items():
            if liveness.live:
                groups = classes(self.decl(pid), liveness.live_bits)
                out[pid] = np.array([g[len(g) // 2] for g in groups], dtype=np.int64)
        return out

    def levels(self, parameter_id: int, live_bits: Sequence[int]) -> int:
        """Distinct values of a control the program can tell apart through ``live_bits``."""
        decl = self.decl(parameter_id)
        return 0 if decl is None or not live_bits else len(classes(decl, tuple(live_bits)))


def model(program: str | Program, library: Library | None = None, **kwargs) -> ProgramModel:
    """The parameter model of a program by device name, program id, or :class:`Program`."""
    library = library or Library.default()
    prog = library.find(program) if isinstance(program, str) else program
    return ProgramModel(prog, library, **kwargs)


def parse_state(model_: ProgramModel, spec: str | None, overrides: Mapping[int, int] = ()) -> np.ndarray:
    """A state from ``initial``, a preset name, or 12 comma-separated values, then per-control overrides."""
    if not spec or spec == "initial":
        state = model_.initial()
    elif "," in spec:
        state = np.asarray([int(v) for v in spec.split(",")], dtype=np.int64)
    else:
        state = model_.preset(spec)
    for pid, value in dict(overrides).items():
        state[pid - 1] = value
    return state

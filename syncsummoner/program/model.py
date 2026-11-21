"""Per-program parameter model: the declarations, the bits each drives, and which bits a state leaves live.

The netlist, its gate-level form and its observability conditions are built once and cached per
program id, version, package digest and netlist source under ``SYNCSUMMONER_PROGRAM_CACHE``
(default ``.program-cache``); every liveness query is then answered in-process from them.
"""

from __future__ import annotations

import hashlib
import json
import multiprocessing
import os
import threading
import time
from collections import OrderedDict
from concurrent.futures import ProcessPoolExecutor
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable, Mapping, Sequence

import numpy as np

from syncsummoner.progress import LOG, track

from . import hdl, observe, sat
from .declared import (
    CORE_REGISTERS,
    LATCHED_REGISTERS,
    PARAM_COUNT,
    REGISTER_BITS,
    REGISTER_COUNT,
    TIMING_REGISTER,
    VALUE_MAX,
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
from .ternary import Ternary
from .vmprog import ParamDecl, SectionType

DEFAULT_CONFIG = "hd_hdmi"
PACKAGE = "tq144:4k"
PCF = "fpga/hardware/rev_b/constraints/pin_map.pcf"
TIMING = "timing"
VIDEO_IN = "ADV7611_D"
#: Name of the video input bits' joint observability condition.
VIDEO = "video"
#: ``registers_in`` bits held as BDD variables: words 0-7 (the controls) and 8 (timing).
CONTROL_BITS = tuple(
    flat_bit(r, b) for r in range(TIMING_REGISTER + 1) for b in reversed(range(REGISTER_BITS))
)


def timing_bits() -> tuple[int, ...]:
    """``registers_in`` bits of the timing word."""
    return tuple(flat_bit(TIMING_REGISTER, b) for b in range(REGISTER_BITS))


#: Core-only words, free inputs of every query.
LOOSE_BITS = tuple(flat_bit(r, b) for r in CORE_REGISTERS for b in range(REGISTER_BITS))
#: Every manual value a control takes.
MANUAL = np.arange(VALUE_MAX + 1)


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


@dataclass(frozen=True)
class Bypass:
    """A setting of one control the model proves returns the input with every other control dead.

    ``value`` is the manual value written to bypass; ``proven`` every manual value the proof covers.
    """

    parameter_id: int
    value: int
    proven: tuple[int, ...]

    def holds(self, manual: np.ndarray | int) -> np.ndarray:
        """Whether each manual value of the control is a proven bypass."""
        return np.isin(np.asarray(manual, dtype=np.int64), self.proven)


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
        method: str | None = None,
    ):
        self.program = program
        self.library = library
        self.config = config
        self.tools = tools or hdl.Tools()
        self.source = source or ("rtl" if program.sources else "bitstream")
        root = Path(cache or os.environ.get("SYNCSUMMONER_PROGRAM_CACHE") or ".program-cache").resolve()
        self.cache = root / program.key / f"{self.source}-{config}"
        self._net: hdl.Netlist | None = None
        self._circuit: observe.Circuit | None = None
        self._conditions: dict[frozenset, observe.Conditions] = {}
        self._ternary: Ternary | None = None
        self._exact: sat.Exact | None = None
        #: ``bdd``, ``sat`` or ``ternary`` to answer by that method alone; None for the BDDs, then SAT.
        self.prefer = method
        self.method: str | None = None
        #: Bytes the BDDs may use; None for what the process may use when an analysis starts.
        self.memory: int | None = None
        self._outgrew = False
        self.timings: dict[str, float] = {}
        self._lock = threading.RLock()
        self._recent: OrderedDict[str, hdl.Netlist] = OrderedDict()
        self.note = ""

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
        tmp = path.with_suffix(f".{os.getpid()}.{threading.get_ident()}.tmp")
        tmp.write_text(json.dumps(value))
        tmp.replace(path)
        return value

    @property
    def netlist(self) -> hdl.Netlist:
        """``program_top`` from source, or the chip from the bitstream, with ``registers_in`` as a port."""
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

    def latched(self) -> dict[int, tuple[int, ...]]:
        """Bits of each control the bitstream latches from SPI: those its synthesised design reads."""
        if self.source != "bitstream":
            raise ValueError("latched bits come from a bitstream netlist")
        _ = self.netlist
        found = json.loads((self.cache / "register-map.json").read_text())
        out = {}
        for decl in self.params:
            f = field(decl.parameter_id)
            out[decl.parameter_id] = tuple(b for b in f.bits if f"r{f.register}[{b}]" in found)
        return out

    def bitstream(self) -> bytes:
        """The package's bitstream for this IO configuration."""
        return self.program.package.bitstreams()[SectionType[self.config.upper()]]

    def _ties(self, state: np.ndarray, timing: int | None, free: Iterable[int]) -> tuple[dict, dict]:
        words = registers(self.params, state, timing)
        loose = set(free)
        named = {f"p{pid}": list(field(pid).flat) for pid in loose if pid != TIMING}
        named_bits = {k for ks in named.values() for k in ks}
        ties = {}
        for reg in range(REGISTER_COUNT):
            if reg in CORE_REGISTERS or (reg == TIMING_REGISTER and (timing is None or TIMING in loose)):
                continue
            word = words.get(reg, 0) if reg < LATCHED_REGISTERS else 0
            for bit in range(REGISTER_BITS):
                k = flat_bit(reg, bit)
                if k not in named_bits:
                    ties[k] = (word >> bit) & 1
        return ties, named

    def ties(self, state: np.ndarray, timing: int | None, free: Iterable[int]) -> dict[int, int]:
        """Each ``registers_in`` bit held at ``state``, by flat position, with ``free`` (and an unset
        ``timing``) left out.
        """
        return self._ties(np.asarray(state, dtype=np.int64), timing, free)[0]

    @property
    def circuit(self) -> observe.Circuit:
        """The netlist in Yosys's gate library: one Yosys run per build, cached as ``gates.json``."""
        with self._lock:
            if self._circuit is None:
                path = self.cache / "gates.json"
                if not path.exists():
                    start = time.monotonic()
                    net = hdl.lower(self.tools, self.netlist, self.cache)
                    net.save(path.with_suffix(".tmp")).replace(path)
                    self.timings["lower"] = time.monotonic() - start
                design = json.loads(path.read_text(encoding="utf-8"))
                self._circuit = observe.Circuit.read(
                    design, "program_top" if self.source == "rtl" else "chip"
                )
        return self._circuit

    def _path(self, key: frozenset) -> Path:
        return self.cache / f"observe-{'-'.join(sorted(map(str, key))) or 'tied'}.json"

    def _job(self, key: frozenset, memory: int | None = None) -> tuple:
        """What one analysis needs, picklable: the circuit, its varying bits, its sources, its file and
        the bytes it may use (default: :attr:`memory`).
        """
        circuit = self.circuit
        nets = circuit.controls()
        video = "data_in" if self.source == "rtl" else VIDEO_IN
        sources = {
            f"c{k}": (f"c{k}", [nets[k]]) for p in key if p != TIMING for k in field(p).flat if k in nets
        }
        sources[VIDEO] = (None, [b for b, n in circuit.inputs_named().items() if n.startswith(video)])
        varying = sorted(k for p in key for k in (timing_bits() if p == TIMING else field(p).flat))
        budget = self.memory if memory is None else memory
        return circuit, varying, sources, self._path(key), self.cache / "observe.exhausted", budget

    def prepare(self, keys: Iterable[Iterable]) -> None:
        """Build the missing analyses of several freed sets at once, one process per CPU."""
        todo = [k for k in {frozenset(k) for k in keys} if not self._path(k).exists()]
        if len(todo) < 2 or self._spent():
            return
        start = time.monotonic()
        workers = min(hdl.cpus(), len(todo))
        jobs = [self._job(k, self._budget() // workers) for k in todo]
        context = multiprocessing.get_context("spawn")
        with ProcessPoolExecutor(max_workers=workers, mp_context=context) as pool:
            list(track(pool.map(_analyse, *zip(*jobs)), total=len(jobs), desc="analyses", unit="set"))
        self.timings["observe"] = self.timings.get("observe", 0.0) + time.monotonic() - start
        self.timings["analyses"] = self.timings.get("analyses", 0) + len(jobs)

    def _budget(self) -> int:
        """Bytes the BDDs may use now: :attr:`memory`, or what the process may use."""
        return observe.memory_limit() if self.memory is None else self.memory

    def _spent(self) -> bool:
        """Whether the BDDs do not answer: another method is preferred, or they outgrew a budget at
        least as large as this one (:func:`outgrown`).
        """
        if self.prefer not in (None, "bdd") or self._outgrew:
            return True
        return outgrown(self.cache / "observe.exhausted") >= self._budget()

    def conditions(self, free: Iterable = ()) -> observe.Conditions | None:
        """Observability conditions over the controls, with those in ``free`` (and ``timing``) varying.

        None once this program's BDDs have outgrown memory, so the SAT miter answers, or the ternary
        simulator where the miter cannot be built (``method``); :class:`~.observe.Exhausted` instead
        when the BDDs alone are to answer.
        """
        key = frozenset(free)
        circuit = self.circuit
        with self._lock:
            if not self._spent() and key not in self._conditions:
                path = self._path(key)
                if not path.exists():
                    start = time.monotonic()
                    _analyse(*self._job(key))
                    self.timings["observe"] = self.timings.get("observe", 0.0) + time.monotonic() - start
                    self.timings["analyses"] = self.timings.get("analyses", 0) + 1
                    self._outgrew = not path.exists()
                if path.exists():
                    self._conditions[key] = observe.Conditions.load(path, CONTROL_BITS)
            method = self.method if self.method in ("sat", "ternary") else "bdd"
            if self._spent() and method == "bdd" and self.prefer == "bdd":
                raise observe.Exhausted(f"{self.program.name}: the BDDs outgrew {self._budget()} bytes")
            if self._spent() and method == "bdd":
                method = "ternary" if self.prefer == "ternary" else self._build_exact(circuit)
            if method != self.method:
                self.method = method
                LOG.info("%s: liveness by %s", self.program.name, method)
            if method == "ternary" and self._ternary is None:
                self._ternary = Ternary(circuit)
            return self._conditions.get(key) if method == "bdd" else None

    def _build_exact(self, circuit: observe.Circuit) -> str:
        """``sat`` once the miter is built, else ``ternary`` with the reason logged."""
        start = time.monotonic()
        try:
            self._exact = sat.Exact(circuit, keep=PARAM_COUNT + 1)
        except (MemoryError, RuntimeError, ValueError) as err:
            LOG.warning("%s: the SAT miter cannot be built (%s)", self.program.name, err)
            return "ternary"
        self.timings["miter"] = time.monotonic() - start
        return "sat"

    @property
    def exact(self) -> sat.Exact:
        """The SAT miter's queries, once the BDDs no longer answer."""
        if self._exact is None:
            self.conditions(())
        if self._exact is None:
            raise RuntimeError(f"{self.program.name} is answered by {self.method}, not SAT")
        return self._exact

    def _reach(self, state: np.ndarray, timing: int | None, free: Sequence) -> dict[int, list[int]]:
        """Live bits of each freed control, with everything else tied to ``state``."""
        return self._reach_many([state], timing, free)[0]

    def _reach_many(
        self, states: Sequence[np.ndarray], timing: int | None, free: Sequence
    ) -> list[dict[int, list[int]]]:
        pids = [p for p in dict.fromkeys(free) if p != TIMING]
        ties = [self._ties(np.asarray(s, dtype=np.int64), timing, free)[0] for s in states]
        cond = self.conditions({*free, *([TIMING] if timing is None else [])})
        if cond is not None:
            seen = [{k: cond.live(f"c{k}", t) for p in pids for k in field(p).flat} for t in ties]
        elif self.method == "sat":
            flat = [k for p in pids for k in field(p).flat]
            seen = [dict(zip(flat, self._exact.live(t, flat))) for t in ties]
        else:
            lanes = self._ternary.live(_lanes(ties))
            seen = [
                {k: bool(lanes[f"c{k}"][i]) for p in pids for k in field(p).flat} for i in range(len(ties))
            ]
        return [
            {p: [b for b, k in zip(field(p).bits, field(p).flat) if hit[k]] for p in pids} for hit in seen
        ]

    def reach(
        self, state: np.ndarray, free: Sequence[int], timing: int | None = None
    ) -> dict[int, list[int]]:
        """Live bits of each control in ``free`` with all of them free together and the rest at ``state``.

        Dead is exact for every value of the freed controls; live means some value reaches the video.
        """
        return self._reach(np.asarray(state, dtype=np.int64), timing, list(free))

    def revivable(
        self, state: np.ndarray, param: int, free: Sequence[int], timing: int | None = None
    ) -> bool:
        """Whether some held setting of the controls in ``free`` makes ``param`` live, the rest at ``state``.

        Exact from ``param``'s own conditions or by SAT over the settings of ``free`` one control at a
        time; the ternary simulator answers with ``free`` left as X.
        """
        state = np.asarray(state, dtype=np.int64)
        cond = self.conditions({param, *([TIMING] if timing is None else [])})
        if cond is None and self.method == "sat":
            return self._settings(state, timing, param, list(free)) is not None
        if cond is None:
            return bool(self._reach(state, timing, [param, *free])[param])
        ties, _ = self._ties(state, timing, [param, *free])
        return any(cond.live(f"c{k}", ties) for k in field(param).flat)

    def _settings(
        self, state: np.ndarray, timing: int | None, param: int, free: Sequence[int]
    ) -> list[int] | None:
        """A held manual value of each control in ``free``, in order, making ``param`` live by SAT,
        or None. Each prefix is kept only when the rest, left free, can still complete it.
        """
        bits = field(param).flat
        held = [k for q in free for k in field(q).flat]
        if not self._exact.alive([self._ties(state, timing, [param, *free])[0]], bits, held)[0]:
            return None
        if not free:
            return []
        head, rest = free[0], list(free[1:])
        for group in track(classes(self.decl(head), field(head).bits), desc=f"P{param} by P{head}"):
            moved = state.copy()
            moved[head - 1] = group[len(group) // 2]
            tail = self._settings(moved, timing, param, rest)
            if tail is not None:
                return [int(moved[head - 1]), *tail]
        return None

    def reach_many(
        self, states: Sequence[np.ndarray], free: Sequence[int], timing: int | None = None
    ) -> list[dict[int, list[int]]]:
        """:meth:`reach` at each of ``states``, in one pass where the ternary simulator answers."""
        return self._reach_many(states, timing, list(free))

    def _key(self, **key) -> dict:
        """``key`` for an analysis cached on disk, marked when the ternary screen alone answers it."""
        return {**key, **({"coarse": True} if self.prefer == "ternary" else {})}

    def resolution(self) -> dict[int, tuple[int, ...]]:
        """Bits of each control the program reads at all, every register free."""

        def compute():
            free = [d.parameter_id for d in self.params] + [TIMING]
            reach = self._reach(np.zeros(PARAM_COUNT, dtype=np.int64), None, free)
            return [[d.parameter_id, list(reach.get(d.parameter_id, ()))] for d in self.params]

        return {p: tuple(bits) for p, bits in self._cached("resolution", self._key(), compute)}

    def live(self, state: np.ndarray | None = None, timing: int | None = None, *, gates: bool = True) -> dict:
        """Each declared control's live bits with every other control held at ``state``.

        A control with no live bits lists the controls (or ``timing``) whose freeing revives it.
        """
        state = self.initial() if state is None else np.asarray(state, dtype=np.int64)
        pids = [d.parameter_id for d in self.params]
        self.prepare({p, *([TIMING] if timing is None else [])} for p in pids)
        out = {p: Liveness(p, tuple(self._reach(state, timing, [p])[p])) for p in pids}
        if not gates:
            return out
        others = [*pids, *([TIMING] if timing is not None else [])]
        dead = [p for p in pids if not out[p].live]
        self.prepare({p, q, *([TIMING] if timing is None else [])} for p in dead for q in others if q != p)
        for p in track([p for p in pids if not out[p].live], desc=f"{self.program.name} gates", unit="param"):
            names = tuple(
                f"P{q}" if q != TIMING else TIMING
                for q in others
                if q != p and self._reach(state, timing, [p, q])[p]
            )
            out[p] = Liveness(p, (), names)
        return out

    def _folded(self, state: np.ndarray, timing: int | None, free: Sequence[int] = ()) -> hdl.Netlist:
        """The design folded at ``state``; the last few are kept, as a step ends where the next starts."""
        ties, named = self._ties(state, timing, free)
        key = _hash({"ties": sorted(ties.items()), "named": sorted((k, v) for k, v in named.items())})
        with self._lock:
            if key in self._recent:
                self._recent.move_to_end(key)
                return self._recent[key]
        names = {f"{n}_b{i}": [k] for n, ks in named.items() for i, k in enumerate(ks)}
        net = hdl.fold(self.tools, hdl.tie(self.netlist, ties, names), self.cache)
        with self._lock:
            self._recent[key] = net
            while len(self._recent) > 2 * hdl.cpus():
                self._recent.popitem(last=False)
        return net

    def visible(
        self, state: np.ndarray, parameter_id: int, a: int, b: int, timing: int | None = None
    ) -> bool:
        """Whether ``a`` and ``b`` differ in any bit still reaching the video with the rest at ``state``."""
        decl = self.decl(parameter_id)
        if decl is None:
            return False
        live = self._reach(np.asarray(state, dtype=np.int64), timing, [parameter_id])[parameter_id]
        mask = sum(1 << bit for bit in live) if len(field(parameter_id).bits) > 1 else int(bool(live))
        return bool((int(register_value(decl, a)) ^ int(register_value(decl, b))) & mask)

    def step_matters(
        self,
        state: np.ndarray | None,
        parameter_id: int,
        a: int,
        b: int,
        timing: int | None = None,
        free: Sequence[int] = (),
    ) -> Step:
        """Whether changing one control from ``a`` to ``b`` can change the video, others held at ``state``.

        Controls in ``free`` are left as inputs: no change is then proven for every value they take.
        """
        state = self.initial() if state is None else np.asarray(state, dtype=np.int64)
        decl = self.decl(parameter_id)
        free = sorted(set(free) - {parameter_id})
        if decl is None:
            return Step(parameter_id, a, b, False, "the program declares no such parameter")
        if not free and not self.visible(state, parameter_id, a, b, timing):
            return Step(parameter_id, a, b, False, "the values differ only in bits no output depends on here")
        sa, sb = state.copy(), state.copy()
        sa[parameter_id - 1], sb[parameter_id - 1] = a, b
        key = {"a": sa.tolist(), "b": sb.tolist(), "timing": timing, **({"free": free} if free else {})}

        def compute():
            clock = "clk" if self.source == "rtl" else hdl.ICE40_PINS["clk"]
            verdict = hdl.compare(
                self.tools, self._folded(sa, timing, free), self._folded(sb, timing, free), clock
            )
            return asdict(verdict)

        verdict = self._cached("step", key, compute)
        return Step(parameter_id, a, b, verdict["differs"], verdict["how"])

    def carries_input(self, state: np.ndarray, timing: int | None = None) -> bool:
        """Whether any video input bit is still in the fan-in cone of the outputs at ``state``."""
        ties, _ = self._ties(np.asarray(state, dtype=np.int64), timing, [])
        prefix = "data_in" if self.source == "rtl" else VIDEO_IN
        cond = self.conditions([TIMING] if timing is None else [])
        if cond is not None:
            return cond.live(VIDEO, ties)
        if self.method == "sat":
            return self._exact.carries([ties])[0]
        lanes = self._ternary.live(_lanes([ties]))
        return any(bool(v[0]) for name, v in lanes.items() if name.startswith(prefix))

    def bypass(self, state: np.ndarray | None = None, timing: int | None = None) -> Bypass | None:
        """The control setting that returns the input with every other control dead, None if there is none.

        A declared switch turned on that leaves every other control dead at ``state`` comes first; else
        the first control with settings :meth:`passes` proves, written at the middle of their longest run.
        """
        base = self.initial() if state is None else np.asarray(state, dtype=np.int64)
        found = self._cached(
            "bypass", self._key(state=base.tolist(), timing=timing), lambda: self._bypass(base, timing)
        )
        return None if found is None else Bypass(found[0], found[1], tuple(found[2]))

    def _bypass(self, base: np.ndarray, timing: int | None) -> list | None:
        """:meth:`bypass` as ``[parameter_id, value, proven]``, computed."""
        for decl in self.params:
            if kind(decl)[0] is not Kind.BOOLEAN:
                continue
            on = base.copy()
            on[decl.parameter_id - 1] = VALUE_MAX
            live = self.live(on, timing, gates=False)
            if not any(l.live for p, l in live.items() if p != decl.parameter_id) and self.carries_input(
                on, timing
            ):
                same = register_value(decl, MANUAL) == register_value(decl, VALUE_MAX)
                return [decl.parameter_id, VALUE_MAX, MANUAL[same].tolist()]
        for decl in self.params:
            proven = self.passes(decl.parameter_id, base, timing)
            if proven.size:
                runs = np.split(proven, np.flatnonzero(np.diff(proven) > 1) + 1)
                run = max(runs, key=len)
                return [decl.parameter_id, int(run[len(run) // 2]), proven.tolist()]
        return None

    def passes(
        self, parameter_id: int, state: np.ndarray | None = None, timing: int | None = None
    ) -> np.ndarray:
        """Manual values of one control at which the input reaches the video and no bit of any other
        declared control does, for every value of them together; undeclared words are held at ``state``.

        Each other control dead alone at ``state`` is necessary, so the joint analysis is built only
        for a control that passes that first.
        """
        base = self.initial() if state is None else np.asarray(state, dtype=np.int64)
        decl = self.decl(parameter_id)
        if decl is None:
            return MANUAL[:0]
        others = [d.parameter_id for d in self.params if d.parameter_id != parameter_id]
        wild = [] if timing is not None else [TIMING]
        self.prepare([{q, *wild} for q in others] + [set(wild)])
        ok = self._heard(base, timing, parameter_id, [], [VIDEO])
        for free in [*([q] for q in others), *([others] if others else [])]:
            if ok is None or not ok.any():
                break
            heard = self._heard(base, timing, parameter_id, free)
            ok = None if heard is None else ok & ~heard
        if ok is None and self.method == "sat":
            return MANUAL[self._passes_sat(base, timing, parameter_id, others)]
        if ok is None:
            return MANUAL[self._passes_ternary(base, timing, parameter_id, others)]
        return MANUAL[ok[register_value(decl, MANUAL)]]

    def _heard(
        self, base: np.ndarray, timing: int | None, param: int, free: Sequence[int], names=None
    ) -> np.ndarray | None:
        """Per register value of ``param``, whether ``names`` (the bits of ``free``) can reach the video
        for some value of ``free``, the rest at ``base``; None once the BDDs have outgrown memory.
        """
        cond = self.conditions({*free, *([TIMING] if timing is None else [])})
        if cond is None:
            return None
        ties, _ = self._ties(base, timing, [*free, param])
        names = names or [f"c{k}" for q in free for k in field(q).flat]
        return cond.heard(names, ties, field(param).flat)

    def _passes_sat(
        self, base: np.ndarray, timing: int | None, param: int, others: Sequence[int]
    ) -> np.ndarray:
        """:meth:`passes` by SAT, per setting of ``param``: the input observable, and no bit of the
        other controls, varying together.
        """
        groups = classes(self.decl(param), field(param).bits)
        states = np.repeat(base[None, :], len(groups), axis=0)
        states[:, param - 1] = [g[len(g) // 2] for g in groups]
        heard = np.flatnonzero(self._exact.carries([self._ties(s, timing, [])[0] for s in states]))
        flat = [k for q in others for k in field(q).flat]
        alive = self._exact.alive([self._ties(states[i], timing, others)[0] for i in heard], flat)
        ok = np.zeros(VALUE_MAX + 1, dtype=bool)
        for i, live in zip(heard, alive):
            ok[groups[i]] = not live
        return ok

    def _passes_ternary(
        self, base: np.ndarray, timing: int | None, param: int, others: Sequence[int]
    ) -> np.ndarray:
        """:meth:`passes` by the ternary simulator, the other controls left as X, per manual value."""
        groups = classes(self.decl(param), field(param).bits)
        states = np.repeat(base[None, :], len(groups), axis=0)
        states[:, param - 1] = [g[0] for g in groups]
        dead = [not any(r.values()) for r in self._reach_many(states, timing, list(others))]
        ties = [self._ties(s, timing, [])[0] for s in states]
        prefix = "data_in" if self.source == "rtl" else VIDEO_IN
        lanes = self._ternary.live(_lanes(ties))
        heard = np.any([v for name, v in lanes.items() if name.startswith(prefix)], axis=0)
        ok = np.zeros(VALUE_MAX + 1, dtype=bool)
        for g, d, h in zip(groups, dead, heard):
            ok[g] = d and bool(h)
        return ok

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


def model(
    program: str | Program, library: Library | None = None, *, source: str | None = None, **kwargs
) -> ProgramModel:
    """The parameter model of a program by device name, program id, or :class:`Program`.

    With no ``source``, the published VHDL is used only where it reads exactly the bits the
    package's own bitstream latches; otherwise the bitstream is, with the mismatch in ``note``.
    """
    library = library or Library.default()
    prog = library.find(program) if isinstance(program, str) else program
    if source is not None or not prog.sources:
        return ProgramModel(prog, library, source=source, **kwargs)
    rtl = ProgramModel(prog, library, source="rtl", **kwargs)
    chip = ProgramModel(prog, library, source="bitstream", **kwargs)
    reads, latched = rtl.resolution(), chip.latched()
    if reads == latched:
        return rtl
    differ = [
        f"P{p} {_bits(reads[p])} where the bitstream latches {_bits(latched[p])}"
        for p in reads
        if reads[p] != latched[p]
    ]
    chip.note = "published source does not match the package: it reads " + "; ".join(differ)
    return chip


def outgrown(spent: Path) -> int:
    """Bytes the BDDs outgrew, from the record :func:`_analyse` leaves; -1 with no such record."""
    try:
        return int(json.loads(spent.read_text(encoding="utf-8"))["budget"])
    except (OSError, ValueError, TypeError, KeyError):
        return -1


def _analyse(
    circuit: observe.Circuit, varying: list, sources: dict, path: Path, spent: Path, memory: int | None
) -> None:
    """One analysis into ``path``, or the budget it outgrew (``memory`` bytes, default what the process
    may use) and why into ``spent``.
    """
    budget = observe.memory_limit() if memory is None else memory
    try:
        analysis = observe.Analysis(circuit, CONTROL_BITS, LOOSE_BITS, limit=budget, varying=varying)
        found = analysis.observe(sources)
        found.save(path)
    except observe.Exhausted as err:
        if budget > outgrown(spent):
            spent.write_text(json.dumps({"budget": budget, "reason": str(err)}), encoding="utf-8")


def _lanes(ties: Sequence[Mapping[int, int]]) -> np.ndarray:
    """Tie sets as a lane-by-bit matrix of 0, 1 and -1 for an untied bit."""
    out = np.full((len(ties), REGISTER_COUNT * REGISTER_BITS), -1, dtype=np.int8)
    for i, t in enumerate(ties):
        if t:
            keys = np.fromiter(t, dtype=np.int64)
            out[i, keys] = np.fromiter(t.values(), dtype=np.int8)
    return out


def _bits(bits: Sequence[int]) -> str:
    return "[" + ",".join(map(str, sorted(bits, reverse=True))) + "]" if bits else "nothing"


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

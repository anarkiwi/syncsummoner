"""The device's modulation layer under MIDI clock, archived once by readback with no video.

Each setting runs its tempo-locked operators (:data:`TEMPO_OPERATORS`) on all 12 slots under a host clock
paced at its tempo, reading every slot's combined output ``o`` beside the transport's beat ``phase``. The
model is ``o = manual + (space - 512) * wave(frac(rate(time) * beats), slope)``, clipped, toggles at 512.
"""

from __future__ import annotations

import json
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Iterator

import numpy as np
from scipy.signal import lombscargle
from scipy.stats import qmc

from syncsummoner.device.profile import PARAM_COUNT, PARAM_MAX
from syncsummoner.progress import track

#: The operators MIDI clock drives: Sync LFO and Clock Div.
TEMPO_OPERATORS = (7, 20)
#: MIDI clock ticks a quarter note.
PPQN = 24
#: ``transport status`` phase units a beat: 16.16 fixed point.
PHASE_ONE = 1 << 16
#: The space value firmware reads as no depth; depth is signed about it.
KNOB_MIDDLE = 512
#: Tempo range swept, in beats a minute, log-uniform.
TEMPO = (40.0, 240.0)
#: A slot's knobs in ``modulation set`` order.
KNOBS = ("manual", "time", "space", "slope")
#: Slots the panel switches drive, P7-P11: under a source each reads on while its modulation is above the
#: middle of its operator's wave for positive depth, below it for negative, whatever its manual value.
TOGGLES = (np.arange(PARAM_COUNT) >= 6) & (np.arange(PARAM_COUNT) <= 10)
#: Cycles over which a rate estimate is trusted, at least, in the window it was read over.
CYCLES = 2.0
#: Beats of clock a setting sends before MIDI Start, so the device's tempo estimate has settled when its
#: beat phase starts running.
PREROLL_BEATS = 2
#: Wave cells: positions over a cycle and slope bands.
CELLS = (128, 32)
#: Positions over a cycle a rate's fold is judged in.
FOLD_BINS = 32
#: One readback: setting, request bounds, transport counters, host ticks sent and every slot's ``o``.
POLL = np.dtype(
    [
        ("setting", "<i4"),
        ("t0_ns", "<i8"),
        ("t1_ns", "<i8"),
        ("field", "<i8"),
        ("phase", "<i8"),
        ("bpm_x100", "<i8"),
        ("playing", "?"),
        ("ticks", "<i8"),
        ("o", "<i8", (PARAM_COUNT,)),
    ]
)


@dataclass(frozen=True)
class Design:
    """Settings to run: a tempo each, and per slot an operator and its knobs, ``(n, 12)`` each."""

    bpm: np.ndarray
    operator: np.ndarray
    manual: np.ndarray
    time: np.ndarray
    space: np.ndarray
    slope: np.ndarray

    @classmethod
    def sobol(cls, settings: int, seed: int) -> Design:
        """A scrambled Sobol design of ``settings``, a power of two, each slot drawing its own knobs."""
        m = int(settings).bit_length() - 1
        if settings < 1 or 1 << m != settings:
            raise ValueError(f"a Sobol design keeps its balance at a power of two settings, not {settings}")
        u = qmc.Sobol(1 + PARAM_COUNT * 5, scramble=True, seed=seed).random_base2(m)
        lo, hi = np.log(TEMPO)
        slots = u[:, 1:].reshape(settings, 5, PARAM_COUNT)
        knobs = np.minimum(slots[:, 1:] * (PARAM_MAX + 1), PARAM_MAX).astype(np.int64)
        ops = np.asarray(TEMPO_OPERATORS)[(slots[:, 0] * len(TEMPO_OPERATORS)).astype(np.int64)]
        return cls(np.round(np.exp(lo + u[:, 0] * (hi - lo)), 2), ops, *knobs.transpose(1, 0, 2))

    @classmethod
    def scan(cls, step: int, tempos: tuple[float, ...] = TEMPO) -> Design:
        """Every ``step``-th time value of each operator at each tempo, at full depth and middle slope.

        The manual value sits just below a toggle's threshold, so a toggle slot crosses it.
        """
        times = np.arange(0, PARAM_MAX + 1, step)
        times = np.r_[times, np.full(-times.size % PARAM_COUNT, times[-1])].reshape(-1, PARAM_COUNT)
        rows = [(bpm, op, t) for bpm in tempos for op in TEMPO_OPERATORS for t in times]
        full = np.full((len(rows), PARAM_COUNT), PARAM_MAX)
        middle = np.full_like(full, KNOB_MIDDLE)
        return cls(
            np.array([r[0] for r in rows]),
            np.array([np.full(PARAM_COUNT, r[1]) for r in rows]),
            middle - 1,
            np.array([r[2] for r in rows]),
            full,
            middle,
        )

    def __len__(self) -> int:
        return int(self.bpm.size)

    def arrays(self) -> dict[str, np.ndarray]:
        """The design's fields by name."""
        return {name: getattr(self, name) for name in ("bpm", "operator", *KNOBS)}

    def slots(self) -> dict[str, np.ndarray]:
        """Per setting and slot, ``(n, 12)``: the setting's tempo broadcast beside each knob."""
        return {"bpm": np.broadcast_to(self.bpm[:, None], self.operator.shape), **self.arrays()}


class Clock(threading.Thread):
    """MIDI clock at ``bpm``: tick ``n`` is due ``n`` tick periods after the first, so no drift builds up."""

    def __init__(
        self,
        midi: Any,
        bpm: float,
        *,
        clock: Callable[[], float] = time.monotonic,
        sleep: Callable[[float], None] = time.sleep,
    ):
        super().__init__(daemon=True)
        self.midi, self.period, self.clock, self.sleep = midi, 60.0 / (bpm * PPQN), clock, sleep
        self.sent, self.halt = 0, threading.Event()

    def run(self) -> None:
        origin = self.clock()
        while not self.halt.is_set():
            delay = origin + self.sent * self.period - self.clock()
            if delay > 0:
                self.sleep(delay)
            self.midi.clock_tick()
            self.sent += 1


def configure(shell: Any, design: Design, k: int) -> None:
    """Setting ``k`` on every slot, its operator first: a slot with no source ignores its knobs."""
    for i in range(PARAM_COUNT):
        shell.command("modulation", "source", str(i), str(int(design.operator[k, i])))
        shell.set_modulation(i, *(int(getattr(design, name)[k, i]) for name in KNOBS))


def snapshot(shell: Any) -> dict:
    """What a sweep changes: the transport, and each slot's manual value, knobs and source."""
    transport = shell.transport_status()
    return {"bpm_x100": int(transport["bpm_x100"]), "state": shell.program_state()}


def restore(shell: Any, found: dict) -> None:
    """Every slot back as found, knobs set under an active source, then the transport stopped at its tempo."""
    state = found["state"]
    for i in range(PARAM_COUNT):
        shell.command("modulation", "source", str(i), str(TEMPO_OPERATORS[0]))
        shell.set_modulation(i, *(int(state[key][i]) for key in ("m", "t", "sp", "sl")))
        shell.command("modulation", "source", str(i), str(int(state["sr"][i])))
    shell.stop()
    shell.set_bpm(found["bpm_x100"] / 100)


def poll(shell: Any, k: int, ticks: int) -> tuple:
    """One ``transport status`` then one ``modulation status``, as a :data:`POLL` row."""
    t0 = time.monotonic_ns()
    said = json.loads(shell.command("transport", "status").payload)
    o = [int(m["o"]) for m in shell.modulation_status()["modulators"]]
    return (
        k,
        t0,
        time.monotonic_ns(),
        int(said["field_count"]),
        int(said["phase"]),
        int(said["bpm_x100"]),
        said.get("state") == "playing",
        ticks,
        o,
    )


def sweep(shell: Any, midi: Any, design: Design, seconds: float) -> np.ndarray:
    """Each setting for ``seconds`` from MIDI Start to Stop under its own clock; the found state is restored
    however the sweep ends. Returns the :data:`POLL` rows."""
    found, rows = snapshot(shell), []
    try:
        for k in track(range(len(design)), desc="modulation settings", total=len(design), unit="setting"):
            configure(shell, design, k)
            clock = Clock(midi, float(design.bpm[k]))
            clock.start()
            time.sleep(PREROLL_BEATS * 60.0 / float(design.bpm[k]))
            base = clock.sent
            midi.start()
            end = time.monotonic() + seconds
            while time.monotonic() < end:
                rows.append(poll(shell, k, clock.sent - base))
            clock.halt.set()
            clock.join()
            midi.stop()
    finally:
        restore(shell, found)
    return np.array(rows, dtype=POLL)


def save(path: str | Path, runs: dict[str, tuple[Design, np.ndarray]]) -> None:
    """Named runs, each a design and its readback, in one ``.npz``."""
    arrays = {f"{name}.{key}": v for name, (d, polls) in runs.items() for key, v in d.arrays().items()}
    np.savez_compressed(path, **arrays, **{f"{name}.polls": polls for name, (_, polls) in runs.items()})


def load(path: str | Path) -> dict[str, tuple[Design, np.ndarray]]:
    """The named runs :func:`save` wrote."""
    with np.load(path) as f:
        names = sorted({key.rsplit(".", 1)[0] for key in f.files})
        fields = ("bpm", "operator", *KNOBS)
        return {n: (Design(*(f[f"{n}.{key}"] for key in fields)), f[f"{n}.polls"]) for n in names}


def started(polls: np.ndarray) -> Iterator[tuple[int, np.ndarray]]:
    """Each setting's polls from MIDI Start on, dropping those still showing the run before's phase."""
    for k in np.unique(polls["setting"]):
        mine = polls[polls["setting"] == k]
        fell = np.flatnonzero(np.diff(mine["phase"]) < 0)
        yield int(k), mine[fell[-1] + 1 :] if fell.size else mine


def beats(polls: np.ndarray) -> np.ndarray:
    """The device's beat position at each poll."""
    return polls["phase"] / PHASE_ONE


def tracking(polls: np.ndarray) -> np.ndarray:
    """Device beats less the beats the host had sent, at every poll once the clock has sent a beat."""
    gaps = [beats(p) - p["ticks"] / PPQN for _, p in started(polls)]
    out = np.concatenate(gaps) if gaps else np.zeros(0)
    ticks = np.concatenate([p["ticks"] for _, p in started(polls)]) if gaps else np.zeros(0)
    return out[ticks >= PPQN]


def rate(b: np.ndarray, o: np.ndarray, fields_per_beat: float) -> float:
    """Cycles a beat of a slot's output, NaN unless it lies between :data:`CYCLES` over the window and the
    Nyquist rate of the device's field sampling."""
    span = float(b[-1] - b[0]) if b.size > 1 else 0.0
    if span <= 0 or np.ptp(o) == 0:
        return float("nan")
    freqs = np.geomspace(CYCLES / span, fields_per_beat / 2, 4096)
    power = lombscargle(b, o - o.mean(), 2 * np.pi * freqs)
    peak = int(np.argmax(power))
    if peak in (0, freqs.size - 1):
        return float("nan")
    fine = np.linspace(freqs[peak - 1], freqs[peak + 1], 512)
    return float(fine[np.argmax(lombscargle(b, o - o.mean(), 2 * np.pi * fine))])


def _fields_per_beat(polls: np.ndarray, bpm: float) -> float:
    dt = (polls["t1_ns"][-1] - polls["t1_ns"][0]) / 1e9
    fields = polls["field"][-1] - polls["field"][0]
    return float(fields / dt * 60.0 / bpm) if dt > 0 and fields > 0 else float("nan")


def folded(r: float, b: np.ndarray, o: np.ndarray, bins: int = FOLD_BINS) -> float:
    """Share of the variance of ``o`` left within ``bins`` positions over a cycle of ``r``."""
    if np.var(o) == 0:
        return 0.0
    cell = np.minimum((np.mod(r * b, 1.0) * bins).astype(np.int64), bins - 1)
    count = np.bincount(cell, minlength=bins)
    mean = np.bincount(cell, weights=o, minlength=bins) / np.maximum(count, 1)
    return float(np.mean((o - mean[cell]) ** 2) / np.var(o))


def refine(r: float, seen: list[tuple[np.ndarray, np.ndarray, float]]) -> tuple[float, float]:
    """The rate within a cycle's drift over the longest window of ``r`` folding every run best, and its score:
    the middle of the rates tied best, since every rate in that interval explains the readings equally."""
    span = max(float(np.ptp(b)) for b, _, _ in seen)
    grid = r + np.linspace(-1.0, 1.0, 4 * CELLS[0] + 1) / span
    scores = np.array([sum(folded(g, b, o) for b, o, _ in seen) for g in grid])
    tied = np.flatnonzero(scores <= scores.min() + np.spacing(1.0) * len(seen))
    return float(np.median(grid[tied])), float(scores.min())


@dataclass(frozen=True)
class Rates:
    """Each operator's cycles a beat by time value, read from a scan: ``knobs`` and ``rates`` per operator."""

    knobs: dict[int, np.ndarray]
    rates: dict[int, np.ndarray]

    @classmethod
    def read(cls, design: Design, polls: np.ndarray) -> Rates:
        """Each scanned time value's rate: of every run's estimate, the one folding all its runs best, so an
        estimate aliased by field sampling at one tempo loses to one that explains every tempo."""
        runs: dict[tuple[int, int], list[tuple[np.ndarray, np.ndarray, float]]] = {}
        for k, mine in started(polls):
            per_beat = _fields_per_beat(mine, float(design.bpm[k]))
            for i in range(PARAM_COUNT):
                key = (int(design.operator[k, i]), int(design.time[k, i]))
                runs.setdefault(key, []).append((beats(mine), mine["o"][:, i].astype(float), per_beat))
        knobs: dict[int, list[int]] = {op: [] for op in TEMPO_OPERATORS}
        rates: dict[int, list[float]] = {op: [] for op in TEMPO_OPERATORS}
        for (op, t), seen in sorted(runs.items()):
            candidates = [r for b, o, f in seen if np.isfinite(r := rate(b, o, f))]
            if candidates and op in knobs:
                knobs[op].append(t)
                rates[op].append(min((refine(r, seen) for r in candidates), key=lambda rs: rs[1])[0])
        return cls({op: np.array(v) for op, v in knobs.items()}, {op: np.array(v) for op, v in rates.items()})

    def extend(self, design: Design, polls: np.ndarray, keep: np.ndarray) -> Rates:
        """These rates with every kept setting's time values added, each taking whichever rate of the scanned
        values either side folds its own readings better: settings between scan steps place each step's edge.
        """
        knobs = {op: list(self.knobs[op]) for op in TEMPO_OPERATORS}
        rates = {op: list(self.rates[op]) for op in TEMPO_OPERATORS}
        for k, mine in started(polls):
            if not keep[k]:
                continue
            for i in range(PARAM_COUNT):
                op, t = int(design.operator[k, i]), int(design.time[k, i])
                scanned = self.knobs[op]
                if scanned.size == 0:
                    continue
                at = int(np.searchsorted(scanned, t))
                sides = {float(self.rates[op][j]) for j in (at - 1, at) if 0 <= j < scanned.size}
                o = mine["o"][:, i].astype(float)
                if len(sides) > 1 and np.ptp(o) > 0:
                    knobs[op].append(t)
                    rates[op].append(min(sides, key=lambda r, b=beats(mine), o=o: folded(r, b, o)))
        order = {op: np.argsort(knobs[op], kind="stable") for op in TEMPO_OPERATORS}
        return Rates(
            {op: np.asarray(knobs[op])[order[op]] for op in TEMPO_OPERATORS},
            {op: np.asarray(rates[op])[order[op]] for op in TEMPO_OPERATORS},
        )

    def at(self, op: np.ndarray, time_: np.ndarray) -> np.ndarray:
        """Cycles a beat at each operator and time value: the nearest known time value's."""
        out = np.full(np.shape(op), np.nan)
        for o in TEMPO_OPERATORS:
            m = np.asarray(op) == o
            if m.any() and self.knobs[o].size:
                near = np.abs(np.asarray(time_)[m][:, None] - self.knobs[o][None, :]).argmin(axis=1)
                out[m] = self.rates[o][near]
        return out


def _samples(rates: Rates, design: Design, polls: np.ndarray, keep: np.ndarray | None = None) -> dict:
    """Every continuous slot's readings with the cycle position, slope, depth and manual it was read at."""
    cols: dict[str, list[np.ndarray]] = {
        key: [] for key in ("op", "x", "slope", "depth", "manual", "o", "setting")
    }
    slots = design.slots()
    for k, mine in started(polls):
        if keep is not None and not keep[k]:
            continue
        b = beats(mine)
        for i in range(PARAM_COUNT):
            r = rates.at(slots["operator"][k : k + 1, i], slots["time"][k : k + 1, i])[0]
            n = b.size
            cols["op"].append(np.full(n, slots["operator"][k, i]))
            cols["x"].append(np.mod(r * b, 1.0))
            cols["slope"].append(np.full(n, slots["slope"][k, i]))
            cols["depth"].append(np.full(n, slots["space"][k, i] - KNOB_MIDDLE))
            cols["manual"].append(np.full(n, slots["manual"][k, i]))
            cols["o"].append(mine["o"][:, i])
            cols["setting"].append(np.full(n, k * PARAM_COUNT + i))
    out = {key: np.concatenate(v) for key, v in cols.items()}
    out["toggle"] = TOGGLES[out["setting"] % PARAM_COUNT]
    return out


def _cell(x: np.ndarray, slope: np.ndarray) -> np.ndarray:
    xs = np.minimum((x * CELLS[0]).astype(np.int64), CELLS[0] - 1)
    ss = np.minimum((slope * CELLS[1] // (PARAM_MAX + 1)).astype(np.int64), CELLS[1] - 1)
    return xs * CELLS[1] + ss


def _bilinear(grid: np.ndarray, x: np.ndarray, slope: np.ndarray) -> np.ndarray:
    """Cell means interpolated between cell centres, periodic over the cycle, over the cells that hold any."""
    g = grid.reshape(CELLS)
    fx, fs = x * CELLS[0] - 0.5, np.clip(slope * CELLS[1] / (PARAM_MAX + 1) - 0.5, 0, CELLS[1] - 1)
    x0, s0 = np.floor(fx).astype(np.int64), np.minimum(np.floor(fs).astype(np.int64), CELLS[1] - 2)
    wx, ws = fx - x0, fs - s0
    total, weight = np.zeros(x.shape), np.zeros(x.shape)
    for dx, dy, w in (
        (0, 0, (1 - wx) * (1 - ws)),
        (1, 0, wx * (1 - ws)),
        (0, 1, (1 - wx) * ws),
        (1, 1, wx * ws),
    ):
        v = g[(x0 + dx) % CELLS[0], s0 + dy]
        ok = np.isfinite(v)
        total += np.where(ok, w * v, 0.0)
        weight += np.where(ok, w, 0.0)
    return np.where(weight > 0, total / np.where(weight > 0, weight, 1.0), np.nan)


@dataclass(frozen=True)
class Model:
    """Each operator's wave over cycle position and slope, from unclipped continuous readings."""

    rates: Rates
    waves: dict[int, np.ndarray]

    @classmethod
    def fit(cls, rates: Rates, design: Design, polls: np.ndarray, keep: np.ndarray | None = None) -> Model:
        """Waves as each cell's mean of ``(o - manual) / depth`` over readings off the rails at depth."""
        s = _samples(rates, design, polls, keep)
        use = (~s["toggle"]) & (s["o"] > 0) & (s["o"] < PARAM_MAX) & (np.abs(s["depth"]) * 4 >= KNOB_MIDDLE)
        use &= np.isfinite(s["x"])
        waves = {}
        for op in TEMPO_OPERATORS:
            m = use & (s["op"] == op)
            cell = _cell(s["x"][m], s["slope"][m])
            w = (s["o"][m] - s["manual"][m]) / s["depth"][m]
            total = np.bincount(cell, weights=w, minlength=np.prod(CELLS))
            count = np.bincount(cell, minlength=np.prod(CELLS))
            waves[op] = np.where(count > 0, total / np.maximum(count, 1), np.nan)
        return cls(rates, waves)

    def predict(self, s: dict) -> np.ndarray:
        """Each reading's predicted ``o`` from its operator, cycle position, slope, depth and manual."""
        w = np.full(s["x"].shape, np.nan)
        for op in TEMPO_OPERATORS:
            m = (s["op"] == op) & np.isfinite(s["x"])
            w[m] = _bilinear(self.waves[op], s["x"][m], s["slope"][m])
        middle = np.zeros(w.shape)
        for op in TEMPO_OPERATORS:
            middle[s["op"] == op] = 0.5 * (np.nanmin(self.waves[op]) + np.nanmax(self.waves[op]))
        o = np.clip(np.rint(s["manual"] + s["depth"] * w), 0, PARAM_MAX)
        return np.where(s["toggle"], np.where(s["depth"] * (w - middle) > 0, PARAM_MAX, 0), o)


@dataclass(frozen=True)
class Accuracy:
    """Held-out error: continuous RMS and median absolute error, toggle agreement, readings predicted."""

    rms: dict[int, float]
    median: dict[int, float]
    toggles: float
    predicted: float

    def lines(self) -> list[str]:
        """The accuracy as report lines."""
        out = [
            f"op {op}: RMS {self.rms[op]:.1f}, median |error| {self.median[op]:.1f} of {PARAM_MAX}"
            for op in TEMPO_OPERATORS
        ]
        return out + [f"toggles agree {self.toggles:.1%}", f"readings predicted {self.predicted:.1%}"]


def evaluate(rates: Rates, design: Design, polls: np.ndarray, folds: int = 4) -> Accuracy:
    """Fit on all but one fold of settings, rate steps placed by those settings too; predict the fold."""
    fold = np.arange(len(design)) % folds
    pieces = []
    for f in range(folds):
        known = rates.extend(design, polls, fold != f)
        model = Model.fit(known, design, polls, fold != f)
        s = _samples(known, design, polls, fold == f)
        pieces.append((s, model.predict(s)))
    s = {key: np.concatenate([p[0][key] for p in pieces]) for key in pieces[0][0]}
    pred = np.concatenate([p[1] for p in pieces])
    ok = np.isfinite(pred)
    err = pred - s["o"]
    cont, tog = ok & ~s["toggle"], ok & s["toggle"]
    rms = {op: float(np.sqrt(np.mean(err[cont & (s["op"] == op)] ** 2))) for op in TEMPO_OPERATORS}
    med = {op: float(np.median(np.abs(err[cont & (s["op"] == op)]))) for op in TEMPO_OPERATORS}
    return Accuracy(rms, med, float(np.mean(err[tog] == 0)) if tog.any() else float("nan"), float(ok.mean()))


def report(runs: dict[str, tuple[Design, np.ndarray]], folds: int = 4) -> list[str]:
    """Rates from the ``scan`` run, then the ``sobol`` run's beat tracking and held-out accuracy."""
    rates = Rates.read(*runs["scan"])
    lines = [
        f"op {op} cycles a beat by time: "
        + " ".join(f"{t}:{r:.4g}" for t, r in zip(rates.knobs[op], rates.rates[op]))
        for op in TEMPO_OPERATORS
    ]
    gap = tracking(runs["sobol"][1])
    lines.append(
        f"device beats less host beats: mean {gap.mean():+.4f}, sd {gap.std():.4f}, "
        f"worst {np.abs(gap).max():.4f}"
    )
    return lines + evaluate(rates, *runs["sobol"], folds=folds).lines()


def add_parser(probe_sub: Any) -> None:
    """The ``probe modulation`` commands."""
    node = probe_sub.add_parser("modulation", help="archive the tempo-locked modulation layer by readback")
    sub = node.add_subparsers(dest="modulation_cmd", required=True)
    run = sub.add_parser("sweep", help="scan each operator's rates, then a Sobol sweep, on the device")
    run.add_argument("--out", required=True, help="the .npz to write")
    run.add_argument("--settings", type=int, default=64, help="Sobol settings, a power of two")
    run.add_argument("--seconds", type=float, default=16.0, help="seconds each setting runs")
    run.add_argument("--scan-step", type=int, default=16, help="time values apart the rate scan reads")
    run.add_argument("--seed", type=int, default=0)
    run.set_defaults(func=_sweep_cmd)
    fit = sub.add_parser("fit", help="rates, beat tracking and held-out accuracy of a sweep's model")
    fit.add_argument("sweep", help="a sweep's .npz")
    fit.add_argument("--folds", type=int, default=4)
    fit.set_defaults(func=_fit_cmd)


def _sweep_cmd(args: Any) -> int:
    from syncsummoner.device.transport import Transport

    scan, sobol = Design.scan(args.scan_step), Design.sobol(args.settings, args.seed)
    print(
        f"{len(scan) + len(sobol)} settings, "
        f"{(len(scan) + len(sobol)) * (args.seconds + PREROLL_BEATS * 60 / TEMPO[0]) / 60:.1f} min at most"
    )
    with Transport.open() as port:
        shell, midi = port.device.shell, port.device.midi
        for p in range(1, PARAM_COUNT + 1):
            midi.set_param(p, 0)
            time.sleep(1.0 / 50)
        runs = {
            name: (d, sweep(shell, midi, d, args.seconds)) for name, d in (("scan", scan), ("sobol", sobol))
        }
    save(args.out, runs)
    print("\n".join(report(runs)))
    return 0


def _fit_cmd(args: Any) -> int:
    print("\n".join(report(load(args.sweep), folds=args.folds)))
    return 0

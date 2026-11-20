"""The modulation layer: its designs, rate scan, wave model and held-out accuracy on a simulated device."""

# pylint: disable=missing-function-docstring,missing-class-docstring

import json
import threading

import numpy as np
import pytest

from syncsummoner import cli
from syncsummoner.probe import modlayer as ml

FPS = 30.0
POLLS_PER_FIELD = 4
LFO_RATES = np.array([1 / 16, 1 / 8, 1 / 4, 1 / 3, 1 / 2, 2 / 3, 1, 4 / 3, 3 / 2, 2, 3, 4, 6, 8, 12, 16])
DIV_RATES = 1.0 / np.arange(1, 17)


def true_rate(op, time_):
    table = LFO_RATES if op == 7 else DIV_RATES
    return table[np.minimum(np.asarray(time_) * table.size // 1024, table.size - 1)]


def wave(op, x, slope):
    if op == 7:
        bend = slope / 1023.0
        return np.where(x < 0.5, 2 * x, 2 - 2 * x) * 2 * bend + np.sin(2 * np.pi * x) * (1 - bend) - bend
    return (x < slope / 1023.0).astype(float)


def simulate(design, seconds, lag_beats=0.0):
    """Polls of a device whose beat phase runs at each setting's tempo from Start, a stale first poll each."""
    rows, n = [], int(seconds * FPS * POLLS_PER_FIELD)
    for k in range(len(design)):
        bpm = float(design.bpm[k])
        field = np.arange(n) // POLLS_PER_FIELD
        b = field * bpm / 60.0 / FPS
        phase = np.round(b * ml.PHASE_ONE).astype(np.int64)
        ticks = np.floor((b + lag_beats) * ml.PPQN).astype(np.int64)
        o = np.empty((n, 12), dtype=np.int64)
        for i in range(12):
            op, t = int(design.operator[k, i]), int(design.time[k, i])
            mod = (design.space[k, i] - 512) * wave(op, np.mod(true_rate(op, t) * b, 1.0), design.slope[k, i])
            raw = design.manual[k, i] + mod
            o[:, i] = np.where(ml.TOGGLES[i], np.where(raw >= 512, 1023, 0), np.clip(np.rint(raw), 0, 1023))
        t_ns = (np.arange(n) * 1e9 / FPS / POLLS_PER_FIELD).astype(np.int64)
        stale = (k, 0, 0, 999, 10 * ml.PHASE_ONE, 12000, True, 0, o[0])
        rows.append(stale)
        rows += [
            (k, a, a + 1000, f, p, int(bpm * 100), True, tk, oo)
            for a, f, p, tk, oo in zip(t_ns, field, phase, ticks, o)
        ]
    return np.array(rows, dtype=ml.POLL)


@pytest.fixture(name="runs", scope="module")
def _runs():
    scan = ml.Design.scan(32)
    sobol = ml.Design.sobol(64, seed=3)
    return {"scan": (scan, simulate(scan, 16.0)), "sobol": (sobol, simulate(sobol, 8.0))}


def test_sobol_design_draws_tempo_operators_and_knobs_in_range():
    d = ml.Design.sobol(64, seed=1)
    assert d.operator.shape == (64, 12) and set(np.unique(d.operator)) == set(ml.TEMPO_OPERATORS)
    assert ml.TEMPO[0] <= d.bpm.min() and d.bpm.max() <= ml.TEMPO[1]
    for name in ml.KNOBS:
        knob = getattr(d, name)
        assert knob.min() >= 0 and knob.max() <= 1023 and np.unique(knob).size > 48
    assert np.median(d.bpm) == pytest.approx(np.sqrt(np.prod(ml.TEMPO)), rel=0.1)
    with pytest.raises(ValueError, match="power of two"):
        ml.Design.sobol(48, seed=1)


def test_scan_reads_every_step_of_each_operator_at_each_tempo():
    d = ml.Design.scan(16)
    for op in ml.TEMPO_OPERATORS:
        for bpm in ml.TEMPO:
            m = (d.operator[:, 0] == op) & (d.bpm == bpm)
            assert set(np.unique(d.time[m])) == set(range(0, 1024, 16))
    assert (d.space == 1023).all() and (d.slope == 512).all()


def test_rates_recover_each_operators_table_from_the_scan(runs):
    rates = ml.Rates.read(*runs["scan"])
    for op in ml.TEMPO_OPERATORS:
        assert rates.knobs[op].size > 0.9 * 32
        drift = np.abs(rates.rates[op] - true_rate(op, rates.knobs[op])) * 64.0
        assert drift.max() < 1.0 / ml.FOLD_BINS + 1.0 / (2 * ml.CELLS[0])


def test_held_out_settings_are_predicted_from_the_others(runs):
    rates = ml.Rates.read(*runs["scan"])
    acc = ml.evaluate(rates, *runs["sobol"])
    assert acc.predicted > 0.95
    assert acc.median[7] < 8 and acc.median[20] < 2
    assert acc.toggles > 0.9


def test_a_wrong_rate_table_is_caught_by_the_held_out_error(runs):
    rates = ml.Rates.read(*runs["scan"])
    wrong = ml.Rates(rates.knobs, {op: r * 1.1 for op, r in rates.rates.items()})
    assert ml.evaluate(wrong, *runs["sobol"]).median[7] > 4 * ml.evaluate(rates, *runs["sobol"]).median[7]


def test_tracking_drops_stale_polls_and_measures_the_device_lead(runs):
    design = runs["sobol"][0]
    gap = ml.tracking(simulate(design, 4.0, lag_beats=-0.25))
    assert gap.mean() == pytest.approx(0.25, abs=1 / ml.PPQN)
    assert np.abs(ml.tracking(runs["sobol"][1])).max() <= 1 / ml.PPQN


def test_rate_is_nan_when_the_window_holds_too_few_cycles_or_none():
    b = np.linspace(0, 4, 400)
    assert np.isnan(ml.rate(b, np.sin(2 * np.pi * b / 16), 15.0))
    assert np.isnan(ml.rate(b, np.zeros_like(b), 15.0))
    assert ml.rate(b, np.sin(2 * np.pi * 1.5 * b), 15.0) == pytest.approx(1.5, rel=1e-3)


def test_save_and_load_round_trip_and_fit_reports(runs, tmp_path, capsys):
    path = tmp_path / "mod.npz"
    ml.save(path, runs)
    back = ml.load(path)
    for name, (d, polls) in runs.items():
        np.testing.assert_array_equal(back[name][0].time, d.time)
        np.testing.assert_array_equal(back[name][1], polls)
    assert cli.main(["probe", "modulation", "fit", str(path), "--folds", "2"]) == 0
    out = capsys.readouterr().out
    assert "op 7 cycles a beat" in out and "toggles agree" in out and "device beats less host beats" in out


class FakeShell:
    """Records commands; answers transport and modulation status from a simulated phase."""

    def __init__(self):
        self.state = {"m": [3] * 12, "t": [100] * 12, "sp": [200] * 12, "sl": [300] * 12, "sr": [0] * 12}
        self.bpm_x100, self.calls, self.playing = 5143, [], False
        self.sources = list(self.state["sr"])

    def transport_status(self):
        return {"bpm_x100": self.bpm_x100}

    def program_state(self):
        return json.loads(json.dumps(self.state))

    def command(self, *words):
        self.calls.append(words)
        if words[:2] == ("modulation", "source"):
            self.sources[int(words[2])] = int(words[3])

        class Reply:
            payload = json.dumps(
                {
                    "field_count": len(self.calls),
                    "phase": 4369 * len(self.calls),
                    "bpm_x100": 12000,
                    "state": "playing",
                }
            )

        return Reply()

    def set_modulation(self, i, manual, *knobs):
        if self.sources[i]:
            for key, v in zip(("t", "sp", "sl"), knobs):
                self.state[key][i] = v
        self.state["m"][i] = manual

    def modulation_status(self):
        return {"modulators": [{"s": s, "m": 0, "o": 512} for s in self.sources]}

    def stop(self):
        self.playing = False

    def set_bpm(self, bpm):
        self.bpm_x100 = int(round(bpm * 100))


class FakeMidi:
    def __init__(self):
        self.ticks, self.starts, self.stops, self.lock = 0, 0, 0, threading.Lock()

    def clock_tick(self, count=1):
        with self.lock:
            self.ticks += count

    def start(self):
        self.starts += 1

    def stop(self):
        self.stops += 1


def test_sweep_runs_every_setting_and_restores_what_it_found(monkeypatch):
    monkeypatch.setattr(ml, "PREROLL_BEATS", 0)
    shell, midi = FakeShell(), FakeMidi()
    found = shell.program_state()
    d = ml.Design.sobol(2, seed=0)
    d = ml.Design(np.array([240.0, 240.0]), d.operator, d.manual, d.time, d.space, d.slope)
    polls = ml.sweep(shell, midi, d, 0.05)
    assert set(np.unique(polls["setting"])) == {0, 1} and midi.starts == midi.stops == 2
    assert shell.program_state() == found and shell.sources == found["sr"] and shell.bpm_x100 == 5143


def test_clock_paces_ticks_from_its_origin_without_drift():
    now = [0.0]
    midi = FakeMidi()

    def sleep(dt):
        now[0] += dt
        if midi.ticks >= 96:
            clock.halt.set()

    clock = ml.Clock(midi, 120.0, clock=lambda: now[0], sleep=sleep)
    clock.run()
    assert clock.sent == 97 and now[0] == pytest.approx(96 * 60 / (120 * 24))

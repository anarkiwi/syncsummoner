"""Locked takes: plans, clocks, calibration and a whole take against a simulated device."""

import contextlib
import json
import types

import numpy as np
import pytest

from syncsummoner.compose import evidence as ev
from syncsummoner.compose import lock as L
from syncsummoner.compose import locktake as LT
from syncsummoner.device import v4l2rec
from syncsummoner.device.playout import FLIP
from syncsummoner.device.profile import PARAM_MAX, ParamSpec, Role
from syncsummoner.device.v4l2rec import FRAME
from syncsummoner.device.wire import READ, SEND, Poller, Sender

FPS = 30.0
P = 1e9 / FPS
#: Tick-to-capture delay the simulated device and card have: tick ``n`` latches played frame ``n + 2``.
DELTA = 1.7 * P
S0 = 5_000_000_000
T0 = S0 + 50_000_000
TAU = T0 + 0.3 * P
SIDE = 4
sleep_until_on = LT._sleep_until  # pylint: disable=protected-access


def specs(roles=None):
    """Twelve parameter specs, ``roles`` mapping index to role."""
    roles = roles or {}
    return [
        ParamSpec(index=i, name=f"p{i}", native_min=0.0, native_max=1.0, role=roles.get(i, Role.EFFECT))
        for i in range(1, 13)
    ]


def info_with(roles=None):
    """A program info carrying ``specs(roles)``."""
    return types.SimpleNamespace(params=specs(roles))


@pytest.mark.parametrize(
    "roles, want",
    [
        ({1: Role.BYPASS, 12: Role.WET}, {1: PARAM_MAX}),
        ({12: Role.WET}, {12: 0}),
        ({5: Role.WET}, None),
        ({}, None),
    ],
)
def test_the_dry_state_is_a_bypass_on_or_a_crossfader_wet_control_off(roles, want):
    """Bypass wins over the crossfader; a wet control off the crossfader slot is no dry state."""
    assert LT.dry_state(info_with(roles)) == want


def test_a_frame_plan_targets_the_first_frame_at_or_after_its_time_less_the_bias():
    """``ceil((t - bias) * fps)``, an exact boundary its own frame, negatives clamped, stable order."""
    auto = types.SimpleNamespace(
        times=np.array([0.1, 0.115, -0.5, 0.0, 0.06, 0.0]),
        indices=np.array([1, 2, 3, 4, 5, 6]),
        values=np.array([10, 20, 30, 40, 50, 60]),
    )
    plan = LT.frame_plan(auto, fps=FPS, early_bias_s=0.015)
    assert plan.dtype == LT.PLAN
    assert plan["target"].tolist() == [0, 0, 0, 2, 3, 3]
    assert plan["param"].tolist() == [3, 4, 6, 5, 1, 2]
    assert plan["value"].tolist() == [30, 40, 60, 50, 10, 20]


def test_the_initial_state_is_the_working_point_then_frame_zeros_writes():
    """Working values in device units, overridden in order by every write targeting frame 0."""
    plan = np.array([(0, 2, 7), (0, 2, 9), (1, 3, 5)], dtype=LT.PLAN)
    assert LT.initial_state({2: 0.5, 3: True, 4: 2000}, plan) == {2: 9, 3: PARAM_MAX, 4: PARAM_MAX}


@pytest.mark.parametrize(
    "value, want",
    [
        (True, PARAM_MAX),
        (np.bool_(False), 0),
        (0.5, 512),
        (1.5, PARAM_MAX),
        (-0.2, 0),
        (np.float32(0.25), 256),
    ]
    + [(-4, 0), (300, 300), (np.int64(5000), PARAM_MAX)],
)
def test_working_values_are_clipped_into_device_units(value, want):
    """Booleans are the extremes, floats a fraction of full scale, integers clipped."""
    assert LT.to_units(value) == want


def test_the_boot_count_is_read_from_its_setting():
    """Setting 9, as the device reports it in JSON."""
    asked = []
    shell = types.SimpleNamespace(
        get_setting=lambda key: asked.append(key) or json.dumps({"key": key, "type": "uint32", "value": 7})
    )
    assert LT.boot_count(shell) == 7 and asked == [LT.BOOT_COUNT_KEY]


def test_small_frames_keeps_only_whole_frames_with_the_strip_cropped(monkeypatch):
    """A partial tail frame is dropped; the crop removes the strip rows."""
    seen = {}
    frame = np.arange(3 * 2 * 3, dtype=np.uint8)

    def run(argv, **kw):
        seen["argv"], seen["kw"] = argv, kw
        return types.SimpleNamespace(stdout=np.r_[frame, frame + 1, frame[:5]].tobytes())

    monkeypatch.setattr(LT.subprocess, "run", run)
    out = LT.small_frames("take.mkv", size=(3, 2), strip_px=8, ffmpeg="ff")
    assert out.shape == (2, 3, 2, 3)
    assert out[1].ravel().tolist() == (frame + 1).tolist()
    assert seen["argv"][0] == "ff" and "crop=iw:ih-8:0:0,scale=3:2:flags=area" in seen["argv"]
    assert seen["kw"]["check"] is False


def run_of(first, last, slot=0):
    """A gap-free run."""
    return L.Run(first=first, last=last, slot=slot, repeats=1.0, agree=1.0, worst_ms=0.0)


def test_frame_starts_follow_each_runs_cadence():
    """Each run's captures counted from its own slot; captures past the size are ignored."""
    starts = LT.frame_starts(9, [run_of(0, 3, slot=1), run_of(5, 12, slot=0)], 2)
    assert starts.tolist() == [False, True, False, True, False, True, False, True, False]


def ticks_for(width=2e6):
    """A tick bracket around the simulated device's true phase."""
    return ev.Ticks(lo=TAU - width / 2, hi=TAU + width / 2, period=P, count=4)


def times_for(lead=10, period=P):
    """Frame times with no measured starts: the line through ``T0 + lead * P``."""
    return ev.FrameTimes(ns=np.full(1, np.nan), lead=lead, origin=T0 + lead * P, period=period)


@pytest.mark.parametrize("target", [-7, 0, 1, 13])
def test_a_clock_sends_each_write_midway_between_ticks_so_it_lands_on_its_frame(target):
    """The first tick after the send latches the target, clear by half a period less the bracket."""
    ticks, times = ticks_for(), times_for()
    clock = LT.Clock(times=times, ticks=ticks, delta=DELTA)
    t = clock.when(target)
    n, clear = ticks.after(np.array([t]))
    assert int(ev.frame_of_tick(n, ticks, times, DELTA)[0]) == target
    assert clear[0] == pytest.approx(P / 2 - ticks.width / 2, abs=1.0)
    assert t == pytest.approx(ev.send_time(np.array([target]), ticks, times, DELTA)[0], abs=1.0)


def test_extending_the_clock_refits_the_period_from_the_growing_frame_log():
    """Captures past the run's end refit the line; a sequence break cuts what is used."""
    lead, size = 10, 400
    k = np.arange(size)
    t = (T0 + k * P / 2).astype(np.int64)
    t[300:] += 10**12
    sequence = np.r_[k[:300], k[300:] + 5]
    clock = LT.Clock(
        times=times_for(lead, period=1.01 * P), ticks=ticks_for(), delta=DELTA, run=run_of(0, 99)
    )
    clock.extend(t, sequence)
    assert clock.times.period == pytest.approx(P, rel=1e-9)
    assert clock.ticks.period == clock.times.period
    assert clock.times.origin == pytest.approx(T0 + lead * P, abs=1.0)
    assert (clock.ticks.lo, clock.ticks.hi) == (ticks_for().lo, ticks_for().hi)


def test_extending_a_clock_with_no_run_changes_nothing():
    """No anchored run, no refit."""
    clock = LT.Clock(times=times_for(period=1.01 * P), ticks=ticks_for(), delta=DELTA)
    clock.extend(np.arange(10), np.arange(10))
    assert clock.times.period == 1.01 * P


def latency_of(delta):
    """A dry take's measured latency."""
    return ev.Latency(delta=delta, spread=1e5, samples=3, period=P)


def test_measured_latencies_accumulate_and_are_read_as_the_midpoint_of_their_extremes(tmp_path):
    """Each dry take adds one; the midpoint has the least worst error; none written is an error."""
    with pytest.raises(LT.NoLatencyError, match="bypass"):
        LT.read_latency(tmp_path)
    LT.write_latency(tmp_path, latency_of(DELTA), "Teletext")
    assert LT.read_latency(tmp_path) == DELTA
    LT.write_latency(tmp_path, latency_of(DELTA + 0.2 * P), "Other")
    LT.write_latency(tmp_path, latency_of(DELTA + 0.1 * P), "Third")
    assert LT.read_latency(tmp_path) == pytest.approx(DELTA + 0.1 * P)
    said = json.loads((tmp_path / LT.LATENCY_FILE).read_text())["takes"]
    assert [t["program"] for t in said] == ["Teletext", "Other", "Third"]
    assert said[0]["frames"] == pytest.approx(1.7)


def test_dry_takes_half_a_frame_apart_contradict_one_card_phase(tmp_path):
    """No single delay is within half a frame of both."""
    LT.write_latency(tmp_path, latency_of(DELTA), "Teletext")
    LT.write_latency(tmp_path, latency_of(DELTA + 0.55 * P), "Other")
    with pytest.raises(LT.LeadInTimeError, match="half a frame"):
        LT.read_latency(tmp_path)


class Recorded:
    """A sender stand-in keeping each send."""

    def __init__(self):
        self.sends = []

    def send(self, param, value, target):
        """Keep it."""
        self.sends.append((param, value, target))


def test_calibration_toggles_the_dry_control_at_random_intervals_ending_dry(monkeypatch):
    """Wet first, then alternating, every write a calibration write, each after a sleep in range."""
    slept = []
    monkeypatch.setattr(LT.time, "sleep", slept.append)
    sender, lead_in = Recorded(), LT.LeadIn(toggles=3, toggle_s=(0.2, 0.3))
    LT.calibrate(sender, {1: PARAM_MAX}, {1: 40, 2: 9}, lead_in)
    assert sender.sends == [(1, v, ev.CALIBRATION) for v in [40, PARAM_MAX] * 3]
    assert len(slept) == 6 and all(0.2 <= s <= 0.3 for s in slept) and len(set(slept)) == 6


def test_probing_steps_every_slot_off_its_value_and_back(monkeypatch):
    """Up by one, or down from full scale, then back."""
    slept = []
    monkeypatch.setattr(LT.time, "sleep", slept.append)
    sender = Recorded()
    LT.probe_ticks(sender, {7: PARAM_MAX, 2: 5}, LT.LeadIn(toggle_s=(0.3, 0.4), probe_s=0.05))
    assert sender.sends == [(2, 6, ev.CALIBRATION), (2, 5, ev.CALIBRATION)] + [
        (7, PARAM_MAX - 1, ev.CALIBRATION),
        (7, PARAM_MAX, ev.CALIBRATION),
    ]
    assert slept == [0.05] * 4


def test_a_takes_rows_return_the_dry_control_to_its_take_value_before_the_plan():
    """Dry takes lead with the wet write at ``wet_frame``; others are the plan alone."""
    plan = np.array([(0, 3, 5), (4, 1, 9)], dtype=LT.PLAN)
    rows = LT.plan_rows(plan, {1: PARAM_MAX}, {1: 40, 3: 5}, wet_frame=-15)
    assert rows.tolist() == [(-15, 1, 40), (0, 3, 5), (4, 1, 9)]
    assert LT.plan_rows(plan, {12: 0}, {}, wet_frame=-3)[0].tolist() == (-3, 12, 0)
    assert LT.plan_rows(plan, None, {1: 40}, wet_frame=-15) is plan


def test_the_state_at_a_frame_applies_every_write_up_to_it():
    """Writes after the frame do not count."""
    plan = np.array([(-3, 1, 5), (0, 2, 6), (0, 1, 7), (4, 3, 8)], dtype=LT.PLAN)
    assert LT.state_at(np.zeros(4), plan, 0).tolist() == [7, 6, 0, 0]
    assert LT.state_at(np.zeros(4), plan, 4).tolist() == [7, 6, 8, 0]


def loaded_wire(*, program="Teletext", state="running", said_t1=-1e3, last=(5, 6, 0), fpga=None):
    """A wire record whose last readback before frame 0's tick is ``last``."""
    ticks, times = ticks_for(), times_for()
    cutoff = ticks.lo + np.rint((times.at(0) - DELTA - ticks.phase) / ticks.period) * ticks.period
    reads = np.zeros(2, dtype=[("t0_ns", "<i8"), ("t1_ns", "<i8"), ("o", "<i8", (3,))])
    reads["t1_ns"] = [cutoff - 5e6, cutoff + 1e6]
    reads["o"] = [last, (9, 9, 9)]
    payload = json.dumps({"state": state, "program": program}) if fpga is None else fpga
    loaded = json.dumps({"fpga": payload, "t1_ns": cutoff + said_t1})
    plan = np.array([(-2, 1, 5), (0, 2, 6), (3, 3, 1)], dtype=LT.PLAN)
    return {"loaded": np.array(loaded), "start": np.zeros(3, np.int64), "plan": plan, "reads": reads}


@pytest.mark.parametrize(
    "kw, ok, said",
    [
        ({}, True, "fpga runs"),
        ({"program": "TELETEXT"}, True, "matches"),
        ({"program": "Other"}, False, "does not run"),
        ({"state": "loading"}, False, "does not run"),
        ({"said_t1": 0.0}, False, "after frame 0"),
        ({"last": (5, 0, 0)}, False, "[5, 0, 0] against [5, 6, 0]"),
        ({"fpga": "not json"}, False, "does not run"),
        ({"fpga": ""}, False, "does not run"),
    ],
)
def test_a_program_is_loaded_before_frame_0_only_if_it_runs_and_read_back_frame_0s_state(kw, ok, said):
    """Running the named program, answered before the tick, last readback equal to frame 0's state."""
    got = LT.loaded_before("vendor/Teletext", loaded_wire(**kw), ticks_for(), times_for(), DELTA)
    assert got["ok"] is ok and said in got["summary"] and got["want"] == [5, 6, 0]


def test_no_readback_before_frame_0_is_not_loaded():
    """Nothing read before the tick cannot confirm frame 0's state."""
    wire = loaded_wire()
    wire["reads"] = wire["reads"][1:]
    got = LT.loaded_before("Teletext", wire, ticks_for(), times_for(), DELTA)
    assert not got["ok"] and "None against" in got["summary"]


def test_a_takes_evidence_sits_beside_it():
    """Named for the take."""
    assert str(LT.evidence_path("/a/take.mkv")) == "/a/take.mkv.evidence.json"


class Sim:
    """A device latching CCs on its own frame tick, a card capturing it, and a clock only sleeps advance.

    Played frame ``f`` reaches the card at ``T0 + f * P`` in two captures and shows the state
    latched at tick ``f - 2``; a write latches on the first tick at or after it is sent.
    """

    def __init__(self, *, lead, count, tail=10, drop=None):
        self.now, self.lead, self.count = S0, lead, count
        self.size = 2 * (lead + count + tail)
        self.writes = []
        self.drop = drop
        self.poller = self.sender = None
        self.next_poll = 0
        rng = np.random.default_rng(3)
        self.base = rng.uniform(0, 60, (lead + count + tail, 3, SIDE, SIDE)).astype(np.float32)
        self.base[:lead] = 30 + rng.uniform(0, 2, (lead, 3, SIDE, SIDE))
        self.frames = np.zeros(self.size, FRAME)
        self.frames["sequence"] = np.arange(self.size)
        self.frames["t_ns"] = np.rint(T0 + np.arange(self.size) * P / 2)

    def clock(self):
        """Simulated ``CLOCK_MONOTONIC``."""
        return self.now

    def write(self, param, value):
        """A CC reaching the device now, unless it is the dropped one."""
        if self.drop is not None and (param, value) == self.drop:
            return
        self.writes.append((self.now, param, value))

    def state(self, at):
        """The 12 values the device holds at host time ``at``."""
        out = np.zeros(12, dtype=np.int64)
        for t, param, value in self.writes:
            if TAU + np.ceil((t - TAU) / P) * P <= at:
                out[param - 1] = value
        return out

    def picture(self, k):
        """Capture ``k``: the clip frame, or its processed form when bypass is off."""
        f = k // 2
        s = self.state(TAU + (f - 2) * P)
        if s[0] == PARAM_MAX:
            return self.base[f]
        return 0.2 * self.base[f] + 100 + 0.1 * s[2]

    def stored(self):
        """Captures the card has delivered by now."""
        return int(np.count_nonzero(self.frames["t_ns"] <= self.now))

    def images(self, size=None):
        """Every capture so far as an image."""
        size = self.stored() if size is None else size
        return np.rint([self.picture(k) for k in range(size)]).astype(np.uint8)

    def attribution(self, size=None):
        """The exact attribution of the first ``size`` captures."""
        size = self.stored() if size is None else size
        picked = 2 * (np.arange(self.count) + self.lead) + 1
        return L.Attribution(
            picked=np.where(picked < size, picked, -1),
            shown=np.arange(size) // 2 - self.lead,
            per_frame=2,
            lead=self.lead,
            runs=(run_of(0, size - 1),),
            anchor=0,
            offset_ns=0.0,
            period_ns=P,
            frame0_ns=T0 + self.lead * P,
            drops=0,
            read=size,
            agreed=size,
        )

    def sleep(self, seconds):
        """Advance, polling and sending whatever falls due on the way."""
        end = self.now + int(seconds * 1e9)
        while True:
            due = []
            if self.poller is not None:
                due.append((self.next_poll, 1))
            if self.sender is not None:
                rows = self.sender._queue.queue  # pylint: disable=protected-access
                while rows and rows[0][0] is None:
                    rows.popleft()
                if rows:
                    due.append((self.sender.when(rows[0][0]), 0))
            if not due or min(due)[0] > end:
                break
            at, which = min(due)
            self.now = max(self.now, int(at))
            if which:
                self.next_poll = at + self.poller.interval_s * 1e9
                self.poller.poll_once()
            else:
                target, param, value = rows.popleft()
                self.sender.send(param, value, target)
        self.now = max(self.now, end)


def rigged(sim, info, *, boots=(7, 7), cc_map=tuple(range(1, 13))):
    """A rig over ``sim``: shell, MIDI port, session, playout and recorder."""
    boots, said, loads = list(boots), [], []
    offsets = [0] * 12
    shell = types.SimpleNamespace(log_lines=[], said=said)

    def status():
        values = sim.state(sim.now)
        sim.now += 1_000_000
        return {"modulators": [{"o": int(v)} for v in values]}

    def command(*words):
        said.append(words)
        if words == ("fpga", "status"):
            return types.SimpleNamespace(payload=json.dumps({"state": "running", "program": "Teletext"}))
        return types.SimpleNamespace(payload="")

    def set_param(param, value):
        offsets[param - 1] = value
        sim.write(param, value)
        word = int(round(value * 16383 / 1023))
        msb = cc_map[param - 1]
        shell.log_lines.extend(
            [
                f"[MIDI USB] CC  ch=1 cc={msb} val={word >> 7}",
                f"[MIDI USB] CC  ch=1 cc={msb + 32} val={word & 127}",
            ]
        )

    shell.get_setting = lambda key: json.dumps({"key": key, "type": "uint32", "value": boots.pop(0)})
    shell.command, shell.modulation_status = command, status
    midi = types.SimpleNamespace(get_param=lambda p: offsets[p - 1], cc_map=list(cc_map))
    transport = types.SimpleNamespace(
        device=types.SimpleNamespace(shell=shell, midi=midi), set_param=set_param
    )

    @contextlib.contextmanager
    def held(*_, **__):
        yield types.SimpleNamespace(status=lambda: {"hold": 2})

    playout = types.SimpleNamespace(playing=held, flips=lambda: np.zeros(3, FLIP))
    rig = types.SimpleNamespace(
        transport=transport, session=types.SimpleNamespace(park_values=[0] * 12), playout=playout
    )

    def load(name):
        loads.append((name, sim.now))
        return info

    return rig, types.SimpleNamespace(recording=held), load, loads


@pytest.fixture(name="simulate")
def fixture_simulate(monkeypatch):
    """Run :func:`locktake.locked_take` against a :class:`Sim`, threads replaced by its sleeps."""

    def build(**kw):
        sim = Sim(**kw)

        class SimPoller(Poller):
            """Polled from the simulation's sleeps."""

            def __init__(self, shell, *, interval_s):
                super().__init__(shell, interval_s=interval_s, clock=sim.clock, sleep=sim.sleep)

            def start(self):
                sim.poller, sim.next_poll = self, sim.now
                return self

            def stop(self):
                sim.poller = None
                self._run_asked()

        class SimSender(Sender):
            """Sent from the simulation's sleeps."""

            def __init__(self, write):
                super().__init__(write, clock=sim.clock, sleep=sim.sleep)

            def start(self):
                sim.sender = self
                return self

            def stop(self):
                sim.sender = None

        monkeypatch.setattr(LT, "Poller", SimPoller)
        monkeypatch.setattr(LT, "Sender", SimSender)
        monkeypatch.setattr(LT, "time", types.SimpleNamespace(monotonic_ns=sim.clock, sleep=sim.sleep))
        monkeypatch.setattr(LT, "_sleep_until", lambda t: sleep_until_on(t, clock=sim.clock, sleep=sim.sleep))
        monkeypatch.setattr(LT, "_read_sidecar", lambda take: sim.frames[: sim.stored()])
        monkeypatch.setattr(LT, "read_live", lambda *a, **k: (np.zeros(sim.stored(), np.int64), sim.images()))
        monkeypatch.setattr(
            LT, "live_attribution", lambda *a, **k: (sim.attribution(), sim.frames[: sim.stored()])
        )
        return sim

    return build


def config_for(lead_in_s):
    """A session's settings as the take reads them."""
    return types.SimpleNamespace(
        fps=FPS, lead_in_s=lead_in_s, strip_px=8, bits=16, tail_s=0.2, ffmpeg="ffmpeg"
    )


LEAD_IN = LT.LeadIn(toggles=3, toggle_s=(0.2, 0.25), settle_s=0.3, wet_s=0.5, poll_s=0.005)
PLAN_ROWS = np.array(
    [(0, 3, 512), (0, 4, 100), (5, 3, 600), (9, 4, 200), (20, 3, 100), (20, 5, 900)], dtype=LT.PLAN
)


def take_and_prove(monkeypatch, simulate, tmp_path, *, roles, working, boots=(7, 7), drop=None):
    """A whole simulated take, then its evidence from the files it left."""
    sim = simulate(lead=120, count=30, drop=drop)
    rig, recorder, load, loads = rigged(sim, info_with(roles), boots=boots)
    out = tmp_path / "take.mkv"
    record = LT.locked_take(
        rig,
        recorder,
        out,
        program="Teletext",
        seconds=1.0,
        config=config_for(4.0),
        plan=PLAN_ROWS,
        working=lambda info: working,
        load=load,
        lead_in=LEAD_IN,
    )
    monkeypatch.setattr(L, "attribute_take", lambda *a, **k: sim.attribution(sim.size))
    monkeypatch.setattr(v4l2rec, "read_log", lambda take: sim.frames)
    clip = sim.base.astype(np.uint8)
    monkeypatch.setattr(
        LT, "small_frames", lambda path, **k: clip if str(path) == "clip.mkv" else sim.images(sim.size)
    )
    proof = LT.take_evidence(out, program="Teletext", clip="clip.mkv", config=config_for(4.0))
    return sim, rig, record, proof, loads


def test_a_dry_take_measures_its_latency_and_proves_every_write_landed(monkeypatch, simulate, tmp_path):
    """Calibrated on its bypass, every planned write lands on its frame, reads back and is received."""
    sim, rig, record, proof, loads = take_and_prove(
        monkeypatch, simulate, tmp_path, roles={1: Role.BYPASS}, working={1: 0, 3: 0.5, 4: 100}
    )
    assert [name for name, _ in loads] == ["Teletext"]
    assert record.delta == pytest.approx(DELTA, abs=2e6)
    assert LT.read_latency(tmp_path) == record.delta
    assert record.plan[0].tolist() == (-15, 1, 0)
    calibration = record.sends[record.sends["target"] == ev.CALIBRATION]
    assert calibration["value"].tolist() == [0, PARAM_MAX] * 3
    assert (tmp_path / "take.mkv.wire.npz").is_file() and (tmp_path / "take.mkv.lock.npz").is_file()
    said = rig.transport.device.shell.said
    assert said[0] == ("midi", "monitor", "on") and said[-1] == ("midi", "monitor", "off")
    assert ("fpga", "status") in said
    assert proof.ok, proof.lines()
    assert proof.landed.ok and proof.applied.ok and proof.received.ok and proof.loaded["ok"]
    assert proof.latency["frames"] == pytest.approx(1.7, abs=0.06)
    assert len(proof.landed.rows) == PLAN_ROWS.size + 1
    assert proof.received.sent == 2 * record.sends.size
    assert {m.param for m in proof.moved} == {3, 4, 5}
    assert json.loads(proof.to_json())["ok"]
    assert sim.sender is None and sim.poller is None


def test_a_write_the_device_drops_fails_the_readback(monkeypatch, simulate, tmp_path):
    """Sent on time but never applied: the readback disagrees on its slot from its frame on."""
    proof = take_and_prove(
        monkeypatch, simulate, tmp_path, roles={1: Role.BYPASS}, working={1: 0, 3: 0.5}, drop=(3, 600)
    )[3]
    assert proof.landed.ok and not proof.applied.ok and not proof.ok
    assert set(proof.applied.disagree["slot"].tolist()) == {3}
    assert proof.applied.disagree["frame"].min() >= 5
    assert set(proof.applied.disagree["want"].tolist()) == {600}


def test_a_take_without_a_dry_control_uses_the_last_measured_latency(monkeypatch, simulate, tmp_path):
    """No latency yet is refused before anything plays; with one, the take probes ticks and holds."""
    sim = simulate(lead=120, count=30)
    rig, recorder, load, loads = rigged(sim, info_with({}))
    kw = {"program": "Teletext", "seconds": 1.0, "config": config_for(4.0), "plan": PLAN_ROWS}
    with pytest.raises(LT.NoLatencyError):
        LT.locked_take(
            rig, recorder, tmp_path / "t.mkv", working=lambda info: {}, load=load, lead_in=LEAD_IN, **kw
        )
    assert not rig.transport.device.shell.said
    LT.write_latency(tmp_path, ev.Latency(delta=DELTA, spread=0.0, samples=1, period=P), "Other")
    _, _, record, proof, loads = take_and_prove(
        monkeypatch, simulate, tmp_path, roles={}, working={3: 1.0, 4: 100}
    )
    assert [name for name, _ in loads] == ["Teletext", "Passthru", "Teletext"]
    assert record.dry is None and record.delta == DELTA
    probes = record.sends[record.sends["target"] == ev.CALIBRATION]
    assert probes[["param", "value"]].tolist() == [(3, 513), (3, 512), (4, 101), (4, 100)]
    assert proof.ok, proof.lines()
    assert proof.latency["samples"] == 0


def test_a_reboot_during_the_take_is_an_error_after_its_record_is_kept(monkeypatch, simulate, tmp_path):
    """The boot count rising fails the take, its wire record saved first."""
    with pytest.raises(LT.RebootError, match="from 7 to 8"):
        take_and_prove(monkeypatch, simulate, tmp_path, roles={1: Role.BYPASS}, working={1: 0}, boots=(7, 8))
    assert (tmp_path / "take.mkv.wire.npz").is_file()


def test_a_lead_in_too_short_for_the_calibration_is_an_error(simulate, tmp_path):
    """The first write due before the clock is known is a lead-in time error, the monitor left off."""
    sim = simulate(lead=30, count=30)
    rig, recorder, load, _ = rigged(sim, info_with({1: Role.BYPASS}))
    with pytest.raises(LT.LeadInTimeError, match="lengthen the lead-in"):
        LT.locked_take(
            rig,
            recorder,
            tmp_path / "t.mkv",
            program="Teletext",
            seconds=1.0,
            config=config_for(1.0),
            plan=PLAN_ROWS,
            working=lambda info: {1: 0},
            load=load,
            lead_in=LEAD_IN,
        )
    assert rig.transport.device.shell.said[-1] == ("midi", "monitor", "off")
    assert (tmp_path / "t.mkv.wire.npz").is_file()
    assert sim.sender is None and sim.poller is None


def test_measured_latency_comes_from_where_each_toggle_showed(simulate):
    """Toggles of the bypass, landing two frames after their tick, give the device's delay."""
    sim = simulate(lead=60, count=4)
    sim.now = T0
    sim.writes.append((T0 + 50_000_000, 1, PARAM_MAX))
    sends = []
    for k, t in enumerate(T0 + np.array([0.3, 0.62, 0.95, 1.3]) * 1e9):
        value = 0 if k % 2 == 0 else PARAM_MAX
        sim.writes.append((int(t), 1, value))
        sends.append((int(t), int(t), 1, value, ev.CALIBRATION))
    toggles = np.array(sends, dtype=SEND)
    sim.now = int(T0 + 1.9e9)
    polls = np.arange(S0, sim.now, 5_000_000)
    reads = np.array([(t, t + 1_000_000, tuple(sim.state(t))) for t in polls], dtype=READ)
    found, images = sim.attribution(), sim.images()
    frames = sim.frames[: sim.stored()]
    times = ev.frame_times(frames["t_ns"], found.shown, found.runs, 2, sim.lead)
    moves = ev.transitions(reads)
    ticks = ev.tick_bracket(moves["a"], moves["b"], times.period)
    got = LT.measure_latency(images, frames, found, toggles, moves, times, ticks)
    assert got.samples == 4 and got.delta == pytest.approx(DELTA, abs=ticks.width)

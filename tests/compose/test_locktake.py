"""Locked takes: plans, clocks, calibration and a whole take against a simulated device."""

import contextlib
import gc
import json
import types

import numpy as np
import pytest

from syncsummoner.compose import evidence as ev
from syncsummoner.compose import lock as L
from syncsummoner.compose import locktake as LT
from syncsummoner.compose import session as SS
from syncsummoner.device import loadtimes, v4l2rec
from syncsummoner.device.profile import PARAM_MAX, ParamSpec
from syncsummoner.device.v4l2rec import FRAME
from syncsummoner.device import wire as W
from syncsummoner.device.wire import SEND, Echoes, Poller, Sender
from syncsummoner.probe import decoded
from syncsummoner.program.model import Bypass
from syncsummoner.compose.timecode import ClipLayout
from tests.compose import locked

FPS = 30.0
P = 1e9 / FPS
S0 = 5_000_000_000
T0 = S0 + 50_000_000
#: The simulated device ticks for played frame ``n`` this long after the Pi flips it.
TAU = T0 + 0.3 * P
#: The simulated device's ``field_count`` on its tick for played frame 0.
FIELD0 = 1000
#: How long a Pi report or reply takes to reach the host.
HEARD_NS = 1_000_000
SIDE = 4


def specs():
    """Twelve parameter specs."""
    return [ParamSpec(index=i, name=f"p{i}", native_min=0.0, native_max=1.0) for i in range(1, 13)]


def switch(slot):
    """A bypass switch on ``slot``, as the program model states one."""
    return Bypass(slot, PARAM_MAX, tuple(range(512, PARAM_MAX + 1)))


def info_with(dry=None):
    """A program info whose program model finds ``dry`` its bypass: a switch's slot or a :class:`Bypass`."""
    found = switch(dry) if isinstance(dry, int) else dry
    return types.SimpleNamespace(params=specs(), name="Teletext", dry=found)


def bypass_of(info):
    """The bypass a test's program info carries, standing in for the program model."""
    return info.dry


@pytest.mark.parametrize(
    "dry, want",
    [(1, {1: PARAM_MAX}), (7, {7: PARAM_MAX}), (Bypass(12, 0, (0, 1)), {12: 0}), (None, None)],
)
def test_the_dry_state_is_the_bypass_the_program_model_finds(dry, want):
    """The dry state is the model's bypass setting; no slot or label stands in for one."""
    assert LT.dry_state(info_with(dry), bypass_of) == want


def test_without_a_declaration_or_a_model_there_is_no_bypass(monkeypatch, tmp_path):
    """A program with no declaration, or one the library cannot find, has no dry state."""
    assert LT.program_bypass(info_with()) is None
    monkeypatch.setenv("SYNCSUMMONER_CARD", str(tmp_path))
    monkeypatch.setenv("SYNCSUMMONER_FIRMWARE", str(tmp_path))
    info = types.SimpleNamespace(name="Nothing", declared=types.SimpleNamespace(program_id="org.none"))
    assert LT.program_bypass(info) is None


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


def test_a_frame_plan_keeps_one_write_per_slot_and_frame_and_drops_repeats():
    """A pre-roll ramp clamped onto frame 0 leaves its last value; a write repeating its slot goes."""
    times = np.r_[np.linspace(-1.0, -0.1, 30), 0.2, 0.3, 0.3]
    auto = types.SimpleNamespace(
        times=times,
        indices=np.r_[np.full(30, 1), 1, 1, 2],
        values=np.r_[np.arange(30) * 10, 290, 7, 7],
    )
    plan = LT.frame_plan(auto, fps=FPS, early_bias_s=0.015)
    assert plan.tolist() == [(0, 1, 290), (9, 1, 7), (9, 2, 7)]


def rows_of(*rows):
    """Plan rows."""
    return np.array(list(rows), dtype=LT.PLAN)


def test_a_schedule_drops_writes_the_slots_already_hold():
    """Against a known state the first write per slot is dropped when it repeats it; order is by target."""
    plan = rows_of((3, 2, 5), (0, 1, 9), (3, 1, 4), (3, 2, 6), (4, 1, 4), (5, 2, 6))
    state = np.zeros(12, dtype=np.int64)
    state[0] = 9
    assert LT.schedule(plan, state).tolist() == [(3, 1, 4), (3, 2, 6)]
    assert LT.schedule(plan).tolist() == [(0, 1, 9), (3, 1, 4), (3, 2, 6)]
    assert LT.schedule(rows_of()).size == 0


def test_a_take_schedule_reports_the_messages_its_frames_carry():
    """Rows sent per frame as scheduled, nothing moved, and the frame the lead-in turns wet on."""
    plan = rows_of((0, 1, 5), (3, 1, 6), (3, 2, 6), (3, 3, 6))
    rows = LT.schedule(plan, np.zeros(12, dtype=np.int64))
    assert rows.tolist() == plan.tolist()
    per_write = np.r_[2, 2, 1, np.full(9, 2)]
    frames, carried = LT.frame_load(rows, per_write)
    assert frames.tolist() == [0, 3] and carried.tolist() == [2, 5]
    said = LT.describe_load(np.zeros(9), plan, rows, per_write)
    assert said == (
        "9 automation writes, 4 frame-plan rows, 4 sent; 2 frames carry writes, 7 messages, "
        "busiest frame 3 with 5, median 3.5 a frame"
    )
    assert LT.describe_load([], rows_of(), rows_of(), per_write).endswith("nothing to send")
    assert LT.wet_at(LT.LeadIn(wet_s=0.5), 30.0) == -15


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






def times_for(lead=10, period=P):
    """Frame times on the line through ``T0 + lead * P``."""
    return ev.FrameTimes(lead=lead, origin=T0 + lead * P, period=period)




def test_a_clock_with_no_line_is_not_extended():
    """Frame times given outright stay as given."""
    clock = LT.Clock(times=times_for(period=1.01 * P))
    clock.extend(np.zeros(10, FRAME))
    assert clock.times == times_for(period=1.01 * P)














class Recorded:
    """A sender stand-in keeping each send."""

    def __init__(self):
        self.sends = []

    def send(self, param, value, target):
        """Keep it."""
        self.sends.append((param, value, target))




def test_a_takes_rows_wet_the_dry_control_for_the_plan_and_dry_it_for_the_lead_out():
    """Dry takes go wet at ``wet_frame`` and back dry at ``dry_frame``; others are the plan alone."""
    plan = np.array([(0, 3, 5), (4, 1, 9)], dtype=LT.PLAN)
    rows = LT.plan_rows(plan, {1: PARAM_MAX}, {1: 40, 3: 5}, wet_frame=-15)
    assert rows.tolist() == [(-15, 1, 40), (0, 3, 5), (4, 1, 9)]
    rows = LT.plan_rows(plan, {1: PARAM_MAX}, {1: 40}, wet_frame=-15, dry_frame=30)
    assert rows.tolist() == [(-15, 1, 40), (0, 3, 5), (4, 1, 9), (30, 1, PARAM_MAX)]
    assert LT.plan_rows(plan, {12: 0}, {}, wet_frame=-3)[0].tolist() == (-3, 12, 0)
    assert LT.plan_rows(plan, None, {1: 40}, wet_frame=-15) is plan


def test_the_state_at_a_frame_applies_every_write_up_to_it():
    """Writes after the frame do not count."""
    plan = np.array([(-3, 1, 5), (0, 2, 6), (0, 1, 7), (4, 3, 8)], dtype=LT.PLAN)
    assert LT.state_at(np.zeros(4), plan, 0).tolist() == [7, 6, 0, 0]
    assert LT.state_at(np.zeros(4), plan, 4).tolist() == [7, 6, 8, 0]


def loaded_wire(*, program="Teletext", state="running", said_t1=-1e3, last=(5, 6, 0), fpga=None):
    """A wire record whose last readback before frame 0 was seen shown is ``last``."""
    cutoff = 7e9
    reads = np.zeros(2, dtype=[("t0_ns", "<i8"), ("t1_ns", "<i8"), ("o", "<i8", (3,))])
    reads["t1_ns"] = [cutoff - 5e6, cutoff + 1e6]
    reads["o"] = [last, (9, 9, 9)]
    payload = json.dumps({"state": state, "program": program}) if fpga is None else fpga
    loaded = json.dumps({"fpga": payload, "t1_ns": cutoff + said_t1})
    plan = np.array([(-2, 1, 5), (0, 2, 6), (3, 3, 1)], dtype=LT.PLAN)
    shown = np.array([(-1, cutoff - 3e7), (0, cutoff), (1, cutoff + 3e7)], dtype=SS.PLAYED)
    return {"loaded": np.array(loaded), "start": np.zeros(3, np.int64), "plan": plan, "reads": reads, "shown": shown}


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
    """Running the named program, answered before frame 0 showed, last readback equal to its state."""
    got = LT.loaded_before("vendor/Teletext", loaded_wire(**kw))
    assert got["ok"] is ok and said in got["summary"] and got["want"] == [5, 6, 0]


def test_no_readback_before_frame_0_is_not_loaded():
    """Nothing read before frame 0 showed cannot confirm its state."""
    wire = loaded_wire()
    wire["reads"] = wire["reads"][1:]
    got = LT.loaded_before("Teletext", wire)
    assert not got["ok"] and "None against" in got["summary"]


def test_a_takes_evidence_sits_beside_it():
    """Named for the take."""
    assert str(LT.evidence_path("/a/take.mkv")) == "/a/take.mkv.evidence.json"


class Sim:
    """A device latching CCs on its own frame tick, a card capturing it, and a clock only sleeps advance.

    Played frame ``f`` is flipped at ``T0 + f * P``, ticked for at ``TAU + f * P``, captured a quarter
    frame either side of ``T0 + (f + 0.5) * P`` and shows the state latched at tick ``f - 1``; a write
    latches on the first tick at or after it is sent; ``counts`` lays the frames out as a session's takes.
    """

    def __init__(self, *, lead, count, tail=20, drop=None, counts=None):
        counts = [count] if counts is None else list(counts)
        self.now, self.lead, self.count, self.counts = S0, lead, sum(counts), counts
        self.dry = switch(1)
        frames = lead + self.count + tail
        self.size = 2 * frames
        self.writes = []
        self.drop = drop
        self.poller = self.sender = None
        self.next_poll = 0
        self.peeking = False
        rng = np.random.default_rng(3)
        self.clip = rng.uniform(0, 60, (max(counts) + tail, 3, SIDE, SIDE)).astype(np.float32)
        self.clip_frame = np.r_[
            np.zeros(lead, np.int64),
            *[np.arange(c) for c in counts],
            counts[-1] + np.arange(tail),
        ]
        self.base = self.clip[self.clip_frame]
        self.base[:lead] = 30 + rng.uniform(0, 2, (lead, 3, SIDE, SIDE))
        code = np.r_[np.arange(-lead, 0), self.clip_frame[lead:]]
        self.frames = np.zeros(self.size, FRAME)
        self.frames["sequence"] = np.arange(self.size)
        self.frames["t_ns"] = np.rint(T0 + (np.arange(self.size) + 0.5) * P / 2)
        self.frames["digest"] = np.arange(self.size) // 2 + 1
        self.codes = np.repeat(code, 2) & locked.WRAP
        self.flips = locked.flip_log(frames, 2)
        self.files = {}

    def source(self, lead):
        """The clip file's pictures: ``lead`` lead-in copies of its first frame, then the clip."""
        return np.r_[np.repeat(self.clip[:1], lead, axis=0), self.clip].astype(np.uint8)

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
        s = self.state(TAU + (f - 1) * P)
        if self.dry.holds(s[self.dry.parameter_id - 1]):
            return self.base[f]
        return 0.2 * self.base[f] + 100 + 0.1 * s[2]

    def field(self):
        """The device's ``field_count`` now."""
        return FIELD0 + int(np.floor((self.now - TAU) / P))

    def flipped(self, at):
        """The last played frame the Pi has flipped by host time ``at``."""
        return int(np.floor((at - T0) / P))

    def tick_at(self, count):
        """When the device's ``field_count`` reaches ``count``."""
        return int(np.ceil(TAU + (count - FIELD0) * P))

    def stored(self):
        """Captures the card has delivered by now."""
        return int(np.count_nonzero(self.frames["t_ns"] <= self.now))

    def images(self, size=None, at=None):
        """Every capture so far as an image, or those ``at``."""
        at = np.arange(self.stored() if size is None else size) if at is None else at
        return np.rint([self.picture(k) for k in at]).astype(np.uint8)

    def attribution(self, size=None):
        """The attribution of the first ``size`` captures from their strips and bytes."""
        size = self.stored() if size is None else size
        frames = self.frames[:size]
        return L.attribute(self.codes[:size], frames, self.flips, hold=2, lead=self.lead, count=self.count)

    def roll(self, path, opens, closes):
        """Captures from ``opens`` go to ``path``, the file before keeping those before ``closes``."""
        last = list(self.files)[-1]
        self.files[last] = (self.files[last][0], closes)
        self.files[str(path)] = (opens, np.inf)

    def captures(self, path):
        """The captures ``path`` holds by now, as indices into the card's."""
        if str(path) not in self.files:
            raise FileNotFoundError(path)
        opens, closes = self.files[str(path)]
        t = self.frames["t_ns"][: self.stored()]
        return np.flatnonzero((t >= opens) & (t < closes))

    def poll_next(self):
        """Advance to the next poll and run it: what a sender waiting on its echo sees."""
        self.now = max(self.now, int(self.next_poll))
        self.next_poll = self.now + self.poller.interval_s * 1e9
        self.poller.poll_once()

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
                    self.peeking = True
                    due.append((self.sender.due(rows[0][0], None), 0))
                    self.peeking = False
            if not due or min(due)[0] > end:
                break
            at, which = min(due)
            self.now = max(self.now, int(at))
            if which:
                self.next_poll = at + self.poller.interval_s * 1e9
                self.poller.poll_once()
            else:
                target, param, value = rows.popleft()
                self.sender.send(param, value, target, int(at))
        self.now = max(self.now, end)


class SimShown:
    """The host's view of the simulated Pi: each flip heard a moment after it, each request answered."""

    def __init__(self, sim):
        self.sim, self.asks = sim, []

    def latest(self):
        frame = self.sim.flipped(self.sim.now - HEARD_NS)
        return frame if frame >= 0 else W.NO_FLOOR

    def ask(self):
        self.asks.append((self.sim.now, self.sim.flipped(self.sim.now + HEARD_NS // 2)))

    def asked(self):
        out = np.zeros(len(self.asks), dtype=SS.ASKED)
        for k, (t0, frame) in enumerate(self.asks):
            out[k] = (t0, t0 + HEARD_NS, frame)
        return out[out["t1_ns"] <= self.sim.now]

    def log(self):
        frames = np.arange(max(self.sim.flipped(self.sim.now - HEARD_NS) + 1, 0))
        out = np.zeros(frames.size, dtype=SS.PLAYED)
        out["frame"], out["t_ns"] = frames, np.ceil(T0 + frames * P + HEARD_NS)
        return out


@contextlib.contextmanager
def following(sim):
    """Follow the simulated Pi for the block."""
    yield SimShown(sim)


def rigged(sim, info, *, boots=(7, 7), cc_map=tuple(range(1, 13)), lose=None):
    """A rig over ``sim``: shell, MIDI port, session, playout and recorder; ``lose`` is lost once."""
    lost = [] if lose is None else [lose]
    boots, said, loads = list(boots), [], []
    offsets = [0] * 12
    shell = types.SimpleNamespace(log_lines=[], said=said, loads=[])

    def status():
        values = sim.state(sim.now)
        sim.now += 1_000_000
        return {"modulators": [{"o": int(v)} for v in values]}

    def command(*words):
        if words == ("transport", "status"):
            count = sim.field()
            sim.now += 1_000_000
            return types.SimpleNamespace(
                payload=json.dumps({"field_count": count, "phase": 0, "bpm_x100": 12000, "state": "stopped"})
            )
        said.append(words)
        if words == ("fpga", "status"):
            return types.SimpleNamespace(payload=json.dumps({"state": "running", "program": "Teletext"}))
        return types.SimpleNamespace(payload="")

    def set_param(param, value):
        offsets[param - 1] = value
        if (param, value) in lost:
            lost.remove((param, value))
            return
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
    played = []

    @contextlib.contextmanager
    def playing(*, fps, spans=None):
        played.append((fps, spans))
        yield types.SimpleNamespace(status=lambda: {"hold": 2}, following=lambda: following(sim))

    @contextlib.contextmanager
    def recording(path):
        sim.files = {str(path): (-np.inf, np.inf)}
        yield "proc"
        sim.now = max(sim.now, int(sim.frames["t_ns"][-1]))

    playout = types.SimpleNamespace(playing=playing, flips=lambda: sim.flips, played=played)
    recorder = types.SimpleNamespace(
        recording=recording, roll=lambda proc, path, opens, closes: sim.roll(path, opens, closes)
    )
    rig = types.SimpleNamespace(
        transport=transport, session=types.SimpleNamespace(park_values=[0] * 12), playout=playout
    )

    def load(name):
        loads.append((name, sim.now))
        shell.loads.append((len(said), sim.poller is None and sim.sender is None))
        return info

    return rig, recorder, load, loads


@pytest.fixture(name="simulate")
def fixture_simulate(monkeypatch):
    """Run :func:`session.locked_session` against a :class:`Sim`, threads replaced by its sleeps."""

    def build(**kw):
        sim = Sim(**kw)

        class SimEchoes(Echoes):
            """Awaited by polling the simulation."""

            def __init__(self):
                super().__init__(idle=sim.poll_next)

        class SimPoller(Poller):
            """Polled from the simulation's sleeps."""

            def __init__(self, shell, *, interval_s, echoes=None):
                super().__init__(
                    shell, interval_s=interval_s, clock=sim.clock, sleep=sim.sleep, echoes=echoes
                )

            def start(self):
                sim.poller, sim.next_poll = self, sim.now
                return self

            def stop(self):
                sim.poller = None
                self._run_asked()

            def wait_count(self, count, stop):
                at = sim.tick_at(count)
                if not sim.peeking:
                    sim.sleep(max(at - sim.now, 0) / 1e9)
                return at

        class SimSender(Sender):
            """Sent from the simulation's sleeps."""

            def __init__(self, write, *, echoes=None, encode=None):
                super().__init__(write, clock=sim.clock, echoes=echoes, encode=encode)

            def start(self):
                sim.sender = self
                return self

            def stop(self):
                sim.sender = None

        def live(path, **_):
            at = sim.captures(path)
            return sim.codes[at], sim.images(at=at)

        monkeypatch.setattr(SS, "Echoes", SimEchoes)
        monkeypatch.setattr(SS, "Poller", SimPoller)
        monkeypatch.setattr(SS, "Sender", SimSender)
        monkeypatch.setattr(SS, "time", types.SimpleNamespace(monotonic_ns=sim.clock, sleep=sim.sleep))
        monkeypatch.setattr(LT, "time", types.SimpleNamespace(monotonic_ns=sim.clock, sleep=sim.sleep))
        monkeypatch.setattr(SS, "_read_sidecar", lambda take: sim.frames[sim.captures(take)])
        monkeypatch.setattr(SS, "read_live", live)
        monkeypatch.setattr(
            SS,
            "live_attribution",
            lambda *a, **k: (L.require_anchor(sim.attribution(), "take"), sim.frames[: sim.stored()]),
        )
        return sim

    return build


def config_for(**kw):
    """A session's settings as the take reads them."""
    return types.SimpleNamespace(fps=FPS, strip_px=8, bits=16, tail_s=0.2, ffmpeg="ffmpeg", **kw)


#: Polls off the frame grid, so no echo of one write ends a whole number of frames after another's.
LEAD_IN = LT.LeadIn(settle_s=0.3, wet_s=0.5, poll_s=0.00493)
PLAN_ROWS = np.array(
    [(0, 3, 512), (0, 4, 100), (5, 3, 600), (9, 4, 200), (20, 3, 100), (20, 5, 900)], dtype=LT.PLAN
)


def ends_for(dry, directory):
    """The lead-in and lead-out a simulated session derives: its loads take no time."""
    loadtimes.record("Passthru", 0, 0)
    return SS.session_frames(LEAD_IN, FPS, dry=dry)


def session_for(simulate, tmp_path, *, dry, counts, **kw):
    """A simulation laid out as the session its takes will derive, and the clip it plays."""
    lead, tail = ends_for(dry, tmp_path)
    sim = simulate(lead=lead, count=sum(counts), tail=tail + 12, counts=counts, **kw)
    return sim, ClipLayout(lead=lead, frames=max(counts), tail=tail + 12)


def take_and_prove(
    monkeypatch,
    simulate,
    tmp_path,
    *,
    dry,
    working,
    boots=(7, 7),
    drop=None,
    config=None,
    lose=None,
    plan=None,
):
    """A whole simulated take, then its evidence from the files it left."""
    info = info_with(dry)
    sim, clip = session_for(simulate, tmp_path, dry=info.dry is not None, counts=[30], drop=drop)
    sim.dry = info.dry or sim.dry
    rig, recorder, load, loads = rigged(sim, info, boots=boots, lose=lose)
    out = tmp_path / "take.mkv"
    record = SS.locked_take(
        rig,
        recorder,
        out,
        program="Teletext",
        seconds=1.0,
        config=config_for() if config is None else config,
        plan=PLAN_ROWS if plan is None else plan,
        working=lambda info: working,
        load=load,
        clip=clip,
        lead_in=LEAD_IN,
        bypass=lambda name: info.dry,
    )
    monkeypatch.setattr(L, "attribute_take", lambda *a, **k: sim.attribution(sim.size))
    monkeypatch.setattr(v4l2rec, "read_log", lambda take: sim.frames)
    clip_pictures = sim.base.astype(np.uint8)
    monkeypatch.setattr(
        LT.decoded,
        "features",
        lambda path, spec, **k: decoded.Features(
            {"small": clip_pictures if str(path) == "clip.mkv" else sim.images(sim.size)}, ""
        ),
    )
    proof = LT.take_evidence(out, program="Teletext", clip="clip.mkv", config=config_for())
    return sim, rig, record, proof, loads


def test_a_take_sends_and_records_toggle_slots_as_the_firmware_reads_them(monkeypatch, simulate, tmp_path):
    """A toggle planned below the threshold goes out, echoes and is recorded off; at it, on."""
    plan = np.r_[PLAN_ROWS, np.array([(10, 8, 300), (15, 8, 512)], dtype=LT.PLAN)]
    _, _, record, proof, _ = take_and_prove(
        monkeypatch, simulate, tmp_path, dry=1, working={1: 0, 8: 1}, plan=plan
    )
    toggled = record.plan[record.plan["param"] == 8]
    assert toggled[["target", "value"]].tolist() == [(15, PARAM_MAX)]
    assert record.sends[record.sends["param"] == 8]["value"].tolist() == [PARAM_MAX]
    assert proof.landed.ok and proof.received.ok


def test_a_dry_take_lands_every_write_by_command_order_and_proves_it(monkeypatch, simulate, tmp_path):
    """Held dry for its ends, every planned write lands on its frame, reads back and is received."""
    sim, rig, record, proof, loads = take_and_prove(
        monkeypatch, simulate, tmp_path, dry=1, working={1: 0, 3: 0.5, 4: 100}
    )
    assert [name for name, _ in loads] == ["Teletext"]
    wire = np.load(tmp_path / "take.mkv.wire.npz")
    assert wire["ticks"].size and wire["asked"].size and wire["shown"].size
    sends = wire["sends"]
    framed = sends["target"] > ev.START
    assert (sends["floor"][framed] < sends["target"][framed]).all(), "each write leaves before its frame shows"
    assert record.plan[0].tolist() == (-15, 1, 0) and record.plan[-1].tolist() == (30, 1, PARAM_MAX)
    lock = np.load(tmp_path / "take.mkv.lock.npz")
    assert (int(lock["lead"]), int(lock["count"]), int(lock["tail"])) == (
        sim.lead,
        30,
        ends_for(True, tmp_path)[1],
    )
    assert rig.playout.played == [(FPS, [(0, sim.lead + 30 + int(lock["tail"]))])]
    assert not (record.sends["target"] == ev.CALIBRATION).any()
    assert (tmp_path / "take.mkv.wire.npz").is_file() and (tmp_path / "take.mkv.lock.npz").is_file()
    said = rig.transport.device.shell.said
    assert said[0] == ("midi", "monitor", "on") and said[-1] == ("midi", "monitor", "off")
    assert ("fpga", "status") in said
    assert proof.ok, proof.lines()
    assert proof.landed.ok and proof.applied.ok and proof.received.ok and proof.loaded["ok"]
    assert not proof.landed.ambiguous and (proof.landed.rows["first"] == proof.landed.rows["target"]).all()
    assert len(proof.landed.rows) == np.count_nonzero(PLAN_ROWS["target"] > 0) + 2
    assert 0 not in record.plan["target"].tolist()
    assert proof.received.sent == 2 * record.sends.size
    assert {m.param for m in proof.moved} == {3, 4, 5}
    assert json.loads(proof.to_json())["ok"]
    assert sim.sender is None and sim.poller is None


def test_a_fader_bypass_toggles_against_the_working_value_and_lands_like_a_switch(
    monkeypatch, simulate, tmp_path
):
    """A fader proven a pass-through at 0 holds the lead-in dry and dries the lead-out."""
    _, _, record, proof, loads = take_and_prove(
        monkeypatch, simulate, tmp_path, dry=Bypass(12, 0, (0,)), working={12: 700, 3: 0.5, 4: 100}
    )
    assert [name for name, _ in loads] == ["Teletext"] and record.dry == {12: 0}
    assert record.plan[0].tolist() == (-15, 12, 700) and record.plan[-1].tolist() == (30, 12, 0)
    assert proof.ok, proof.lines()


def test_every_write_waits_for_the_previous_ones_echo_and_lands_on_its_frame(monkeypatch, simulate, tmp_path):
    """One write in flight: each starts after the last one's echo; frame 20's two writes both land there."""
    collecting, original = [], Sim.write

    def watched(self, param, value):
        collecting.append(gc.isenabled())
        original(self, param, value)

    monkeypatch.setattr(Sim, "write", watched)
    sim, _, record, proof, _ = take_and_prove(
        monkeypatch, simulate, tmp_path, dry=1, working={1: 0, 3: 0.5, 4: 100, 6: 0}
    )
    sends = record.sends
    assert sends[sends["target"] == ev.START]["param"].tolist() == [1, 3, 4]
    echo = W.echoed(sends, record.arrays["receipts"], record.arrays["reads"], record.cc_map)
    assert np.isfinite(echo).all() and (sends["t0_ns"][1:] >= echo[:-1, 1]).all()
    assert record.resent.size == 0
    rows = proof.landed.rows
    assert sorted(rows[rows["target"] == 20][["landed", "param"]].tolist()) == [(20, 3), (20, 5)]
    assert proof.ok, proof.lines()
    assert collecting and not any(collecting) and gc.isenabled()
    assert sim.sender is None


def test_a_write_lost_on_the_link_is_resent_once_and_lands_from_its_copy(monkeypatch, simulate, tmp_path):
    """No echo past the take's longest one: written again, the copy recorded, its echo landing the write."""
    _, _, record, proof, _ = take_and_prove(
        monkeypatch, simulate, tmp_path, dry=1, working={1: 0, 3: 0.5}, lose=(3, 600)
    )
    (copy,) = record.resent.tolist()
    sends = record.sends
    assert (
        sends[copy][["param", "value", "target"]].tolist()
        == sends[copy - 1][["param", "value", "target"]].tolist()
    )
    assert sends[copy]["t0_ns"] > sends[copy - 1]["t1_ns"]
    row = proof.landed.rows[(proof.landed.rows["target"] == 5) & (proof.landed.rows["param"] == 3)][0]
    assert row["echo_ns"] > sends[copy]["t1_ns"] and row["landed"] in (5, 6)
    assert proof.landed.unechoed == 0 and not proof.landed.duplicated and proof.applied.ok


def test_a_write_the_device_drops_fails_the_readback(monkeypatch, simulate, tmp_path):
    """Sent on time but never applied: the readback disagrees on its slot from its frame on."""
    proof = take_and_prove(monkeypatch, simulate, tmp_path, dry=1, working={1: 0, 3: 0.5}, drop=(3, 600))[3]
    assert proof.landed.ok and not proof.applied.ok and not proof.ok
    assert set(proof.applied.disagree["slot"].tolist()) == {3}, proof.applied.disagree.tolist()
    assert proof.applied.disagree["frame"].min() >= 5
    assert set(proof.applied.disagree["want"].tolist()) == {600}




def test_a_take_without_a_dry_control_lands_its_writes_with_no_other_takes_measure(monkeypatch, simulate, tmp_path):
    """Passthru at the ends only; the take's own events land every write, no other take consulted."""
    _, rig, record, proof, loads = take_and_prove(
        monkeypatch, simulate, tmp_path, dry=None, working={3: 1.0, 4: 100}
    )
    assert [name for name, _ in loads] == ["Passthru", "Teletext", "Passthru"]
    shell = rig.transport.device.shell
    at, quiet = shell.loads[-1]
    assert quiet and shell.said[at - 1] == ("midi", "monitor", "off")
    timed = json.loads(str(np.load(tmp_path / "take.mkv.wire.npz")["loads"]))
    assert [x["program"] for x in timed] == ["Teletext", "Passthru"]
    assert all(x["t0_ns"] <= x["t1_ns"] for x in timed) and timed[0]["t1_ns"] <= timed[1]["t0_ns"]
    lead = ends_for(False, tmp_path)[0]
    assert loads[-1][1] >= T0 + (lead + 30) * P and loads[1][1] < T0 + lead * P
    assert record.dry is None
    assert not (record.sends["target"] == ev.CALIBRATION).any()
    assert proof.ok, proof.lines()


def test_a_reboot_during_the_take_is_an_error_after_its_record_is_kept(monkeypatch, simulate, tmp_path):
    """The boot count rising fails the take, its wire record saved first."""
    with pytest.raises(LT.RebootError, match="from 7 to 8"):
        take_and_prove(monkeypatch, simulate, tmp_path, dry=1, working={1: 0}, boots=(7, 8))
    assert (tmp_path / "take.mkv.wire.npz").is_file()


def test_a_clock_measured_slower_than_real_time_is_a_lead_in_error(monkeypatch, simulate, tmp_path):
    """Decoding the lead-in slower than it was recorded leaves the first write due; the monitor left off."""
    sim, clip = session_for(simulate, tmp_path, dry=True, counts=[30])
    rig, recorder, load, _ = rigged(sim, info_with(1))

    def slow(*_a, **_k):
        found = sim.attribution()
        sim.now += 10 * sim.lead * int(P)
        return found, sim.frames[: sim.stored()]

    monkeypatch.setattr(SS, "live_attribution", slow)
    with pytest.raises(LT.LeadInTimeError, match="past the lead-in this session derived"):
        SS.locked_take(
            rig,
            recorder,
            tmp_path / "t.mkv",
            program="Teletext",
            seconds=1.0,
            config=config_for(),
            plan=PLAN_ROWS,
            working=lambda info: {1: 0},
            load=load,
            clip=clip,
            lead_in=LEAD_IN,
            bypass=lambda name: info_with(1).dry,
        )
    assert rig.transport.device.shell.said[-1] == ("midi", "monitor", "off")
    assert (tmp_path / "t.mkv.wire.npz").is_file()
    assert sim.sender is None and sim.poller is None


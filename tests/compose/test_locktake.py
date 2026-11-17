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
from syncsummoner.device import v4l2rec
from syncsummoner.device.playout import FLIP
from syncsummoner.device.profile import PARAM_MAX, ParamSpec
from syncsummoner.device.v4l2rec import FRAME
from syncsummoner.device import wire as W
from syncsummoner.device.wire import SEND, Echoes, Poller, Sender
from syncsummoner.probe import decoded
from syncsummoner.program.model import Bypass
from tests.compose import locked

FPS = 30.0
P = 1e9 / FPS
#: Tick-to-capture delay the simulated device and card have: tick ``n`` latches played frame ``n + 2``.
DELTA = 1.7 * P
S0 = 5_000_000_000
T0 = S0 + 50_000_000
TAU = T0 + 0.3 * P
SIDE = 4
sleep_until_on = LT._sleep_until  # pylint: disable=protected-access


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


def ticks_for(width=2e6):
    """A tick bracket around the simulated device's true phase."""
    return ev.Ticks(lo=TAU - width / 2, hi=TAU + width / 2, period=P, count=4)


def lat_for(width=2e6, samples=4):
    """A send-to-capture band ``width`` wide around the simulated device's latency."""
    return ev.Latency(lo=DELTA - width / 2, hi=DELTA + width / 2, samples=samples, period=P)


def times_for(lead=10, period=P):
    """Frame times on the line through ``T0 + lead * P``."""
    return ev.FrameTimes(lead=lead, origin=T0 + lead * P, period=period)


@pytest.mark.parametrize("target", [-7, 0, 1, 13])
def test_a_clock_sends_each_write_midway_between_ticks_so_late_wakes_still_land(target):
    """Due half a frame before the target's tick; a wake short of that, less the bracket, lands it."""
    times = times_for()
    clock = LT.Clock(times=times, lat=lat_for())
    ticks = clock.ticks
    assert ticks == ev.grid(times, lat_for()) and ticks.width == pytest.approx(2e6)
    t = clock.when(target)
    assert t == pytest.approx(ev.send_time(np.array([target]), ticks, times, DELTA)[0], abs=1.0)
    for late in (1.0, P / 4, (P - ticks.width) / 2 - 1.0):
        n, clear = ticks.after(np.array([t + late]))
        assert int(ev.frame_of_tick(n, ticks, times, DELTA)[0]) == target
        assert clear[0] == pytest.approx((P - ticks.width) / 2 - late, abs=1.0)


def test_a_clock_with_no_line_is_not_extended():
    """Frame times given outright stay as given."""
    clock = LT.Clock(times=times_for(period=1.01 * P), lat=lat_for())
    clock.extend(np.zeros(10, FRAME))
    assert clock.times == times_for(period=1.01 * P)


@pytest.mark.parametrize("ppm", [-70.0, 60.0])
def test_the_live_clock_keeps_sends_mid_tick_through_a_drift_and_a_slip(ppm):
    """Attributed on the lead-in, carried on as the log grows; every logged frame's send lands on it.

    Rate-locked cadence counting, extrapolated from the lead-in, lands writes on the wrong frame.
    """
    lead, count = 1200, 10500
    played = locked.take(
        ((0, 2 * (lead + count) + 99, 0.0),), ppm=ppm, jitter_ns=500_000, lead=lead, count=count, hold=1
    )
    capture = locked.VBLANK_NS / 2
    period = 2 * capture * (1 + ppm * 1e-6)
    origin = locked.PI_T0 + locked.CARD_NS - 0.5 * capture

    def true(j):
        return origin + (j + lead) * period

    m = int(np.flatnonzero(played.frame == lead)[0])
    starts = np.flatnonzero(np.diff(played.frame[m:]) > 0) + 1
    assert np.unique((starts - 2 * played.frame[m:][starts]) % 2).size == 2
    found = L.attribute(played.codes[:m], played.frames[:m], played.flips, hold=1, lead=lead, count=count)
    delta = 2.3 * period
    lat = ev.Latency(lo=delta - 1e5, hi=delta + 1e5, samples=3, period=period)
    clock = LT.Clock.of(found, lat)
    assert (clock.first, clock.seen) == (0, m)
    cadence = true(0) + np.arange(-lead, count) * 2 * found.capture_ns
    for size in np.linspace(m, played.frames.size, 7).astype(np.int64):
        clock.extend(played.frames[:size])
        j = np.arange(-lead, count)
        j = j[true(j) < played.frames["t_ns"][size - 1]]
        when = ev.send_time(j, clock.ticks, clock.times, lat.delta)
        assert clock.when(int(j[-1])) == int(when[-1])
        gap = true(j) - delta - when
        assert np.array_equal(np.ceil((when + delta - true(0)) / period), j)
        assert np.minimum(gap, period - gap).min() > (lat.hi - lat.lo) / 2 + 500_000
    assert clock.seen == played.frames.size - 2
    old = cadence - delta - period / 2
    assert not np.array_equal(np.ceil((old + delta - true(0)) / period), np.arange(-lead, count))


def latency_of(delta, line=(0.0, 0.0)):
    """A dry take's measured latency band, its frame starts ``line`` off its line."""
    return ev.Latency(lo=delta - 1e5, hi=delta + 1e5, samples=3, period=P, line_lo=line[0], line_hi=line[1])


def test_dry_takes_bound_the_latency_where_every_one_admits_it(tmp_path):
    """Each take's band widened by its line's error; the band read is where they all meet."""
    with pytest.raises(LT.NoLatencyError, match="bypass"):
        LT.read_latency(tmp_path)
    LT.write_latency(tmp_path, latency_of(DELTA, (-2e6, 1e6)), "Teletext")
    one = LT.read_latency(tmp_path)
    assert (one.lo, one.hi, one.samples, one.period) == (DELTA - 2.1e6, DELTA + 1.1e6, 0, P)
    LT.write_latency(tmp_path, latency_of(DELTA + 5e5, (-1e6, 3e6)), "Other")
    both = LT.read_latency(tmp_path)
    assert (both.lo, both.hi) == (DELTA - 6e5, DELTA + 1.1e6)
    said = json.loads((tmp_path / LT.LATENCY_FILE).read_text())["takes"]
    assert [(t["program"], t["basis"]) for t in said] == [("Teletext", ev.BASIS), ("Other", ev.BASIS)]
    assert said[0]["frames"] == pytest.approx(1.7) and said[0]["line_lo"] == -2e6


def test_a_programs_latency_is_replaced_and_takes_measured_otherwise_ignored(tmp_path):
    """One entry a program and basis, the latest; entries with no start basis bound nothing."""
    old = {"program": "Teletext", "lo": 0.0, "hi": 1.0, "period": P}
    (tmp_path / LT.LATENCY_FILE).write_text(json.dumps({"takes": [old]}))
    with pytest.raises(LT.NoLatencyError, match="start-basis"):
        LT.read_latency(tmp_path)
    LT.write_latency(tmp_path, latency_of(DELTA), "Teletext")
    LT.write_latency(tmp_path, latency_of(DELTA + 3e5), "Teletext")
    said = json.loads((tmp_path / LT.LATENCY_FILE).read_text())["takes"]
    assert said[0] == old and len(said) == 2 and said[1]["lo"] == DELTA + 2e5
    assert LT.read_latency(tmp_path).delta == pytest.approx(DELTA + 3e5)


def test_dry_takes_that_contradict_are_refused_and_a_wide_band_only_widens(tmp_path):
    """Bands that do not meet say nothing; a band of any width is the band every take admits."""
    LT.write_latency(tmp_path, latency_of(DELTA, (-0.3 * P, 0.3 * P)), "Other")
    wide = LT.read_latency(tmp_path)
    assert wide.hi - wide.lo == pytest.approx(0.6 * P + 2e5)
    LT.write_latency(tmp_path, latency_of(DELTA + 0.4 * P), "Teletext")
    with pytest.raises(LT.LeadInTimeError, match="empty"):
        LT.read_latency(tmp_path)


@pytest.mark.parametrize(
    "live, kept, want",
    [
        ({"lo": 1.0, "hi": 3.0, "basis": ev.BASIS}, None, (1.0, 3.0)),
        ({"lo": 1.0, "hi": 3.0}, None, "no CC-to-picture latency"),
        ({"lo": 1.0, "hi": 3.0}, latency_of(DELTA, (-1e5, 1e5)), (DELTA - 2e5, DELTA + 2e5)),
        ({}, latency_of(DELTA, (-0.3 * P, 0.3 * P)), (DELTA - 1e5 - 0.3 * P, DELTA + 1e5 + 0.3 * P)),
    ],
    ids=["live", "unmeasured", "on-file", "wide"],
)
def test_a_take_without_a_dry_control_lands_its_writes_by_a_start_basis_band(tmp_path, live, kept, want):
    """The band it was sent by when measured to frame starts, else what the dry takes on file admit."""
    if kept is not None:
        LT.write_latency(tmp_path, kept, "Other")
    wire = {"sends": np.zeros(0, SEND), "latency": np.array(json.dumps(live)), "dry": np.array("null")}
    seen = (None, None, None, times_for())
    if isinstance(want, str):
        with pytest.raises(ev.EvidenceError, match=want):
            LT.take_latency(tmp_path / "t.mkv", wire, seen, None)
    else:
        lat = LT.take_latency(tmp_path / "t.mkv", wire, seen, None)
        assert (lat.lo, lat.hi, lat.period) == (*want, P)


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

    Played frame ``f`` starts at ``T0 + f * P``, is captured a quarter frame either side of
    ``T0 + (f + 0.5) * P`` and shows the state latched at tick ``f - 2``; a write latches on the
    first tick at or after it is sent.
    """

    def __init__(self, *, lead, count, tail=20, drop=None):
        self.now, self.lead, self.count = S0, lead, count
        self.dry = switch(1)
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
        self.frames["t_ns"] = np.rint(T0 + (np.arange(self.size) + 0.5) * P / 2)
        self.frames["digest"] = np.arange(self.size) // 2 + 1
        self.codes = (np.arange(self.size) // 2 - lead) & locked.WRAP
        self.flips = locked.flip_log(lead + count + tail, 2)

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
        if self.dry.holds(s[self.dry.parameter_id - 1]):
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
        """The attribution of the first ``size`` captures from their strips and bytes."""
        size = self.stored() if size is None else size
        frames = self.frames[:size]
        return L.attribute(self.codes[:size], frames, self.flips, hold=2, lead=self.lead, count=self.count)

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
                self.sender.send(param, value, target, int(at))
        self.now = max(self.now, end)


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

    @contextlib.contextmanager
    def held(*_, **__):
        yield types.SimpleNamespace(status=lambda: {"hold": 2})

    playout = types.SimpleNamespace(playing=held, flips=lambda: np.zeros(3, FLIP))
    rig = types.SimpleNamespace(
        transport=transport, session=types.SimpleNamespace(park_values=[0] * 12), playout=playout
    )

    def load(name):
        loads.append((name, sim.now))
        shell.loads.append((len(said), sim.poller is None and sim.sender is None))
        return info

    return rig, types.SimpleNamespace(recording=held), load, loads


@pytest.fixture(name="simulate")
def fixture_simulate(monkeypatch):
    """Run :func:`locktake.locked_take` against a :class:`Sim`, threads replaced by its sleeps."""

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

        class SimSender(Sender):
            """Sent from the simulation's sleeps."""

            def __init__(self, write, *, echoes=None, encode=None):
                super().__init__(write, clock=sim.clock, sleep=sim.sleep, echoes=echoes, encode=encode)

            def start(self):
                sim.sender = self
                return self

            def stop(self):
                sim.sender = None

        monkeypatch.setattr(LT, "Echoes", SimEchoes)
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


def config_for(lead_in_s, **kw):
    """A session's settings as the take reads them."""
    return types.SimpleNamespace(
        fps=FPS, lead_in_s=lead_in_s, strip_px=8, bits=16, tail_s=0.2, ffmpeg="ffmpeg", **kw
    )


LEAD_IN = LT.LeadIn(toggles=3, toggle_s=(0.2, 0.25), settle_s=0.3, wet_s=0.5, poll_s=0.005)
PLAN_ROWS = np.array(
    [(0, 3, 512), (0, 4, 100), (5, 3, 600), (9, 4, 200), (20, 3, 100), (20, 5, 900)], dtype=LT.PLAN
)


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
    sim = simulate(lead=120, count=30, drop=drop)
    info = info_with(dry)
    sim.dry = info.dry or sim.dry
    rig, recorder, load, loads = rigged(sim, info, boots=boots, lose=lose)
    out = tmp_path / "take.mkv"
    record = LT.locked_take(
        rig,
        recorder,
        out,
        program="Teletext",
        seconds=1.0,
        config=config_for(4.0) if config is None else config,
        plan=PLAN_ROWS if plan is None else plan,
        working=lambda info: working,
        load=load,
        lead_in=LEAD_IN,
        bypass=bypass_of,
    )
    monkeypatch.setattr(L, "attribute_take", lambda *a, **k: sim.attribution(sim.size))
    monkeypatch.setattr(v4l2rec, "read_log", lambda take: sim.frames)
    clip = sim.base.astype(np.uint8)
    monkeypatch.setattr(
        LT.decoded,
        "features",
        lambda path, spec, **k: decoded.Features(
            {"small": clip if str(path) == "clip.mkv" else sim.images(sim.size)}, ""
        ),
    )
    proof = LT.take_evidence(out, program="Teletext", clip="clip.mkv", config=config_for(4.0))
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


def test_a_dry_take_measures_its_latency_and_proves_every_write_landed(monkeypatch, simulate, tmp_path):
    """Calibrated on its bypass, every planned write lands on its frame, reads back and is received."""
    sim, rig, record, proof, loads = take_and_prove(
        monkeypatch, simulate, tmp_path, dry=1, working={1: 0, 3: 0.5, 4: 100}
    )
    assert [name for name, _ in loads] == ["Teletext"]
    band = json.loads(str(np.load(tmp_path / "take.mkv.wire.npz")["latency"]))
    assert band["lo"] <= DELTA <= band["hi"]
    (kept,) = json.loads((tmp_path / LT.LATENCY_FILE).read_text())["takes"]
    assert kept["program"] == "Teletext" and kept["lo"] <= DELTA <= kept["hi"]
    assert kept["line_lo"] == pytest.approx(-P / 4, abs=2.0) and kept["line_hi"] == pytest.approx(
        P / 4, abs=2.0
    )
    wide = LT.read_latency(tmp_path)
    assert wide.lo == pytest.approx(kept["lo"] + kept["line_lo"]) and wide.hi == pytest.approx(
        kept["hi"] + kept["line_hi"]
    )
    assert record.plan[0].tolist() == (-15, 1, 0) and record.plan[-1].tolist() == (30, 1, PARAM_MAX)
    lock = np.load(tmp_path / "take.mkv.lock.npz")
    assert (int(lock["lead"]), int(lock["count"]), int(lock["tail"])) == (120, 30, 120)
    calibration = record.sends[record.sends["target"] == ev.CALIBRATION]
    assert calibration["value"].tolist() == [0, PARAM_MAX] * 3
    assert (tmp_path / "take.mkv.wire.npz").is_file() and (tmp_path / "take.mkv.lock.npz").is_file()
    said = rig.transport.device.shell.said
    assert said[0] == ("midi", "monitor", "on") and said[-1] == ("midi", "monitor", "off")
    assert ("fpga", "status") in said
    assert proof.ok, proof.lines()
    assert proof.landed.ok and proof.applied.ok and proof.received.ok and proof.loaded["ok"]
    assert proof.latency["lo"] <= DELTA <= proof.latency["hi"]
    assert len(proof.landed.rows) == np.count_nonzero(PLAN_ROWS["target"] > 0) + 2
    assert 0 not in record.plan["target"].tolist()
    assert proof.received.sent == 2 * record.sends.size
    assert {m.param for m in proof.moved} == {3, 4, 5}
    assert json.loads(proof.to_json())["ok"]
    assert sim.sender is None and sim.poller is None


def test_a_fader_bypass_toggles_against_the_working_value_and_lands_like_a_switch(
    monkeypatch, simulate, tmp_path
):
    """A fader proven a pass-through at 0 holds the lead-in dry, times the device and dries the lead-out."""
    _, _, record, proof, loads = take_and_prove(
        monkeypatch, simulate, tmp_path, dry=Bypass(12, 0, (0,)), working={12: 700, 3: 0.5, 4: 100}
    )
    assert [name for name, _ in loads] == ["Teletext"] and record.dry == {12: 0}
    calibration = record.sends[record.sends["target"] == ev.CALIBRATION]
    assert calibration["param"].tolist() == [12] * 6 and calibration["value"].tolist() == [700, 0] * 3
    assert record.plan[0].tolist() == (-15, 12, 700) and record.plan[-1].tolist() == (30, 12, 0)
    assert proof.ok, proof.lines()
    assert proof.latency["lo"] <= DELTA <= proof.latency["hi"]


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
    assert set(proof.applied.disagree["slot"].tolist()) == {3}
    assert proof.applied.disagree["frame"].min() >= 5
    assert set(proof.applied.disagree["want"].tolist()) == {600}


def test_a_calibration_the_picture_cannot_show_lands_by_the_band_on_file(monkeypatch, simulate, tmp_path):
    """A start state whose wet picture is the dry one hides every toggle; the dry takes' band lands it."""

    def unseen(*_a, **_k):
        raise ev.EvidenceError("no calibration write showed in the picture")

    monkeypatch.setattr(LT, "measure_latency", unseen)
    LT.write_latency(tmp_path, lat_for(width=0.0, samples=1), "Other")
    record = take_and_prove(monkeypatch, simulate, tmp_path, dry=1, working={1: 0})[2]
    assert record.delta == pytest.approx(DELTA)
    said = json.loads((tmp_path / LT.LATENCY_FILE).read_text())["takes"]
    assert [t["program"] for t in said] == ["Other"]


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
    LT.write_latency(tmp_path, lat_for(width=0.0, samples=1), "Other")
    _, rig, record, proof, loads = take_and_prove(
        monkeypatch, simulate, tmp_path, dry=None, working={3: 1.0, 4: 100}
    )
    assert [name for name, _ in loads] == ["Teletext", "Passthru", "Teletext", "Passthru"]
    shell = rig.transport.device.shell
    at, quiet = shell.loads[-1]
    assert quiet and shell.said[at - 1] == ("midi", "monitor", "off")
    timed = json.loads(str(np.load(tmp_path / "take.mkv.wire.npz")["loads"]))
    assert [x["program"] for x in timed] == ["Teletext", "Passthru"]
    assert all(x["t0_ns"] <= x["t1_ns"] for x in timed) and timed[0]["t1_ns"] <= timed[1]["t0_ns"]
    assert loads[-1][1] >= T0 + (120 + 30) * P
    assert record.dry is None and record.delta == pytest.approx(DELTA)
    assert not (record.sends["target"] == ev.CALIBRATION).any()
    assert proof.ok, proof.lines()
    assert proof.latency["samples"] == 0 and (proof.latency["lo"], proof.latency["hi"]) == (DELTA, DELTA)


def test_a_reboot_during_the_take_is_an_error_after_its_record_is_kept(monkeypatch, simulate, tmp_path):
    """The boot count rising fails the take, its wire record saved first."""
    with pytest.raises(LT.RebootError, match="from 7 to 8"):
        take_and_prove(monkeypatch, simulate, tmp_path, dry=1, working={1: 0}, boots=(7, 8))
    assert (tmp_path / "take.mkv.wire.npz").is_file()


def test_a_lead_in_too_short_for_the_calibration_is_an_error(simulate, tmp_path):
    """The first write due before the clock is known is a lead-in time error, the monitor left off."""
    sim = simulate(lead=30, count=30)
    rig, recorder, load, _ = rigged(sim, info_with(1))
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
            bypass=bypass_of,
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
        sends.append((int(t), int(t), 1, value, ev.CALIBRATION, int(t)))
    toggles = np.array(sends, dtype=SEND)
    sim.now = int(T0 + 1.9e9)
    found, images = sim.attribution(), sim.images()
    frames = sim.frames[: sim.stored()]
    times = ev.frame_times(found)
    got = LT.measure_latency(images, frames, found, toggles, times)
    assert got.samples == 4 and got.lo < DELTA <= got.hi and got.hi - got.lo <= P / 2 + 1
    assert got.line_lo == pytest.approx(-P / 4, abs=2.0) and got.line_hi == pytest.approx(P / 4, abs=2.0)
    assert got.bounds == (got.lo + got.line_lo, got.hi + got.line_hi)

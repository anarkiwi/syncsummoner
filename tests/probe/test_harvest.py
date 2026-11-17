"""Harvest: one recording per program, frames attributed to setpoints by capture time."""

# pylint: disable=missing-function-docstring

import dataclasses
from contextlib import contextmanager, nullcontext
from pathlib import Path

import numpy as np
import pytest
from pyvmancer import CcMapError, DeviceGoneError, PortBusyError, ProgramLoadError, ShellError

from syncsummoner.device.profile import ParamKind
from syncsummoner.device import recorder as recorder_mod
from syncsummoner.device.recorder import BlankTakeError, InputRouteError, TakeReport, inspect_take
from syncsummoner.device.session import Session
from syncsummoner.device.transport import ProgramInfo
from syncsummoner.probe import harvest as H
from syncsummoner.probe.archive import CODEFRAMES, GAP, same_stimulus
from syncsummoner.probe.store import KeyKind, ProgramKey, slug
from tests import transmitter as tx
from tests.device.conftest import COLORBARS_DECLARED, COLORBARS_INFO, FakeClock, FakeTransport, Readback
from tests.device.test_liveness import mjpeg, needs_ffmpeg
from tests.probe.covers import FakeCover, fake_plan

FIRMWARE = "1.0.0-rc.37"
INFO = ProgramInfo.from_json(COLORBARS_INFO, COLORBARS_DECLARED)
KEY = ProgramKey("Alpha", FIRMWARE, KeyKind.NAME_FIRMWARE)
CONFIG = H.HarvestConfig(
    width=64,
    height=48,
    capture_fps=4,
    settle_s=0.25,
    dwell_s=1.0,
    startup_s=0.0,
    health_poll_s=0.0,
    loop_frames=2,
)
#: A take that carries a moving picture, and one that is black however long it ran.
LIVE = TakeReport(frames=30, picture=30, spread=240, motion=0.6, frozen=False, luma=0.42)
DARK = TakeReport(frames=30, luma=0.063)
#: Frames per program at ``capture_fps`` over two dwells and their settles.
PER_PROGRAM = 10
MEASURED = 9
BASE = tuple(range(12))


@pytest.fixture(autouse=True)
def no_ffmpeg(monkeypatch):
    """No rig and no ffmpeg: the picture always returns, and every take carries one."""
    monkeypatch.setattr(H, "settle", lambda recorder, transport, **kwargs: LIVE)
    monkeypatch.setattr(H, "inspect_take", lambda path, **kwargs: LIVE)


def blind(monkeypatch, *, canary=True, take=DARK, canary_probe=DARK):
    """Every take reads black; ``canary`` says whether the passthrough still carries one.

    A dark canary settles to ``canary_probe``, or fails its settle when that is None.
    """
    monkeypatch.setattr(H, "inspect_take", lambda path, **kwargs: take)

    def probe(recorder, transport, *, program, **kwargs):
        del recorder, transport, kwargs
        if program == H.WEDGE_PROBE and not canary:
            if canary_probe is not None:
                return canary_probe
            raise BlankTakeError(f"{program}: no picture")
        return LIVE

    monkeypatch.setattr(H, "settle", probe)


class Port(Readback):
    """Transport stub: answers program queries, and can refuse to hold a program."""

    def __init__(self, programs=("Alpha", "Beta"), *, holds=True, info=None, video=()):
        super().__init__()
        self.transmitter = tx.Transmitter(*video)
        self.programs_list = list(programs)
        self.holds = holds
        self.info = COLORBARS_INFO if info is None else info
        self.loads = []
        self.closed = False
        self.hashes = 0
        self.corrupt = False
        self.verified = 0
        self.load_errors = {}

    def route(self):
        return self.transmitter.route()

    def verify_cc_map(self):
        self.verified += 1
        if self.corrupt:
            raise CcMapError("device CC map differs from the CCs sent (P12 CC0)")

    def programs(self):
        return list(self.programs_list)

    def video_status(self):
        return self.transmitter.video_status()

    def firmware(self):
        return FIRMWARE

    def load_program(self, name):
        self.loads.append(name)
        if self.load_errors.get(name):
            raise self.load_errors[name].pop(0)
        if not self.holds:
            raise ShellError(4, "program not found", f"program load {name}")

    def watch(self, what):
        return self.transmitter.watch(what)

    def program_payload(self, name=None):
        del name
        if not self.holds:
            raise RuntimeError("[3] no program loaded")
        return dict(self.info)

    def program_info(self, name=None):
        return self.describe(self.program_payload(name))

    @staticmethod
    def describe(payload):
        return ProgramInfo.from_json(payload, COLORBARS_DECLARED)

    def file_hash(self, path):
        del path
        self.hashes += 1
        raise AssertionError("hashing a binary over the serial shell is what wedges the device")

    def close(self):
        self.closed = True


class Rec:
    """Recorder stub: ffmpeg's file, and the card times it would report for it."""

    ffmpeg = "ffmpeg"

    def __init__(self, clock, *, fps=CONFIG.capture_fps):
        self.clock = clock
        self.fps = float(fps)
        self.recordings = []
        self.spans = []

    @contextmanager
    def recording(self, path, *, seconds=None, settle_s=1.5):
        """Run for the length of the block, leaving one file behind."""
        del seconds, settle_s
        start = self.clock()
        Path(path).write_bytes(b"one continuous recording")
        self.recordings.append(Path(path))
        yield self
        self.spans.append((start, self.clock()))

    def timestamps(self, path):
        """One capture time per delivered frame, across the whole recording."""
        del path
        start, end = self.spans[-1]
        return np.arange(start, end, 1.0 / self.fps)


class Archive:
    """Frame archive stub keyed exactly as the real one, publishing by rename."""

    def __init__(self, directory, stored=()):
        self.directory = Path(directory)
        self.stored = {program: key.digest for program, key in stored}
        self.stimuli = {}
        self.schedules = {}
        self.rows = {}
        self.infos = {}
        self.keys = {}
        self.commits = []
        self.geometry = {}
        self.marked = set()
        self.marked_key = {}
        self.dark_luma = {}

    def has(self, program, key, stimulus=None):
        held = self.stimuli.get(program, CODEFRAMES)
        return self.stored.get(program) == key.digest and (stimulus is None or same_stimulus(held, stimulus))

    def dark(self, program, key):
        return self.marked_key.get(program) == key.digest

    def clear_dark(self, program):
        self.marked.discard(program)
        self.marked_key.pop(program, None)

    def mark_dark(self, program, key, luma):
        self.marked.add(program)
        self.marked_key[program] = key.digest
        self.dark_luma[program] = luma

    def paths(self, program):
        return tuple(self.directory / f"{slug(program)}{suffix}" for suffix in (".mkv", ".parquet", ".json"))

    def scratch(self, program):
        self.directory.mkdir(parents=True, exist_ok=True)
        return self.directory / f"{slug(program)}.scratch.mkv"

    def commit(
        self,
        program,
        key,
        video,
        rows,
        *,
        width,
        height,
        fps,
        program_info=None,
        stimulus=None,
        schedule=None,
    ):
        assert Path(video).exists(), "the recording is published, not re-encoded"
        self.schedules[program] = schedule
        self.stimuli[program] = dict(stimulus or CODEFRAMES)
        self.infos[program] = program_info
        self.keys[program] = key
        self.rows[program] = list(rows)
        self.geometry[program] = (width, height, fps)
        self.commits.append(program)
        Path(video).replace(self.paths(program)[0])
        self.paths(program)[1].write_bytes(b"sidecar")
        self.paths(program)[2].write_bytes(b"meta")


class Player:
    """Loop player stub recording upload, start and stop."""

    def __init__(self, running=True):
        self.uploaded = 0
        self.running = running
        self.starts = 0
        self.stopped = 0
        self.sent = None
        self.rates = []

    def upload(self, frames):
        self.uploaded += 1 if isinstance(frames, str) else len(list(frames))
        self.sent = frames
        return self.uploaded

    def is_running(self):
        return self.running

    def start(self, *, fps):
        self.rates.append(fps)
        self.starts += 1
        self.running = True
        return 1

    def stop(self):
        self.stopped += 1
        self.running = False
        return True

    def playing(self, *, fps):
        self.start(fps=fps)
        return self

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.stop()
        return False


class Sess:
    """Session stub recording loads, their link and every parameter vector, raising the port's load errors."""

    def __init__(self, transport, **kwargs):
        self.transport = transport
        self.kwargs = kwargs
        self.loads = []
        self.vectors = []

    def load_program(self, name, *, park=True, link=None):
        self.loads.append((name, park, link))
        if self.transport.load_errors.get(name):
            raise self.transport.load_errors[name].pop(0)

    def working_point(self, info):
        return {p.index: 0.5 for p in info.params if p.kind is not ParamKind.UNUSED}

    def set_params(self, values):
        self.vectors.append(dict(values))
        self.transport.state.update(values)


def run(archive, *, port=None, recorder=None, sessions=None, clock=None, **kwargs):
    """Drive a harvest against stubs, returning the report."""
    port = Port() if port is None else port
    clock = FakeClock() if clock is None else clock
    made = [] if sessions is None else sessions

    def factory(transport, **kw):
        made.append(Sess(transport, **kw))
        return made[-1]

    kwargs.setdefault("config", CONFIG)
    kwargs.setdefault("plan", fake_plan)
    kwargs.setdefault("session_factory", factory)
    return H.harvest(
        archive,
        open_transport=lambda: port,
        recorder=Rec(clock) if recorder is None else recorder,
        sleep=clock.sleep,
        clock=clock,
        **kwargs,
    )


def window(setpoint, start, end):
    """One held setpoint, with a vector that names it."""
    return H.Window(setpoint, tuple([setpoint] * 12), start, end)


def test_a_window_opens_only_once_the_parameters_have_settled():
    """A window that opened on the write would claim frames from the previous setpoint."""
    clock = FakeClock()
    session = Sess(Readback())
    windows = H.sweep(session, {}, [{1: 0.0}, {1: 1.0}], config=CONFIG, sleep=clock.sleep, clock=clock)
    assert [w.setpoint for w in windows] == [0, 1]
    assert [(w.start, w.end) for w in windows] == [(0.25, 1.25), (1.5, 2.5)]
    assert session.vectors == [{1: 0.0}, {1: 1.0}], "one write per setpoint, before its window"


def test_a_window_carries_the_whole_commanded_state_not_only_the_swept_slots():
    clock = FakeClock()
    session = Sess(Readback())
    base = session.working_point(INFO)
    session.set_params(base)
    windows = H.sweep(session, base, [{2: 0.0}], config=CONFIG, sleep=clock.sleep, clock=clock)
    held = H.raw_params(base)[2]
    assert held and len(windows[0].params) == 12
    assert windows[0].params[2] == held and windows[0].params[1] == 0


def test_a_frame_is_attributed_to_the_setpoint_that_was_being_held_when_it_arrived():
    windows = [window(0, 1.0, 2.0), window(1, 2.5, 3.5)]
    rows = H.attribute(np.array([1.0, 1.9, 2.5, 3.5]), windows, "Alpha", base=BASE)
    assert [r.setpoint for r in rows] == [0, 0, 1, 1], "the ends of a window are inside it"
    assert [r.frame for r in rows] == [0, 1, 2, 3]
    assert [r.params for r in rows] == [windows[0].params] * 2 + [windows[1].params] * 2
    assert [r.captured for r in rows] == [1.0, 1.9, 2.5, 3.5]
    assert {r.program for r in rows} == {"Alpha"}


def test_a_frame_captured_between_windows_is_kept_as_a_gap():
    """The sidecar stays one row per frame, so the archive needs no re-encode."""
    windows = [window(0, 1.0, 2.0), window(1, 2.5, 3.5)]
    rows = H.attribute(np.array([0.5, 2.25, 9.0]), windows, "Alpha", base=BASE)
    assert [r.setpoint for r in rows] == [GAP, GAP, GAP], "before, between and after all count"
    assert [r.params for r in rows] == [BASE] * 3, "a gap row carries the base state"


def test_attribution_does_not_depend_on_the_order_windows_were_recorded_in():
    windows = [window(1, 2.5, 3.5), window(0, 1.0, 2.0)]
    rows = H.attribute(np.array([1.5, 3.0]), windows, "Alpha", base=BASE)
    assert [(r.setpoint, r.params) for r in rows] == [(0, windows[1].params), (1, windows[0].params)]


def test_attribution_without_a_window_archives_nothing():
    """A sweep that never opened a window has nothing to say about any frame."""
    assert not H.attribute(np.array([1.0, 2.0]), [], "Alpha", base=BASE)


def test_attribution_of_a_recording_that_holds_no_frames():
    assert not H.attribute(np.array([]), [window(0, 1.0, 2.0)], "Alpha", base=BASE)


def test_holds_program_is_the_load_completing():
    """The load waits on liveness, so its outcome is the verdict: refused is no, gone propagates."""
    assert H.holds_program(Port())
    assert not H.holds_program(Port(holds=False))
    port = Port()
    port.load_errors[H.WEDGE_PROBE] = [ProgramLoadError("Passthru", {"state": "running"}, "did not load")]
    assert not H.holds_program(port)
    port.load_errors[H.WEDGE_PROBE] = [DeviceGoneError("loading Passthru", "shell busy 3.0s")]
    with pytest.raises(DeviceGoneError):
        H.holds_program(port)


def test_wait_healthy_reopens_the_transport_across_a_power_cycle():
    """A wedge only clears on a power cycle, which re-enumerates the serial node."""
    ports = [Port(holds=False), Port(holds=False), Port()]
    queue = list(ports)
    got = H.wait_healthy(lambda: queue.pop(0), config=CONFIG, sleep=lambda _s: None, clock=FakeClock())
    assert got is ports[-1] and ports[0].closed and not got.closed


def test_wait_healthy_gives_up_rather_than_blocking_forever():
    clock = FakeClock()
    config = H.HarvestConfig(health_timeout_s=2.0, health_poll_s=1.0)
    with pytest.raises(H.HarvestError, match="hold a program"):
        H.wait_healthy(lambda: Port(holds=False), config=config, sleep=clock.sleep, clock=clock)


def test_wait_healthy_tolerates_a_transport_that_will_not_even_open():
    """A power cycle re-enumerates the serial node, so opening fails for a while."""
    attempts = []

    def opener():
        attempts.append(len(attempts))
        if len(attempts) < 3:
            raise OSError("no such device")
        return Port()

    got = H.wait_healthy(opener, config=CONFIG, sleep=lambda _s: None, clock=FakeClock())
    assert isinstance(got, Port) and len(attempts) == 3


def test_wait_healthy_stops_at_a_port_another_process_holds():
    """A held tty is not a power cycle, so waiting out the health deadline would only hide it."""
    attempts = []

    def opener():
        attempts.append(1)
        raise PortBusyError("/dev/ttyACM0")

    with pytest.raises(PortBusyError):
        H.wait_healthy(opener, config=CONFIG, sleep=lambda _s: None, clock=FakeClock())
    assert len(attempts) == 1


def test_the_canary_loads_a_passthrough_and_watches_for_the_picture():
    clock = FakeClock()
    port = Port()
    session = Sess(port)
    assert H.carries_stimulus(session, Rec(clock), port, config=CONFIG, sleep=clock.sleep, clock=clock)
    assert session.loads == [(H.WEDGE_PROBE, True, None)]
    assert session.vectors, "a parked passthrough with no working point carries nothing either"


def test_the_canary_restarts_the_stimulus_before_blaming_the_rig():
    """A dead loop is a black picture too, and it is the host's fault, not the device's."""
    clock = FakeClock()
    port = Port()
    player = Player(running=False)
    assert H.carries_stimulus(
        Sess(port), Rec(clock), port, config=CONFIG, player=player, sleep=clock.sleep, clock=clock
    )
    assert player.starts == 1 and player.running


def test_the_canary_is_false_only_for_a_transmitted_flat_field(monkeypatch):
    """A passthrough driving a flat field means the rig is dark; a capture that never came is no verdict."""
    blind(monkeypatch, canary=False)
    clock = FakeClock()
    port = Port()
    assert not H.carries_stimulus(Sess(port), Rec(clock), port, config=CONFIG, sleep=clock.sleep, clock=clock)
    blind(monkeypatch, canary=False, canary_probe=None)
    with pytest.raises(BlankTakeError):
        H.carries_stimulus(Sess(port), Rec(clock), port, config=CONFIG, sleep=clock.sleep, clock=clock)


def test_a_passthrough_on_the_wrong_input_is_a_route_error_not_a_dark_rig(monkeypatch):
    """A device left on analog is black for a reason the route check names before the picture is judged."""
    blind(monkeypatch, canary=False)
    clock = FakeClock()
    port = Port(video=[dict(tx.LIVE, source="analog")])
    with pytest.raises(InputRouteError, match="reads input 'analog'"):
        H.carries_stimulus(Sess(port), Rec(clock), port, config=CONFIG, sleep=clock.sleep, clock=clock)


def test_the_canary_judges_the_stimulus_only_once_the_transmitter_drives_it(monkeypatch):
    """The real settle: the transmitter is waited for on liveness, and a device that leaves USB ends it."""
    monkeypatch.setattr(H, "settle", recorder_mod.settle)
    monkeypatch.setattr(recorder_mod, "inspect_take", lambda path, **kwargs: LIVE)
    clock = FakeClock()
    alive, dead = Port(video=tx.blackout(300)), Port(video=[tx.DEAD])
    recorder = Rec(clock)
    assert H.carries_stimulus(Sess(alive), recorder, alive, config=CONFIG, sleep=clock.sleep, clock=clock)
    assert alive.transmitter.polls == 302 and len(recorder.recordings) == 1
    dead.transmitter.attached = (True, False)
    with pytest.raises(DeviceGoneError):
        H.carries_stimulus(Sess(dead), Rec(clock), dead, config=CONFIG, sleep=clock.sleep, clock=clock)


def test_a_program_is_swept_into_exactly_one_recording(tmp_path):
    """The card is read once per program; the host only writes setpoints while it runs."""
    clock = FakeClock()
    recorder = Rec(clock)
    archive = Archive(tmp_path)
    port = Port()
    session = Sess(port)
    result = H.harvest_program(
        session,
        recorder,
        archive,
        port,
        "Alpha",
        KEY,
        config=CONFIG,
        plan=fake_plan,
        sleep=clock.sleep,
        clock=clock,
    )
    assert len(recorder.recordings) == 1 and archive.commits == ["Alpha"]
    assert not recorder.recordings[0].exists(), "the scratch recording was published, not left behind"
    assert archive.paths("Alpha")[0].exists()
    assert archive.geometry["Alpha"] == (CONFIG.width, CONFIG.height, CONFIG.capture_fps)
    assert (result.frames, result.measured) == (PER_PROGRAM, MEASURED)
    assert result.seconds == pytest.approx(2.5) and result.report is LIVE and not result.dark
    assert len(session.vectors) == 1 + len(FakeCover().vectors()), "the working point, then every setpoint"
    assert archive.infos["Alpha"] == COLORBARS_INFO, "the device's reply is archived with the frames"


def test_every_captured_frame_is_archived_against_the_vector_it_was_captured_under(tmp_path):
    archive = Archive(tmp_path)
    report = run(archive)
    assert [r.program for r in report.results] == ["Alpha", "Beta"]
    assert report.frames == 2 * PER_PROGRAM
    for program in ("Alpha", "Beta"):
        rows = archive.rows[program]
        assert [r.frame for r in rows] == list(range(PER_PROGRAM))
        assert sorted({r.setpoint for r in rows}) == [GAP, 0, 1]
        assert all(len(r.params) == 12 for r in rows), "rows carry the full raw vector"
        assert len({r.params for r in rows if r.setpoint != GAP}) == 2, "one vector per setpoint"
        assert [r.captured for r in rows] == sorted(r.captured for r in rows)


def test_harvest_keys_on_name_and_firmware_not_the_program_binary(tmp_path):
    """Hashing binaries runs over the serial shell, which is what wedges this device."""
    archive = Archive(tmp_path)
    port = Port()
    run(archive, port=port)
    assert port.hashes == 0
    for program, key in archive.keys.items():
        assert key == ProgramKey(program, FIRMWARE, KeyKind.NAME_FIRMWARE)


def test_harvest_drops_the_link_on_every_program_load(tmp_path):
    link = object()
    sessions = []
    run(Archive(tmp_path), sessions=sessions, link=link)
    assert sessions[0].loads == [("Alpha", True, link), ("Beta", True, link)]


def test_harvest_resumes_by_skipping_what_is_already_archived(tmp_path):
    archive = Archive(tmp_path, stored=[("Alpha", ProgramKey("Alpha", FIRMWARE, KeyKind.NAME_FIRMWARE))])
    report = run(archive)
    assert [r.cached for r in report.results] == [True, False]
    assert "Alpha" not in archive.rows and report.frames == PER_PROGRAM


def test_harvest_reflashed_firmware_invalidates_a_stored_archive(tmp_path):
    stale = ProgramKey("Alpha", "0.9.0", KeyKind.NAME_FIRMWARE)
    archive = Archive(tmp_path, stored=[("Alpha", stale)])
    assert not run(archive).results[0].cached


def test_harvest_over_mixed_authors_keeps_each_program_distinct(tmp_path):
    names = ("Teletext", "czartroniks/Teletext", "boneoh/RGB Bit Crush")
    archive = Archive(tmp_path)
    report = run(archive, port=Port(programs=names))
    assert sorted(r.program for r in report.results) == sorted(names)
    assert set(archive.rows) == set(names)
    assert len({archive.paths(n)[0] for n in names}) == 3
    assert all(path.parent == tmp_path for n in names for path in archive.paths(n))


def test_harvest_only_probes_the_programs_it_was_given(tmp_path):
    report = run(Archive(tmp_path), programs=["Beta"])
    assert [r.program for r in report.results] == ["Beta"]


def test_harvest_stops_once_the_device_leaves_usb(tmp_path):
    """A device gone from USB needs a power cycle, so the remaining programs must not be burnt through."""
    port = Port(programs=("Alpha", "Beta", "Gamma"))
    gone = DeviceGoneError("loading Beta", "shell busy 8.0s, usb gone")
    port.load_errors = {"Beta": [gone]}
    report = run(Archive(tmp_path), port=port, session_factory=passthrough_raises(gone))
    assert report.gone and [r.program for r in report.results] == ["Alpha", "Beta"]
    assert report.failures[0].error.startswith("DeviceGoneError") and report.fault.startswith(H.GONE_NOTE)
    assert port.closed


def test_a_load_that_merely_runs_long_never_stops_the_run(tmp_path):
    """A refused or wrong load on a device still on USB fails that program only."""
    port = Port(programs=("Alpha", "Beta"))
    port.load_errors = {"Alpha": [ProgramLoadError("Alpha", {"state": "idle"}, "did not load")]}
    report = run(Archive(tmp_path), port=port)
    assert not report.stopped and [r.program for r in report.failures] == ["Alpha"]
    assert report.results[1].frames == PER_PROGRAM


def test_a_device_left_on_another_input_fails_the_pass_not_the_run(tmp_path):
    """The selected input is verified after every load; a mismatch is that pass's failure."""
    port, archive = Port(programs=("Alpha", "Beta"), video=[dict(tx.LIVE, source="analog")]), Archive(
        tmp_path
    )
    said = []
    report = run(archive, port=port, log=said.append)
    assert [r.program for r in report.failures] == ["Alpha", "Beta"] and not report.stopped
    assert all(r.error.startswith("InputRouteError") for r in report.failures) and not archive.commits
    assert any("Passthru check raised InputRouteError" in line for line in said)


def test_a_wrong_route_before_the_run_archives_nothing(tmp_path):
    """Vid Route Mode is checked before the first program, and a mismatch is the run's fault."""
    port, archive = Port(), Archive(tmp_path)
    port.transmitter.routed = 4
    report = run(archive, port=port)
    assert report.stopped and report.fault.startswith(H.ROUTE_NOTE) and not report.results
    assert "reads 4 (Auto), the session needs 1 (HDMI In)" in report.fault
    assert port.loads == [H.WEDGE_PROBE] and not archive.commits and port.closed


def test_a_route_that_changes_during_a_sweep_fails_the_pass_and_stops_the_run(tmp_path):
    """A route wrong after the sweep fails that program unarchived, and the control blames the route."""
    port, archive = Port(programs=("Alpha", "Beta")), Archive(tmp_path)
    routes = iter([1, 1])
    port.route = lambda: next(routes, 4)
    report = run(archive, port=port)
    assert [r.program for r in report.results] == ["Alpha"] and not archive.commits
    assert report.results[0].error.startswith("RouteError: Alpha: Vid Route Mode reads 4 (Auto)")
    assert report.stopped and report.fault.startswith(H.ROUTE_NOTE) and not archive.dark_luma


def test_the_route_check_can_be_skipped(tmp_path):
    """``route=None`` never reads the setting."""
    port = Port(programs=("Alpha",))
    port.route = lambda: pytest.fail("the route was read")
    report = run(Archive(tmp_path), port=port, config=dataclasses.replace(CONFIG, route=None))
    assert not report.stopped and report.results[0].frames == PER_PROGRAM


def test_a_cc_map_changed_by_the_end_of_a_sweep_fails_the_program_unarchived(tmp_path):
    """A take driven through a changed map measures the wrong slots, so it is never committed."""

    class Corrupting(Sess):
        """Session whose load of Alpha leaves the device CC map changed."""

        def load_program(self, name, *, park=True, link=None):
            super().load_program(name, park=park, link=link)
            self.transport.corrupt = name == "Alpha"

    port, archive = Port(), Archive(tmp_path)
    report = run(archive, port=port, session_factory=Corrupting)
    assert [r.program for r in report.failures] == ["Alpha"]
    assert report.failures[0].error.startswith("CcMapError")
    assert archive.commits == ["Beta"] and port.verified == 2
    assert not report.gone and not report.blacked


@pytest.mark.parametrize(
    ("video", "when"),
    [([tx.DEAD], "as the sweep began"), ([tx.LIVE, tx.LIVE, tx.at("720p60"), tx.LIVE], "as the sweep ended")],
    ids=["dropped-before", "dropped-during"],
)
def test_a_transmitter_lost_at_a_sweep_boundary_fails_the_program_unarchived(tmp_path, video, when):
    """Frames recorded while the transmitter was down are the card's, never the program's."""
    port, archive = Port(video=video), Archive(tmp_path)
    report = run(archive, port=port, programs=["Alpha"])
    assert report.failures[0].error.startswith("SignalLostError") and when in report.failures[0].error
    assert not archive.commits and not archive.paths("Alpha")[0].exists()
    assert not report.stopped, "the canary is what decides whether the rig is dark"


def test_harvest_records_nothing_until_the_transmitter_is_back(tmp_path, monkeypatch):
    """The real settle waits out a load blackout over serial before the card is read at all."""
    monkeypatch.setattr(H, "settle", recorder_mod.settle)
    monkeypatch.setattr(recorder_mod, "inspect_take", lambda path, **kwargs: LIVE)
    clock = FakeClock()
    port = Port(video=tx.blackout(4))

    class Watched(Rec):
        """Notes how many status polls preceded each recording."""

        polls = []

        def recording(self, path, **kwargs):
            self.polls.append(port.transmitter.polls)
            return super().recording(path, **kwargs)

    recorder = Watched(clock)
    report = run(Archive(tmp_path), port=port, recorder=recorder, clock=clock, programs=["Alpha"])
    assert not report.failures and report.results[0].frames
    assert recorder.polls[0] == 5, "the probe waited for the fifth poll, the first to see the transmitter"


def test_a_failure_with_a_live_canary_carries_on(tmp_path):
    """One broken program on a healthy rig must not stop the run."""

    class Failing(Sess):
        """Session that cannot load one particular program."""

        def load_program(self, name, *, park=True, link=None):
            super().load_program(name, park=park, link=link)
            if name == "Alpha":
                raise RuntimeError("that one is broken")

    report = run(Archive(tmp_path), session_factory=Failing)
    assert not report.stopped and [r.program for r in report.results] == ["Alpha", "Beta"]
    assert report.failures and report.results[1].frames == PER_PROGRAM


def passthrough_raises(*errors):
    """A session factory whose passthrough loads raise ``errors`` in turn; the device itself opens fine."""
    queue = list(errors)

    class Refusing(Sess):
        """Session whose passthrough load raises the next queued error."""

        def load_program(self, name, *, park=True, link=None):
            super().load_program(name, park=park, link=link)
            if name == H.WEDGE_PROBE and queue:
                raise queue.pop(0)

    return Refusing


def blank_once(monkeypatch, program):
    """``program`` shows no picture after its load; everything else, the passthrough included, does."""

    def probe(recorder, transport, *, program: str, **kwargs):
        del recorder, transport, kwargs
        if program == blank:
            raise BlankTakeError(f"{program}: transmitting 1080p30 but nothing captured within 40.0s")
        return LIVE

    blank = program
    monkeypatch.setattr(H, "settle", probe)


def test_a_passthrough_load_that_raises_on_a_healthy_device_does_not_end_the_run(tmp_path, monkeypatch):
    """A blank program, then a passthrough load misread before it completed: the next program still runs."""
    blank_once(monkeypatch, "Alpha")
    port = Port(programs=("Alpha", "Beta"))
    refusing = passthrough_raises(ShellError(3, "no program loaded", "program info"))
    said = []
    report = run(Archive(tmp_path), port=port, log=said.append, session_factory=refusing)
    assert not report.stopped and not report.fault
    assert [r.program for r in report.results] == ["Alpha", "Beta"] and report.results[
        1
    ].frames == PER_PROGRAM
    assert report.failures[0].error.startswith("BlankTakeError")
    assert any("Passthru check raised ShellError" in line for line in said)


def test_a_passthrough_check_that_finds_the_device_gone_stops_the_run(tmp_path, monkeypatch):
    """The device that went away mid-check is a rig fault, and the run stops there."""
    blank_once(monkeypatch, "Alpha")
    gone = passthrough_raises(DeviceGoneError("loading Passthru", "shell busy 2.0s, usb gone"))
    report = run(Archive(tmp_path), port=Port(programs=("Alpha", "Beta")), session_factory=gone)
    assert report.gone and report.fault.startswith(H.GONE_NOTE)
    assert [r.program for r in report.results] == ["Alpha"]


def test_a_failed_program_loses_its_dark_verdict_so_a_resume_retries_it(tmp_path, monkeypatch):
    """A retried dark program that now fails has no verdict, and the next plain resume runs it."""
    archive = Archive(tmp_path)
    archive.mark_dark("Alpha", KEY, 0.0)
    blank_once(monkeypatch, "Alpha")
    report = run(archive, programs=["Alpha"], retry_dark=True)
    assert report.failures and not archive.marked
    monkeypatch.setattr(H, "settle", lambda recorder, transport, **kwargs: LIVE)
    assert run(archive, programs=["Alpha"]).results[0].frames == PER_PROGRAM


def test_the_summary_lists_each_failed_program_with_its_error_and_the_fault(tmp_path, monkeypatch):
    """Dark discards are verdicts, not failures; a rig fault is named last."""
    blank_once(monkeypatch, "Alpha")
    report = run(Archive(tmp_path), port=Port(programs=("Alpha", "Beta")))
    assert report.summary() == [
        f"{PER_PROGRAM} frames from 2 programs in {report.seconds / 60:.1f} min, 1 failed",
        "  Alpha: BlankTakeError: Alpha: transmitting 1080p30 but nothing captured within 40.0s",
    ]
    blind(monkeypatch, canary=False)
    report = run(Archive(tmp_path / "b"), port=Port(programs=("Alpha", "Beta")))
    assert report.summary()[-1].startswith(
        f"stopped: {H.BLACKED_NOTE} (Passthru: the passthrough shows no picture"
    )
    assert ", 0 failed" in report.summary()[0], "a dark discard is not a failure"


def test_a_failure_on_a_dark_rig_stops_the_run(tmp_path, monkeypatch):
    """A raising program on a faulted rig must be diagnosed, not repeated 49 times."""
    blind(monkeypatch, canary=False)

    class Failing(Sess):
        """Session that cannot load the first program."""

        def load_program(self, name, *, park=True, link=None):
            super().load_program(name, park=park, link=link)
            if name == "Alpha":
                raise RuntimeError("that one is broken")

    report = run(Archive(tmp_path), session_factory=Failing)
    assert report.blacked and not report.gone, "a dark rig is not a device gone from USB"
    assert [r.program for r in report.results] == ["Alpha"], "it stopped rather than trying the rest"


def test_a_dark_program_is_discarded_when_the_canary_still_carries_the_source(tmp_path, monkeypatch):
    """One black program is legitimate, so the run keeps going."""
    blind(monkeypatch, canary=True)
    archive = Archive(tmp_path)
    report = run(archive)
    assert report.blacked is False and report.stopped is False
    assert [r.program for r in report.results] == ["Alpha", "Beta"], "every program was attempted"
    assert all("discarded" in r.error for r in report.results)
    assert report.frames == 0, "black frames are not counted as harvested"
    assert archive.dark_luma == {"Alpha": DARK.luma, "Beta": DARK.luma}


def test_black_output_with_a_dark_canary_stops_the_run(tmp_path, monkeypatch):
    """A passthrough that carries nothing means the rig, not the program, is dark: no dark verdict."""
    blind(monkeypatch, canary=False)
    archive = Archive(tmp_path)
    report = run(archive)
    assert report.blacked is True and report.stopped is True and not archive.dark_luma
    assert "Vid Route Mode 1 (HDMI In)" in report.fault and "transmitter vic 34 de 2FE90F" in report.fault
    assert report.gone is False, "a black output is not a device gone from USB, and needs its own diagnosis"
    assert [r.program for r in report.results] == ["Alpha"], "it stopped rather than archiving more"


def test_a_discarded_program_has_its_archive_files_removed(tmp_path, monkeypatch):
    """Black frames must not stay committed, or a resume would skip re-probing them."""
    blind(monkeypatch, canary=True)
    archive = Archive(tmp_path)
    run(archive, programs=["Alpha"])
    assert not any(path.exists() for path in archive.paths("Alpha"))


def test_a_dark_program_is_not_probed_again_on_resume(tmp_path, monkeypatch):
    """Loads are the scarce resource: a program that is itself black has no archive to resume."""
    blind(monkeypatch, canary=True)
    archive = Archive(tmp_path)
    run(archive, programs=["Alpha"])
    assert archive.marked == {"Alpha"}, "the verdict is recorded against the key"
    report = run(archive, programs=["Alpha"])
    assert [r.cached for r in report.results] == [True] and report.frames == 0


def test_a_dark_verdict_from_other_firmware_is_not_trusted(tmp_path, monkeypatch):
    blind(monkeypatch, canary=True)
    archive = Archive(tmp_path)
    archive.marked_key["Alpha"] = "0.9.0-digest"
    assert not run(archive, programs=["Alpha"]).results[0].cached, "a reflash re-probes it"


def test_a_dark_program_is_retried_and_rejudged_when_asked(tmp_path, monkeypatch):
    """``retry_dark`` ignores the verdict, so a program a stricter criterion condemned runs again."""
    blind(monkeypatch, canary=True)
    archive = Archive(tmp_path)
    run(archive, programs=["Alpha"])
    monkeypatch.setattr(H, "inspect_take", lambda path, **kwargs: LIVE)
    report = run(archive, programs=["Alpha"], retry_dark=True)
    assert not report.results[0].cached and not report.results[0].dark
    assert archive.commits == ["Alpha", "Alpha"] and archive.paths("Alpha")[0].exists()


def sweep_take(tmp_path, uniform_every):
    """MJPEG sweep of achromatic bars of varying pitch, every ``uniform_every``-th setpoint flat."""
    width = np.arange(320)
    frames = [
        (
            np.full((180, 320, 3), 16, np.uint8)
            if uniform_every and i % uniform_every == 0
            else np.repeat(
                np.where((width // (8 + i)) % 2, 255, 0).astype(np.uint8)[None, :, None], 180, 0
            ).repeat(3, 2)
        )
        for i in range(30)
    ]
    return inspect_take(mjpeg(tmp_path / "sweep.mkv", frames), samples=60)


@needs_ffmpeg
def test_a_sweep_with_flat_setpoints_and_achromatic_bars_is_archived_not_discarded(tmp_path, monkeypatch):
    """A third of the setpoints flat and the rest black-and-white bars: still data."""
    take = sweep_take(tmp_path, uniform_every=3)
    assert take.picture == 20 and take.ever_structured, "achromatic bars are picture"
    assert not take.usable, "a render would reject it, which is not the archive's standard"
    blind(monkeypatch, canary=True, take=take)
    archive = Archive(tmp_path / "archive")
    report = run(archive, programs=["Alpha"])
    assert not report.results[0].dark and not report.results[0].error
    assert archive.paths("Alpha")[0].exists() and not archive.marked


@needs_ffmpeg
def test_an_all_uniform_program_is_dark(tmp_path, monkeypatch):
    take = sweep_take(tmp_path, uniform_every=1)
    blind(monkeypatch, canary=True, take=take)
    archive = Archive(tmp_path / "archive")
    report = run(archive, programs=["Alpha"])
    assert "discarded" in report.results[0].error and archive.marked == {"Alpha"}


def test_results_report_what_the_take_actually_held():
    measured = H.ProgramResult("P", frames=10, measured=9, seconds=12.0, report=LIVE)
    assert not measured.dark and "9/10 frames measured in 12s" in str(measured)
    assert "30 picture" in str(measured) and "motion +0.60" in str(measured)
    assert H.ProgramResult("P", frames=10, report=DARK).dark and "UNUSABLE" in str(
        H.ProgramResult("P", frames=10, report=DARK)
    )
    held_flat = TakeReport(frames=30, picture=1, frozen=True)
    assert not held_flat.usable and not H.ProgramResult("P", frames=10, report=held_flat).dark
    assert str(H.ProgramResult("P")) == "P: cached"
    assert "FAILED" in str(H.ProgramResult("P", error="RuntimeError: x"))


def test_upload_stimulus_builds_a_loop_at_the_playout_geometry():
    player = Player()
    config = H.HarvestConfig(width=64, height=48, loop_frames=3)
    assert H.upload_stimulus(player, config) == 3


def test_harvest_uploads_the_stimulus_once_and_restarts_a_dead_player(tmp_path):
    player = Player(running=False)
    messages = []
    run(Archive(tmp_path), player=player, log=messages.append)
    assert player.uploaded == CONFIG.loop_frames, "the loop is invariant, so it is paid for once"
    assert player.stopped and any("Alpha" in m for m in messages)


def test_harvest_restarts_a_player_that_died_mid_run(tmp_path):
    """The loop is the stimulus; without it every archived frame is of nothing."""

    class Dead(Player):
        """Player whose loop never stays up."""

        def is_running(self):
            return False

    player = Dead()
    run(Archive(tmp_path), player=player)
    assert player.starts == 3, "once for the run, then once per program that found it down"


def test_a_real_session_holds_the_link_down_for_the_load_command_only(tmp_path):
    """End to end against the real Session: the link must be down for the load itself."""
    calls = []

    class Link:
        """Link stub logging when it is dropped and restored."""

        def quiet(self):
            """Quiet."""
            return self

        def __enter__(self):
            calls.append("down")
            return self

        def __exit__(self, *exc):
            calls.append("up")
            return False

    clock = FakeClock()
    port = FakeTransport()
    port.programs = lambda: ["Alpha"]
    port.firmware = lambda: FIRMWARE
    port.program_info = lambda name=None: INFO
    port.program_payload = lambda name=None: dict(COLORBARS_INFO)
    port.describe = Port.describe
    port.close = lambda: calls.append("close")

    def load(name, hold=None):
        with nullcontext() if hold is None else hold():
            calls.append(f"load {name}")
        calls.append(f"running {name}")

    port.load_program = load
    report = run(
        Archive(tmp_path),
        port=port,
        clock=clock,
        link=Link(),
        session_factory=lambda t, **kw: Session(t, sleep=clock.sleep, clock=clock, **kw),
    )
    assert report.frames and not report.failures
    start = calls.index("down")
    assert calls[start : start + 4] == [
        "down",
        "load Alpha",
        "up",
        "running Alpha",
    ], "down for the command only"

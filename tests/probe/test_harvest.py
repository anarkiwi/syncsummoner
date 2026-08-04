"""Harvest: one recording per program, frames attributed to setpoints by capture time."""

# pylint: disable=missing-function-docstring

from contextlib import contextmanager
from pathlib import Path

import numpy as np
import pytest

from syncsummoner.device.profile import CROSSFADER_INDEX, ParamKind
from syncsummoner.device.recorder import BlankTakeError, TakeReport
from syncsummoner.device.session import Session
from syncsummoner.device.transport import ProgramInfo
from syncsummoner.probe import harvest as H
from syncsummoner.probe.archive import GAP
from syncsummoner.probe.store import KeyKind, ProgramKey
from tests.device.conftest import COLORBARS_INFO, FakeClock, FakeTransport

FIRMWARE = "1.0.0-rc.37"
INFO = ProgramInfo.from_json(COLORBARS_INFO)
KEY = ProgramKey("Alpha", FIRMWARE, KeyKind.NAME_FIRMWARE)
CONFIG = H.HarvestConfig(
    width=64,
    height=48,
    capture_fps=4,
    setpoints=2,
    settle_s=0.25,
    dwell_s=1.0,
    startup_s=0.0,
    health_poll_s=0.0,
    wedge_settle_s=0.0,
    loop_frames=2,
)
#: A take that carries a moving picture, and one that is black however long it ran.
LIVE = TakeReport(frames=30, distinct=24, blank=0, luma=0.42)
DARK = TakeReport(frames=30, distinct=1, blank=30, luma=0.001)
#: Frames per program at ``capture_fps`` over two dwells and their settles.
PER_PROGRAM = 10
MEASURED = 9
BASE = tuple(range(12))


@pytest.fixture(autouse=True)
def no_ffmpeg(monkeypatch):
    """No rig and no ffmpeg: the picture always returns, and every take carries one."""
    monkeypatch.setattr(H, "settle", lambda recorder, **kwargs: LIVE)
    monkeypatch.setattr(H, "inspect_take", lambda path, **kwargs: LIVE)


def blind(monkeypatch, *, canary=True, take=DARK):
    """Every take reads black; ``canary`` says whether the passthrough still carries one."""
    monkeypatch.setattr(H, "inspect_take", lambda path, **kwargs: take)

    def probe(recorder, *, program, **kwargs):
        del recorder, kwargs
        if program == H.WEDGE_PROBE and not canary:
            raise BlankTakeError(f"{program}: no picture")
        return LIVE

    monkeypatch.setattr(H, "settle", probe)


class Port:
    """Transport stub: answers program queries, and can refuse to hold a program."""

    def __init__(self, programs=("Alpha", "Beta"), *, holds=True, info=INFO):
        self.programs_list = list(programs)
        self.holds = holds
        self.info = info
        self.loads = []
        self.closed = False
        self.hashes = 0

    def programs(self):
        return list(self.programs_list)

    def firmware(self):
        return FIRMWARE

    def load_program(self, name):
        self.loads.append(name)

    def program_info(self, name=None):
        del name
        if not self.holds:
            raise RuntimeError("[3] no program loaded")
        return self.info

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
        self.rows = {}
        self.keys = {}
        self.commits = []
        self.geometry = {}
        self.marked = set()
        self.marked_key = {}
        self.dark_luma = {}

    def has(self, program, key):
        return self.stored.get(program) == key.digest

    def dark(self, program, key):
        return self.marked_key.get(program) == key.digest

    def mark_dark(self, program, key, luma):
        self.marked.add(program)
        self.marked_key[program] = key.digest
        self.dark_luma[program] = luma

    def paths(self, program):
        return tuple(self.directory / f"{program}{suffix}" for suffix in (".mkv", ".parquet", ".json"))

    def scratch(self, program):
        self.directory.mkdir(parents=True, exist_ok=True)
        return self.directory / f"{program}.scratch.mkv"

    def commit(self, program, key, video, rows, *, width, height, fps):
        assert Path(video).exists(), "the recording is published, not re-encoded"
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

    def upload(self, frames):
        self.uploaded += len(list(frames))
        return self.uploaded

    def is_running(self):
        return self.running

    def start(self, *, fps):
        del fps
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
    """Session stub recording loads, their link, and every parameter vector."""

    def __init__(self, transport, **kwargs):
        self.transport = transport
        self.kwargs = kwargs
        self.loads = []
        self.vectors = []

    def load_program(self, name, *, park=True, link=None):
        self.loads.append((name, park, link))

    def working_point(self, info):
        return {p.index: 0.5 for p in info.params if p.kind is not ParamKind.UNUSED}

    def set_params(self, values):
        self.vectors.append(dict(values))


def run(archive, *, port=None, recorder=None, sessions=None, clock=None, **kwargs):
    """Drive a harvest against stubs, returning the report."""
    port = Port() if port is None else port
    clock = FakeClock() if clock is None else clock
    made = [] if sessions is None else sessions

    def factory(transport, **kw):
        made.append(Sess(transport, **kw))
        return made[-1]

    kwargs.setdefault("config", CONFIG)
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


def test_sweep_includes_the_crossfader_like_any_other_continuous_effect():
    """Pinning P12 open produced false blank takes; it now sweeps 0..100% too."""
    vectors = H.sweep_vectors(INFO, CONFIG, np.random.default_rng(0))
    assert vectors and all(CROSSFADER_INDEX in v for v in vectors)
    values = {v[CROSSFADER_INDEX] for v in vectors}
    assert len(values) > 1, "a fixed value would be the same regression as pinning it open"
    booleans = {p.index for p in INFO.params if p.kind is ParamKind.BOOLEAN}
    assert booleans and all(booleans <= set(v) for v in vectors), "booleans change the picture"


def test_sweep_still_forces_the_crossfader_even_when_the_program_calls_every_slot_null():
    """It gates output even where ``program info`` names it ``Null 12``."""
    info = ProgramInfo.from_json({"name": "Null", "parameters": [{"name": "-", "min": 0, "max": 100}] * 12})
    vectors = H.sweep_vectors(info, CONFIG, np.random.default_rng(0))
    assert vectors and all(set(v) == {CROSSFADER_INDEX} for v in vectors)


def test_sweep_of_a_program_reporting_no_parameters_at_all_is_one_empty_vector():
    info = ProgramInfo.from_json({"name": "Null", "parameters": []})
    assert H.sweep_vectors(info, CONFIG, np.random.default_rng(0)) == [{}]


def test_a_window_opens_only_once_the_parameters_have_settled():
    """A window that opened on the write would claim frames from the previous setpoint."""
    clock = FakeClock()
    session = Sess(None)
    windows = H.sweep(session, {}, [{1: 0.0}, {1: 1.0}], config=CONFIG, sleep=clock.sleep, clock=clock)
    assert [w.setpoint for w in windows] == [0, 1]
    assert [(w.start, w.end) for w in windows] == [(0.25, 1.25), (1.5, 2.5)]
    assert session.vectors == [{1: 0.0}, {1: 1.0}], "one write per setpoint, before its window"


def test_a_window_carries_the_whole_commanded_state_not_only_the_swept_slots():
    """P12 gates the output and is held open off-sweep, so a zero there misreads the state."""
    clock = FakeClock()
    base = Sess(None).working_point(INFO)
    windows = H.sweep(Sess(None), base, [{1: 0.0}], config=CONFIG, sleep=clock.sleep, clock=clock)
    parked = H.raw_params(base)[CROSSFADER_INDEX - 1]
    assert parked and len(windows[0].params) == 12
    assert windows[0].params[CROSSFADER_INDEX - 1] == parked


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


def test_wedged_distinguishes_a_live_device_from_one_that_holds_nothing():
    assert not H.wedged(Port(), sleep=lambda _s: None)
    assert H.wedged(Port(holds=False), sleep=lambda _s: None)


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


def test_the_canary_is_false_when_the_picture_never_comes_back(monkeypatch):
    """A passthrough that carries nothing means the rig, not the program, is dark."""
    blind(monkeypatch, canary=False)
    clock = FakeClock()
    port = Port()
    assert not H.carries_stimulus(Sess(port), Rec(clock), port, config=CONFIG, sleep=clock.sleep, clock=clock)


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
        rng=np.random.default_rng(0),
        sleep=clock.sleep,
        clock=clock,
    )
    assert len(recorder.recordings) == 1 and archive.commits == ["Alpha"]
    assert not recorder.recordings[0].exists(), "the scratch recording was published, not left behind"
    assert archive.paths("Alpha")[0].exists()
    assert archive.geometry["Alpha"] == (CONFIG.width, CONFIG.height, CONFIG.capture_fps)
    assert (result.frames, result.measured) == (PER_PROGRAM, MEASURED)
    assert result.seconds == pytest.approx(2.5) and result.report is LIVE and not result.dark
    assert len(session.vectors) == 1 + CONFIG.setpoints, "the working point, then every setpoint"


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


def test_harvest_only_probes_the_programs_it_was_given(tmp_path):
    report = run(Archive(tmp_path), programs=["Beta"])
    assert [r.program for r in report.results] == ["Beta"]


def test_harvest_stops_once_the_device_will_not_hold_a_program(tmp_path):
    """No software recovery exists, so the remaining programs must not be burnt through."""

    class Failing(Sess):
        """Session whose second load leaves the device holding nothing."""

        def load_program(self, name, *, park=True, link=None):
            super().load_program(name, park=park, link=link)
            if name == "Beta":
                self.transport.holds = False
                raise RuntimeError("load failed")

    port = Port(programs=("Alpha", "Beta", "Gamma"))
    report = run(Archive(tmp_path), port=port, session_factory=Failing)
    assert report.wedged and [r.program for r in report.results] == ["Alpha", "Beta"]
    assert report.failures[0].error.startswith("RuntimeError")
    assert port.closed


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
    assert report.blacked and not report.wedged, "a dark rig is not a wedge"
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
    """A passthrough that carries nothing means the rig, not the program, is dark."""
    blind(monkeypatch, canary=False)
    report = run(Archive(tmp_path))
    assert report.blacked is True and report.stopped is True
    assert report.wedged is False, "a black output is not a wedge, and needs its own diagnosis"
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


def test_results_report_what_the_take_actually_held():
    measured = H.ProgramResult("P", frames=10, measured=9, seconds=12.0, report=LIVE)
    assert not measured.dark and "9/10 frames measured in 12s" in str(measured)
    assert "24 distinct" in str(measured)
    assert H.ProgramResult("P", frames=10, report=DARK).dark and "UNUSABLE" in str(
        H.ProgramResult("P", frames=10, report=DARK)
    )
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


def test_a_real_session_holds_the_link_down_across_the_whole_load(tmp_path):
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
    port.close = lambda: calls.append("close")
    port.load_program = lambda name: calls.append(f"load {name}")
    report = run(
        Archive(tmp_path),
        port=port,
        clock=clock,
        link=Link(),
        session_factory=lambda t, **kw: Session(t, sleep=clock.sleep, clock=clock, **kw),
    )
    assert report.frames and not report.failures
    start = calls.index("down")
    assert calls[start : start + 3] == ["down", "load Alpha", "up"], "down for the whole load"

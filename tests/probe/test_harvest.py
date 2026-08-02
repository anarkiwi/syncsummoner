"""Harvest: whole-library native frame archiving, resume, and wedge detection."""

# pylint: disable=missing-function-docstring

import numpy as np
import pytest

from syncsummoner.device.profile import CROSSFADER_INDEX, ParamKind
from syncsummoner.device.session import Session
from syncsummoner.device.transport import ProgramInfo
from syncsummoner.probe import harvest as H
from syncsummoner.probe.store import KeyKind, ProgramKey
from tests.device.conftest import COLORBARS_INFO, FakeClock, FakeTransport

FIRMWARE = "1.0.0-rc.37"
INFO = ProgramInfo.from_json(COLORBARS_INFO)
CONFIG = H.HarvestConfig(
    width=64,
    height=48,
    setpoints=2,
    frames_per_point=2,
    burst_settle=1,
    settle_s=0.0,
    startup_s=0.0,
    health_poll_s=0.0,
    wedge_settle_s=0.0,
    loop_frames=2,
)


def frame(value, size=(48, 64)):
    """A packed native frame of one constant sample value."""
    return np.full(size + (2,), value, dtype=np.uint8)


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


class Cap:
    """Capture stub cycling a fixed native frame sequence, recording the call order."""

    def __init__(self, frames=(64, 200), calls=None):
        self.values = list(frames)
        self.reads = 0
        self.calls = [] if calls is None else calls

    def read_native(self):
        self.reads += 1
        return frame(self.values[self.reads % len(self.values)])

    def wait_for_content(self, timeout_s=15.0):
        del timeout_s
        self.calls.append("wait_for_content")
        return True

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False


class Writer:
    """Open-writer stub checking geometry and keeping every row it was handed."""

    def __init__(self, rows, shape):
        self.rows = rows
        self.shape = shape

    def write(self, native, *, params, setpoint, loop_index=None, captured=None):
        del loop_index, captured
        assert native.shape == self.shape, "the archive stores the card's own buffer"
        self.rows.append((int(native[0, 0, 0]), tuple(params), setpoint))
        return len(self.rows) - 1

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False


class Archive:
    """Frame archive stub keyed exactly as the real one, with no encoder."""

    def __init__(self, directory, stored=()):
        self.directory = directory
        self.stored = {program: key.digest for program, key in stored}
        self.rows = {}
        self.keys = {}
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

    def writer(self, program, key, *, width, height, fps=25):
        del fps
        self.keys[program] = key
        return Writer(self.rows.setdefault(program, []), (height, width, 2))


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
        self.calls = kwargs.pop("calls", None)

    def load_program(self, name, *, park=True, link=None):
        self.loads.append((name, park, link))

    def working_point(self, info):
        return {p.index: 0.5 for p in info.params if p.kind is not ParamKind.UNUSED}

    def set_params(self, values):
        self.vectors.append(dict(values))


def run(archive, *, port=None, capture=None, sessions=None, **kwargs):
    """Drive a harvest against stubs, returning the report."""
    port = Port() if port is None else port
    made = [] if sessions is None else sessions

    def factory(transport, **kw):
        made.append(Sess(transport, **kw))
        return made[-1]

    kwargs.setdefault("config", CONFIG)
    return H.harvest(
        archive,
        open_transport=lambda: port,
        open_capture=lambda: Cap() if capture is None else capture,
        session_factory=factory,
        sleep=lambda _s: None,
        clock=FakeClock(),
        **kwargs,
    )


def test_sweep_excludes_the_crossfader_but_keeps_booleans():
    """P12 only gates output and a CC offset is non-negative, so it cannot be closed."""
    vectors = H.sweep_vectors(INFO, CONFIG, np.random.default_rng(0))
    assert vectors and all(CROSSFADER_INDEX not in v for v in vectors)
    booleans = {p.index for p in INFO.params if p.kind is ParamKind.BOOLEAN}
    assert booleans and all(booleans <= set(v) for v in vectors), "booleans change the picture"


def test_sweep_of_a_program_with_no_used_parameters_is_one_empty_vector():
    info = ProgramInfo.from_json({"name": "Null", "parameters": [{"name": "-", "min": 0, "max": 100}] * 12})
    assert H.sweep_vectors(info, CONFIG, np.random.default_rng(0)) == [{}]


def test_native_burst_discards_a_settle_run():
    got = H.native_burst(Cap(), 3, settle=2, clock=FakeClock())
    assert len(got) == 3 and all(f.dtype == np.uint8 for f in got)


def test_native_burst_gives_up_at_the_deadline():
    class Dead:
        """Capture that never produces a frame."""

        def read_native(self):
            """Read native."""
            clock.now += 1.0

    clock = FakeClock()
    assert not H.native_burst(Dead(), 5, timeout_s=3.0, clock=clock)


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


def test_harvest_archives_every_setpoint_with_the_vector_that_produced_it(tmp_path):
    archive = Archive(tmp_path)
    report = run(archive)
    assert [r.program for r in report.results] == ["Alpha", "Beta"]
    assert report.frames == 8, "two programs, two setpoints, two frames each"
    for program in ("Alpha", "Beta"):
        rows = archive.rows[program]
        assert [row[2] for row in rows] == [0, 0, 1, 1]
        assert all(len(row[1]) == 12 for row in rows), "rows carry the full raw vector"
        assert len({row[1] for row in rows}) == 2, "one vector per setpoint"


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


def test_harvest_waits_for_content_before_reading_a_single_frame(tmp_path):
    """A too-short settle after a load archives pure black; the wait is what fixes it."""
    calls = []
    capture = Cap(calls=calls)
    run(Archive(tmp_path), capture=capture)
    assert calls == ["wait_for_content", "wait_for_content"]


def test_harvest_resumes_by_skipping_what_is_already_archived(tmp_path):
    key = ProgramKey("Alpha", FIRMWARE, KeyKind.NAME_FIRMWARE)
    archive = Archive(tmp_path, stored=[("Alpha", key)])
    report = run(archive)
    assert [r.cached for r in report.results] == [True, False]
    assert "Alpha" not in archive.rows and report.frames > 0


def test_harvest_reflashed_firmware_invalidates_a_stored_archive(tmp_path):
    stale = ProgramKey("Alpha", "0.9.0", KeyKind.NAME_FIRMWARE)
    archive = Archive(tmp_path, stored=[("Alpha", stale)])
    assert not run(archive).results[0].cached


def test_harvest_stops_once_the_device_will_not_hold_a_program(tmp_path):
    """No software recovery exists, so the remaining programs must not be burnt through."""

    class Failing(Sess):
        """Session whose second load leaves the device holding nothing."""

        def load_program(self, name, *, park=True, link=None):
            super().load_program(name, park=park, link=link)
            if name == "Beta":
                self.transport.holds = False
                raise RuntimeError("load failed")

    sessions = []
    port = Port(programs=("Alpha", "Beta", "Gamma"))
    report = H.harvest(
        Archive(tmp_path),
        open_transport=lambda: port,
        open_capture=Cap,
        session_factory=lambda t, **kw: sessions.append(Failing(t, **kw)) or sessions[-1],
        config=CONFIG,
        sleep=lambda _s: None,
        clock=FakeClock(),
    )
    assert report.wedged and [r.program for r in report.results] == ["Alpha", "Beta"]
    assert report.failures[0].error.startswith("RuntimeError")
    assert port.closed


def test_harvest_carries_on_past_a_program_that_failed_without_wedging(tmp_path):
    class Failing(Sess):
        """Session that cannot load one particular program."""

        def load_program(self, name, *, park=True, link=None):
            super().load_program(name, park=park, link=link)
            if name == "Alpha":
                raise RuntimeError("that one is broken")

    report = H.harvest(
        Archive(tmp_path),
        open_transport=Port,
        open_capture=Cap,
        session_factory=Failing,
        config=CONFIG,
        sleep=lambda _s: None,
        clock=FakeClock(),
    )
    assert not report.wedged and len(report.results) == 2
    assert report.failures and report.results[1].frames == 4


def test_harvest_uploads_the_stimulus_once_and_restarts_a_dead_player(tmp_path):
    player = Player(running=False)
    messages = []
    run(Archive(tmp_path), player=player, log=messages.append)
    assert player.uploaded == CONFIG.loop_frames, "the loop is invariant, so it is paid for once"
    assert player.stopped and any("Alpha" in m for m in messages)


def test_harvest_only_probes_the_programs_it_was_given(tmp_path):
    archive = Archive(tmp_path)
    report = run(archive, programs=["Beta"])
    assert [r.program for r in report.results] == ["Beta"]


def test_dark_and_cached_results_report_themselves():
    dark = H.ProgramResult("P", frames=4, seconds=1.0, luma=H.DARK_LUMA - 1)
    assert dark.dark and "DARK" in str(dark)
    assert not H.ProgramResult("P", frames=4, seconds=1.0, luma=H.DARK_LUMA).dark
    assert str(H.ProgramResult("P")) == "P: cached"
    assert "FAILED" in str(H.ProgramResult("P", error="RuntimeError: x"))
    assert "Y=" in str(H.ProgramResult("P", frames=3, seconds=1.0, luma=99.0))


def test_upload_stimulus_builds_a_loop_at_the_playout_geometry():
    player = Player()
    config = H.HarvestConfig(width=64, height=48, loop_frames=3)
    assert H.upload_stimulus(player, config) == 3


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
    archive = Archive(tmp_path)
    report = H.harvest(
        archive,
        open_transport=lambda: port,
        open_capture=Cap,
        link=Link(),
        config=CONFIG,
        sleep=clock.sleep,
        clock=clock,
        session_factory=lambda t, **kw: Session(t, sleep=clock.sleep, clock=clock, **kw),
    )
    assert report.frames == 4
    start = calls.index("down")
    assert calls[start : start + 3] == ["down", "load Alpha", "up"], "down for the whole load"


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


def test_harvest_restarts_a_player_that_died_mid_run(tmp_path):
    """The loop is the stimulus; without it every archived frame is of nothing."""

    class Dead(Player):
        """Player whose loop never stays up."""

        def is_running(self):
            return False

    player = Dead()
    run(Archive(tmp_path), player=player)
    assert player.starts == 3, "once for the run, then once per program that found it down"


class SwitchCap(Cap):
    """Capture stub whose luma the session flips, so a canary can differ from a program."""

    def __init__(self, box, calls=None):
        super().__init__(calls=calls)
        self.box = box

    def read_native(self):
        self.reads += 1
        return frame(self.box["luma"])


def dark_run(archive, *, canary_luma, programs=("Alpha", "Beta")):
    """Harvest where every program is black and the canary reports ``canary_luma``."""
    box = {"luma": 0}
    port = Port(programs=programs)
    capture = SwitchCap(box)

    class Canary(Sess):
        """Session stub that lifts the capture out of black only for the canary load."""

        def load_program(self, name, *, park=True, link=None):
            super().load_program(name, park=park, link=link)
            box["luma"] = canary_luma if name == H.WEDGE_PROBE else 0

    return H.harvest(
        archive,
        open_transport=lambda: port,
        open_capture=lambda: capture,
        config=CONFIG,
        sleep=lambda _s: None,
        clock=FakeClock(),
        session_factory=Canary,
    )


def test_a_dark_program_is_discarded_when_the_canary_still_carries_the_source(tmp_path):
    """One black program is legitimate, so the run keeps going."""
    archive = Archive(tmp_path)
    report = dark_run(archive, canary_luma=200)
    assert report.blacked is False and report.stopped is False
    assert [r.program for r in report.results] == ["Alpha", "Beta"], "every program was attempted"
    assert all("discarded" in r.error for r in report.results)
    assert report.frames == 0, "black frames are not counted as harvested"


def test_black_output_with_a_dark_canary_stops_the_run(tmp_path):
    """A passthrough that carries nothing means the rig, not the program, is dark."""
    archive = Archive(tmp_path)
    report = dark_run(archive, canary_luma=0)
    assert report.blacked is True and report.stopped is True
    assert report.wedged is False, "a black output is not a wedge, and needs its own diagnosis"
    assert [r.program for r in report.results] == ["Alpha"], "it stopped rather than archiving more"


def test_a_discarded_program_has_its_archive_files_removed(tmp_path):
    """Black frames must not stay committed, or a resume would skip re-probing them."""
    archive = Archive(tmp_path)
    for path in archive.paths("Alpha"):
        path.write_bytes(b"stale")
    dark_run(archive, canary_luma=200, programs=("Alpha",))
    assert not any(path.exists() for path in archive.paths("Alpha"))


def test_native_burst_drops_a_read_that_repeats_the_previous_frame():
    """Measured, an unpaced 30-frame burst spanned 0.25s and held one distinct frame."""

    class Stuck:
        """Capture whose reads outrun the card and return the same buffer."""

        def __init__(self):
            self.reads = 0

        def read_native(self):
            """Read native."""
            self.reads += 1
            return frame(64)

    clock = FakeClock()
    capture = Stuck()
    got = H.native_burst(capture, 5, settle=0, interval_s=0.1, sleep=clock.sleep, clock=clock)
    assert len(got) == 1 and capture.reads == 5, "it kept reading, but stored one measurement"


def test_native_burst_paces_itself_across_the_dwell():
    """Unpaced, the burst samples the few milliseconds the reads take, not the setpoint."""
    clock = FakeClock()
    got = H.native_burst(Cap(), 8, settle=0, interval_s=0.05, sleep=clock.sleep, clock=clock)
    assert len(got) == 8 and clock.now == pytest.approx(0.4), "one stored frame per interval"


def test_a_program_whose_picture_never_moves_is_archived_and_flagged(tmp_path):
    """A generator program is still by design: measured, five of six failed a motion gate."""

    class Blank(Cap):
        """Capture whose picture never moves before the sweep."""

        def wait_for_content(self, timeout_s=15.0):
            del timeout_s
            return False

    archive = Archive(tmp_path)
    report = run(archive, capture=Blank())
    assert not report.failures and report.frames == 8, "a still picture is still a measurement"
    assert [r.program for r in report.unsettled] == ["Alpha", "Beta"]
    assert "UNSETTLED" in str(report.results[0])


def test_a_settled_program_is_not_flagged(tmp_path):
    report = run(Archive(tmp_path))
    assert not report.unsettled and "UNSETTLED" not in str(report.results[0])


def test_rows_carry_the_commanded_state_not_only_the_swept_slots(tmp_path):
    """P12 gates the output and is held open off-sweep, so a zero there misreads the state."""
    archive = Archive(tmp_path)
    run(archive)
    parked = H.raw_params(Sess(None).working_point(INFO))[CROSSFADER_INDEX - 1]
    assert parked, "the working point holds the crossfader open"
    assert all(row[1][CROSSFADER_INDEX - 1] == parked for row in archive.rows["Alpha"])


def test_a_failure_on_a_dark_rig_stops_the_run(tmp_path):
    """A raising program on a faulted rig must be diagnosed, not repeated 49 times."""

    class Dark(Cap):
        """Capture that reads black, so the passthrough canary fails too."""

        def read_native(self):
            self.reads += 1
            return frame(0)

    class Failing(Sess):
        """Session that cannot load the first program."""

        def load_program(self, name, *, park=True, link=None):
            super().load_program(name, park=park, link=link)
            if name == "Alpha":
                raise RuntimeError("that one is broken")

    report = H.harvest(
        Archive(tmp_path),
        open_transport=Port,
        open_capture=Dark,
        session_factory=Failing,
        config=CONFIG,
        sleep=lambda _s: None,
        clock=FakeClock(),
    )
    assert report.blacked and not report.wedged, "a dark rig is not a wedge"
    assert [r.program for r in report.results] == ["Alpha"], "it stopped rather than trying the rest"


def test_a_failure_with_a_live_canary_carries_on(tmp_path):
    """One broken program on a healthy rig must not stop the run."""

    class Failing(Sess):
        """Session that cannot load the first program."""

        def load_program(self, name, *, park=True, link=None):
            super().load_program(name, park=park, link=link)
            if name == "Alpha":
                raise RuntimeError("that one is broken")

    report = H.harvest(
        Archive(tmp_path),
        open_transport=Port,
        open_capture=Cap,
        session_factory=Failing,
        config=CONFIG,
        sleep=lambda _s: None,
        clock=FakeClock(),
    )
    assert not report.stopped and [r.program for r in report.results] == ["Alpha", "Beta"]


def test_a_dark_program_is_not_probed_again_on_resume(tmp_path):
    """Loads are the scarce resource: a program that is itself black has no archive to resume."""
    archive = Archive(tmp_path)
    dark_run(archive, canary_luma=200, programs=("Alpha",))
    assert archive.marked == {"Alpha"}, "the verdict is recorded against the key"
    report = dark_run(archive, canary_luma=200, programs=("Alpha",))
    assert [r.cached for r in report.results] == [True] and report.frames == 0


def test_a_dark_verdict_from_other_firmware_is_not_trusted(tmp_path):
    archive = Archive(tmp_path)
    archive.marked_key["Alpha"] = "0.9.0-digest"
    report = dark_run(archive, canary_luma=200, programs=("Alpha",))
    assert not report.results[0].cached, "a reflash re-probes it"

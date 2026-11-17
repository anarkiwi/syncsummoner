"""CLI wiring: the standing rule that every program load drops the source link."""

# pylint: disable=missing-function-docstring,protected-access

from pathlib import Path
from types import SimpleNamespace

import pytest
from pyvmancer import CcMapError

from syncsummoner import cli
from syncsummoner.device import capture as capture_mod
from syncsummoner.device import link as link_mod
from syncsummoner.device import recorder as recorder_mod
from syncsummoner.device import session as session_mod
from syncsummoner.device import transport as transport_mod
from syncsummoner.probe import harvest as harvest_mod
from syncsummoner.probe import runner as runner_mod
from syncsummoner.probe import store as store_mod


class Port:
    """Transport stub carrying one program and no manifest."""

    corrupt = False

    def __init__(self, serial=None, stall=None):
        self.serial, self.stall = serial, stall
        self.closed = False

    @classmethod
    def open(cls, *, serial=None, stall=None):
        """Open."""
        return cls(serial, stall)

    def programs(self):
        """Programs."""
        return ["Colorbars"]

    def firmware(self):
        """Firmware."""
        return "1.0.0-rc.37"

    def program_manifest(self):
        """Program manifest."""
        return None

    def program_info(self, name=None):
        """Program info."""
        del name
        return type("Info", (), {"params": []})()

    def verify_cc_map(self):
        """Verify the CC map, failing once ``corrupt`` is set."""
        if type(self).corrupt:
            raise CcMapError("device CC map differs from the CCs sent")

    def close(self):
        """Close."""
        self.closed = True


class Cap:
    """Capture stub recording the geometry it was opened at."""

    def __init__(self, device=None, *, width=720, height=576, fps=50):
        self.device, self.width, self.height, self.fps = device, width, height, fps

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False


class Rec:
    """Recorder stub recording how the card was asked to be read."""

    def __init__(self, **kwargs):
        self.kwargs = kwargs


class Sess:
    """Session stub recording the link every load was given."""

    loads = []

    def __init__(self, transport, **kwargs):
        self.transport, self.kwargs = transport, kwargs

    def load_program(self, name, *, park=True, link=None):
        """Load program."""
        del park
        type(self).loads.append((name, link))

    def ensure_live(self, capture, **kwargs):
        """Ensure live."""


class Link:
    """Link stub standing in for HDMI control of the stimulus source."""

    def __init__(self, host=None):
        self.host = host


@pytest.fixture(name="rig")
def rig_fixture(monkeypatch):
    """Patch every hardware endpoint the probe path reaches for."""
    Sess.loads = []
    Port.corrupt = False
    calls = {"plans": []}
    monkeypatch.setattr(transport_mod, "Transport", Port)
    monkeypatch.setattr(capture_mod, "Capture", Cap)
    monkeypatch.setattr(session_mod, "Session", Sess)
    monkeypatch.setattr(link_mod, "Link", Link)
    monkeypatch.setattr(runner_mod, "run_plan", lambda *a, **kw: calls["plans"].append(kw) or [])
    return calls


@pytest.mark.usefixtures("rig")
def test_probe_run_drops_the_link_across_every_program_load(tmp_path, capsys):
    """Standing rule: a Videomancer program change happens with the HDMI link down."""
    assert cli.main(["probe", "run", "--plan", "oat", "--out", str(tmp_path)]) == 0
    assert len(Sess.loads) == 1
    name, link = Sess.loads[0]
    assert name == "Colorbars" and isinstance(link, Link), "load_program must be handed a Link"
    assert capsys.readouterr().out.strip() == "[]"


@pytest.mark.usefixtures("rig")
def test_probe_run_honours_an_explicit_source_host_and_opting_out(tmp_path):
    cli.main(["probe", "run", "--out", str(tmp_path), "--source-host", "pi@other"])
    assert Sess.loads[0][1].host == "pi@other"
    cli.main(["probe", "run", "--out", str(tmp_path), "--no-link"])
    assert Sess.loads[1][1] is None


@pytest.mark.usefixtures("rig")
@pytest.mark.parametrize("corrupt", [False, True])
def test_probe_run_stores_nothing_measured_under_a_changed_cc_map(tmp_path, monkeypatch, capsys, corrupt):
    stored = []
    monkeypatch.setattr(store_mod.ResultStore, "put", lambda _store, name, *_a: stored.append(name))
    monkeypatch.setattr(runner_mod, "run_plan", lambda *a, **kw: ["record"])
    Port.corrupt = corrupt
    cli.main(["probe", "run", "--out", str(tmp_path)])
    assert stored == ([] if corrupt else ["Colorbars"])
    assert ("CC map differs" in capsys.readouterr().err) is corrupt


def test_probe_run_measures_through_the_capture_and_archives_nothing(rig, tmp_path):
    """Archiving is its own subcommand now: the plan runner only measures."""
    cli.main(["probe", "run", "--out", str(tmp_path)])
    assert "archive" not in rig["plans"][0]


@pytest.fixture(name="harvested")
def harvested_fixture(monkeypatch):
    """Patch the archive run's endpoints, returning what ``harvest`` was handed."""
    seen = {}

    def fake_harvest(archive, **kwargs):
        seen.update(kwargs, archive=archive)
        return harvest_mod.HarvestReport(
            results=[
                harvest_mod.ProgramResult("Colorbars", frames=4),
                harvest_mod.ProgramResult("Roll Shutter", error="BlankTakeError: no picture"),
            ],
            seconds=60.0,
        )

    monkeypatch.setattr(transport_mod, "Transport", Port)
    monkeypatch.setattr(link_mod, "Link", Link)
    monkeypatch.setattr(recorder_mod, "Recorder", Rec)
    monkeypatch.setattr(recorder_mod, "require_ffmpeg", lambda *a, **kw: "ffmpeg")
    monkeypatch.setattr(harvest_mod, "harvest", fake_harvest)
    return seen


def test_probe_archive_drives_a_harvest_run_with_link_and_stimulus(harvested, tmp_path, capsys):
    args = ["--out", str(tmp_path), "--setpoints", "3", "--dwell", "2.5", "--capture", "/dev/video9"]
    assert cli.main(["probe", "archive"] + args) == 0
    assert isinstance(harvested["link"], Link) and harvested["player"] is not None
    assert harvested["config"].setpoints == 3 and harvested["config"].dwell_s == 2.5
    assert harvested["programs"] is None
    err = capsys.readouterr().err
    assert (
        "4 frames from 2 programs in 1.0 min, 1 failed" in err
    ), "the run summary is a stage line, on stderr"
    assert "Roll Shutter: BlankTakeError: no picture" in err, "every failed program is named with its error"


def test_probe_archive_without_footage_plays_the_codeframe_loop(harvested, tmp_path):
    from syncsummoner.device.playout import LoopPlayer

    assert cli.main(["probe", "archive", "--out", str(tmp_path)]) == 0
    assert isinstance(harvested["player"], LoopPlayer) and harvested["config"].footage is None


def test_probe_archive_with_footage_loops_the_prepared_clip_at_the_session_rate(
    harvested, tmp_path, monkeypatch, capsys
):
    from syncsummoner.device.playout import ClipPlayer

    asked = []

    def fake_prepare(source, directory, **kwargs):
        asked.append((source, directory, kwargs["start"], kwargs["seconds"], kwargs["render"].timing))
        return {"clip": "stimulus.mkv", "frames": 90, "fps": 30.0, "source": source, "sha256": "f00d" * 16}

    monkeypatch.setattr(harvest_mod, "prepare_footage", fake_prepare)
    argv = ["probe", "archive", "--out", str(tmp_path), "--footage", "film.mp4", "--footage-start", "4"]
    assert cli.main(argv + ["--footage-seconds", "3"]) == 0
    assert asked == [("film.mp4", str(tmp_path), 4.0, 3.0, "1080p30")]
    player, config = harvested["player"], harvested["config"]
    assert isinstance(player, ClipPlayer) and player.loop
    assert config.footage == str(tmp_path / "stimulus.mkv") and config.loop_fps == 30.0
    assert "90 frames (3.0s) of film.mp4, sha256 f00df00df00d" in capsys.readouterr().err


def test_the_stall_guard_is_off_unless_asked_for(harvested, tmp_path):
    """Device waits run on liveness; ``--stall`` is the opt-in guard an unattended run may set."""
    assert cli.main(["probe", "archive", "--out", str(tmp_path)]) == 0
    assert harvested["open_transport"]().stall is None
    assert cli.main(["--stall", "600", "probe", "archive", "--out", str(tmp_path)]) == 0
    assert harvested["open_transport"]().stall == 600.0


def test_probe_archive_lists_dark_programs_and_retries_them_when_asked(harvested, tmp_path, capsys):
    from syncsummoner.probe.archive import FrameArchive
    from syncsummoner.probe.store import KeyKind, ProgramKey

    FrameArchive(tmp_path).mark_dark("Lumarian", ProgramKey("Lumarian", "1", KeyKind.NAME_FIRMWARE), 0.4)
    assert cli.main(["probe", "archive", "--out", str(tmp_path)]) == 0
    assert harvested["retry_dark"] is False and "skipping dark: Lumarian" in capsys.readouterr().err
    assert cli.main(["probe", "archive", "--out", str(tmp_path), "--retry-dark"]) == 0
    assert harvested["retry_dark"] is True and "retrying dark: Lumarian" in capsys.readouterr().err


def test_probe_archive_records_the_card_losslessly(harvested, tmp_path):
    assert cli.main(["probe", "archive", "--out", str(tmp_path), "--capture", "/dev/video9"]) == 0
    kwargs = harvested["recorder"].kwargs
    assert kwargs["mode"] is recorder_mod.FFV1
    assert kwargs["device"] == "/dev/video9"
    assert (kwargs["width"], kwargs["height"], kwargs["fps"]) == (1920, 1080, 30)
    assert harvested["config"].timing == "1080p30"


def test_probe_archive_takes_raster_and_timing_from_one_format(harvested, tmp_path):
    """The transmitter is judged against the same format the card is recorded at."""
    assert cli.main(["probe", "archive", "--out", str(tmp_path), "--format", "ntsc"]) == 0
    config = harvested["config"]
    assert (config.width, config.height, config.capture_fps, config.timing) == (720, 480, 30, "ntsc")
    assert cli.main(["probe", "archive", "--out", str(tmp_path), "--format", "4k"]) == 1


def test_probe_archive_reports_a_device_gone_from_usb_as_a_failure(monkeypatch, tmp_path):
    """A wedge needs a power cycle, so the exit status has to say so."""
    monkeypatch.setattr(transport_mod, "Transport", Port)
    monkeypatch.setattr(link_mod, "Link", Link)
    monkeypatch.setattr(recorder_mod, "Recorder", Rec)
    monkeypatch.setattr(recorder_mod, "require_ffmpeg", lambda *a, **kw: "ffmpeg")
    monkeypatch.setattr(harvest_mod, "harvest", lambda *a, **kw: harvest_mod.HarvestReport(gone=True))
    assert cli.main(["probe", "archive", "--out", str(tmp_path)]) == 1


def test_probe_archive_fails_fast_with_no_ffmpeg_and_never_touches_the_rig(monkeypatch, tmp_path):
    """A missing tool must be a clear error before any rig time is spent, not a bare traceback."""

    def unreachable(*a, **kw):
        raise AssertionError("the rig must not be touched before require_ffmpeg is checked")

    monkeypatch.setattr(transport_mod, "Transport", SimpleNamespace(open=unreachable))
    monkeypatch.setattr(recorder_mod.shutil, "which", lambda name: None)
    assert cli.main(["probe", "archive", "--out", str(tmp_path)]) == 1


def test_compose_passes_density_through(monkeypatch, tmp_path):
    """`search` always took density; the composer could not say it."""
    seen = {}

    def fake_search(profiles, features, **kwargs):
        del profiles, features
        seen.update(kwargs)
        return SimpleNamespace(
            layers=[], duration=1.0, save=lambda path: Path(path).write_text("{}", encoding="utf-8")
        )

    monkeypatch.setattr("syncsummoner.compose.planner.search", fake_search)
    monkeypatch.setattr(
        "syncsummoner.compose.features.analyze", lambda *a, **k: SimpleNamespace(audio=None, video=None)
    )
    monkeypatch.setattr("syncsummoner.cli._profiles_in", lambda directory: {"p": object()})
    cli.main(
        [
            "compose",
            "clip.mkv",
            "--profiles",
            str(tmp_path),
            "--density",
            "0.9",
            "-o",
            str(tmp_path / "score.yaml"),
        ]
    )
    assert seen["density"] == 0.9


@pytest.mark.parametrize(
    "command, flags",
    [
        (
            "render",
            [
                "--played",
                "--prepared",
                "--format",
                "--scratch",
                "--source",
                "--cut-programs",
                "--takes",
                "--reuse-takes",
            ],
        ),
        ("render", ["--audio", "--fade", "--fade-in", "--fade-out", "--no-master", "--take"]),
        ("audition", ["--seconds", "--start", "--audio", "--fade", "--cut-programs"]),
        ("compose", ["--density", "--style", "--budget", "--passes", "--format"]),
        ("probe refit", ["--archive", "--jobs", "--ffmpeg"]),
        ("probe names", ["--archive", "--program"]),
        ("score show", ["--profiles"]),
        ("lockcheck", ["--take", "--scratch", "--prepared", "--judge", "--program", "--lead-in", "--route"]),
        (
            "probe archive",
            [
                "--capture",
                "--setpoints",
                "--dwell",
                "--format",
                "--no-link",
                "--footage",
                "--footage-seconds",
            ],
        ),
    ],
)
def test_every_documented_flag_is_actually_registered(command, flags, capsys):
    """A flag that silently failed to land takes an rig run to discover."""
    with pytest.raises(SystemExit):
        cli.main(command.split() + ["--help"])
    text = capsys.readouterr().out
    missing = [flag for flag in flags if flag not in text]
    assert not missing, f"{command} is missing {missing}"


def _render_stubs(monkeypatch, seen):
    """Stand in for the rig and for ffmpeg, recording what each was asked to do."""
    monkeypatch.setattr(recorder_mod, "require_ffmpeg", lambda *a, **k: "ffmpeg")
    monkeypatch.setattr(cli, "_profiles_in", lambda directory: {"g": object()})

    def fake_render(score, source, out, **kwargs):
        seen["render"] = (score, source, out, kwargs)
        Path(out).write_text("take", encoding="utf-8")
        return SimpleNamespace(usable=True, __str__=lambda self: "ok")

    def fake_master(take, audio, out, **kwargs):
        seen["master"] = (take, audio, out, kwargs)
        return kwargs.get("seconds") or 42.0

    monkeypatch.setattr("syncsummoner.compose.render.render_played", fake_render)
    monkeypatch.setattr("syncsummoner.compose.render.take_start", lambda path, **kw: 1.5)
    monkeypatch.setattr("syncsummoner.compose.master.master", fake_master)


def _score_file(tmp_path, sections=1):
    from syncsummoner.compose.score import GestureInstance, Layer, Score, Section

    span = 120.0 / sections
    score = Score(
        duration=120.0,
        sections=[Section(span * i, span * (i + 1), "A") for i in range(sections)],
        layers=[Layer("g", 0, [GestureInstance("hold", 10.0, 1.0)])],
    )
    path = tmp_path / "score.yaml"
    score.save(path)
    return str(path)


def test_render_masters_the_take_into_the_output(monkeypatch, tmp_path):
    seen = {}
    _render_stubs(monkeypatch, seen)
    out = tmp_path / "final.mp4"
    assert (
        cli.main(
            [
                "render",
                _score_file(tmp_path),
                "--source",
                "clip.mkv",
                "--audio",
                "t.flac",
                "-o",
                str(out),
                "--fade",
                "2.5",
            ]
        )
        == 0
    )
    take, audio, mastered, kwargs = seen["master"]
    assert (Path(take).name, audio, mastered) == ("final.take.mp4", "t.flac", str(out))
    assert (kwargs["fade_in"], kwargs["fade_out"], kwargs["seconds"]) == (2.5, 2.5, None)
    assert kwargs["video_start"] == 1.5, "the finished clip starts where the picture does"


def test_render_can_stop_at_the_raw_take(monkeypatch, tmp_path):
    seen = {}
    _render_stubs(monkeypatch, seen)
    take = tmp_path / "raw.mkv"
    assert (
        cli.main(
            [
                "render",
                _score_file(tmp_path),
                "--source",
                "clip.mkv",
                "--take",
                str(take),
                "--no-master",
                "-o",
                str(tmp_path / "final.mp4"),
            ]
        )
        == 0
    )
    assert seen["render"][2] == str(take)
    assert "master" not in seen


def test_audition_windows_the_score_and_masters_that_length(monkeypatch, tmp_path):
    seen = {}
    _render_stubs(monkeypatch, seen)
    assert (
        cli.main(
            [
                "audition",
                _score_file(tmp_path),
                "--source",
                "clip.mkv",
                "--audio",
                "t.flac",
                "--seconds",
                "20",
                "--start",
                "5",
                "-o",
                str(tmp_path / "aud.mp4"),
            ]
        )
        == 0
    )
    assert seen["render"][0].duration == pytest.approx(20.0)
    assert seen["master"][3]["seconds"] == pytest.approx(20.0)
    assert seen["master"][3]["audio_start"] == pytest.approx(5.0)


def test_a_blank_take_is_never_mastered(monkeypatch, tmp_path):
    seen = {}
    _render_stubs(monkeypatch, seen)
    monkeypatch.setattr(
        "syncsummoner.compose.render.render_played",
        lambda *a, **k: SimpleNamespace(usable=False, __str__=lambda self: "blank"),
    )
    assert (
        cli.main(["render", _score_file(tmp_path), "--source", "clip.mkv", "-o", str(tmp_path / "final.mp4")])
        == 1
    )
    assert "master" not in seen


def test_an_existing_take_is_remastered_without_a_pass(monkeypatch, tmp_path):
    seen = {}
    _render_stubs(monkeypatch, seen)
    take = tmp_path / "raw.mkv"
    take.write_bytes(b"take")
    argv = ["render", _score_file(tmp_path), "--source", "clip.mkv", "--take", str(take)]
    assert cli.main(argv + ["--reuse-takes", "-o", str(tmp_path / "final.mp4")]) == 0
    assert "render" not in seen and seen["master"][0] == str(take)
    assert seen["master"][3]["video_start"] == 1.5


def test_a_take_with_no_lock_record_is_not_mastered(monkeypatch, tmp_path, capsys):
    from syncsummoner.compose import lock

    seen = {}
    _render_stubs(monkeypatch, seen)
    monkeypatch.setattr("syncsummoner.compose.render.take_start", lock.take_start)
    argv = ["render", _score_file(tmp_path), "--source", "clip.mkv", "-o", str(tmp_path / "final.mp4")]
    assert cli.main(argv) == 1
    assert "master" not in seen and "no lock record" in capsys.readouterr().err


def test_a_cut_render_asks_the_device_for_programs_only_through_render_cuts(monkeypatch, tmp_path):
    """Asking up front would open the device even when every take is reused."""
    seen = {}
    _render_stubs(monkeypatch, seen)

    def fake_cuts(*_args, **kwargs):
        seen["cuts"] = kwargs
        return []

    monkeypatch.setattr("syncsummoner.compose.render.render_cuts", fake_cuts)
    monkeypatch.setattr("syncsummoner.compose.render.device_programs", lambda: pytest.fail("device opened"))
    argv = ["render", _score_file(tmp_path), "--source", "clip.mkv", "--cut-programs", "g"]
    assert cli.main(argv + ["--reuse-takes", "--no-master", "-o", str(tmp_path / "final.mp4")]) == 0
    assert seen["cuts"]["reuse"] and callable(seen["cuts"]["available"])


def test_default_fades_come_from_the_library(monkeypatch, tmp_path):
    seen = {}
    _render_stubs(monkeypatch, seen)
    from syncsummoner.compose.master import DEFAULT_FADE_S

    cli.main(
        [
            "render",
            _score_file(tmp_path),
            "--source",
            "clip.mkv",
            "--fade-out",
            "0",
            "-o",
            str(tmp_path / "final.mp4"),
        ]
    )
    assert (seen["master"][3]["fade_in"], seen["master"][3]["fade_out"]) == (DEFAULT_FADE_S, 0.0)


def test_the_package_version_is_the_one_that_was_packaged():
    """A release tag is checked against pyproject; this checks the import agrees."""
    tomllib = pytest.importorskip("tomllib")  # 3.11+; the assertion holds on every version

    import syncsummoner

    pyproject = Path(__file__).resolve().parent.parent / "pyproject.toml"
    packaged = tomllib.loads(pyproject.read_text(encoding="utf-8"))["project"]["version"]
    assert syncsummoner.__version__ == packaged


def test_score_show_prints_the_audit_and_fails_on_a_flag(monkeypatch, tmp_path, capsys):
    from syncsummoner.compose.score import Layer, Score

    path = tmp_path / "score.yaml"
    Score(duration=4.0, layers=[Layer("missing", 0, [])]).save(path)
    monkeypatch.setattr(cli, "_profiles_in", lambda directory: {})
    assert cli.main(["score", "show", str(path)]) == 1
    assert "layer 0: missing  ! no profile" in capsys.readouterr().out


def test_render_timing_flags_reach_the_config(monkeypatch, tmp_path):
    seen = {}
    _render_stubs(monkeypatch, seen)
    argv = [
        "render",
        _score_file(tmp_path),
        "--source",
        "clip.mkv",
        "--no-master",
        "-o",
        str(tmp_path / "f.mp4"),
    ]
    assert cli.main(argv + ["--format", "1080p30"]) == 0
    config = seen["render"][3]["config"]
    assert config.fps == 30.0 and config.cc_budget_hz == session_mod.CC_BUDGET_HZ
    assert cli.main(argv + ["--cc-budget", "150"]) == 0
    assert seen["render"][3]["config"].cc_budget_hz == 150.0
    with pytest.raises(SystemExit):
        cli.main(argv + ["--latency-ms", "25"])


def test_evidence_proves_each_take_and_fails_on_any_unproven(monkeypatch, tmp_path, capsys):
    import numpy as np

    from syncsummoner.compose import locktake
    from syncsummoner.device.wire import save_wire

    takes = [tmp_path / "take-Derez.mkv", tmp_path / "take-Lorenz.mkv"]
    for take in takes:
        save_wire(take, program=np.array(take.stem.split("-")[1]))
    seen = []

    def take_evidence(take, *, program, clip, config):
        seen.append((Path(take).name, program, clip, config.fps, config.strip_px))
        ok = program != "Lorenz"
        return SimpleNamespace(
            ok=ok, lines=lambda: [f"{program} proven {ok}"], to_json=lambda: f'{{"ok": {str(ok).lower()}}}'
        )

    monkeypatch.setattr(locktake, "take_evidence", take_evidence)
    argv = ["evidence", *map(str, takes), "--source", "tc.mkv", "--strip-px", "4"]
    assert cli.main(argv) == 1
    assert seen == [
        ("take-Derez.mkv", "Derez", "tc.mkv", 30.0, 4),
        ("take-Lorenz.mkv", "Lorenz", "tc.mkv", 30.0, 4),
    ]
    assert capsys.readouterr().out.splitlines() == ["Derez proven True", "Lorenz proven False"]
    assert locktake.evidence_path(takes[1]).read_text(encoding="utf-8") == '{"ok": false}'
    assert cli.main(["evidence", str(takes[0]), "--source", "tc.mkv", "--program", "Other"]) == 0
    assert seen[-1][1] == "Other"


def _check_stubs(monkeypatch, seen, *, failed=False):
    """Stand in for the take check and the splice, recording what each was handed."""
    import numpy as np

    from syncsummoner.compose import render as render_mod
    from syncsummoner.compose import takecheck

    def check(score, plan, takes, clip, **kw):
        del score
        seen["check"] = (list(plan), dict(takes), clip, kw)
        traces = {name: takecheck.Trace(np.arange(3), None, None, None, None, "strip") for name in takes}
        dead = [takecheck.DeadSpan(plan[0].program, 0.0, 1.0, takecheck.Fault.FLAT, None)] if failed else []
        covered = [render_mod.Cut(0.0, 120.0, "Covering")]
        return takecheck.CheckReport(cuts=covered, spans=dead, traces=traces)

    monkeypatch.setattr(takecheck, "check_takes", check)
    monkeypatch.setattr(
        render_mod,
        "splice",
        lambda cuts, takes, picked, out, **kw: seen.update(splice=(cuts, picked, out, kw)),
    )


def test_a_master_is_checked_on_the_identity_clock(monkeypatch, tmp_path, capsys):
    from syncsummoner.probe.footage import ClockModel

    seen = {}
    _check_stubs(monkeypatch, seen, failed=True)
    argv = [
        "check",
        _score_file(tmp_path),
        "--source",
        "clip.mkv",
        "--master",
        "m.mp4",
        "--cut-programs",
        "A,B",
    ]
    assert cli.main(argv) == 1
    plan, takes, clip, kw = seen["check"]
    assert takes == {"master": "m.mp4"} and clip == "clip.mkv" and [c.program for c in plan] == ["master"]
    assert kw["models"] == {"master": ClockModel.identity(60.0)} and "splice" not in seen
    assert cli.main(argv + ["--splice", "x.mp4"]) == 1
    assert "--splice cuts takes, not a master" in capsys.readouterr().err


def test_check_splices_the_covered_plan_or_the_plan_as_rendered(monkeypatch, tmp_path):
    seen = {}
    _check_stubs(monkeypatch, seen)
    out = str(tmp_path / "v3.mp4")
    argv = ["check", _score_file(tmp_path), "--source", "clip.mkv", "--takes", "t", "--cut-programs", "A"]
    assert cli.main(argv + ["--splice", out]) == 0
    cuts, picked, written, kw = seen["splice"]
    assert [c.program for c in cuts] == ["Covering"] and written == out and set(picked) == {"A"}
    assert (kw["config"].width, kw["config"].fps) == (1920, 60.0)
    assert seen["check"][1] == {"A": str(Path("t") / "take-A.mkv")}
    assert cli.main(argv + ["--splice", out, "--as-planned"]) == 0
    assert [c.program for c in seen["splice"][0]] == ["A"]


def test_check_cuts_by_the_locked_attribution_beside_a_take_and_warns_without_one(
    monkeypatch, tmp_path, capsys
):
    import numpy as np

    seen = {}
    _check_stubs(monkeypatch, seen)
    np.save(tmp_path / "take-A.mkv.picked.npy", np.array([4, 6, 8]))
    argv = [
        "check",
        _score_file(tmp_path, 2),
        "--source",
        "clip.mkv",
        "--takes",
        str(tmp_path),
        "--cut-programs",
    ]
    assert cli.main(argv + ["A,B"]) == 0
    attributions = seen["check"][3]["attributions"]
    assert set(attributions) == {"A"} and attributions["A"].tolist() == [4, 6, 8]
    err = capsys.readouterr().err
    assert "B: no locked attribution beside" in err and "A: no locked" not in err


def _lockcheck_stubs(monkeypatch, seen, ok=True):
    """Stand in for the timecoding, the locked play and the verdict, recording what each was handed."""
    from syncsummoner.compose import lockcheck as lockcheck_mod
    from syncsummoner.compose import render as render_mod

    report = SimpleNamespace(ok=ok, summary=lambda: "LOCKED" if ok else "NOT LOCKED")
    monkeypatch.setattr(recorder_mod, "require_ffmpeg", lambda *a, **k: "ffmpeg")
    monkeypatch.setattr(
        render_mod,
        "write_timecoded",
        lambda source, out, **kw: seen.update(timecoded=(source, out, kw)) or 90,
    )
    monkeypatch.setattr(
        lockcheck_mod, "lockcheck", lambda take, **kw: seen.update(played=(take, kw)) or report
    )
    monkeypatch.setattr(lockcheck_mod, "judge", lambda take, **kw: seen.update(judged=(take, kw)) or report)


@pytest.mark.parametrize("ok", [True, False])
def test_lockcheck_timecodes_the_footage_plays_it_and_exits_on_the_verdict(monkeypatch, tmp_path, capsys, ok):
    seen = {}
    _lockcheck_stubs(monkeypatch, seen, ok)
    take, clip = tmp_path / "t" / "take.mkv", str(tmp_path / "c.mkv")
    argv = ["lockcheck", "film.mp4", "--take", str(take), "--scratch", clip, "--format", "1080p30"]
    assert cli.main(argv + ["--seconds", "4", "--start", "2", "--lead-in", "2"]) == (0 if ok else 1)
    source, out, kw = seen["timecoded"]
    assert (source, out, kw["seconds"], kw["start"], kw["lead"]) == ("film.mp4", clip, 4.0, 2.0, 60)
    played, kw = seen["played"]
    assert (played, kw["clip"], kw["program"], kw["seconds"]) == (str(take), clip, None, 3.0)
    assert kw["config"].lead_in_s == 2.0 and take.parent.is_dir() and "judged" not in seen
    assert capsys.readouterr().out.strip() == ("LOCKED" if ok else "NOT LOCKED")


@pytest.mark.parametrize(
    "flags,clip",
    [(["--judge"], "c.mkv"), (["--judge", "--program", "Teletext"], None)],
    ids=["passthru", "program"],
)
def test_lockcheck_judges_a_recorded_take_alone(monkeypatch, tmp_path, flags, clip):
    seen = {}
    _lockcheck_stubs(monkeypatch, seen)
    argv = ["lockcheck", "--take", str(tmp_path / "take.mkv"), "--scratch", "c.mkv"]
    assert cli.main(argv + flags) == 0
    assert seen["judged"][1]["clip"] == clip and "played" not in seen and "timecoded" not in seen


def test_lockcheck_plays_a_prepared_clip_as_it_is(monkeypatch, tmp_path):
    seen = {}
    _lockcheck_stubs(monkeypatch, seen)
    argv = ["lockcheck", "--prepared", "--take", str(tmp_path / "take.mkv"), "--scratch", "c.mkv"]
    assert cli.main(argv + ["--program", "Teletext"]) == 0
    assert "timecoded" not in seen and seen["played"][1]["seconds"] is None
    assert seen["played"][1]["program"] == "Teletext"

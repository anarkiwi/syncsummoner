"""CLI wiring: the standing rule that every program load drops the source link."""

# pylint: disable=missing-function-docstring

from pathlib import Path
from types import SimpleNamespace

import pytest

from syncsummoner import cli
from syncsummoner.device import capture as capture_mod
from syncsummoner.device import link as link_mod
from syncsummoner.device import recorder as recorder_mod
from syncsummoner.device import session as session_mod
from syncsummoner.device import transport as transport_mod
from syncsummoner.probe import harvest as harvest_mod
from syncsummoner.probe import runner as runner_mod


class Port:
    """Transport stub carrying one program and no manifest."""

    def __init__(self, serial=None):
        self.serial = serial
        self.closed = False

    @classmethod
    def open(cls, *, serial=None):
        """Open."""
        return cls(serial)

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
            results=[harvest_mod.ProgramResult("Colorbars", frames=4)], seconds=60.0
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
    assert "4 frames" in capsys.readouterr().err, "the run summary is a stage line, on stderr"


def test_probe_archive_records_the_card_losslessly_with_the_hosts_own_clock(harvested, tmp_path):
    """Under ``copyts`` the stored frame times are what ``time.monotonic`` reads."""
    assert cli.main(["probe", "archive", "--out", str(tmp_path), "--capture", "/dev/video9"]) == 0
    kwargs = harvested["recorder"].kwargs
    assert kwargs["mode"] is recorder_mod.FFV1 and kwargs["copyts"] is True
    assert kwargs["device"] == "/dev/video9"
    assert (kwargs["width"], kwargs["height"], kwargs["fps"]) == (1920, 1080, 30)


def test_probe_archive_reports_a_wedged_device_as_a_failure(monkeypatch, tmp_path):
    """A wedge needs a power cycle, so the exit status has to say so."""
    monkeypatch.setattr(transport_mod, "Transport", Port)
    monkeypatch.setattr(link_mod, "Link", Link)
    monkeypatch.setattr(recorder_mod, "Recorder", Rec)
    monkeypatch.setattr(recorder_mod, "require_ffmpeg", lambda *a, **kw: "ffmpeg")
    monkeypatch.setattr(harvest_mod, "harvest", lambda *a, **kw: harvest_mod.HarvestReport(wedged=True))
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
            ],
        ),
        ("render", ["--audio", "--fade", "--fade-in", "--fade-out", "--no-master", "--take"]),
        ("audition", ["--seconds", "--start", "--audio", "--fade", "--cut-programs"]),
        ("compose", ["--density", "--style", "--budget", "--passes", "--format"]),
        ("probe refit", ["--archive", "--jobs", "--ffmpeg"]),
        ("probe archive", ["--capture", "--setpoints", "--dwell", "--width", "--no-link"]),
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
    monkeypatch.setattr("syncsummoner.compose.render.picture_start", lambda path, **kw: 1.5)
    monkeypatch.setattr("syncsummoner.compose.master.master", fake_master)


def _score_file(tmp_path):
    from syncsummoner.compose.score import GestureInstance, Layer, Score, Section

    score = Score(
        duration=120.0,
        sections=[Section(0.0, 120.0, "A")],
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
    import tomllib

    import syncsummoner

    pyproject = Path(__file__).resolve().parent.parent / "pyproject.toml"
    packaged = tomllib.loads(pyproject.read_text(encoding="utf-8"))["project"]["version"]
    assert syncsummoner.__version__ == packaged

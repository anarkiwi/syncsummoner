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
    assert "4 frames" in capsys.readouterr().out


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
        ("compose", ["--density", "--style", "--budget"]),
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

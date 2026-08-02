"""CLI wiring: the standing rule that every program load drops the source link."""

# pylint: disable=missing-function-docstring

from pathlib import Path
from types import SimpleNamespace

import pytest

from syncsummoner import cli
from syncsummoner.device import capture as capture_mod
from syncsummoner.device import link as link_mod
from syncsummoner.device import session as session_mod
from syncsummoner.device import transport as transport_mod
from syncsummoner.probe import archive as archive_mod
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
    """Capture stub recording the pixel mode it was opened in."""

    def __init__(self, device=None, *, width=720, height=576, fps=50, mode=None):
        self.device, self.width, self.height, self.fps, self.mode = device, width, height, fps, mode

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False


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


def test_probe_run_without_archive_keeps_the_default_capture_path(rig, tmp_path):
    cli.main(["probe", "run", "--out", str(tmp_path)])
    assert rig["plans"][0]["archive"] is None


def test_probe_run_archives_native_frames_when_asked(rig, tmp_path, monkeypatch):
    """``--archive`` is opt-in; it switches the capture into the card's own format."""
    opened = {}

    class Archive:
        """FrameArchive stub recording the writer it was asked for."""

        def __init__(self, directory):
            opened["directory"] = directory

        def writer(self, program, key, *, width, height):
            """Writer."""
            opened["writer"] = (program, key, width, height)
            return _Writer()

    class _Writer:
        """Open writer stub."""

        def __enter__(self):
            return self

        def __exit__(self, *exc):
            return False

    monkeypatch.setattr(archive_mod, "FrameArchive", Archive)
    cli.main(["probe", "run", "--out", str(tmp_path), "--archive", str(tmp_path / "frames")])
    assert opened["writer"][0] == "Colorbars" and opened["writer"][2:] == (720, 576)
    assert rig["plans"][0]["archive"] is not None


def test_probe_archive_drives_a_harvest_run_with_link_and_stimulus(monkeypatch, tmp_path, capsys):
    seen = {}

    def fake_harvest(archive, **kwargs):
        seen.update(kwargs, archive=archive)
        return harvest_mod.HarvestReport(results=[harvest_mod.ProgramResult("Colorbars", 4)], seconds=60.0)

    monkeypatch.setattr(transport_mod, "Transport", Port)
    monkeypatch.setattr(capture_mod, "Capture", Cap)
    monkeypatch.setattr(link_mod, "Link", Link)
    monkeypatch.setattr(harvest_mod, "harvest", fake_harvest)
    assert cli.main(["probe", "archive", "--out", str(tmp_path), "--setpoints", "3"]) == 0
    assert isinstance(seen["link"], Link) and seen["player"] is not None
    assert seen["config"].setpoints == 3 and seen["programs"] is None
    assert seen["open_capture"]().mode is capture_mod.PixelMode.YVYU
    assert "4 frames" in capsys.readouterr().out


def test_probe_archive_reports_a_wedged_device_as_a_failure(monkeypatch, tmp_path):
    """A wedge needs a power cycle, so the exit status has to say so."""
    monkeypatch.setattr(transport_mod, "Transport", Port)
    monkeypatch.setattr(capture_mod, "Capture", Cap)
    monkeypatch.setattr(link_mod, "Link", Link)
    monkeypatch.setattr(harvest_mod, "harvest", lambda *a, **kw: harvest_mod.HarvestReport(wedged=True))
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

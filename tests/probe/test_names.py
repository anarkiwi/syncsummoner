"""Names back-fill: each archive gets its program's own reply, resumably, and refit reads it."""

# pylint: disable=missing-function-docstring

import json

import pytest
from pyvmancer.errors import DeviceGoneError, ShellError

from syncsummoner.device.declared import UndeclaredProgramError
from syncsummoner.device.profile import ParamKind
from syncsummoner.probe import names as N
from syncsummoner.probe.archive import PROGRAM_INFO, ArchiveError, FrameArchive
from syncsummoner.probe.harvest import GONE_NOTE
from tests.device.conftest import COLORBARS_INFO, DECLARED
from tests.probe.test_archive import KEY, rows


def info(name):
    return {**COLORBARS_INFO, "name": name}


class Port:
    """Transport stub: switches program a set number of replies after each load."""

    def __init__(self, *, lag=0, fail=(), gone=False):
        self.lag, self.fail, self.gone = lag, set(fail), gone
        self.current, self.pending, self.loads = "Passthru", None, []

    def load_program(self, name):
        self.loads.append(name)
        if self.gone:
            raise DeviceGoneError(f"loading {name}", "shell busy 4.0s, usb gone")
        if name in self.fail:
            raise ShellError(4, "program not found", f"program load {name}")
        self.pending = (name.rpartition("/")[2], self.lag)

    def program_payload(self, name=None):
        del name
        if self.pending is not None:
            name, wait = self.pending
            self.pending = (name, wait - 1) if wait else None
            if not wait:
                self.current = name
        return info(self.current)


@pytest.fixture(name="archive")
def archive_fixture(tmp_path):
    """Three committed archives, one already annotated, written through the real archive."""
    archive = FrameArchive(tmp_path)
    for program in ("Alpha", "author/Beta", "Gamma"):
        video = archive.scratch(program)
        video.write_bytes(b"ffv1")
        archive.commit(program, KEY, video, rows(2, program=program), width=32, height=24)
    archive.annotate("Gamma", **{PROGRAM_INFO: info("Gamma")})
    return archive


def test_plain_name_drops_the_author():
    assert N.plain_name("czartroniks/Fray") == "Fray"
    assert N.plain_name("Fray") == "Fray"


def test_the_reply_after_a_completed_load_names_the_program():
    assert N.read_program_info(Port(), "author/Beta")["name"] == "Beta"


def test_a_reply_naming_another_program_is_an_error_at_once():
    """The load returns only once the device runs the program, so another name is a wrong answer."""
    port = Port(lag=1)
    with pytest.raises(LookupError, match="device reports 'Passthru' after loading 'Alpha'"):
        N.read_program_info(port, "Alpha")


def test_backfill_annotates_what_lacks_a_reply_and_skips_the_rest(archive):
    port = Port()
    report = N.backfill(archive, port)
    assert sorted(report.annotated) == ["Alpha", "author/Beta"] and report.skipped == ["Gamma"]
    assert "Gamma" not in port.loads
    meta = json.loads(archive.paths("author/Beta")[2].read_text(encoding="utf-8"))
    assert meta[PROGRAM_INFO]["name"] == "Beta" and meta["frames"] == 2, "frames metadata kept"
    again = N.backfill(archive, Port())
    assert not again.annotated and len(again.skipped) == 3, "a rerun resumes, it does not repeat"


def test_one_failed_program_is_reported_and_the_run_goes_on(archive):
    said = []
    report = N.backfill(archive, Port(fail={"Alpha"}), log=said.append)
    assert set(report.failed) == {"Alpha"} and report.annotated == ["author/Beta"]
    assert not report.gone and "Alpha" in said[0]


def test_a_device_gone_from_usb_ends_the_run(archive):
    said = []
    report = N.backfill(
        archive, Port(gone=True), programs=["Alpha", "author/Beta", "Missing"], log=said.append
    )
    assert report.gone and list(report.failed) == ["Alpha"] and not report.annotated
    assert said[-1] == GONE_NOTE


def test_archived_specs_carry_device_names_and_declared_kinds(archive):
    assert N.archived_specs(archive.meta("Alpha"), "Alpha", DECLARED) is None
    specs = N.archived_specs(archive.meta("Gamma"), "Colorbars", DECLARED)
    assert [s.name for s in specs] == [p["name"] for p in COLORBARS_INFO["parameters"]]
    assert specs[10].kind is ParamKind.BOOLEAN and specs[0].kind is ParamKind.UNUSED
    assert specs[11].kind is ParamKind.CONTINUOUS and specs[11].native_max == 100
    with pytest.raises(UndeclaredProgramError):
        N.archived_specs(archive.meta("Gamma"), "Gamma", DECLARED)


def test_a_commit_can_carry_the_reply_itself(tmp_path):
    archive = FrameArchive(tmp_path)
    video = archive.scratch("Alpha")
    video.write_bytes(b"ffv1")
    archive.commit("Alpha", KEY, video, rows(2), width=32, height=24, program_info=info("Alpha"))
    assert archive.meta("Alpha")[PROGRAM_INFO] == info("Alpha")


def test_annotating_an_uncommitted_archive_is_refused(tmp_path):
    with pytest.raises(ArchiveError, match="no committed archive"):
        FrameArchive(tmp_path).annotate("Nothing", note=1)

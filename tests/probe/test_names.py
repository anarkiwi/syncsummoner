"""Names back-fill: each archive gets its program's own reply, resumably, and refit reads it."""

# pylint: disable=missing-function-docstring

import json

import pytest

from syncsummoner.device.profile import ParamKind, Role
from syncsummoner.probe import names as N
from syncsummoner.probe.archive import PROGRAM_INFO, ArchiveError, FrameArchive
from tests.device.conftest import COLORBARS_INFO, FakeClock
from tests.probe.test_archive import KEY, rows


def info(name):
    return {**COLORBARS_INFO, "name": name}


class Port:
    """Transport stub: switches program a set number of replies after each load."""

    def __init__(self, *, lag=0, fail=(), holds=True):
        self.lag, self.fail, self.holds = lag, set(fail), holds
        self.current, self.pending, self.loads = "Passthru", None, []

    def load_program(self, name):
        self.loads.append(name)
        if name in self.fail or not self.holds:
            raise RuntimeError(f"no response loading {name}")
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


def test_a_stale_reply_is_asked_again_until_it_names_the_program():
    clock = FakeClock()
    port = Port(lag=2)
    payload = N.read_program_info(port, "author/Beta", sleep=clock.sleep, clock=clock)
    assert payload["name"] == "Beta"
    assert clock.now == pytest.approx(1.0), "two stale replies, half a second apart"


def test_a_reply_that_never_switches_is_an_error():
    clock = FakeClock()
    with pytest.raises(LookupError, match="still reports 'Passthru'"):
        N.read_program_info(Port(lag=99), "Alpha", timeout_s=2.0, sleep=clock.sleep, clock=clock)


def test_backfill_annotates_what_lacks_a_reply_and_skips_the_rest(archive):
    port = Port()
    report = N.backfill(archive, port, sleep=lambda _s: None)
    assert sorted(report.annotated) == ["Alpha", "author/Beta"] and report.skipped == ["Gamma"]
    assert "Gamma" not in port.loads
    meta = json.loads(archive.paths("author/Beta")[2].read_text(encoding="utf-8"))
    assert meta[PROGRAM_INFO]["name"] == "Beta" and meta["frames"] == 2, "frames metadata kept"
    again = N.backfill(archive, Port())
    assert not again.annotated and len(again.skipped) == 3, "a rerun resumes, it does not repeat"


def test_one_failed_program_is_reported_and_the_run_goes_on(archive, monkeypatch):
    monkeypatch.setattr(N, "wedged", lambda transport: False)
    said = []
    report = N.backfill(archive, Port(fail={"Alpha"}), log=said.append)
    assert set(report.failed) == {"Alpha"} and report.annotated == ["author/Beta"]
    assert not report.wedged and "Alpha" in said[0]


def test_a_device_that_holds_nothing_ends_the_run(archive):
    report = N.backfill(archive, Port(holds=False), programs=["Alpha", "author/Beta", "Missing"])
    assert report.wedged and list(report.failed) == ["Alpha"] and not report.annotated


def test_archived_specs_carry_names_ranges_and_roles(archive):
    assert N.archived_specs(archive.meta("Alpha")) is None
    specs = N.archived_specs(archive.meta("Gamma"))
    assert [s.name for s in specs] == [p["name"] for p in COLORBARS_INFO["parameters"]]
    assert specs[10].role is Role.BYPASS and specs[11].role is Role.WET
    assert specs[11].kind is ParamKind.CONTINUOUS and specs[11].native_max == 100


def test_a_commit_can_carry_the_reply_itself(tmp_path):
    archive = FrameArchive(tmp_path)
    video = archive.scratch("Alpha")
    video.write_bytes(b"ffv1")
    archive.commit("Alpha", KEY, video, rows(2), width=32, height=24, program_info=info("Alpha"))
    assert archive.meta("Alpha")[PROGRAM_INFO] == info("Alpha")


def test_annotating_an_uncommitted_archive_is_refused(tmp_path):
    with pytest.raises(ArchiveError, match="no committed archive"):
        FrameArchive(tmp_path).annotate("Nothing", note=1)

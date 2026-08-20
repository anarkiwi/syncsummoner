"""Stage logging and progress: a long run must say what it is doing, or degrade quietly."""

# pylint: disable=missing-function-docstring

import io
import logging

import pytest

from syncsummoner import progress


@pytest.fixture(name="logs")
def _logs():
    stream = io.StringIO()
    progress.configure(0, stream=stream)
    yield stream
    logging.getLogger("syncsummoner").handlers.clear()


def test_a_stage_reports_its_outcome_and_elapsed_time(logs):
    with progress.stage("pass", program="Derez") as done:
        done["writes"] = 12
    text = logs.getvalue()
    assert "pass (program=Derez)" in text
    assert "pass done in" in text and "writes=12" in text


def test_quiet_silences_progress(logs):
    progress.configure(-1, stream=logs)
    with progress.stage("pass"):
        pass
    assert logs.getvalue() == ""


@pytest.mark.parametrize(
    "seconds, expected", [(9.4, "9s"), (89.0, "89s"), (150.0, "2.5 min"), (7200.0, "2.0 h")]
)
def test_durations_read_as_the_unit_that_suits_them(seconds, expected):
    assert progress.human(seconds) == expected


def test_tracking_yields_every_item_with_no_terminal():
    assert list(progress.track(range(4), desc="passes", total=4)) == [0, 1, 2, 3]


def test_tracking_draws_a_bar_on_a_terminal(monkeypatch):
    drawn = {}

    def fake_tqdm(iterable, **kwargs):
        drawn.update(kwargs)
        return iterable

    monkeypatch.setattr(progress.sys.stderr, "isatty", lambda: True, raising=False)
    monkeypatch.setattr(progress, "tqdm", fake_tqdm)
    progress.configure(0)
    assert list(progress.track(range(3), desc="passes", total=3, unit="program")) == [0, 1, 2]
    assert (drawn["desc"], drawn["total"], drawn["unit"]) == ("passes", 3, "program")
    logging.getLogger("syncsummoner").handlers.clear()


def test_tracking_degrades_without_tqdm(monkeypatch):
    monkeypatch.setattr(progress.sys.stderr, "isatty", lambda: True, raising=False)
    monkeypatch.setattr(progress, "tqdm", None)
    assert list(progress.track(iter([1, 2]), desc="passes")) == [1, 2]


def test_a_failed_stage_says_so_rather_than_reporting_done(logs):
    with pytest.raises(RuntimeError, match="denied"):
        with progress.stage("upload"):
            raise RuntimeError("pi@videopi: denied")
    text = logs.getvalue()
    assert "upload failed after" in text and "denied" in text
    assert "upload done" not in text


def test_a_meter_logs_a_heartbeat_when_there_is_no_bar_to_draw(logs):
    with progress.meter(300.0, desc="pass") as elapsed:
        for position in (5.0, 29.0, 31.0, 62.0):
            elapsed.to(position)
    text = logs.getvalue()
    assert "pass 31/300s" in text and "pass 62/300s" in text
    assert "pass 5/300s" not in text, "a heartbeat is periodic, not per update"


def test_a_meter_draws_a_bar_on_a_terminal(monkeypatch):
    drawn = {}

    class Bar:
        """tqdm stand-in recording where it was moved to."""

        n = 0.0

        def refresh(self):
            """Refresh."""
            drawn["n"] = self.n

        def close(self):
            """Close."""
            drawn["closed"] = True

    monkeypatch.setattr(progress.sys.stderr, "isatty", lambda: True, raising=False)
    monkeypatch.setattr(progress, "tqdm", lambda **kwargs: drawn.setdefault("kw", kwargs) and Bar() or Bar())
    progress.configure(0)
    with progress.meter(10.0, desc="pass") as elapsed:
        elapsed.to(4.0)
        elapsed.to(99.0)
    assert drawn["n"] == 10.0, "a meter cannot run past its total"
    assert drawn["closed"] and drawn["kw"]["desc"] == "pass"
    logging.getLogger("syncsummoner").handlers.clear()

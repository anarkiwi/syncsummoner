"""Journal: a durable trail of what preceded a device fault."""

# pylint: disable=missing-function-docstring

import json
import types

import numpy as np

from syncsummoner.device import journal as jn


class Ticker:
    """Monotonic fake clock."""

    def __init__(self):
        self.now = 1000.0

    def __call__(self):
        self.now += 1.0
        return self.now


def test_events_are_flushed_to_disk_as_they_happen(tmp_path):
    """A killed sweep must still leave the trail behind."""
    path = tmp_path / "sub" / "journal.jsonl"
    log = jn.Journal(path, clock=Ticker())
    log.record("call", verb="load_program", args=["Isotherm"])
    log.record("health", ok=True, mean=0.4)
    lines = [json.loads(x) for x in path.read_text(encoding="utf-8").splitlines()]
    assert [x["kind"] for x in lines] == ["call", "health"]
    assert lines[0]["verb"] == "load_program"


def test_since_last_good_narrows_the_suspects():
    log = jn.Journal(clock=Ticker())
    log.record("call", verb="a")
    log.record("health", ok=True)
    log.record("call", verb="b")
    log.record("call", verb="c")
    assert [e.get("verb") for e in log.since_last_good()] == ["b", "c"]


def test_everything_is_suspect_before_the_first_healthy_probe():
    log = jn.Journal(clock=Ticker())
    log.record("call", verb="a")
    log.record("health", ok=False)
    assert len(log.since_last_good()) == 2
    assert "never observed healthy" in log.report()


def test_window_bounds_memory_but_not_the_file(tmp_path):
    path = tmp_path / "j.jsonl"
    log = jn.Journal(path, window=3, clock=Ticker())
    for i in range(10):
        log.record("call", verb=str(i))
    assert len(log.events) == 3
    assert len(path.read_text(encoding="utf-8").splitlines()) == 10


def test_report_names_the_actions_and_their_age():
    log = jn.Journal(clock=Ticker())
    log.record("health", ok=True)
    log.record("call", verb="set_video_timing", args=["720p60"])
    text = log.report()
    assert "set_video_timing" in text and "720p60" in text and "last healthy" in text


def test_health_is_decided_by_frames_not_by_the_flags():
    """hdmi.connected reads false on this rig even when the link is fine."""
    status = types.SimpleNamespace(timing="1080p30", input_source="hdmi", locked=True, source_locked=True)
    transport = types.SimpleNamespace(video_status=lambda: status, current_program=lambda: "Passthru")
    rng = np.random.default_rng(0)
    live = types.SimpleNamespace(
        frames=lambda n, **_kw: [rng.random((8, 8, 3)).astype(np.float32) for _ in range(n)],
        chroma_fraction=lambda _f: 0.8,
    )
    assert jn.probe_health(transport, live)["ok"] is True

    dark = types.SimpleNamespace(
        frames=lambda n, **_kw: [np.zeros((8, 8, 3), np.float32) for _ in range(n)],
        chroma_fraction=lambda _f: 0.0,
    )
    result = jn.probe_health(transport, dark)
    assert result["ok"] is False, "black frames are a fault however healthy the flags look"
    assert result["source_locked"] is True


def test_health_survives_a_dead_serial_link():
    def boom():
        raise RuntimeError("no response")

    transport = types.SimpleNamespace(video_status=boom, current_program=boom)
    result = jn.probe_health(transport)
    assert result["ok"] is False and "status_error" in result


def test_watched_records_the_verbs_it_is_given():
    log = jn.Journal(clock=Ticker())
    calls = []
    port = types.SimpleNamespace(load_program=calls.append, firmware=lambda: "rc.40", tag="x")
    seen = jn.watched(port, log, ["load_program"])
    seen.load_program("Isotherm")
    seen.firmware()
    assert seen.tag == "x"
    assert calls == ["Isotherm"]
    assert [e["verb"] for e in log.events] == ["load_program"]

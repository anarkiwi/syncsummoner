"""The lock acceptance check: cadence, drops, strips and tears judged capture by capture."""

# pylint: disable=missing-function-docstring

import subprocess
import types

import numpy as np
import pytest

from syncsummoner.compose import lock as L
from syncsummoner.compose import lockcheck as LC
from syncsummoner.compose import render as R
from syncsummoner.compose.session import NoLoadTimeError
from syncsummoner.device import loadtimes
from syncsummoner.compose.timecode import ClipLayout
from syncsummoner.device import recorder as recorder_mod
from syncsummoner.device.v4l2rec import sidecar
from syncsummoner.probe import decoded
from tests.compose.locked import CLEAN, STANDARD, take

CONFIG = R.RenderConfig(width=160, height=24, fps=30.0, strip_px=8, bits=16)


def stack(n, *, seed=0, static=False):
    rng = np.random.default_rng(seed)
    frames = rng.uniform(20, 230, (1 if static else n, 12, 16))
    return np.repeat(frames, n, axis=0) if static else frames


def test_a_capture_showing_the_next_frame_in_its_top_band_is_torn():
    source = stack(6)
    captured = 0.8 * source + 10
    captured[2, :4] = 0.8 * source[3, :4] + 10
    assert LC.tears(captured, source) == (8, 1)
    assert LC.tears(0.8 * source + 10, source) == (8, 0)


def test_frames_that_do_not_change_are_not_judged():
    source = stack(6, static=True)
    assert LC.tears(source.copy(), source) == (0, 0)


def recorded(tmp_path, monkeypatch, played, *, luma=None):
    path = tmp_path / "take.mkv"
    L.save_lock(path, played.flips, **played.kw)
    sidecar(path).write_bytes(played.frames.tobytes())
    luma = luma or {}

    def features(p, spec, **kw):
        del spec, kw
        small = luma.get(str(p))
        arrays = {"codes": played.codes}
        if small is not None:
            arrays["small"] = np.stack([small] * 3, axis=1)
        return decoded.Features(arrays, "")

    monkeypatch.setattr(LC.decoded, "features", features)
    return path


@pytest.mark.parametrize("strips,read", [((0,), 0), ((0, 1), 12)])
def test_a_clean_take_is_locked(tmp_path, monkeypatch, strips, read):
    path = recorded(tmp_path, monkeypatch, take(CLEAN, strips=strips))
    report = LC.judge(path, config=CONFIG)
    assert report.ok and (report.frames, report.attributed, report.read, report.agreed) == (6, 6, read, read)
    assert (report.drops, report.gaps, report.off_cadence, report.judged, report.torn) == (0, 0, 0, 0, 0)
    assert report.flips == {"flips": 30, "segments": 1, "missed": 0, "out_of_cadence": 0}
    assert report.summary().startswith("LOCKED: 6/6 clip frames attributed; Pi 30 flips")


def test_a_lost_frame_in_the_clip_is_not_locked(tmp_path, monkeypatch):
    path = recorded(tmp_path, monkeypatch, take(STANDARD))
    report = LC.judge(path, config=CONFIG)
    assert not report.ok and (report.drops, report.gaps, report.off_cadence) == (1, 1, 0)
    assert (
        report.summary().startswith("NOT LOCKED: 6/6")
        and "card 1 lost frames, 1 gaps and 0" in report.summary()
    )


def test_a_picture_settling_off_a_frame_start_is_not_locked(tmp_path, monkeypatch):
    played = take(CLEAN)
    k = int(np.flatnonzero(played.shown() == 2)[0])
    assert played.shown()[k + 1] == 2
    played.frames["digest"][k + 1] = 999
    report = LC.judge(recorded(tmp_path, monkeypatch, played), config=CONFIG)
    assert not report.ok and (report.drops, report.gaps, report.off_cadence) == (0, 0, 1)
    assert report.attributed == 6 and report.line.startswith("6/6 clip frames attributed")


@pytest.mark.parametrize("torn", [None, 3])
def test_a_passthru_take_is_judged_for_tears_on_the_captures_attributed(tmp_path, monkeypatch, torn):
    played = take(CLEAN)
    clip = stack(played.lead + played.count + 1, seed=1)
    captures = clip[played.frame].copy()
    if torn is not None:
        captures[played.shown() == torn, :4] = clip[played.lead + torn + 1, :4]
    path = recorded(
        tmp_path, monkeypatch, played, luma={"clip.mkv": clip, str(tmp_path / "take.mkv"): captures}
    )
    report = LC.judge(path, config=CONFIG, clip="clip.mkv")
    assert (report.judged, report.torn, report.ok) == (8, 0 if torn is None else 1, torn is None)


def encode(path, n, fps=30):
    frames = (stack(n, seed=2)[..., None].repeat(3, axis=3)).astype(np.uint8)
    argv = ["ffmpeg", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", "16x12"]
    argv += ["-r", str(fps), "-i", "pipe:0", "-c:v", "ffv1", str(path)]
    subprocess.run(argv, input=frames.tobytes(), check=True)


def test_a_clip_is_measured_between_its_lead_in_and_lead_out(monkeypatch):
    monkeypatch.setattr(LC.timecode, "clip_layout", lambda clip: ClipLayout(lead=3, frames=6, tail=3))
    assert LC.clip_seconds("clip.mkv", R.RenderConfig(fps=30.0)) == pytest.approx(6 / 30)


def test_a_lockcheck_clip_holds_a_load_in_each_end_or_what_it_was_asked():
    with pytest.raises(NoLoadTimeError):
        LC.check_frames(30.0)
    loadtimes.record("Passthru", 0, 2_000_000_000)
    lead, tail = LC.check_frames(30.0)
    assert lead > 2 * 60 and tail > 60, "each end holds the longest load timed and its anchor"
    assert LC.check_frames(30.0, 0.5) == (15, 15)


@pytest.mark.parametrize("program", [None, "Teletext"])
def test_lockcheck_plays_the_clip_and_judges_tears_only_under_passthru(monkeypatch, program):
    seen = {}
    report = types.SimpleNamespace(summary=lambda: "verdict")
    monkeypatch.setattr(
        R, "play_take", lambda rig, recorder, take_, **kw: seen.update(kw, take=take_, layout=kw["clip"])
    )
    monkeypatch.setattr(LC, "judge", lambda take_, *, config, clip: seen.update(clip=clip) or report)
    monkeypatch.setattr(LC.timecode, "clip_layout", lambda clip: ClipLayout(lead=8, frames=60, tail=8))
    rig = types.SimpleNamespace(
        playout=types.SimpleNamespace(upload=lambda path: seen.update(upload=path) or 5),
        transport=types.SimpleNamespace(close=lambda: pytest.fail("a rig handed in is not closed")),
    )
    got = LC.lockcheck(
        "t.mkv", clip="c.mkv", config=CONFIG, program=program, rig=rig, recorder="rec", seconds=2
    )
    assert got is report and (seen["upload"], seen["take"], seen["program"], seen["seconds"]) == (
        "c.mkv",
        "t.mkv",
        program,
        2,
    )
    assert seen["clip"] == (None if program else "c.mkv")
    assert seen["layout"] == ClipLayout(lead=8, frames=60, tail=8), "the take plays the clip as it lies"


def test_lockcheck_opens_and_closes_its_own_rig(monkeypatch):
    seen = []
    rig = types.SimpleNamespace(
        playout=types.SimpleNamespace(upload=lambda path: 1),
        transport=types.SimpleNamespace(close=lambda: seen.append("closed")),
    )
    monkeypatch.setattr(R, "open_rig", lambda config, capture: seen.append(capture) or rig)
    monkeypatch.setattr(recorder_mod, "Recorder", lambda **kw: seen.append(kw["mode"]) or "rec")
    monkeypatch.setattr(LC, "clip_seconds", lambda clip, config: 3.0)
    monkeypatch.setattr(LC.timecode, "clip_layout", lambda clip: ClipLayout(lead=8, frames=90, tail=8))

    def overran(*args, **kwargs):
        del args
        seen.append(kwargs["seconds"])
        raise R.LeadInError("overran")

    monkeypatch.setattr(R, "play_take", overran)
    with pytest.raises(R.LeadInError):
        LC.lockcheck("t.mkv", clip="c.mkv", config=CONFIG)
    assert seen == [False, recorder_mod.MJPEG, 3.0, "closed"]

"""``probe effects``: archived settings' intensity and safety, paired with the clip frames they show."""

# pylint: disable=missing-function-docstring

import json
import shutil
import subprocess
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import pytest

from syncsummoner import cli
from syncsummoner.device.profile import ProgramProfile, Source
from syncsummoner.probe import effects as fx
from syncsummoner.probe.archive import GAP, FrameRow
from syncsummoner.probe.fit import load_profile, save_profile

H, W = 192, 200


def picture(n, seed=0):
    rng = np.random.default_rng(seed)
    return (rng.random((n, H, W, 3)) * 255).astype(np.uint8)


class Reader:
    """An archive reader stub: its rows and the size, rate and stimulus they were recorded at."""

    def __init__(self, rows, count):
        self.rows, self.count = rows, count
        self.width, self.height, self.fps, self.video = W, H, 30.0, Path("unused.mkv")
        self.meta = {"stimulus": {"kind": "footage", "clip": "clip.mkv", "strip_px": 0}}


class Archive:
    """An archive stub holding one reader."""

    def __init__(self, reader, directory="."):
        self._reader, self.directory = reader, directory

    def reader(self, _program):
        return self._reader


def rows_for(setpoints):
    """Captures two a clip frame: the first of each pair attributed, the second a gap."""
    out, frame, shown = [], 0, 0
    for setpoint, count in setpoints:
        for _ in range(count):
            out.append(FrameRow(frame, "p", (setpoint,) * 12, setpoint, shown=shown))
            out.append(FrameRow(frame + 1, "p", (setpoint,) * 12, GAP, shown=shown))
            frame, shown = frame + 2, shown + 1
    return out


def test_the_ssim_scale_follows_the_reference_downsampling_rule():
    assert fx.factor(1920, 1072) == 4 and fx.factor(320, 100) == 1
    assert fx.scaled(1920, 1080, 8) == (480, 268)
    assert fx.scaled(201, 99, 0) == (200, 98)


def test_clip_kind_reads_the_stimulus_saturation():
    record = {"stimulus": {"coverage": {"saturation": [0.0, 0.0]}}}
    assert fx.clip_kind(record) == (fx.BW, [0.0, 0.0])
    record["stimulus"]["coverage"]["saturation"] = [0.54, 0.54]
    assert fx.clip_kind(record)[0] == fx.COLOUR


def test_each_setpoint_is_measured_against_the_clip_frames_its_captures_show():
    clip = picture(12)
    rows = rows_for([(0, 6), (1, 6)])
    captures = np.zeros((len(rows), H, W, 3), np.uint8)
    for row in rows:
        captures[row.frame] = clip[row.shown] if row.setpoint in (0, GAP) else 128
    thumbs = {}
    found = fx.program_effects(
        Archive(Reader(rows, len(rows))), "p", clip, stream=iter(captures), thumbs=thumbs
    )
    assert [f["setpoint"] for f in found] == [0, 1] and found[0]["params"] == [0] * 12
    assert found[0]["intensity"] == pytest.approx(0.0, abs=1e-6) and found[0]["safe"]
    assert found[1]["blank"] and found[1]["static"] and not found[1]["safe"]
    assert thumbs[0].shape == (H // 4, 4 * (W // 4), 3)


def test_a_setpoint_is_measured_on_its_longest_consecutive_run():
    clip = picture(10)
    rows = [FrameRow(i, "p", (0,) * 12, 0, shown=s) for i, s in enumerate([0, 1, 4, 5, 6, 7])]
    stream = iter([clip[r.shown] for r in rows])
    found = fx.program_effects(Archive(Reader(rows, len(rows))), "p", clip, stream=stream)
    assert [f["frames"] for f in found] == [4]


def test_captures_missing_from_the_video_are_warned_about(caplog):
    clip = picture(4)
    rows = rows_for([(0, 2), (1, 2)])
    stream = iter(np.repeat(clip[:1], 3, axis=0))
    with caplog.at_level("WARNING", logger="syncsummoner"):
        found = fx.program_effects(Archive(Reader(rows, 8)), "p", clip, stream=stream)
    assert len(found) == 1 and "past the end" in caplog.text


def test_a_program_with_no_archive_is_an_error():
    archive = Archive(None, directory="/nowhere")
    with pytest.raises(FileNotFoundError, match="no committed archive"):
        fx.program_effects(archive, "p", picture(1))


def setting(setpoint, params, intensity, **flags):
    names = ("blank", "static", "flashing", "red_flashing")
    found = {name: bool(flags.get(name, False)) for name in names}
    return {
        "setpoint": setpoint,
        "params": params,
        "intensity": intensity,
        **found,
        "safe": not any(found.values()),
    }


def test_a_summary_counts_flags_and_spans_safe_intensity():
    settings = [setting(0, [0], 0.1), setting(1, [1], 0.9, flashing=True), setting(2, [2], 0.5, blank=True)]
    said = fx.summarise(settings)
    assert said["settings"] == 3 and said["safe"] == 1 and said["flashing"] == 1 and said["blank"] == 1
    assert said["intensity"] == [0.1, 0.9] and said["safe_intensity"] == [0.1, 0.1]
    assert fx.summarise([setting(0, [0], 0.2, static=True)])["safe_intensity"] is None


def test_a_lookup_takes_the_nearest_intensity_and_is_unsafe_beside_an_unsafe_setting():
    settings = [setting(k, [k * 100, 7], k / 10, flashing=k == 5) for k in range(10)]
    far = fx.lookup(settings, [110, 7])
    assert far["nearest"] == 1 and far["intensity"] == pytest.approx(0.1) and far["safe"]
    beside = fx.lookup(settings, [440, 7])
    assert beside["nearest"] == 4 and not beside["safe"]
    assert fx.lookup(settings[:1], [0, 7])["safe"]


def test_a_sheet_orders_settings_by_intensity(tmp_path):
    import cv2  # pylint: disable=import-outside-toplevel

    thumbs = {k: np.full((10, 40, 3), 40 * k, np.uint8) for k in range(5)}
    settings = [setting(k, [k], 1 - k / 10, static=k == 2) for k in range(5)]
    fx.sheet(settings, thumbs, tmp_path / "s.png", columns=2)
    drawn = cv2.imread(str(tmp_path / "s.png"))  # pylint: disable=no-member
    assert drawn.shape == (3 * 31, 2 * 46, 3)
    assert drawn[25, 10].tolist() == [160] * 3, "the least intense setting comes first"


needs_ffmpeg = pytest.mark.skipif(shutil.which("ffmpeg") is None, reason="needs ffmpeg")


def make_clip(path, frames=6):
    argv = ["ffmpeg", "-loglevel", "error", "-f", "lavfi", "-i", f"testsrc=size={W}x{H}:rate=30"]
    subprocess.run([*argv, "-frames:v", str(frames), "-c:v", "ffv1", str(path)], check=True)


@needs_ffmpeg
def test_a_clip_is_decoded_once_cropped_and_scaled_then_memory_mapped(tmp_path):
    clip = tmp_path / "clip.mkv"
    make_clip(clip)
    frames = fx.clip_frames(clip, width=W, height=H, strip_px=8)
    assert frames.shape == (6, 184, 200, 3) and isinstance(frames, np.memmap)
    again = fx.clip_frames(clip, width=W, height=H, strip_px=8)
    assert np.array_equal(frames, again)
    with pytest.raises(RuntimeError, match="failed"):
        list(fx.decode(tmp_path / "missing.mkv", width=W, height=H, strip_px=0))


@needs_ffmpeg
def test_an_exchanged_recording_is_decoded_put_right(tmp_path):
    clip, exchanged = tmp_path / "clip.mkv", tmp_path / "exchanged.mkv"
    make_clip(clip)
    argv = ["ffmpeg", "-loglevel", "error", "-i", str(clip), "-vf", "swapuv", "-c:v", "ffv1", str(exchanged)]
    subprocess.run(argv, check=True)
    played = np.stack(list(fx.decode(clip, width=W, height=H, strip_px=0)))
    back = np.stack(list(fx.decode(exchanged, width=W, height=H, strip_px=0)))
    righted = np.stack(list(fx.decode(exchanged, width=W, height=H, strip_px=0, swap=True)))
    error = [np.abs(x.astype(int) - played).mean() for x in (back, righted)]
    assert error[1] < error[0] / 10


def test_effects_read_an_archive_whose_chroma_came_back_exchanged_put_right(monkeypatch):
    clip = picture(2)
    rows = rows_for([(0, 2)])
    reader = Reader(rows, len(rows))
    reader.meta["chroma_exchanged"] = True
    asked = {}

    def stream(*_a, **kw):
        asked.update(kw)
        return iter(clip[[r.shown for r in rows]])

    monkeypatch.setattr(fx, "decode", stream)
    assert len(fx.program_effects(Archive(reader), "p", clip)) == 1 and asked["swap"] is True


def committed(tmp_path, monkeypatch, saturation):
    """An archive directory whose one program measures as canned, and its profile."""
    (tmp_path / "clip.mkv.stimulus.json").write_text(
        json.dumps({"stimulus": {"coverage": {"saturation": saturation}}}), encoding="utf-8"
    )
    meta = {"width": W, "height": H, "stimulus": {"clip": "clip.mkv", "strip_px": 8, "source": "/x"}}

    class Committed:
        """An archive with the one program committed."""

        def __init__(self, directory, **_kw):
            self.directory = directory

        def committed(self):
            return {"P": meta}

    monkeypatch.setattr(fx, "FrameArchive", Committed)
    np.save(tmp_path / "m.npy", np.zeros((1, 2, 2, 3), np.uint8))
    monkeypatch.setattr(fx, "clip_frames", lambda *a, **k: np.load(tmp_path / "m.npy", mmap_mode="r"))

    def canned(_archive, _program, _clip, *, ffmpeg="ffmpeg", thumbs=None):
        del ffmpeg
        if thumbs is not None:
            thumbs[0] = np.zeros((4, 16, 3), np.uint8)
        return [setting(0, [0] * 12, 0.3)]

    monkeypatch.setattr(fx, "program_effects", canned)
    profiles = tmp_path / "profiles"
    profiles.mkdir()
    save_profile(ProgramProfile("P", "fw", "a", Source.HW), profiles / "P.yaml")
    return profiles


def test_measure_runs_each_program_and_draws_its_sheet(tmp_path, monkeypatch):
    committed(tmp_path, monkeypatch, [0.0, 0.0])
    found = fx.measure(tmp_path, sheets=tmp_path / "sheets", executor=ThreadPoolExecutor)
    assert found["P"][1][0]["intensity"] == 0.3 and (tmp_path / "sheets" / "P.png").is_file()


def test_the_command_writes_effects_into_the_profile_by_clip_colour(tmp_path, monkeypatch, capsys):
    profiles = committed(tmp_path, monkeypatch, [0.0, 0.0])
    real = fx.measure
    monkeypatch.setattr(fx, "measure", lambda *a, **k: real(*a, **k, executor=ThreadPoolExecutor))
    argv = ["probe", "effects", "--archive", str(tmp_path), "--profiles", str(profiles)]
    assert cli.main(argv) == 0
    said = json.loads(capsys.readouterr().out)
    assert said["kind"] == fx.BW and said["summary"]["safe"] == 1
    (tmp_path / "clip.mkv.stimulus.json").write_text(
        json.dumps({"stimulus": {"coverage": {"saturation": [0.5, 0.5]}}}), encoding="utf-8"
    )
    assert cli.main(argv) == 0
    effects = load_profile(profiles / "P.yaml").effects
    assert sorted(effects) == [fx.BW, fx.COLOUR] and effects[fx.BW]["settings"][0]["intensity"] == 0.3
    (profiles / "P.yaml").unlink()
    assert cli.main(argv) == 1


def test_a_footage_archive_without_a_stimulus_record_cannot_be_classed(tmp_path):
    with pytest.raises(FileNotFoundError, match="synthetic clip"):
        fx.stimulus_record(tmp_path, {"clip": "c.mkv", "source": str(tmp_path / "c.mkv")})

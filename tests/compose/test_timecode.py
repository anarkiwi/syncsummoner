"""The timecoded source: conform and frame choice against the float reference, strip codes, caching."""

# pylint: disable=missing-function-docstring,no-member

import io
import logging
import os
import subprocess
import types
from dataclasses import replace
from fractions import Fraction

import cv2
import numpy as np
import pytest

from syncsummoner.compose import lock as L
from syncsummoner.compose import render as R
from syncsummoner.compose import timecode as T
from syncsummoner.compose.features import read_frames
from syncsummoner.probe import decoded
from syncsummoner.probe.patterns import read_timecode

CONFIG = R.RenderConfig(width=160, height=96, fps=30.0, strip_px=8, bits=16)


def write_source(path, frames, fps):
    """Encode RGB ``uint8`` frames losslessly, so a decode returns them exactly."""
    height, width = frames[0].shape[:2]
    subprocess.run(
        [
            "ffmpeg",
            "-loglevel",
            "error",
            "-y",
            "-f",
            "rawvideo",
            "-pix_fmt",
            "rgb24",
            "-s",
            f"{width}x{height}",
        ]
        + ["-r", str(fps), "-i", "pipe:0", "-c:v", "ffv1", "-pix_fmt", "bgr0", str(path)],
        input=np.stack(frames).tobytes(),
        check=True,
    )
    return path


def synthetic(tmp_path, *, count=20, size=(48, 70), fps=25.0, name="source.mkv"):
    """A smooth, distinct picture per frame, so resizing and frame choice both show."""
    rng = np.random.default_rng(count)
    yy, xx = np.mgrid[: size[0], : size[1]]
    frames = []
    for i in range(count):
        phase = rng.uniform(0, 2 * np.pi, 3)
        wave = 0.5 + 0.4 * np.sin(xx[..., None] / (3 + i % 5) + yy[..., None] / 4 + phase)
        frames.append((wave * 255).astype(np.uint8))
    return write_source(tmp_path / name, frames, fps)


def reference(source, config, *, seconds=None, start=0.0):
    """Session frames by float area resize of an OpenCV decode, each the first source frame at or after it."""
    stack = [frame for _, frame in read_frames(source)]
    rate = float(cv2.VideoCapture(str(source)).get(cv2.CAP_PROP_FPS))
    skip = int(round(start * rate))
    total = int(round(seconds * config.fps)) if seconds else None
    height, width = stack[0].shape[:2]
    scale = min(config.width / width, config.height / height)
    fit = (max(1, round(width * scale)), max(1, round(height * scale)))
    left, top = (config.width - fit[0]) // 2, (config.height - fit[1]) // 2
    out = []
    for index in range(total or len(stack) * 4):
        held = -(-index * Fraction(rate) // Fraction(config.fps)) + skip
        if held >= len(stack):
            break
        frame = np.zeros((config.height, config.width, 3), np.float32)
        frame[top : top + fit[1], left : left + fit[0]] = cv2.resize(
            stack[held], fit, interpolation=cv2.INTER_AREA
        )
        out.append(np.clip(frame * 255, 0, 255))
    return out


@pytest.mark.parametrize(
    "fps, size, start, seconds",
    [
        (25.0, (48, 70), 0.0, None),
        (25.0, (48, 70), 0.2, 0.5),
        (50.0, (40, 200), 0.0, None),
        (30.0, (96, 160), 0.1, None),
    ],
)
def test_the_stream_is_the_float_area_conform_of_the_first_source_frame_at_or_after_each_time(
    tmp_path, fps, size, start, seconds
):
    """ffmpeg decodes and ``uint8`` area resizing pads: within a level of the float conform, same frames."""
    source = synthetic(tmp_path, size=size, fps=fps)
    want = reference(source, CONFIG, seconds=seconds, start=start)
    total, stream = R.source_stream(source, CONFIG, seconds=seconds, start=start)
    got = [frame.copy() for frame in stream]
    assert total == (int(round(seconds * CONFIG.fps)) if seconds else 0)
    assert len(got) == len(want) and got[0].dtype == np.uint8
    assert max(np.abs(g.astype(np.float32) - w).max() for g, w in zip(got, want)) <= 1.0


def test_a_source_at_the_session_raster_passes_through_exactly(tmp_path):
    source = synthetic(tmp_path, size=(CONFIG.height, CONFIG.width), fps=CONFIG.fps)
    _, stream = R.source_stream(source, CONFIG)
    got = [frame.copy() for frame in stream]
    want = [(frame * 255).round().astype(np.uint8) for _, frame in read_frames(source)]
    assert len(got) == len(want) and all((g == w).all() for g, w in zip(got, want))


def test_an_unreadable_source_has_no_frames(tmp_path):
    (tmp_path / "empty.mkv").write_bytes(b"")
    total, stream = R.source_stream(tmp_path / "empty.mkv", CONFIG)
    assert total == 0 and not list(stream)


def test_a_decoder_that_fails_says_so(tmp_path):
    source = synthetic(tmp_path)
    config = types.SimpleNamespace(**vars(CONFIG), ffmpeg="false")
    with pytest.raises(RuntimeError, match="decoding .* failed: no output"):
        list(R.source_stream(source, config)[1])


@pytest.fixture(name="encoded")
def encoded_fixture(monkeypatch):
    """Frames as handed to the encoder, the decoder left real."""
    written, real = [], subprocess.Popen

    class Pipe:
        """Encoder stdin, keeping the frames it is handed."""

        def write(self, data):
            """Write."""
            written.append(np.frombuffer(bytes(data), np.uint8).reshape(CONFIG.height, CONFIG.width, 3))

        def close(self):
            """Close."""

    def popen(argv, **kw):
        if not any(T.TIMECODE_TAG in arg for arg in argv):
            return real(argv, **kw)
        open(argv[-1], "wb").close()  # pylint: disable=consider-using-with
        return types.SimpleNamespace(stdin=Pipe(), stderr=io.BytesIO(b""), wait=lambda: 0, returncode=0)

    monkeypatch.setattr(T.subprocess, "Popen", popen)
    return written


@pytest.mark.parametrize("lead, tail", [(0, 0), (3, 0), (3, 2)])
def test_every_frame_is_stamped_between_a_wrapped_lead_in_and_a_lead_out_of_identical_pictures(
    encoded, tmp_path, lead, tail
):
    source = synthetic(tmp_path, count=10)
    clip = R.write_timecoded(source, tmp_path / "tc.mkv", config=CONFIG, lead=lead, tail=tail, cache=tmp_path)
    assert clip == 11 and len(encoded) == clip + lead + tail
    codes = [read_timecode(f / 255.0, bits=CONFIG.bits, strip_px=CONFIG.strip_px) for f in encoded]
    assert codes == [65536 - lead + i for i in range(lead)] + list(range(clip + tail))
    body = [f[: -CONFIG.strip_px] for f in encoded]
    assert all((b == body[lead]).all() for b in body[: lead + 1])
    assert all((b == body[-1]).all() for b in body[lead + clip - 1 :])
    assert not (body[lead] == body[lead + 1]).all()


def test_an_encoded_clip_reads_back_its_wrapped_lead_in_every_frame_and_its_lead_out(tmp_path):
    source = synthetic(tmp_path, count=5)
    out = tmp_path / "tc.mkv"
    assert R.write_timecoded(source, out, config=CONFIG, lead=4, tail=2, cache=tmp_path / "cache") == 5
    features = decoded.features(out, decoded.Spec.of(CONFIG))
    assert L.signed(features.codes, CONFIG.bits).tolist() == list(range(-4, 7))
    assert (np.round(features.times * 1000) == np.round(np.arange(11) * 1000 / CONFIG.fps)).all()
    assert T.clip_layout(out) == T.ClipLayout(lead=4, frames=5, tail=2), "the clip says how its frames lie"


def test_a_clip_without_its_layout_tags_is_a_layout_error(tmp_path):
    """A clip timecoded before its layout was tagged cannot say where its lead-in ends."""
    plain = tmp_path / "plain.mkv"
    subprocess.run(
        [
            "ffmpeg",
            "-loglevel",
            "error",
            "-f",
            "lavfi",
            "-i",
            "color=c=gray:s=16x16:d=0.2",
            "-c:v",
            "ffv1",
            str(plain),
        ],
        check=True,
    )
    with pytest.raises(T.LayoutError, match="timecode it again"):
        T.clip_layout(plain)


def test_a_program_that_turns_the_picture_over_leaves_no_legible_strip(tmp_path):
    """A geometric program moves the strip off the edge; that is unread, not a frame index."""
    clip, flipped = tmp_path / "tc.mkv", tmp_path / "flipped.mkv"
    R.write_timecoded(synthetic(tmp_path, count=5), clip, config=CONFIG, lead=4, tail=2, cache=tmp_path)
    subprocess.run(
        ["ffmpeg", "-loglevel", "error", "-i", str(clip), "-vf", "vflip", "-c:v", "ffv1", str(flipped)],
        check=True,
    )
    assert (decoded.features(flipped, decoded.Spec.of(CONFIG)).codes == -1).all()


def test_a_dead_encoder_is_reported_as_what_it_said(monkeypatch, tmp_path):
    """Writing to a dead encoder raises a broken pipe, which says nothing about why."""
    source, real = synthetic(tmp_path), subprocess.Popen

    class Pipe:
        """Encoder stdin that has already gone away."""

        def write(self, data):
            """Write."""
            del data
            raise BrokenPipeError

        def close(self):
            """Close."""

    def popen(argv, **kw):
        if not any(T.TIMECODE_TAG in arg for arg in argv):
            return real(argv, **kw)
        said = io.BytesIO(b"Error opening output timecoded.mkv: Permission denied\n")
        return types.SimpleNamespace(stdin=Pipe(), stderr=said, wait=lambda: 1, returncode=1)

    monkeypatch.setattr(T.subprocess, "Popen", popen)
    with pytest.raises(RuntimeError, match="Permission denied"):
        R.write_timecoded(source, tmp_path / "tc.mkv", config=CONFIG, lead=2, tail=2, cache=tmp_path)
    assert not list(tmp_path.glob("timecoded-*"))


def test_a_clip_is_encoded_once_per_key_whatever_the_scratch_is_named(monkeypatch, tmp_path, caplog):
    source, cache = synthetic(tmp_path, count=5), tmp_path / "cache"
    encodes, real = [], T._encode_timecoded  # pylint: disable=protected-access
    monkeypatch.setattr(T, "_encode_timecoded", lambda *a, **k: encodes.append(k["key"]) or real(*a, **k))
    kw = {"config": CONFIG, "lead": 2, "tail": 1, "cache": cache}
    first = R.write_timecoded(source, tmp_path / "a.mkv", **kw)
    with caplog.at_level(logging.INFO, logger="syncsummoner"):
        again = R.write_timecoded(source, tmp_path / "b.mkv", **kw)
        R.write_timecoded(source, tmp_path / "b.mkv", **kw)
    assert first == again == 5 and len(encodes) == 1
    assert (tmp_path / "a.mkv").read_bytes() == (tmp_path / "b.mkv").read_bytes()
    assert sum("reusing timecoded" in r.getMessage() for r in caplog.records) == 2
    tag = subprocess.run(
        ["ffprobe", "-v", "error", "-show_entries", f"format_tags={T.TIMECODE_TAG}", "-of", "csv=p=0"]
        + [str(tmp_path / "b.mkv")],
        capture_output=True,
        text=True,
        check=True,
    ).stdout.strip()
    assert tag == encodes[0]


def test_a_changed_setting_source_or_entry_is_encoded_afresh(monkeypatch, tmp_path):
    source, cache = synthetic(tmp_path, count=5), tmp_path / "cache"
    encodes, real = [], T._encode_timecoded  # pylint: disable=protected-access
    monkeypatch.setattr(T, "_encode_timecoded", lambda *a, **k: encodes.append(k["key"]) or real(*a, **k))
    kw = {"lead": 2, "tail": 1, "cache": cache}
    R.write_timecoded(source, tmp_path / "tc.mkv", config=CONFIG, **kw)
    R.write_timecoded(source, tmp_path / "tc.mkv", config=replace(CONFIG, bits=12), **kw)
    R.write_timecoded(source, tmp_path / "tc.mkv", config=CONFIG, start=0.2, **kw)
    stat = source.stat()
    os.utime(source, ns=(stat.st_atime_ns, stat.st_mtime_ns + 1))
    R.write_timecoded(source, tmp_path / "tc.mkv", config=CONFIG, **kw)
    entry = cache / f"timecoded-{encodes[-1]}.mkv"
    with entry.open("ab") as tampered:
        tampered.write(b"\0")
    R.write_timecoded(source, tmp_path / "tc.mkv", config=CONFIG, **kw)
    assert len(set(encodes)) == 4 and len(encodes) == 5 and encodes[-1] == encodes[-2]


def test_a_scratch_on_another_filesystem_is_a_copy(monkeypatch, tmp_path):
    def cross_device(*args):
        raise OSError(18, "Invalid cross-device link", *args)

    monkeypatch.setattr(T.os, "link", cross_device)
    source = synthetic(tmp_path, count=5)
    R.write_timecoded(source, tmp_path / "tc.mkv", config=CONFIG, lead=1, tail=1, cache=tmp_path / "cache")
    (entry,) = (tmp_path / "cache").glob("*.mkv")
    assert not (tmp_path / "tc.mkv").samefile(entry)
    assert (tmp_path / "tc.mkv").read_bytes() == entry.read_bytes()


def test_the_default_cache_is_named_by_the_environment(monkeypatch, tmp_path):
    monkeypatch.setenv("SYNCSUMMONER_TIMECODE_CACHE", str(tmp_path))
    assert T.timecode_cache() == tmp_path.resolve()
    monkeypatch.delenv("SYNCSUMMONER_TIMECODE_CACHE")
    monkeypatch.chdir(tmp_path)
    assert T.timecode_cache() == (tmp_path / ".timecode-cache").resolve()

"""The timecoded source: the clip at session geometry with every frame's index in its edge strip.

ffmpeg decodes and encodes; between them each frame stays ``uint8`` and only its strip rows are
written per frame. A clip is a pure function of the source file and the settings in
:func:`timecode_key`, so it is kept under that key and reused (docs/workflow.md).
"""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import subprocess
import tempfile
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterator

import numpy as np

from syncsummoner.probe.patterns import timecode_row
from syncsummoner.progress import LOG, track

#: Bumped whenever what a timecoded clip holds, or how it is encoded, changes; older clips are then rebuilt.
TIMECODE_SCHEMA = 2
#: How a timecoded clip is encoded; part of its cache key.
TIMECODE_X264 = ("-c:v", "libx264", "-crf", "14", "-preset", "veryfast", "-pix_fmt", "yuv420p")
#: Matroska tag naming the key a timecoded clip was built for.
TIMECODE_TAG = "SYNCSUMMONER_TIMECODE"
#: Matroska tags holding a timecoded clip's lead-in and lead-out lengths, in frames.
LEAD_TAG, TAIL_TAG = "SYNCSUMMONER_LEAD", "SYNCSUMMONER_TAIL"


class LayoutError(RuntimeError):
    """A clip does not say how its lead-in, clip and lead-out frames lie."""


@dataclass(frozen=True)
class ClipLayout:
    """A timecoded clip's frames: ``lead`` lead-in, ``frames`` of the clip, ``tail`` lead-out."""

    lead: int
    frames: int
    tail: int


def clip_layout(path: str | Path, *, ffprobe: str = "ffprobe") -> ClipLayout:
    """How a timecoded clip's frames lie, from the tags it was written with and its packet count."""
    argv = [ffprobe, "-v", "error", "-select_streams", "v:0", "-count_packets", "-of", "json"]
    argv += ["-show_entries", "stream=nb_read_packets:format_tags", str(path)]
    said = json.loads(subprocess.run(argv, check=True, capture_output=True, text=True).stdout)
    tags = {k.upper(): v for k, v in said.get("format", {}).get("tags", {}).items()}
    if LEAD_TAG not in tags or TAIL_TAG not in tags:
        raise LayoutError(f"{path} carries no {LEAD_TAG}/{TAIL_TAG} tags: timecode it again")
    lead, tail = int(tags[LEAD_TAG]), int(tags[TAIL_TAG])
    return ClipLayout(lead, int(said["streams"][0]["nb_read_packets"]) - lead - tail, tail)


def timecode_key(
    source: Any, config: Any, *, seconds: float | None, start: float, lead: int, tail: int
) -> str:
    """SHA-256 over the source file's identity and every setting a timecoded clip depends on."""
    path = Path(source).resolve()
    stat = path.stat()
    token = {
        "schema": TIMECODE_SCHEMA,
        "source": [str(path), stat.st_size, stat.st_mtime_ns],
        "raster": [config.width, config.height, config.fps],
        "strip": [config.strip_px, config.bits],
        "span": [seconds, start, lead, tail],
        "x264": list(TIMECODE_X264),
    }
    return hashlib.sha256(json.dumps(token, sort_keys=True).encode()).hexdigest()


def timecode_cache(cache: str | Path | None = None) -> Path:
    """Where clips are kept: ``cache``, else ``SYNCSUMMONER_TIMECODE_CACHE``, else ``.timecode-cache``."""
    return Path(cache or os.environ.get("SYNCSUMMONER_TIMECODE_CACHE") or ".timecode-cache").resolve()


def _cached(entry: Path) -> int | None:
    """Clip frames of a complete cache entry, or None when it is absent or was changed after it was built."""
    try:
        record = json.loads(entry.with_suffix(".json").read_text())
        stat = entry.stat()
    except (OSError, ValueError):
        return None
    if [stat.st_size, stat.st_mtime_ns] != record.get("stat"):
        return None
    return int(record["frames"])


def _publish(entry: Path, out: Path) -> None:
    """Make ``out`` the entry: a hard link where the filesystem allows one, else a copy, swapped in whole."""
    if out.exists() and out.samefile(entry):
        return
    tmp = out.with_name(f".{out.name}.{uuid.uuid4().hex}.tmp")
    try:
        os.link(entry, tmp)
    except OSError:
        shutil.copyfile(entry, tmp)
    os.replace(tmp, out)


def write_timecoded(
    source: Any,
    out: str | Path,
    *,
    config: Any,
    seconds: float | None = None,
    start: float = 0.0,
    lead: int,
    tail: int,
    cache: str | Path | None = None,
) -> int:
    """Write the source at session geometry, each frame's index burnt in, between a lead-in and a lead-out.

    The lead-in is ``lead`` copies of the first frame numbered ``-lead..-1``, wrapped to the strip's
    width, and the lead-out ``tail`` copies of the last numbered on from it.
    Returns the clip frames, neither included.

    A clip the playout can identify frame by frame is what lets the source be
    played from the other end, where it can run at rate. ``start`` skips into it,
    which is how an excerpt gets the footage it was composed against. The clip is
    kept in :func:`timecode_cache` under :func:`timecode_key`, so the same source
    and settings are encoded once whatever ``out`` is named; an entry whose file
    no longer has the size and mtime it was built with is rebuilt.
    """
    out = Path(out)
    key = timecode_key(source, config, seconds=seconds, start=start, lead=lead, tail=tail)
    entry = timecode_cache(cache) / f"timecoded-{key}.mkv"
    frames = _cached(entry)
    if frames is None:
        entry.parent.mkdir(parents=True, exist_ok=True)
        tmp = entry.with_name(f".{entry.name}.{uuid.uuid4().hex}.tmp")
        try:
            frames = _encode_timecoded(
                source, tmp, config=config, seconds=seconds, start=start, lead=lead, tail=tail, key=key
            )
            os.replace(tmp, entry)
        finally:
            tmp.unlink(missing_ok=True)
        stat = entry.stat()
        entry.with_suffix(".json").write_text(
            json.dumps({"frames": frames, "stat": [stat.st_size, stat.st_mtime_ns]})
        )
        LOG.info(
            "timecoded %d frames between a %d and a %d frame lead-in and -out at %dx%d into %s",
            frames,
            lead,
            tail,
            config.width,
            config.height,
            entry,
        )
    else:
        LOG.info("reusing timecoded %s for %s (key %s, %d frames)", entry, out, key[:12], frames)
    _publish(entry, out)
    return frames


def _encode_timecoded(
    source: Any,
    out: Path,
    *,
    config: Any,
    seconds: float | None,
    start: float,
    lead: int,
    tail: int,
    key: str,
) -> int:
    """Encode :func:`source_stream` with its strip codes into ``out``; returns the clip frames.

    An encoder that dies takes its own diagnosis with it, so that is what is raised
    rather than the broken pipe writing to it produces.
    """
    total, frames = source_stream(source, config, seconds=seconds, start=start)
    argv = [
        getattr(config, "ffmpeg", "ffmpeg"),
        "-loglevel",
        "error",
        "-y",
        "-f",
        "rawvideo",
        "-pix_fmt",
        "rgb24",
        "-s",
        f"{config.width}x{config.height}",
        "-r",
        str(config.fps),
        "-i",
        "pipe:0",
        *TIMECODE_X264,
        "-metadata",
        f"{TIMECODE_TAG}={key}",
        "-metadata",
        f"{LEAD_TAG}={lead}",
        "-metadata",
        f"{TAIL_TAG}={tail}",
        "-f",
        "matroska",
        str(out),
    ]
    proc = subprocess.Popen(  # pylint: disable=consider-using-with
        argv, stdin=subprocess.PIPE, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE
    )
    wrap, strip, sent = (1 << config.bits) - 1, config.height - config.strip_px, 0
    numbered = _numbered(frames, lead, tail)
    try:
        for index, frame in track(
            numbered, desc="timecoding", total=total and lead + total + tail or None, unit="frame"
        ):
            frame[strip:] = (timecode_row(index & wrap, config.width, bits=config.bits) * 255).astype(
                np.uint8
            )[:, None]
            try:
                proc.stdin.write(frame.data)
            except BrokenPipeError:
                break
            sent += 1
    finally:
        numbered.close()
        proc.stdin.close()
        said = (proc.stderr.read() or b"").decode("utf-8", "replace").strip()
        proc.stderr.close()
        proc.wait()
    written = max(sent - lead - tail, 0)
    if proc.returncode:
        raise RuntimeError(f"timecoding {out} failed: {said.splitlines()[-1] if said else 'no output'}")
    return written or total


def _numbered(frames: Iterator[np.ndarray], lead: int, tail: int) -> Iterator[tuple[int, np.ndarray]]:
    """Each frame with its strip index: the first also ``lead`` times before 0, the last ``tail`` after it."""
    index, frame = -1, None
    try:
        for index, frame in enumerate(frames):
            for before in range(-lead, 0) if index == 0 else ():
                yield before, frame
            yield index, frame
        for after in range(index + 1, index + 1 + tail) if frame is not None else ():
            yield after, frame
    finally:
        frames.close()


def source_stream(source: Any, config: Any, *, seconds: float | None = None, start: float = 0.0):
    """Return ``(total, frames)``: how many session frames a take is, and them one at a time.

    ffmpeg decodes the source to RGB at its own size and rate, skipping ``start``
    seconds of frames; each session frame then holds the first source frame at or
    after its time, area-resized to fit the raster, centred and padded with black,
    since the aspect a source was shot at is not the rig's to change. Frames are
    ``uint8`` and one buffer is reused, so each is valid until the next is drawn.
    """
    # pylint: disable=no-member
    import cv2

    cap = cv2.VideoCapture(str(source))
    rate = float(cap.get(cv2.CAP_PROP_FPS) or 30.0)
    width, height = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    cap.release()
    if width <= 0 or height <= 0:
        return 0, (frame for frame in ())
    src_fps = rate if rate > 0 else config.fps
    total = int(round((seconds or 0.0) * config.fps)) if seconds else 0
    skip = max(0, int(round(start * src_fps)))
    scale = min(config.width / width, config.height / height)
    fit = (max(1, round(width * scale)), max(1, round(height * scale)))
    left, top = (config.width - fit[0]) // 2, (config.height - fit[1]) // 2
    argv = [
        getattr(config, "ffmpeg", "ffmpeg"),
        "-loglevel",
        "error",
        "-nostdin",
        "-i",
        str(source),
        "-map",
        "0:v:0",
    ]
    argv += ["-vf", f"trim=start_frame={skip}"] if skip else []
    argv += ["-fps_mode", "passthrough", "-f", "rawvideo", "-pix_fmt", "rgb24", "pipe:1"]

    def frames() -> Iterator[np.ndarray]:
        frame = np.zeros((config.height, config.width, 3), np.uint8)
        same = (width, height) == (config.width, config.height)
        decoded = frame if same else np.empty((height, width, 3), np.uint8)
        resized = np.empty((fit[1], fit[0], 3), np.uint8)
        body = frame[top : top + fit[1], left : left + fit[0]]
        index, source_index = 0, skip - 1
        with tempfile.TemporaryFile() as said:
            proc = subprocess.Popen(  # pylint: disable=consider-using-with
                argv, stdin=subprocess.DEVNULL, stdout=subprocess.PIPE, stderr=said
            )
            try:
                while proc.stdout.readinto(decoded.data) == decoded.nbytes:
                    source_index += 1
                    if index * src_fps / config.fps + skip > source_index:
                        continue
                    if not same:
                        body[...] = cv2.resize(decoded, fit, dst=resized, interpolation=cv2.INTER_AREA)
                    while index * src_fps / config.fps + skip <= source_index:
                        yield frame
                        index += 1
                        if total and index >= total:
                            return
                if proc.wait():
                    said.seek(0)
                    text = said.read().decode("utf-8", "replace").strip().splitlines()
                    raise RuntimeError(f"decoding {source} failed: {text[-1] if text else 'no output'}")
            finally:
                proc.stdout.close()
                if proc.poll() is None:
                    proc.kill()
                proc.wait()

    return total, frames()

"""Lossless raw-frame archive: intra-only FFV1 video plus a per-frame state sidecar.

Probing is expensive and the rig dies intermittently, so the card's own 4:2:2
frames are archived once with the state that produced them, and later metrics are
recomputed offline. Keyed like the result store, so a reflash invalidates both.
"""

from __future__ import annotations

import json
import os
import subprocess
import tempfile
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Iterator, Mapping, Sequence

import numpy as np

from syncsummoner.device.profile import PARAM_COUNT
from syncsummoner.probe.fit import load_measurements, save_measurements
from syncsummoner.probe.store import ProgramKey, atomic_write, slug, temp_sibling

__all__ = [
    "ARCHIVE_SCHEMA_VERSION",
    "DARK_SUFFIX",
    "ArchiveError",
    "FrameArchive",
    "FrameReader",
    "FrameRow",
    "FrameWriter",
]

#: Version 2 tags the pipe YVYU; version 1 tagged it YUYV and stored the chroma planes exchanged.
ARCHIVE_SCHEMA_VERSION = 2
#: Nominal container rate; real times are in the sidecar, and 40 ms spacing keeps seeks off the 1 ms timebase.
ARCHIVE_FPS = 25
CONTAINER = "matroska"
VIDEO_SUFFIX = ".mkv"
#: Marks a program measured black on a healthy rig, so a resume does not re-probe it.
DARK_SUFFIX = ".dark.json"
#: What crosses the pipe in both directions: the card's measured byte order, packed 4:2:2.
PIPE_PIX_FMT = "yvyu422"
#: Planar 4:2:2 is the same samples deinterleaved into the canonical plane order.
STORED_PIX_FMT = "yuv422p"
#: Packed 4:2:2 carries two bytes per pixel: luma, and one of the shared chroma pair.
CHANNELS = 2
#: FFV1 level 3 with per-slice CRCs, storing the card's samples as-is with no upsample or padding.
ENCODE_ARGS = (
    "-c:v",
    "ffv1",
    "-level",
    "3",
    "-coder",
    "1",
    "-context",
    "1",
    "-slices",
    "4",
    "-slicecrc",
    "1",
    "-pix_fmt",
    STORED_PIX_FMT,
)
_PARAM_COLUMNS = tuple(f"p{i + 1}" for i in range(PARAM_COUNT))


class ArchiveError(RuntimeError):
    """The archive could not be written, read, or was not encoded losslessly."""


@dataclass(frozen=True)
class FrameRow:
    """One archived frame: where it sits in the stream, and what produced it.

    ``loop_index`` is the decoded codeframe position, None when it did not
    decode; ``captured`` is wall-clock epoch seconds.
    """

    frame: int
    program: str
    params: tuple[int, ...]
    setpoint: int
    loop_index: int | None = None
    captured: float = 0.0

    def as_row(self) -> dict[str, Any]:
        """Flatten to a Parquet-friendly row, sharing ``MeasurementRecord`` param naming."""
        row: dict[str, Any] = {
            "frame": self.frame,
            "program": self.program,
            "setpoint": self.setpoint,
            "loop_index": self.loop_index,
            "captured": self.captured,
        }
        row.update(dict(zip(_PARAM_COLUMNS, self.params)))
        return row

    @classmethod
    def from_row(cls, row: Mapping[str, Any]) -> "FrameRow":
        """Rebuild from the flat form produced by :meth:`as_row`."""
        loop_index = row.get("loop_index")
        return cls(
            frame=int(row["frame"]),
            program=str(row["program"]),
            params=tuple(int(row[name]) for name in _PARAM_COLUMNS),
            setpoint=int(row["setpoint"]),
            loop_index=None if loop_index is None else int(loop_index),
            captured=float(row.get("captured") or 0.0),
        )


def _params(vector: Sequence[int]) -> tuple[int, ...]:
    """Validate and normalise a raw parameter vector."""
    values = tuple(int(v) for v in vector)
    if len(values) != PARAM_COUNT:
        raise ArchiveError(f"parameter vector has {len(values)} values, expected {PARAM_COUNT}")
    return values


class FrameWriter:
    """Streams the card's native YVYU frames into an ffmpeg FFV1 encoder over stdin.

    Frames go to the encoder as they arrive and are never accumulated; only one
    small row per frame is held, and that is what the sidecar commits when the
    context exits cleanly.
    """

    def __init__(self, archive: "FrameArchive", program: str, key: ProgramKey, *, width, height, fps):
        self.archive = archive
        self.program = program
        self.key = key
        self.width = int(width)
        self.height = int(height)
        self.fps = int(fps)
        self.rows: list[FrameRow] = []
        self.log = ""
        self._proc: Any = None
        self._errors: Any = None
        self._tmp: Path | None = None
        if self.width % 2:
            raise ArchiveError(f"4:2:2 shares chroma across pixel pairs; width {self.width} is odd")

    @property
    def count(self) -> int:
        """Frames written so far."""
        return len(self.rows)

    @property
    def shape(self) -> tuple[int, int, int]:
        """Shape every archived frame must have."""
        return (self.height, self.width, CHANNELS)

    def command(self, path: Path) -> list[str]:
        """Encoder argv writing lossless FFV1 to ``path``."""
        return [
            self.archive.ffmpeg,
            "-loglevel",
            "error",
            "-y",
            "-f",
            "rawvideo",
            "-pix_fmt",
            PIPE_PIX_FMT,
            "-s",
            f"{self.width}x{self.height}",
            "-r",
            str(self.fps),
            "-i",
            "pipe:0",
            *ENCODE_ARGS,
            "-f",
            CONTAINER,
            str(path),
        ]

    def __enter__(self) -> "FrameWriter":
        self.archive.directory.mkdir(parents=True, exist_ok=True)
        self._tmp = temp_sibling(self.archive.paths(self.program)[0])
        self._errors = tempfile.TemporaryFile()
        self._proc = self.archive.popen(
            self.command(self._tmp),
            stdin=subprocess.PIPE,
            stdout=subprocess.DEVNULL,
            stderr=self._errors,
        )
        return self

    def write(
        self,
        frame: np.ndarray,
        *,
        params: Sequence[int],
        setpoint: int,
        loop_index: int | None = None,
        captured: float | None = None,
    ) -> int:
        """Encode one native YVYU frame, returning the index it takes in the stream."""
        if self._proc is None:
            raise ArchiveError("archive writer is not open")
        if not isinstance(frame, np.ndarray) or frame.dtype != np.uint8:
            raise ArchiveError("archived frames must be uint8, exactly as the card delivered them")
        if frame.shape != self.shape:
            raise ArchiveError(f"frame {frame.shape} does not match {self.shape}")
        row = FrameRow(
            frame=len(self.rows),
            program=self.program,
            params=_params(params),
            setpoint=int(setpoint),
            loop_index=None if loop_index is None else int(loop_index),
            captured=float(self.archive.clock() if captured is None else captured),
        )
        try:
            self._proc.stdin.write(np.ascontiguousarray(frame).data)
        except (BrokenPipeError, OSError, ValueError) as exc:
            raise ArchiveError(f"ffmpeg stopped accepting frames after {len(self.rows)}") from exc
        self.rows.append(row)
        return row.frame

    def _reap(self) -> int:
        """Close stdin, wait for the encoder, and keep whatever it said."""
        try:
            self._proc.stdin.close()
        except (BrokenPipeError, OSError):
            pass
        returncode = int(self._proc.wait() or 0)
        if self._errors is not None:
            self._errors.seek(0)
            self.log = self._errors.read().decode("utf-8", "replace").strip()
            self._errors.close()
            self._errors = None
        self._proc = None
        return returncode

    def _meta(self, video: Path, data: Path) -> dict[str, Any]:
        """Self-describing metadata: identity, geometry, codec and yield."""
        return {
            "schema_version": ARCHIVE_SCHEMA_VERSION,
            "program": self.program,
            "key_kind": self.key.kind.value,
            "key_material": self.key.material,
            "key_digest": self.key.digest,
            "codec": "ffv1",
            "container": CONTAINER,
            "pix_fmt": PIPE_PIX_FMT,
            "stored_pix_fmt": STORED_PIX_FMT,
            "channels": CHANNELS,
            "width": self.width,
            "height": self.height,
            "fps": self.fps,
            "frames": len(self.rows),
            "bytes_per_frame": video.stat().st_size / len(self.rows),
            "video": video.name,
            "data": data.name,
            "created": datetime.fromtimestamp(self.archive.clock(), timezone.utc).isoformat(),
        }

    def _commit(self) -> None:
        """Publish the video, then the sidecar, then the metadata marker."""
        video, data, meta_path = self.archive.paths(self.program)
        os.replace(self._tmp, video)
        atomic_write(data, lambda tmp: save_measurements(self.rows, tmp))
        payload = json.dumps(self._meta(video, data), indent=2)
        atomic_write(meta_path, lambda tmp: tmp.write_text(payload, encoding="utf-8"))

    def __exit__(self, exc_type, exc, traceback) -> bool:
        returncode = self._reap()
        try:
            if exc_type is not None:
                return False
            if returncode:
                raise ArchiveError(f"ffmpeg exited {returncode}: {self.log or 'no diagnostic'}")
            if not self.rows:
                raise ArchiveError(f"no frames archived for {self.program!r}")
            self._commit()
        finally:
            if self._tmp is not None:
                self._tmp.unlink(missing_ok=True)
                self._tmp = None
        return False


class FrameReader:
    """Random access into one archived program: its rows, and any frame by index."""

    def __init__(
        self,
        video: Path,
        meta: Mapping[str, Any],
        rows: Sequence[FrameRow],
        *,
        ffmpeg,
        run,
        popen: Callable[..., Any] = subprocess.Popen,
    ):
        self.video = Path(video)
        self.meta = dict(meta)
        self.rows = list(rows)
        self.width = int(self.meta["width"])
        self.height = int(self.meta["height"])
        self.fps = float(self.meta["fps"])
        self._ffmpeg = ffmpeg
        self._run = run
        self._popen = popen

    @property
    def count(self) -> int:
        """Frames in the stream."""
        return int(self.meta["frames"])

    @property
    def shape(self) -> tuple[int, int, int]:
        """Shape of every archived frame."""
        return (self.height, self.width, CHANNELS)

    def command(self, index: int) -> list[str]:
        """Decoder argv emitting frame ``index`` as packed YVYU on stdout.

        Every frame is a keyframe, so the input seek lands on it directly rather
        than decoding the stream from the start.
        """
        offset = max(0.0, index - 0.5) / self.fps
        return [
            self._ffmpeg,
            "-loglevel",
            "error",
            "-ss",
            f"{offset:.6f}",
            "-i",
            str(self.video),
            "-frames:v",
            "1",
            "-f",
            "rawvideo",
            "-pix_fmt",
            PIPE_PIX_FMT,
            "pipe:1",
        ]

    def frame(self, index: int) -> np.ndarray:
        """Decode frame ``index`` as packed YVYU, byte for byte as it was written."""
        if not 0 <= index < self.count:
            raise IndexError(f"frame {index} outside 0..{self.count - 1}")
        result = self._run(self.command(index), capture_output=True, check=False)
        if result.returncode or len(result.stdout) != self.height * self.width * CHANNELS:
            raise ArchiveError(f"decoding frame {index} of {self.video} gave {len(result.stdout)} bytes")
        return np.frombuffer(bytearray(result.stdout), dtype=np.uint8).reshape(self.shape)

    def stream(self, *, start: int = 0, count: int | None = None) -> "Iterator[np.ndarray]":
        """Yield frames in stream order from one decoder, rather than a seek per frame.

        Reading the whole archive back frame by frame costs a process and a seek
        each time; recomputing metrics offline reads it in order, so it does not.
        """
        want = self.count - start if count is None else min(int(count), self.count - start)
        if want <= 0:
            return
        size = self.height * self.width * CHANNELS
        argv = [
            self._ffmpeg,
            "-loglevel",
            "error",
            "-ss",
            f"{max(0.0, start - 0.5) / self.fps:.6f}",
            "-i",
            str(self.video),
            "-frames:v",
            str(want),
            "-f",
            "rawvideo",
            "-pix_fmt",
            PIPE_PIX_FMT,
            "pipe:1",
        ]
        proc = self._popen(argv, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL)
        try:
            for _ in range(want):
                buf = proc.stdout.read(size)
                if buf is None or len(buf) < size:
                    return
                yield np.frombuffer(bytearray(buf), dtype=np.uint8).reshape(self.shape)
        finally:
            proc.stdout.close()
            proc.wait()

    def select(
        self,
        *,
        params: Sequence[int] | None = None,
        setpoint: int | None = None,
        loop_index: int | None = None,
    ) -> list[FrameRow]:
        """Rows matching every criterion given, in stream order."""
        wanted = None if params is None else _params(params)
        return [
            row
            for row in self.rows
            if (wanted is None or row.params == wanted)
            and (setpoint is None or row.setpoint == setpoint)
            and (loop_index is None or row.loop_index == loop_index)
        ]


class FrameArchive:
    """Native frame archives under ``directory``, one video plus sidecar per program.

    The JSON metadata is written last and is the commit marker, as in the result
    store: a run killed mid-encode leaves at most an uncommitted temp file.
    """

    def __init__(
        self,
        directory: str | os.PathLike[str],
        *,
        clock: Callable[[], float] = time.time,
        popen: Callable[..., Any] = subprocess.Popen,
        run: Callable[..., Any] = subprocess.run,
        ffmpeg: str = "ffmpeg",
    ):
        self.directory = Path(directory)
        self.clock = clock
        self.popen = popen
        self.run = run
        self.ffmpeg = ffmpeg

    def paths(self, program: str) -> tuple[Path, Path, Path]:
        """Video, sidecar and metadata paths for one program."""
        stem = slug(program)
        return (
            self.directory / f"{stem}{VIDEO_SUFFIX}",
            self.directory / f"{stem}.parquet",
            self.directory / f"{stem}.json",
        )

    def meta(self, program: str) -> dict[str, Any] | None:
        """Committed metadata for ``program``, or None when nothing usable is stored."""
        video, data, meta_path = self.paths(program)
        try:
            meta = json.loads(meta_path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            return None
        if str(meta.get("pix_fmt")) != PIPE_PIX_FMT:
            return None
        if int(meta.get("schema_version", 0)) > ARCHIVE_SCHEMA_VERSION:
            return None
        return meta if video.exists() and data.exists() else None

    def has(self, program: str, key: ProgramKey) -> bool:
        """True when a committed archive for ``program`` was keyed on ``key``."""
        meta = self.meta(program)
        return meta is not None and meta.get("key_digest") == key.digest

    def dark_path(self, program: str) -> Path:
        """Where a program's dark verdict is recorded."""
        return self.directory / f"{slug(program)}{DARK_SUFFIX}"

    def mark_dark(self, program: str, key: ProgramKey, luma: float) -> None:
        """Record that ``program`` is itself black on a rig that still carries the source.

        A dark program has no archive to resume from, so without this a resume
        re-probes it every time, and loads are the scarce resource on this device.
        """
        payload = json.dumps(
            {
                "program": program,
                "key_digest": key.digest,
                "luma": float(luma),
                "created": datetime.fromtimestamp(self.clock(), timezone.utc).isoformat(),
            },
            indent=2,
        )
        self.directory.mkdir(parents=True, exist_ok=True)
        atomic_write(self.dark_path(program), lambda tmp: tmp.write_text(payload, encoding="utf-8"))

    def dark(self, program: str, key: ProgramKey) -> bool:
        """True when ``program`` was measured dark under ``key``, so it need not run again."""
        try:
            verdict = json.loads(self.dark_path(program).read_text(encoding="utf-8"))
        except (OSError, ValueError):
            return False
        return str(verdict.get("key_digest")) == key.digest

    def committed(self) -> dict[str, dict[str, Any]]:
        """Metadata of every complete archive, keyed by program name."""
        found = {}
        for path in sorted(self.directory.glob("*.json")):
            try:
                program = str(json.loads(path.read_text(encoding="utf-8")).get("program", ""))
            except (OSError, ValueError):
                continue
            meta = self.meta(program) if program else None
            if meta is not None:
                found[program] = meta
        return found

    def rows(self, program: str) -> list[FrameRow]:
        """Sidecar rows of a committed archive, in stream order."""
        if self.meta(program) is None:
            return []
        return [FrameRow.from_row(row) for row in load_measurements(self.paths(program)[1])]

    def writer(
        self, program: str, key: ProgramKey, *, width: int, height: int, fps: int = ARCHIVE_FPS
    ) -> FrameWriter:
        """Context manager encoding one program's raw frames."""
        return FrameWriter(self, program, key, width=width, height=height, fps=fps)

    def reader(self, program: str, key: ProgramKey | None = None) -> FrameReader | None:
        """Reader for a committed archive, or None when absent or keyed differently."""
        meta = self.meta(program)
        if meta is None or (key is not None and meta.get("key_digest") != key.digest):
            return None
        return FrameReader(
            self.paths(program)[0],
            meta,
            self.rows(program),
            ffmpeg=self.ffmpeg,
            run=self.run,
            popen=self.popen,
        )

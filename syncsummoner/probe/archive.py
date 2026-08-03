"""Lossless raw-frame archive: intra-only FFV1 video plus a per-frame state sidecar.

Probing is expensive and the rig dies intermittently, so ffmpeg records the card
once per program and this stores that recording with the state each frame was
captured under. Keyed like the result store, so a reflash invalidates both.
"""

from __future__ import annotations

import json
import os
import subprocess
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
    "GAP",
    "ArchiveError",
    "FrameArchive",
    "FrameReader",
    "FrameRow",
]

#: Version 3 is recorded by ffmpeg with the chroma pair put back; 1 and 2 stored it exchanged.
ARCHIVE_SCHEMA_VERSION = 3
#: Fallback container rate; the capture time of every frame is in the sidecar.
ARCHIVE_FPS = 30
CONTAINER = "matroska"
VIDEO_SUFFIX = ".mkv"
#: Marks a program measured black on a healthy rig, so a resume does not re-probe it.
DARK_SUFFIX = ".dark.json"
#: Setpoint of a frame captured between setpoints, kept for continuity but not measured.
GAP = -1
#: What the card is asked for; the recorder swaps the chroma pair back before encoding.
CAPTURE_PIX_FMT = "yuyv422"
#: Planar 4:2:2 is the same samples deinterleaved into the canonical plane order.
STORED_PIX_FMT = "yuv422p"
#: Read back converted, so nothing downstream repeats a 4:2:2 upsample in Python.
READ_PIX_FMT = "rgb24"
CHANNELS = 3
_PARAM_COLUMNS = tuple(f"p{i + 1}" for i in range(PARAM_COUNT))


class ArchiveError(RuntimeError):
    """The archive could not be written, read, or was not encoded losslessly."""


@dataclass(frozen=True)
class FrameRow:
    """One archived frame: where it sits in the stream, and what produced it.

    ``setpoint`` is :data:`GAP` for a frame captured while the sweep was moving
    between setpoints; ``captured`` is the card's monotonic capture time.
    """

    frame: int
    program: str
    params: tuple[int, ...]
    setpoint: int
    captured: float = 0.0

    def as_row(self) -> dict[str, Any]:
        """Flatten to a Parquet-friendly row, sharing ``MeasurementRecord`` param naming."""
        row: dict[str, Any] = {
            "frame": self.frame,
            "program": self.program,
            "setpoint": self.setpoint,
            "captured": self.captured,
        }
        row.update(dict(zip(_PARAM_COLUMNS, self.params)))
        return row

    @classmethod
    def from_row(cls, row: Mapping[str, Any]) -> "FrameRow":
        """Rebuild from the flat form produced by :meth:`as_row`."""
        return cls(
            frame=int(row["frame"]),
            program=str(row["program"]),
            params=tuple(int(row[name]) for name in _PARAM_COLUMNS),
            setpoint=int(row["setpoint"]),
            captured=float(row.get("captured") or 0.0),
        )


def _params(vector: Sequence[int]) -> tuple[int, ...]:
    """Validate and normalise a raw parameter vector."""
    values = tuple(int(v) for v in vector)
    if len(values) != PARAM_COUNT:
        raise ArchiveError(f"parameter vector has {len(values)} values, expected {PARAM_COUNT}")
    return values


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

    def shape(self, width: int | None = None) -> tuple[int, int, int]:
        """Shape frames come back in, at full size or scaled to ``width``."""
        if not width or width >= self.width:
            return (self.height, self.width, CHANNELS)
        return (2 * round(self.height * width / self.width / 2), int(width), CHANNELS)

    def command(self, *, start: int = 0, count: int = 1, width: int | None = None) -> list[str]:
        """Decoder argv emitting ``count`` frames from ``start`` as RGB on stdout.

        Every frame is a keyframe, so the input seek lands on one directly rather
        than decoding from the beginning, and ffmpeg does any scaling asked for.
        """
        argv = [
            self._ffmpeg,
            "-loglevel",
            "error",
            "-ss",
            f"{max(0.0, start - 0.5) / self.fps:.6f}",
            "-i",
            str(self.video),
            "-frames:v",
            str(int(count)),
        ]
        if width and width < self.width:
            argv += ["-vf", f"scale={int(width)}:-2"]
        return argv + ["-f", "rawvideo", "-pix_fmt", READ_PIX_FMT, "pipe:1"]

    def frame(self, index: int, *, width: int | None = None) -> np.ndarray:
        """Decode one frame as RGB, scaled to ``width`` by the decoder if asked."""
        if not 0 <= index < self.count:
            raise IndexError(f"frame {index} outside 0..{self.count - 1}")
        shape = self.shape(width)
        result = self._run(self.command(start=index, width=width), capture_output=True, check=False)
        if result.returncode or len(result.stdout) != int(np.prod(shape)):
            raise ArchiveError(f"decoding frame {index} of {self.video} gave {len(result.stdout)} bytes")
        return np.frombuffer(bytearray(result.stdout), dtype=np.uint8).reshape(shape)

    def stream(
        self, *, start: int = 0, count: int | None = None, width: int | None = None
    ) -> "Iterator[np.ndarray]":
        """Yield frames in stream order from one decoder, rather than a seek per frame.

        Reading the whole archive back frame by frame costs a process and a seek
        each time; recomputing metrics offline reads it in order, so it does not.
        """
        want = self.count - start if count is None else min(int(count), self.count - start)
        if want <= 0:
            return
        shape = self.shape(width)
        size = int(np.prod(shape))
        argv = self.command(start=start, count=want, width=width)
        proc = self._popen(argv, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL)
        try:
            for _ in range(want):
                buf = proc.stdout.read(size)
                if buf is None or len(buf) < size:
                    return
                yield np.frombuffer(bytearray(buf), dtype=np.uint8).reshape(shape)
        finally:
            proc.stdout.close()
            proc.wait()

    def select(self, *, params: Sequence[int] | None = None, setpoint: int | None = None) -> list[FrameRow]:
        """Rows matching every criterion given, in stream order."""
        wanted = None if params is None else _params(params)
        return [
            row
            for row in self.rows
            if (wanted is None or row.params == wanted) and (setpoint is None or row.setpoint == setpoint)
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
        if str(meta.get("pix_fmt")) != CAPTURE_PIX_FMT:
            return None
        if int(meta.get("schema_version", 0)) != ARCHIVE_SCHEMA_VERSION:
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

    def scratch(self, program: str) -> Path:
        """Where a recording is made so that committing it is a rename, not a copy."""
        self.directory.mkdir(parents=True, exist_ok=True)
        return temp_sibling(self.paths(program)[0])

    def commit(
        self,
        program: str,
        key: ProgramKey,
        video: str | os.PathLike[str],
        rows: Sequence[FrameRow],
        *,
        width: int,
        height: int,
        fps: float = ARCHIVE_FPS,
    ) -> dict[str, Any]:
        """Publish a recording as this program's archive: video, sidecar, then marker.

        The metadata is written last and is the commit marker, as in the result
        store: a run killed mid-recording leaves at most an uncommitted temp file.
        """
        if not rows:
            raise ArchiveError(f"no frames archived for {program!r}")
        if len(rows) != len({row.frame for row in rows}):
            raise ArchiveError(f"{program!r} has repeated frame indices in its sidecar")
        for row in rows:
            _params(row.params)
        target, data, meta_path = self.paths(program)
        self.directory.mkdir(parents=True, exist_ok=True)
        os.replace(Path(video), target)
        atomic_write(data, lambda tmp: save_measurements(rows, tmp))
        meta = {
            "schema_version": ARCHIVE_SCHEMA_VERSION,
            "program": program,
            "key_kind": key.kind.value,
            "key_material": key.material,
            "key_digest": key.digest,
            "codec": "ffv1",
            "container": CONTAINER,
            "pix_fmt": CAPTURE_PIX_FMT,
            "stored_pix_fmt": STORED_PIX_FMT,
            "width": int(width),
            "height": int(height),
            "fps": float(fps),
            "frames": len(rows),
            "measured": sum(1 for row in rows if row.setpoint != GAP),
            "bytes_per_frame": target.stat().st_size / len(rows),
            "video": target.name,
            "data": data.name,
            "created": datetime.fromtimestamp(self.clock(), timezone.utc).isoformat(),
        }
        payload = json.dumps(meta, indent=2)
        atomic_write(meta_path, lambda tmp: tmp.write_text(payload, encoding="utf-8"))
        return meta

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

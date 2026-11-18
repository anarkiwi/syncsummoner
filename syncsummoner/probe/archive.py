"""Lossless raw-frame archive: intra-only FFV1 video plus a per-frame state sidecar.

Probing is expensive and the rig dies intermittently, so ffmpeg records the card
once per program and this stores that recording with the state each frame was
captured under. Keyed like the result store, so a reflash invalidates both.
"""

from __future__ import annotations

import errno
import json
import os
import shutil
import subprocess
import threading
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Iterable, Iterator, Mapping, Sequence

import numpy as np

from syncsummoner.device.profile import PARAM_COUNT
from syncsummoner.probe.fit import load_measurements, save_measurements
from syncsummoner.probe.store import ProgramKey, atomic_write, slug, temp_sibling

__all__ = [
    "ARCHIVE_SCHEMA_VERSION",
    "CODEFRAMES",
    "DARK_SUFFIX",
    "DECODERS",
    "GAP",
    "MJPEG_PIX_FMT",
    "PROGRAM_INFO",
    "READABLE_SCHEMAS",
    "STIMULUS",
    "ArchiveError",
    "FrameArchive",
    "FrameReader",
    "FrameRow",
    "container_duration",
    "is_footage",
    "link",
    "listing",
    "same_stimulus",
    "stimulus_of",
]

#: 4 names its stimulus (3 played the codeframe loop); 3 put the chroma pair back, 1 and 2 swapped it.
ARCHIVE_SCHEMA_VERSION = 4
READABLE_SCHEMAS = frozenset({3, ARCHIVE_SCHEMA_VERSION})
#: Metadata key naming what was played while an archive was recorded.
STIMULUS = "stimulus"
CODEFRAMES = {"kind": "codeframes"}
#: Fallback container rate; the capture time of every frame is in the sidecar.
ARCHIVE_FPS = 30
CONTAINER = "matroska"
VIDEO_SUFFIX = ".mkv"
#: A video of several parts, an ffconcat listing the concat demuxer reads as one stream.
LISTING_SUFFIX = ".ffconcat"
LISTING_HEADER = "ffconcat version 1.0"
#: Marks a program measured black on a healthy rig, so a resume does not re-probe it.
DARK_SUFFIX = ".dark.json"
#: Setpoint of a frame captured between setpoints, kept for continuity but not measured.
GAP = -1
#: What the card is asked for; the recorder swaps the chroma pair back before encoding.
CAPTURE_PIX_FMT = "yuyv422"
#: The card's own compressed stream, as a locked take records it.
MJPEG_PIX_FMT = "mjpeg"
#: Card formats an archive may hold.
CAPTURE_PIX_FMTS = frozenset({CAPTURE_PIX_FMT, MJPEG_PIX_FMT})
#: Metadata key holding the device's own ``program info`` reply for the archived program.
PROGRAM_INFO = "program_info"
#: Planar 4:2:2 is the same samples deinterleaved into the canonical plane order.
STORED_PIX_FMT = "yuv422p"
#: Read back converted, so nothing downstream repeats a 4:2:2 upsample in Python.
READ_PIX_FMT = "rgb24"
CHANNELS = 3
_PARAM_COLUMNS = tuple(f"p{i + 1}" for i in range(PARAM_COUNT))
#: Raw RGB on a pipe, one output frame per decoded frame whatever the timestamps say. A raw
#: encoder is a copy, so it runs on one thread rather than a pool per core per output.
_RAW_OUT = ("-threads", "1", "-fps_mode", "passthrough", "-f", "rawvideo", "-pix_fmt", READ_PIX_FMT)
#: Decoder input flags by backend. Vulkan decodes FFV1 in compute shaders and leaves each frame
#: on the GPU, so the graph downloads it first; a decoder that fell back to the CPU then fails
#: the download rather than passing for a GPU decode.
DECODERS = {
    "cpu": (),
    "vulkan": (
        "-init_hw_device",
        "vulkan=vk",
        "-hwaccel",
        "vulkan",
        "-hwaccel_device",
        "vk",
        "-hwaccel_output_format",
        "vulkan",
    ),
}
#: Frames the automatic choice decodes both ways before trusting the GPU with an archive.
PROBE_FRAMES = 8


def stimulus_of(meta: Mapping[str, Any]) -> dict[str, Any]:
    """What an archive was recorded against: its metadata's own record, else the codeframe loop."""
    return dict(meta.get(STIMULUS) or CODEFRAMES)


def is_footage(meta: Mapping[str, Any]) -> bool:
    """Whether an archive was recorded against a timecoded footage clip."""
    return stimulus_of(meta).get("kind") == "footage"


def same_stimulus(stored: Mapping[str, Any], asked: Mapping[str, Any]) -> bool:
    """Whether two stimulus records name the same picture: one kind, and for footage one clip digest."""
    fields = ("kind", "sha256") if asked.get("kind") == "footage" else ("kind",)
    return all(stored.get(name) == asked.get(name) for name in fields)


class ArchiveError(RuntimeError):
    """The archive could not be written, read, or was not encoded losslessly."""


def container_duration(path: str | Path, *, ffprobe: str = "ffprobe", run: Callable[..., Any]) -> str:
    """A part's container duration as ffprobe prints it, to the microsecond the concat demuxer keeps."""
    argv = [ffprobe, "-v", "error", "-show_entries", "format=duration", "-of", "default=nw=1:nk=1"]
    result = run([*argv, str(path)], capture_output=True, check=False)
    text = result.stdout.decode() if isinstance(result.stdout, bytes) else str(result.stdout or "")
    try:
        if not result.returncode and float(text) > 0:
            return text.strip()
    except ValueError:
        pass
    raise ArchiveError(f"no container duration for {path}: {_last_line(result.stderr)}")


def link(source: str | Path, target: Path) -> None:
    """Publish ``source`` at ``target`` as a hard link, copying it only across filesystems."""
    tmp = temp_sibling(target)
    tmp.unlink(missing_ok=True)
    try:
        try:
            os.link(source, tmp)
        except OSError as err:
            if err.errno != errno.EXDEV:
                raise
            shutil.copy2(source, tmp)
        os.replace(tmp, target)
    finally:
        tmp.unlink(missing_ok=True)


def listing(parts: Sequence[str], durations: Sequence[str]) -> str:
    """An ffconcat listing of ``parts`` by relative name, each with its duration so the whole is seekable."""
    body = "".join(f"file '{name}'\nduration {span}\n" for name, span in zip(parts, durations))
    return f"{LISTING_HEADER}\n{body}"


@dataclass(frozen=True)
class FrameRow:
    """One archived frame: where it sits in the stream, and what produced it.

    ``setpoint`` is :data:`GAP` for a frame captured while the sweep was moving
    between setpoints; ``captured`` is the card's monotonic capture time, ``shown`` the frame of
    the stimulus clip it shows when a locked take attributed it, else :data:`GAP`; ``ambiguity`` how many
    frames earlier the take's capture clock also admits it shows.
    """

    frame: int
    program: str
    params: tuple[int, ...]
    setpoint: int
    captured: float = 0.0
    shown: int = GAP
    ambiguity: int = 0

    def as_row(self) -> dict[str, Any]:
        """Flatten to a Parquet-friendly row, sharing ``MeasurementRecord`` param naming."""
        row: dict[str, Any] = {
            "frame": self.frame,
            "program": self.program,
            "setpoint": self.setpoint,
            "captured": self.captured,
            "shown": self.shown,
            "ambiguity": self.ambiguity,
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
            shown=GAP if row.get("shown") is None else int(row["shown"]),
            ambiguity=int(row.get("ambiguity") or 0),
        )


def _params(vector: Sequence[int]) -> tuple[int, ...]:
    """Validate and normalise a raw parameter vector."""
    values = tuple(int(v) for v in vector)
    if len(values) != PARAM_COUNT:
        raise ArchiveError(f"parameter vector has {len(values)} values, expected {PARAM_COUNT}")
    return values


def _drain(pipe: Any, shape: tuple, positions: Sequence[int], into: dict) -> None:
    """Read the frame at each of ``positions`` off ``pipe`` into ``into``, stopping if it ends."""
    size = int(np.prod(shape))
    for position in positions:
        buf = pipe.read(size)
        if len(buf) < size:
            return
        into[position] = np.frombuffer(bytearray(buf), dtype=np.uint8).reshape(shape)


def _drain_all(pipe: Any, shape: tuple, into: list) -> None:
    """Read every frame of ``shape`` off ``pipe`` into ``into``, until it ends."""
    size = int(np.prod(shape))
    while len(buf := pipe.read(size)) == size:
        into.append(np.frombuffer(bytearray(buf), dtype=np.uint8).reshape(shape))


def _drained(fd: int, drain: Callable[[Any], None]) -> None:
    """Run ``drain`` on the read end of a pipe, closing it after."""
    with os.fdopen(fd, "rb") as pipe:
        drain(pipe)


def _last_line(stderr: bytes | str | None) -> str:
    """The last thing a decoder said, which is where it says why it stopped."""
    text = stderr.decode(errors="replace") if isinstance(stderr, bytes) else stderr or ""
    return (text.strip().splitlines() or ["no message"])[-1]


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
        threads: int | None = None,
        decode: str = "cpu",
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
        self._threads = (
            []
            if threads is None
            else [
                flag
                for name in ("-threads", "-filter_threads", "-filter_complex_threads")
                for flag in (name, str(int(threads)))
            ]
        )
        self._hwaccel = DECODERS[decode]
        self._download = (
            f"hwdownload,format={self.meta.get('stored_pix_fmt', STORED_PIX_FMT)}," if self._hwaccel else ""
        )

    @property
    def count(self) -> int:
        """Frames in the stream."""
        return int(self.meta["frames"])

    def shape(self, width: int | None = None, strip_px: int = 0) -> tuple[int, int, int]:
        """Shape frames come back in, at full size or scaled to ``width``, less ``strip_px`` native rows."""
        height = self.height - int(strip_px)
        if not width or width >= self.width:
            return (height, self.width, CHANNELS)
        return (2 * round(height * width / self.width / 2), int(width), CHANNELS)

    def command(self, *, start: int = 0, count: int = 1, width: int | None = None) -> list[str]:
        """Decoder argv emitting ``count`` frames from ``start`` as RGB on stdout.

        Every frame is a keyframe, so the input seek lands on one directly rather
        than decoding from the beginning, and ffmpeg does any scaling asked for.
        """
        argv = [
            *self._head(),
            "-ss",
            f"{max(0.0, start - 0.5) / self.fps:.6f}",
            "-i",
            str(self.video),
            "-frames:v",
            str(int(count)),
        ]
        if self._download or (width and width < self.width):
            argv += ["-vf", self._download + self._scale(width)]
        return argv + [*_RAW_OUT, "pipe:1"]

    def _head(self) -> list[str]:
        """Decoder argv up to the input's own options."""
        return [self._ffmpeg, "-loglevel", "error", *self._threads, *self._hwaccel]

    def _scale(self, width: int | None) -> str:
        """Filter taking a native frame to ``width``, the one :meth:`command` asks for."""
        return f"scale={int(width)}:-2" if width and width < self.width else "null"

    def passes(
        self,
        *,
        width: int | None = None,
        native: Iterable[int] = (),
        into: dict | None = None,
        strip_px: int = 0,
        strips: list | None = None,
    ) -> Iterator[np.ndarray]:
        """Yield every frame at ``width`` in stream order, keeping those at ``native`` full size.

        One decoder feeds every output through a split filter graph, so the archive is read once;
        each side output is drained by its own thread, a decoder writing several pipes stalling on
        whichever is not read. ``strip_px`` crops the strip off what is yielded; ``strips`` keeps it.
        """
        keep = sorted({int(i) for i in native if 0 <= int(i) < self.count})
        crop = int(strip_px) if strip_px and strip_px > 0 else 0
        sides = []
        if keep:
            picked = "+".join(f"eq(n\\,{i})" for i in keep)
            held = {} if into is None else into
            sides.append((f"select='{picked}'", lambda pipe: _drain(pipe, self.shape(), keep, held)))
        if crop and strips is not None:
            shape = (crop, self.width, CHANNELS)
            sides.append((f"crop=iw:{crop}:0:ih-{crop}", lambda pipe: _drain_all(pipe, shape, strips)))
        body = (f"crop=iw:ih-{crop}:0:0," if crop else "") + self._scale(width)
        heads = "".join(f"[o{i}]" for i in range(len(sides) + 1))
        graph = (
            f"[0:v]{self._download}" + (f"split={len(sides) + 1}{heads};[o0]" if sides else "") + f"{body}[s]"
        )
        graph += "".join(f";[o{i + 1}]{chain}[x{i}]" for i, (chain, _) in enumerate(sides))
        argv = [
            *self._head(),
            "-i",
            str(self.video),
            "-filter_complex",
            graph,
            "-map",
            "[s]",
            *_RAW_OUT,
            "pipe:1",
        ]
        fds, threads = [], []
        for i, (_, drain) in enumerate(sides):
            read_fd, write_fd = os.pipe()
            argv += ["-map", f"[x{i}]", *_RAW_OUT, f"pipe:{write_fd}"]
            fds.append(write_fd)
            threads.append(threading.Thread(target=_drained, args=(read_fd, drain), daemon=True))
        try:
            yield from self._frames(
                argv,
                self.shape(width, crop),
                self.count,
                tuple(fds),
                started=lambda: [t.start() for t in threads],
            )
        finally:
            for thread in threads:
                if thread.is_alive():
                    thread.join()

    def _frames(
        self,
        argv: list[str],
        shape: tuple,
        want: int,
        fds: tuple,
        *,
        started: Callable[[], None] = lambda: None,
    ) -> Iterator[np.ndarray]:
        """Frames of ``shape`` from a decoder's stdout, ending early if it does."""
        size = int(np.prod(shape))
        proc = self._popen(argv, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, pass_fds=fds)
        for fd in fds:
            os.close(fd)
        started()
        try:
            for _ in range(want):
                buf = proc.stdout.read(size)
                if buf is None or len(buf) < size:
                    return
                yield np.frombuffer(bytearray(buf), dtype=np.uint8).reshape(shape)
        finally:
            proc.stdout.close()
            proc.wait()

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
        yield from self._frames(
            self.command(start=start, count=want, width=width), self.shape(width), want, ()
        )

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
        ffprobe: str = "ffprobe",
        threads: int | None = None,
        decode: str = "cpu",
    ):
        self.directory = Path(directory)
        self.clock = clock
        self.popen = popen
        self.run = run
        self.ffmpeg = ffmpeg
        self.ffprobe = ffprobe
        self.threads = threads
        self.decode = decode

    def paths(self, program: str) -> tuple[Path, Path, Path]:
        """Single-file video, sidecar and metadata paths for one program."""
        stem = slug(program)
        return (
            self.directory / f"{stem}{VIDEO_SUFFIX}",
            self.directory / f"{stem}.parquet",
            self.directory / f"{stem}.json",
        )

    def files(self, program: str, meta: Mapping[str, Any]) -> list[Path]:
        """The video an archive's metadata names, a file or a listing, then any parts it lists."""
        names = [meta.get("video") or self.paths(program)[0].name, *meta.get("parts", ())]
        return [self.directory / str(name) for name in names]

    def meta(self, program: str) -> dict[str, Any] | None:
        """Committed metadata for ``program``, or None when nothing usable is stored."""
        _, data, meta_path = self.paths(program)
        try:
            meta = json.loads(meta_path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            return None
        if str(meta.get("pix_fmt")) not in CAPTURE_PIX_FMTS:
            return None
        if int(meta.get("schema_version", 0)) not in READABLE_SCHEMAS:
            return None
        return meta if data.exists() and all(path.exists() for path in self.files(program, meta)) else None

    def has(self, program: str, key: ProgramKey, stimulus: Mapping[str, Any] | None = None) -> bool:
        """True when a committed archive for ``program`` was keyed on ``key``.

        With ``stimulus`` it must also have been recorded against the same one:
        the same kind and, for footage, the same clip by digest.
        """
        meta = self.meta(program)
        if meta is None or meta.get("key_digest") != key.digest:
            return False
        return stimulus is None or same_stimulus(stimulus_of(meta), stimulus)

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

    def clear_dark(self, program: str) -> None:
        """Drop any dark verdict for ``program``, so a resume runs it again."""
        self.dark_path(program).unlink(missing_ok=True)

    def dark(self, program: str, key: ProgramKey) -> bool:
        """True when ``program`` was measured dark under ``key``, so it need not run again."""
        try:
            verdict = json.loads(self.dark_path(program).read_text(encoding="utf-8"))
        except (OSError, ValueError):
            return False
        return str(verdict.get("key_digest")) == key.digest

    def darkened(self) -> dict[str, dict[str, Any]]:
        """Every recorded dark verdict, keyed by program name."""
        found = {}
        for path in sorted(self.directory.glob(f"*{DARK_SUFFIX}")):
            try:
                verdict = json.loads(path.read_text(encoding="utf-8"))
            except (OSError, ValueError):
                continue
            found[str(verdict.get("program", path.name[: -len(DARK_SUFFIX)]))] = verdict
        return found

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
        """Where a recording is made so that committing it is a rename, not a copy.

        It keeps the video suffix because the recorder infers the container from
        it, and hides behind a dot so an uncommitted one is never mistaken for an
        archive.
        """
        self.directory.mkdir(parents=True, exist_ok=True)
        return self.directory / f".{slug(program)}.recording{VIDEO_SUFFIX}"

    def commit(
        self,
        program: str,
        key: ProgramKey,
        video: str | os.PathLike[str] | Sequence[str | os.PathLike[str]],
        rows: Sequence[FrameRow],
        *,
        width: int,
        height: int,
        fps: float = ARCHIVE_FPS,
        program_info: Mapping[str, Any] | None = None,
        stimulus: Mapping[str, Any] | None = None,
        schedule: Sequence[Mapping[str, Any]] | None = None,
        pix_fmt: str = CAPTURE_PIX_FMT,
    ) -> dict[str, Any]:
        """Publish a recording as this program's archive: video, sidecar, then marker.

        ``video`` is one recording, moved in, or a sequence of them, linked in as parts beside an
        ffconcat listing of them so nothing is re-written. Any earlier marker goes first and the
        metadata is written last as the commit marker, as in the result store: a run killed
        mid-commit leaves no archive rather than a mismatched one, and files only an earlier commit
        named go once the new one is published.
        ``program_info`` is the device's reply for the program, kept so a refit
        knows each parameter's name and native range, and ``stimulus`` names what was
        played; ``schedule`` lists each setpoint with the full state it was stepped from, and
        ``pix_fmt`` names the card format recorded, raw (FFV1) or the card's MJPEG.
        A committed archive supersedes any earlier dark verdict for the program.
        """
        if not rows:
            raise ArchiveError(f"no frames archived for {program!r}")
        if len(rows) != len({row.frame for row in rows}):
            raise ArchiveError(f"{program!r} has repeated frame indices in its sidecar")
        for row in rows:
            _params(row.params)
        target, data, meta_path = self.paths(program)
        sources = None if isinstance(video, (str, os.PathLike)) else [Path(v) for v in video]
        spans = [container_duration(v, ffprobe=self.ffprobe, run=self.run) for v in sources or ()]
        self.directory.mkdir(parents=True, exist_ok=True)
        previous = self.meta(program)
        stale = set(self.files(program, previous)) if previous else set()
        meta_path.unlink(missing_ok=True)
        if sources is None:
            os.replace(Path(video), target)
            parts = [target]
        else:
            parts = [
                target.with_name(f"{target.stem}.part{i:02d}{VIDEO_SUFFIX}") for i in range(len(sources))
            ]
            for source, part in zip(sources, parts):
                link(source, part)
            target = target.with_suffix(LISTING_SUFFIX)
            text = listing([p.name for p in parts], spans)
            atomic_write(target, lambda tmp: tmp.write_text(text, encoding="utf-8"))
        atomic_write(data, lambda tmp: save_measurements(rows, tmp))
        meta = {
            "schema_version": ARCHIVE_SCHEMA_VERSION,
            "program": program,
            "key_kind": key.kind.value,
            "key_material": key.material,
            "key_digest": key.digest,
            "codec": "mjpeg" if pix_fmt == MJPEG_PIX_FMT else "ffv1",
            "container": CONTAINER,
            "pix_fmt": pix_fmt,
            "stored_pix_fmt": STORED_PIX_FMT,
            "width": int(width),
            "height": int(height),
            "fps": float(fps),
            "frames": len(rows),
            "measured": sum(1 for row in rows if row.setpoint != GAP),
            "bytes_per_frame": sum(p.stat().st_size for p in parts) / len(rows),
            "video": target.name,
            "data": data.name,
            "created": datetime.fromtimestamp(self.clock(), timezone.utc).isoformat(),
        }
        if program_info is not None:
            meta[PROGRAM_INFO] = dict(program_info)
        if stimulus is not None:
            meta[STIMULUS] = dict(stimulus)
        if schedule is not None:
            meta["schedule"] = [dict(entry) for entry in schedule]
        if sources is not None:
            meta["parts"] = [p.name for p in parts]
        self._write_meta(meta_path, meta)
        for path in stale - set(self.files(program, meta)):
            path.unlink(missing_ok=True)
        self.dark_path(program).unlink(missing_ok=True)
        return meta

    def annotate(self, program: str, **fields: Any) -> dict[str, Any]:
        """Merge ``fields`` into a committed archive's metadata, leaving its frames alone."""
        meta = self.meta(program)
        if meta is None:
            raise ArchiveError(f"no committed archive for {program!r}")
        meta.update(fields)
        self._write_meta(self.paths(program)[2], meta)
        return meta

    @staticmethod
    def _write_meta(path: Path, meta: Mapping[str, Any]) -> None:
        payload = json.dumps(meta, indent=2)
        atomic_write(path, lambda tmp: tmp.write_text(payload, encoding="utf-8"))

    def reader(
        self, program: str, key: ProgramKey | None = None, *, decode: str | None = None
    ) -> FrameReader | None:
        """Reader for a committed archive, or None when absent or keyed differently."""
        meta = self.meta(program)
        if meta is None or (key is not None and meta.get("key_digest") != key.digest):
            return None
        return FrameReader(
            self.files(program, meta)[0],
            meta,
            self.rows(program),
            ffmpeg=self.ffmpeg,
            run=self.run,
            popen=self.popen,
            threads=self.threads,
            decode=self.decode if decode is None else decode,
        )

    def choose_decode(
        self,
        choice: str = "auto",
        programs: Iterable[str] | None = None,
        *,
        timer: Callable[[], float] = time.monotonic,
    ) -> tuple[str, str]:
        """The backend to decode with and why, without setting it.

        ``auto`` takes Vulkan only if it decodes the first readable program's opening
        frames byte for byte as the CPU does and at least as fast per stream: a GPU
        slower per stream cannot win with one stream per worker. Asking for Vulkan
        outright takes it whatever its pace, and raises where ``auto`` falls back.
        """
        if choice == "cpu":
            return "cpu", "asked for"
        names = sorted(self.committed()) if programs is None else list(programs)
        name = next((name for name in names if self.meta(name) is not None), None)
        if name is None:
            why = "no committed archive to compare it on"
        else:
            (cpu, cpu_rate), (gpu, gpu_rate) = (self._probe(name, d, timer) for d in ("cpu", "vulkan"))
            if cpu.returncode or not cpu.stdout:
                raise ArchiveError(f"cannot decode {name} on the CPU: {_last_line(cpu.stderr)}")
            if gpu.returncode:
                why = f"Vulkan decode failed: {_last_line(gpu.stderr)}"
            elif gpu.stdout != cpu.stdout:
                why = f"Vulkan decode of {name} differs from the CPU's"
            else:
                pace = f"{gpu_rate:.1f} frames/s per stream against the CPU's {cpu_rate:.1f}"
                if choice == "vulkan" or gpu_rate >= cpu_rate:
                    return "vulkan", f"bit-identical to the CPU on {name}, {pace}"
                return "cpu", f"Vulkan is bit-identical on {name} but slower, {pace}"
        if choice == "vulkan":
            raise ArchiveError(why)
        return "cpu", why

    def _probe(self, name: str, decode: str, timer: Callable[[], float]) -> tuple[Any, float]:
        """A backend's decode of the opening frames, and its rate with start-up cancelled out.

        One frame and then :data:`PROBE_FRAMES` are decoded, and the rate is taken
        from the difference, since device and decoder set-up dwarf a few frames.
        """
        reader = self.reader(name, decode=decode)
        results, spans = [], []
        for count in (1, PROBE_FRAMES):
            start = timer()
            results.append(self.run(reader.command(count=count), capture_output=True, check=False))
            spans.append(timer() - start)
        failed = next((result for result in results if result.returncode), results[-1])
        return failed, (PROBE_FRAMES - 1) / max(spans[1] - spans[0], 1e-9)

"""Every recording decoded once: per capture its time, strip code and the small pictures judging reads.

Keyframe-cut segments decode side by side, each seeked exactly, so the joined stream is one
decode's; the result is kept beside the recording, keyed by the SHA-256 of its bytes and frame
log, :data:`SCHEMA` and the :class:`Spec` it was decoded to (docs/judging.md).
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import subprocess
import threading
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict, dataclass
from fractions import Fraction
from pathlib import Path
from typing import Any, Callable, Mapping

import numpy as np

from syncsummoner.device.recorder import INSPECT_SIZE
from syncsummoner.device.stored import Stored, captures
from syncsummoner.device.v4l2rec import read_log, sidecar
from syncsummoner.probe.patterns import read_codes
from syncsummoner.progress import LOG, meter

__all__ = [
    "CALIBRATION_SIZE",
    "Features",
    "SCHEMA",
    "Spec",
    "cache_path",
    "calibration_colour",
    "decode",
    "features",
    "segment",
]

#: Bumped whenever what a decode stores, or how, changes; older caches are then decoded afresh.
SCHEMA = 1
#: Size the lead-in's pictures are compared at to see where a calibration write landed, in planar YUV.
CALIBRATION_SIZE = (96, 54)
#: Captures read into one strip-reading block.
CHUNK = 512
#: Bytes hashed per read.
HASH_BLOCK = 1 << 24

_PTS = re.compile(rb"pts_time:\s*(-?[0-9.]+)")
_COLOUR = re.compile(rb"color_range:(\w+) color_space:(\w+)")


@dataclass(frozen=True)
class Spec:
    """What each capture is decoded to: the strip read at full width, the body area-scaled twice."""

    strip_px: int = 8
    bits: int = 16
    survival: tuple[int, int] = INSPECT_SIZE
    calibration: tuple[int, int] = CALIBRATION_SIZE

    @classmethod
    def of(cls, config: Any) -> "Spec":
        """The spec a render config's strip calls for."""
        return cls(strip_px=int(config.strip_px), bits=int(config.bits))

    def graph(self) -> str:
        """One filter graph: survival-size luma, calibration-size YUV and the full-width strip.

        Every branch converts its own crop, so no branch's format is negotiated up to the whole frame;
        ``showinfo`` checksums the small branch, the cheapest to time each frame by.
        """
        (sw, sh), px = self.survival, self.strip_px
        return (
            "[0:v]split=3[a][b][c];"
            f"[a]crop=iw:ih-{px}:0:0,scale={sw}:{sh}:flags=area,format=gray[s];"
            f"[b]{self.colour()}[e];"
            f"[c]crop=iw:{px}:0:ih-{px},scale,format=rgb24[t]"
        )

    def colour(self) -> str:
        """The calibration branch of :meth:`graph`; its ``showinfo`` gives each picture's range and matrix."""
        (cw, ch), px = self.calibration, self.strip_px
        return f"crop=iw:ih-{px}:0:0,scale={cw}:{ch}:flags=area,format=yuv444p,showinfo"

    def token(self) -> dict:
        """The spec as stored with a cache."""
        fields = {k: list(v) if isinstance(v, tuple) else v for k, v in asdict(self).items()}
        return {"schema": SCHEMA, **fields}


class Features:
    """A recording's per-capture features, read lazily from a cache or held from a fresh decode.

    ``times`` are presentation seconds, ``codes`` strip codes (-1 unreadable), ``luma`` survival-size
    luma ``(n, h, w)``, ``small`` calibration-size YUV ``(n, 3, h, w)``, ``log`` the V4L2 frame log or None.
    """

    def __init__(self, arrays: Mapping[str, np.ndarray], key: str):
        self._arrays, self.key, self._got = arrays, key, {}

    def _get(self, name: str) -> np.ndarray | None:
        if name not in self._got:
            self._got[name] = self._arrays[name] if name in self._arrays else None
        return self._got[name]

    @property
    def times(self) -> np.ndarray:
        """Each capture's presentation time, in seconds."""
        return self._get("times")

    @property
    def codes(self) -> np.ndarray:
        """Each capture's strip code, -1 where unreadable."""
        return self._get("codes")

    @property
    def luma(self) -> np.ndarray:
        """Each capture's survival-size luma."""
        return self._get("luma")

    @property
    def small(self) -> np.ndarray:
        """Each capture's calibration-size planar YUV."""
        return self._get("small")

    @property
    def log(self) -> np.ndarray | None:
        """The recording's V4L2 frame log, as it stood when decoded."""
        return self._get("log")

    def __len__(self) -> int:
        return int(self.codes.size)


def cache_path(path: str | Path) -> Path:
    """Where a recording's features are kept."""
    return Path(f"{path}.features.npz")


def content_key(path: str | Path) -> str:
    """SHA-256 of a recording's bytes, then of its frame log's when it has one."""
    digest = hashlib.sha256()
    for part in (Path(path), sidecar(path)):
        if part.is_file():
            with open(part, "rb") as handle:
                while block := handle.read(HASH_BLOCK):
                    digest.update(block)
    return digest.hexdigest()


def _stat(path: str | Path) -> list[int]:
    found = [Path(path).stat()] + ([sidecar(path).stat()] if sidecar(path).is_file() else [])
    return [v for s in found for v in (s.st_size, s.st_mtime_ns)]


def ffprobe_for(ffmpeg: str) -> str:
    """The ffprobe installed beside an ffmpeg."""
    head, slash, tail = ffmpeg.rpartition("/")
    return head + slash + tail.replace("ffmpeg", "ffprobe")


def packets(path: str | Path, *, ffmpeg: str = "ffmpeg") -> tuple[np.ndarray, np.ndarray, Fraction, int]:
    """Video packet times in presentation order, which are keyframes, the time base and the frame width."""
    probe = [ffprobe_for(ffmpeg), "-v", "error", "-select_streams", "v:0"]
    stream = subprocess.run(
        probe + ["-show_entries", "stream=time_base,width", "-of", "json", str(path)],
        check=True,
        capture_output=True,
        text=True,
    )
    info = json.loads(stream.stdout)["streams"][0]
    listed = subprocess.run(
        probe + ["-show_entries", "packet=pts,flags", "-of", "csv=p=0", str(path)],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.split()
    rows = [line.split(",") for line in listed if not line.startswith("N/A")]
    pts = np.array([int(r[0]) for r in rows], dtype=np.int64)
    key = np.array(["K" in r[1] for r in rows], dtype=bool)
    order = np.argsort(pts, kind="stable")
    return pts[order], key[order], Fraction(info["time_base"]), int(info["width"])


def segments(key: np.ndarray, parts: int) -> np.ndarray:
    """Start indices of up to ``parts`` segments of about equal length, each on a keyframe, then the end."""
    starts = np.r_[0, np.flatnonzero(key)]
    want = np.linspace(0, key.size, max(1, parts) + 1)[:-1]
    picked = starts[np.clip(np.searchsorted(starts, want), 0, starts.size - 1)]
    return np.r_[np.unique(picked), key.size]


def _drain(pipe, into: list[bytes]) -> None:
    with pipe:
        while chunk := pipe.read(1 << 20):
            into.append(chunk)


def _run_segment(
    source: list[str],
    spec: Spec,
    *,
    count: int,
    width: int,
    ffmpeg: str,
    done: Callable[[int], None],
    copyts: bool,
) -> tuple[list[bytes], bytes, list[bytes], list[np.ndarray], int]:
    """Decode ``count`` captures from ``source``: ffmpeg's log, the two pictures, strip codes, exit status."""
    argv = [ffmpeg, "-hide_banner", "-nostdin", "-nostats", "-loglevel", "info", "-threads", "1"]
    argv += [
        *(["-copyts"] if copyts else []),
        *source,
        "-filter_threads",
        "1",
        "-filter_complex",
        spec.graph(),
    ]
    small_r, small_w = os.pipe()
    strip_r, strip_w = os.pipe()
    for label, target in (("s", "pipe:1"), ("e", f"pipe:{small_w}"), ("t", f"pipe:{strip_w}")):
        argv += ["-map", f"[{label}]", "-fps_mode", "passthrough", "-frames:v", str(count)]
        argv += ["-f", "rawvideo", target]
    proc = subprocess.Popen(  # pylint: disable=consider-using-with
        argv, stdout=subprocess.PIPE, stderr=subprocess.PIPE, pass_fds=(small_w, strip_w)
    )
    os.close(small_w)
    os.close(strip_w)
    px = spec.strip_px
    stride = px * width * 3
    small: list[bytes] = []
    said: list[bytes] = []
    codes: list[np.ndarray] = []

    def strips() -> None:
        with os.fdopen(strip_r, "rb") as pipe:
            while block := pipe.read(stride * CHUNK):
                rows = np.frombuffer(block, np.uint8, len(block) - len(block) % stride)
                rows = rows.reshape(-1, px, width, 3)
                codes.append(read_codes(rows / np.float32(255.0), bits=spec.bits))
                done(rows.shape[0])

    readers = [
        threading.Thread(target=_drain, args=(os.fdopen(small_r, "rb"), small), daemon=True),
        threading.Thread(target=strips, daemon=True),
        threading.Thread(target=lambda: said.extend(proc.stderr), daemon=True),
    ]
    for reader in readers:
        reader.start()
    body = proc.stdout.read()
    proc.wait()
    for reader in readers:
        reader.join()
    return said, body, small, codes, proc.returncode


def segment(
    path: str | Path,
    spec: Spec,
    *,
    start_s: Fraction | None,
    count: int,
    width: int,
    ffmpeg: str,
    done: Callable[[int], None],
) -> dict[str, np.ndarray]:
    """One segment's ``count`` captures from ``start_s`` on: times, codes, luma and calibration YUV."""
    with captures(path, start_s=start_s, ffmpeg=ffmpeg) as source:
        said, body, small, codes, returncode = _run_segment(
            source, spec, count=count, width=width, ffmpeg=ffmpeg, done=done, copyts=start_s is not None
        )
    (sw, sh), (cw, ch) = spec.survival, spec.calibration
    if returncode:
        tail = b"".join(said).decode("utf-8", "replace").strip().splitlines()
        raise RuntimeError(f"decoding {path} failed with exit {returncode}: {tail[-1] if tail else ''}")
    times = np.array([float(m.group(1)) for line in said for m in [_PTS.search(line)] if m], dtype=np.float64)
    code = np.concatenate(codes) if codes else np.zeros(0, dtype=np.int64)
    blob = b"".join(small)
    n = min(times.size, code.size, len(body) // (sw * sh), len(blob) // (3 * cw * ch))
    return {
        "times": times[:n],
        "codes": code[:n].astype(np.int64),
        "luma": np.frombuffer(body, np.uint8, n * sw * sh).reshape(n, sh, sw),
        "small": np.frombuffer(blob, np.uint8, n * 3 * cw * ch).reshape(n, 3, ch, cw),
    }


def decode(
    path: str | Path, spec: Spec = Spec(), *, workers: int | None = None, ffmpeg: str = "ffmpeg"
) -> dict[str, np.ndarray]:
    """Decode a recording to its per-capture features, ``workers`` segments at once.

    A segment keeps only the captures timed before the next one's start, so none is handed twice;
    joined times other than the demuxer's packet times, one for one, are said.
    """
    workers = workers or len(os.sched_getaffinity(0))
    pts, key, base, width = packets(path, ffmpeg=ffmpeg)
    stored = Stored.of(path)
    if stored is not None:
        shown = stored.index[stored.index < key.size]
        pts, key, base = stored.rows["pts"][: shown.size], key[shown], stored.layout.tb
    if not pts.size:
        raise RuntimeError(f"{path} holds no video packets")
    bounds = segments(key, workers)
    edges = [pts[i] * base for i in bounds[:-1]]
    lock, total, span = (
        threading.Lock(),
        [0],
        float(pts.size / max(1, pts.size - 1) * (pts[-1] - pts[0]) * base),
    )
    with meter(span, desc=f"decoding {Path(path).name}") as progress:

        def done(n: int) -> None:
            with lock:
                total[0] += n
                progress.to(total[0] * span / pts.size)

        def one(index: int) -> dict[str, np.ndarray]:
            found = segment(
                path,
                spec,
                start_s=edges[index] if index else None,
                count=int(bounds[index + 1] - bounds[index]),
                width=width,
                ffmpeg=ffmpeg,
                done=done,
            )
            if index + 1 == len(edges):
                return found
            keep = found["times"] < float(edges[index + 1]) - 0.5 * float(base)
            return {k: v[keep] for k, v in found.items()}

        with ThreadPoolExecutor(max_workers=len(edges)) as pool:
            parts = list(pool.map(one, range(len(edges))))
    out = {k: np.concatenate([p[k] for p in parts]) for k in parts[0]}
    expected = pts * float(base)
    if out["times"].size != expected.size or not np.allclose(out["times"], expected, atol=0.5 * float(base)):
        LOG.warning("%s: %d frames decoded from %d packets", path, out["times"].size, pts.size)
    if sidecar(path).is_file():
        out["log"] = read_log(path)
    return out


def calibration_colour(path: str | Path, spec: Spec = Spec(), *, ffmpeg: str = "ffmpeg") -> tuple[bool, str]:
    """Whether the calibration YUV :func:`decode` makes of ``path`` is full range, and its matrix tag.

    Read from the first picture through the same branch, as ffmpeg converts it: a YUV stream keeps its own
    range, and RGB is scaled to studio swing.
    """
    with captures(path, ffmpeg=ffmpeg) as source:
        argv = [ffmpeg, "-hide_banner", "-nostdin", "-nostats", "-loglevel", "info", *source]
        done = subprocess.run(
            [*argv, "-vf", spec.colour(), "-frames:v", "1", "-f", "null", "-"],
            check=False,
            capture_output=True,
        )
    found = _COLOUR.search(done.stderr)
    if done.returncode or found is None:
        raise RuntimeError(
            f"cannot read the colour of {path}: {done.stderr.decode('utf-8', 'replace')[-200:]}"
        )
    return found.group(1) == b"pc", found.group(2).decode()


def _cached(path: str | Path, spec: Spec) -> Features | None:
    """The cache beside ``path`` if it was decoded to ``spec`` from the bytes there now."""
    at = cache_path(path)
    if not at.is_file():
        return None
    try:
        stored = np.load(at)
        meta = json.loads(str(stored["meta"]))
    except (OSError, ValueError, KeyError) as err:
        LOG.warning("%s: unreadable feature cache, decoding afresh: %s", at, err)
        return None
    if meta.get("spec") != spec.token():
        return None
    if meta.get("stat") != _stat(path) and meta.get("key") != content_key(path):
        return None
    return Features(stored, meta["key"])


def features(
    path: str | Path,
    spec: Spec = Spec(),
    *,
    workers: int | None = None,
    ffmpeg: str = "ffmpeg",
    store: bool = True,
) -> Features:
    """A recording's :class:`Features`, decoded once and then read from the cache beside it.

    The content hash is taken while the decode runs, and the stat it was taken at lets an untouched
    recording skip rehashing. A cache that cannot be written is said and skipped.
    """
    found = _cached(path, spec)
    if found is not None:
        LOG.info("%s: features from %s", path, cache_path(path))
        return found
    stat = _stat(path)
    with ThreadPoolExecutor(max_workers=1) as pool:
        hashing = pool.submit(content_key, path)
        arrays = decode(path, spec, workers=workers, ffmpeg=ffmpeg)
        key = hashing.result()
    if store:
        _store(path, arrays, {"spec": spec.token(), "key": key, "stat": stat})
    return Features(arrays, key)


def _store(path: str | Path, arrays: Mapping[str, np.ndarray], meta: dict) -> None:
    at = cache_path(path)
    partial = at.with_name(f"{at.name}.{os.getpid()}.{threading.get_ident()}.partial")
    try:
        with open(partial, "wb") as handle:
            np.savez(handle, meta=np.array(json.dumps(meta)), **arrays)
        os.replace(partial, at)
    except OSError as err:
        partial.unlink(missing_ok=True)
        LOG.warning("%s: features not cached: %s", path, err)

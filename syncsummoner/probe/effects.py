"""``probe effects``: each archived setting's intensity and safety, captures against the clip frames shown.

Captures and clip frames are compared at SSIM's own viewing scale (:func:`factor`); each setting's profile
entry is keyed by whether its clip was black and white or colour, so one profile carries both.
"""

# pylint: disable=no-member  ; cv2 is a compiled extension pylint cannot introspect

from __future__ import annotations

import argparse
import json
import logging
import multiprocessing
import os
import subprocess
from concurrent.futures import Executor, ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Callable, Iterable, Iterator, Sequence

import cv2
import numpy as np
from threadpoolctl import threadpool_limits

from syncsummoner import aesthetics
from syncsummoner.device.stored import captures
from syncsummoner.probe.archive import GAP, FrameArchive, stimulus_of
from syncsummoner.probe.store import slug
from syncsummoner.progress import LOG, configure, track

__all__ = [
    "BW",
    "COLOUR",
    "clip_frames",
    "clip_kind",
    "decode",
    "factor",
    "lookup",
    "measure",
    "program_effects",
    "sheet",
    "summarise",
    "thumb",
]

#: Profile keys for a clip with no chroma, and for one with.
BW, COLOUR = "bw", "colour"
#: Pixels SSIM's reference implementation scales a picture's short side towards (Wang et al.).
SSIM_SIDE = 256
#: Threads each worker's numerical libraries may use.
THREADS = 2


def factor(width: int, height: int) -> int:
    """Wang et al.'s SSIM downsampling: the short side over :data:`SSIM_SIDE`, rounded, at least one."""
    return max(1, int(round(min(width, height) / SSIM_SIDE)))


def scaled(width: int, height: int, strip_px: int) -> tuple[int, int]:
    """The picture's size less its strip, at :func:`factor`, both even."""
    body = height - strip_px
    k = factor(width, body)
    return 2 * (width // (2 * k)), 2 * (body // (2 * k))


def decode(
    path: str | Path, *, width: int, height: int, strip_px: int, ffmpeg: str = "ffmpeg"
) -> Iterator[np.ndarray]:
    """Every frame of ``path`` in stream order, strip cropped, area-scaled to :func:`scaled`, as RGB."""
    w, h = scaled(width, height, strip_px)
    size = w * h * 3
    graph = f"crop=iw:ih-{strip_px}:0:0,scale={w}:{h}:flags=area"
    # pylint: disable-next=contextmanager-generator-missing-cleanup
    with captures(path, ffmpeg=ffmpeg) as source:
        argv = [ffmpeg, "-loglevel", "error", "-nostdin", *source, "-vf", graph]
        argv += ["-fps_mode", "passthrough", "-f", "rawvideo", "-pix_fmt", "rgb24", "pipe:1"]
        with subprocess.Popen(argv, stdout=subprocess.PIPE) as proc:
            while len(buf := proc.stdout.read(size)) == size:
                yield np.frombuffer(buf, np.uint8).reshape(h, w, 3)
            proc.stdout.close()
            if proc.wait():
                raise RuntimeError(f"decoding {path} failed with exit {proc.returncode}")


def clip_frames(
    clip: str | Path, *, width: int, height: int, strip_px: int, ffmpeg: str = "ffmpeg"
) -> np.ndarray:
    """Every frame of a played clip at :func:`scaled`, memory-mapped from a cache beside it, made once."""
    w, h = scaled(width, height, strip_px)
    clip = Path(clip)
    cache = clip.with_name(f"{clip.name}.effects-{w}x{h}.npy")
    stamp = clip.stat()
    key = {"size": stamp.st_size, "mtime_ns": stamp.st_mtime_ns, "strip_px": strip_px}
    said = cache.with_suffix(".json")
    if cache.is_file() and said.is_file() and json.loads(said.read_text(encoding="utf-8")) == key:
        return np.load(cache, mmap_mode="r")
    frames = list(
        track(decode(clip, width=width, height=height, strip_px=strip_px, ffmpeg=ffmpeg), desc="clip")
    )
    partial = cache.with_name(f"{cache.name}.{os.getpid()}.partial")
    with open(partial, "wb") as handle:
        np.save(handle, np.stack(frames))
    os.replace(partial, cache)
    said.write_text(json.dumps(key), encoding="utf-8")
    return np.load(cache, mmap_mode="r")


def clip_kind(stimulus: dict) -> tuple[str, list[float]]:
    """:data:`BW` or :data:`COLOUR` for a synthetic clip's record, and its saturation range."""
    saturation = [float(v) for v in stimulus["stimulus"]["coverage"]["saturation"]]
    return (BW if max(saturation) <= 0.0 else COLOUR), saturation


def _groups(rows: Sequence[Any]) -> dict[int, tuple[tuple[int, ...], list[tuple[int, int]]]]:
    """Each setpoint's parameters and its ``(capture, clip frame)`` pairs in clip order."""
    out: dict[int, tuple[tuple[int, ...], list[tuple[int, int]]]] = {}
    for row in rows:
        if row.setpoint != GAP and row.shown >= 0:
            out.setdefault(row.setpoint, (row.params, []))[1].append((row.frame, row.shown))
    return {k: (params, sorted(pairs, key=lambda p: p[1])) for k, (params, pairs) in sorted(out.items())}


def _runs(shown: np.ndarray) -> list[np.ndarray]:
    """Index runs of consecutive clip frames."""
    return np.split(np.arange(shown.size), np.flatnonzero(np.diff(shown) != 1) + 1)


def program_effects(
    archive: FrameArchive,
    program: str,
    clip: np.ndarray,
    *,
    ffmpeg: str = "ffmpeg",
    stream: Iterable[np.ndarray] | None = None,
    thumbs: dict[int, np.ndarray] | None = None,
) -> list[dict[str, Any]]:
    """Each setpoint's parameters and :class:`aesthetics.Effect`, on its longest run of consecutive frames.

    ``stream`` stands in for the archive's decode; ``thumbs``, when given, gets each setpoint's :func:`thumb`.
    """
    reader = archive.reader(program)
    if reader is None:
        raise FileNotFoundError(f"{program}: no committed archive in {archive.directory}")
    stimulus = stimulus_of(reader.meta)
    groups = _groups(reader.rows)
    wanted = {frame: k for k, (_, pairs) in groups.items() for frame, _ in pairs}
    held: dict[int, np.ndarray] = {}
    out = []
    pending = iter(groups)
    current = next(pending, None)
    if stream is None:
        stream = decode(
            reader.video,
            width=reader.width,
            height=reader.height,
            strip_px=int(stimulus["strip_px"]),
            ffmpeg=ffmpeg,
        )
    for index, frame in enumerate(track(stream, desc=program, total=reader.count, unit="frame")):
        if index in wanted:
            held[index] = frame
        while current is not None and all(f in held for f, _ in groups[current][1]):
            out.append(_measure(current, groups[current], held, clip, reader.fps, thumbs))
            current = next(pending, None)
    while current is not None:
        LOG.warning("%s: setpoint %d has captures past the end of its video", program, current)
        current = next(pending, None)
    return out


def thumb(source: np.ndarray, output: np.ndarray, shrink: int = 4) -> np.ndarray:
    """The middle source frame beside the first, middle and last output frames, ``shrink`` times smaller."""
    picks = [source[len(source) // 2], output[0], output[len(output) // 2], output[-1]]
    return np.concatenate([np.asarray(p)[::shrink, ::shrink] for p in picks], axis=1)


def _measure(
    setpoint: int, group: tuple, held: dict, clip: np.ndarray, fps: float, thumbs: dict | None
) -> dict[str, Any]:
    """One setpoint's effect, its captures released once measured."""
    params, pairs = group
    frames = np.array([f for f, _ in pairs])
    shown = np.array([s for _, s in pairs])
    run = max(_runs(shown), key=len)
    taken = [held.pop(int(f)) for f in frames]
    output = np.stack(taken[run[0] : run[-1] + 1])
    source = np.asarray(clip[shown[run]])
    found = aesthetics.effect(source, output, fps=fps)
    if thumbs is not None:
        thumbs[int(setpoint)] = thumb(source, output)
    return {"setpoint": int(setpoint), "params": [int(p) for p in params], **found.as_dict()}


def summarise(settings: Sequence[dict[str, Any]]) -> dict[str, Any]:
    """Counts of each flag, how many settings are safe, and the intensity range all and safe ones span."""

    def span(values: list[float]) -> list[float] | None:
        return [float(min(values)), float(max(values))] if values else None

    flags = ("blank", "static", "flashing", "red_flashing")
    return {
        "settings": len(settings),
        "safe": sum(bool(s["safe"]) for s in settings),
        **{flag: sum(bool(s[flag]) for s in settings) for flag in flags},
        "intensity": span([s["intensity"] for s in settings]),
        "safe_intensity": span([s["intensity"] for s in settings if s["safe"]]),
    }


#: Tile borders, BGR, by the first flag a setting raises.
FLAG_COLOURS = {
    "flashing": (0, 0, 255),
    "red_flashing": (255, 0, 255),
    "blank": (255, 0, 0),
    "static": (0, 255, 255),
}
SAFE_COLOUR = (0, 160, 0)


def sheet(
    settings: Sequence[dict[str, Any]], thumbs: dict[int, np.ndarray], path: str | Path, columns: int = 4
) -> None:
    """A contact sheet of every setting by rising intensity, bordered by its first flag, written as PNG.

    Each tile is :func:`thumb`'s source frame and three output frames, labelled with setpoint, intensity and
    flags; green borders are safe settings.
    """
    tiles = []
    for found in sorted(settings, key=lambda s: s["intensity"]):
        flags = [f for f in FLAG_COLOURS if found[f]]
        colour = FLAG_COLOURS[flags[0]] if flags else SAFE_COLOUR
        tile = cv2.cvtColor(np.ascontiguousarray(thumbs[found["setpoint"]]), cv2.COLOR_RGB2BGR)
        tile = cv2.copyMakeBorder(tile, 18, 3, 3, 3, cv2.BORDER_CONSTANT, value=colour)
        label = f"s{found['setpoint']} i={found['intensity']:.3f} {' '.join(flags) or 'safe'}"
        cv2.putText(tile, label, (4, 13), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
        tiles.append(tile)
    blank = np.zeros_like(tiles[0])
    tiles += [blank] * (-len(tiles) % columns)
    rows = [np.concatenate(tiles[i : i + columns], axis=1) for i in range(0, len(tiles), columns)]
    cv2.imwrite(str(path), np.concatenate(rows, axis=0))


def lookup(settings: Sequence[dict[str, Any]], params: Sequence[int]) -> dict[str, Any]:
    """The nearest measured setting's intensity at ``params``, safe only if every setting as near is.

    Distance is over the parameters the settings varied, each over the span they took; "as near" is
    within the median nearest-neighbour spacing of the settings, the design's own resolution.
    """
    grid = np.array([s["params"] for s in settings], dtype=np.float64)
    lo, hi = grid.min(axis=0), grid.max(axis=0)
    live = hi > lo
    unit = (grid[:, live] - lo[live]) / (hi - lo)[live]
    query = (np.asarray(params, dtype=np.float64)[live] - lo[live]) / (hi - lo)[live]
    near = np.linalg.norm(unit - query, axis=1)
    spacing = np.linalg.norm(unit[:, None] - unit[None], axis=2)
    np.fill_diagonal(spacing, np.inf)
    radius = float(np.median(spacing.min(axis=1))) if len(settings) > 1 else 0.0
    nearest = int(np.argmin(near))
    within = near <= max(radius, float(near[nearest]))
    return {
        "intensity": float(settings[nearest]["intensity"]),
        "safe": bool(all(settings[i]["safe"] for i in np.flatnonzero(within))),
        "nearest": int(settings[nearest]["setpoint"]),
        "distance": float(near[nearest]),
    }


def _worker(verbosity: int) -> None:
    """A fresh worker logging as its parent does, OpenCV on one thread."""
    configure(verbosity)
    cv2.setNumThreads(1)


def _one(directory: str, program: str, clip_path: str, ffmpeg: str, sheets: str | None) -> list[dict]:
    """A worker's program, its native libraries held to :data:`THREADS`, its sheet drawn into ``sheets``."""
    with threadpool_limits(THREADS):
        clip = np.load(clip_path, mmap_mode="r")
        thumbs: dict[int, np.ndarray] | None = {} if sheets else None
        found = program_effects(
            FrameArchive(directory, ffmpeg=ffmpeg), program, clip, ffmpeg=ffmpeg, thumbs=thumbs
        )
        if sheets and found:
            sheet(found, thumbs, Path(sheets) / f"{slug(program)}.png")
        return found


def measure(
    directory: str | Path,
    programs: Iterable[str] | None = None,
    *,
    jobs: int = 1,
    ffmpeg: str = "ffmpeg",
    sheets: str | Path | None = None,
    executor: Callable[..., Executor] = ProcessPoolExecutor,
) -> dict[str, tuple[dict, list[dict]]]:
    """Every named program's stimulus record and setting effects, ``jobs`` programs at a time.

    Each fresh worker logs as this process does; with ``sheets``, each program's :func:`sheet` goes there.
    """
    if sheets:
        Path(sheets).mkdir(parents=True, exist_ok=True)
    archive = FrameArchive(directory, ffmpeg=ffmpeg)
    committed = archive.committed()
    names = sorted(committed) if programs is None else list(programs)
    verbosity = {logging.DEBUG: 1, logging.INFO: 0}.get(LOG.getEffectiveLevel(), -1)
    options = (
        {
            "mp_context": multiprocessing.get_context("forkserver"),
            "initializer": _worker,
            "initargs": (verbosity,),
        }
        if executor is ProcessPoolExecutor
        else {}
    )
    out: dict[str, tuple[dict, list[dict]]] = {}
    with executor(max(1, jobs), **options) as pool:
        futures = {}
        for name in names:
            meta = committed[name]
            stimulus = stimulus_of(meta)
            frames = clip_frames(
                Path(directory) / stimulus["clip"],
                width=int(meta["width"]),
                height=int(meta["height"]),
                strip_px=int(stimulus["strip_px"]),
                ffmpeg=ffmpeg,
            )
            job = pool.submit(
                _one, str(directory), name, str(frames.filename), ffmpeg, str(sheets) if sheets else None
            )
            futures[job] = (name, stimulus)
        for future in track(as_completed(futures), desc="effects", total=len(futures), unit="program"):
            name, stimulus = futures[future]
            out[name] = (stimulus, future.result())
            LOG.info("%s: %s", name, json.dumps(summarise(out[name][1])))
    return out


def stimulus_record(directory: str | Path, stimulus: dict) -> dict:
    """The synthetic clip's own record beside the clip an archive played."""
    for path in (
        Path(directory) / f"{stimulus['clip']}.stimulus.json",
        Path(f"{stimulus['source']}.stimulus.json"),
    ):
        if path.is_file():
            return json.loads(path.read_text(encoding="utf-8"))
    raise FileNotFoundError(f"no stimulus record for {stimulus['clip']}; effects need a synthetic clip")


def _effects_cmd(args: argparse.Namespace) -> int:
    from syncsummoner.probe.fit import load_profile, save_profile  # pylint: disable=import-outside-toplevel

    programs = None if args.program == "all" else args.program.split(",")
    found = measure(args.archive, programs, jobs=args.jobs, ffmpeg=args.ffmpeg, sheets=args.sheets)
    written = []
    for program, (stimulus, settings) in sorted(found.items()):
        kind, saturation = clip_kind(stimulus_record(args.archive, stimulus))
        entry = {"clip": stimulus["clip"], "saturation": saturation, "summary": summarise(settings)}
        path = Path(args.profiles) / f"{slug(program)}.yaml"
        if path.is_file():
            profile = load_profile(path)
            profile.effects = {**profile.effects, kind: {**entry, "settings": settings}}
            save_profile(profile, path)
            written.append(str(path))
        else:
            LOG.warning("%s: no profile at %s to write its effects into", program, path)
        print(json.dumps({"program": program, "kind": kind, **entry}))
    return 0 if written else 1


def add_parser(probe_sub: Any) -> None:
    """The ``probe effects`` command."""
    cmd = probe_sub.add_parser(
        "effects", help="each archived setting's intensity and safety, into its program's profile"
    )
    cmd.add_argument("--archive", required=True, help="the committed archive directory")
    cmd.add_argument("--profiles", required=True, help="the profiles to write each program's effects into")
    cmd.add_argument("--program", default="all", help="comma-separated program names, or all")
    cmd.add_argument("--jobs", type=int, default=1, help="programs measured at once")
    cmd.add_argument("--ffmpeg", default="ffmpeg", help="decoder to read the archive and clip with")
    cmd.add_argument("--sheets", help="a directory for each program's contact sheet of its settings")
    cmd.set_defaults(func=_effects_cmd)

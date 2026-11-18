"""Takes that store every capture, rewritten to store each run of byte-identical captures once.

The rewrite goes beside the take and is checked capture for capture, by bytes and by decoded frame,
before it replaces the take and every hard-linked name of it; any mismatch keeps the original.
"""

from __future__ import annotations

import json
import os
import stat
import subprocess
import threading
from dataclasses import dataclass
from fractions import Fraction
from pathlib import Path
from typing import Iterable, Iterator, Sequence

import numpy as np

from syncsummoner.device.stored import Stored, digest_of
from syncsummoner.device.v4l2rec import (
    FOURCC,
    STORED,
    Layout,
    encoder_argv,
    ivf_frame,
    read_sidecar,
    sidecar,
)
from syncsummoner.probe.decoded import ffprobe_for
from syncsummoner.progress import LOG, track

#: The fourcc a stored codec is framed as, by ffprobe's codec name and pixel format.
CODECS = {("mjpeg", None): FOURCC["mjpeg"], ("rawvideo", "yuyv422"): FOURCC["yuyv422"]}


class Refused(RuntimeError):
    """A take that cannot be rewritten exactly, so is left as it is."""


@dataclass(frozen=True)
class Packets:
    """A take's video packets in file order, with the layout they are stored on."""

    pts: np.ndarray
    sizes: np.ndarray
    layout: Layout


@dataclass
class Outcome:
    """What deduplicating one take did, or would do."""

    take: str
    status: str
    captures: int = 0
    stored: int = 0
    before: int = 0
    after: int = 0
    why: str = ""

    def line(self) -> str:
        """One table row."""
        saved = 0 if not self.before else 100 * (1 - self.after / self.before)
        return (
            f"{self.status:<12} {self.captures:>7} {self.stored:>7} {self.before:>15,} {self.after:>15,} "
            f"{saved:>5.1f}%  {self.take}{'  ' + self.why if self.why else ''}"
        )


_COLUMNS = f"{'status':<12} {'captures':>7} {'stored':>7}"
HEADING = f"{_COLUMNS} {'bytes before':>15} {'bytes after':>15} {'saved':>6}  take"


def packets(path: str | Path, *, ffmpeg: str = "ffmpeg") -> Packets:
    """The take's one video stream: every packet's timestamp and size, refused unless each is a keyframe."""
    argv = [ffprobe_for(ffmpeg), "-v", "error", "-of", "json"]
    argv += ["-show_entries", "stream=codec_type,codec_name,pix_fmt,width,height,time_base"]
    streams = json.loads(
        subprocess.run([*argv, str(path)], capture_output=True, check=True, text=True).stdout
    )
    streams = streams.get("streams", [])
    if len(streams) != 1 or streams[0].get("codec_type") != "video":
        raise Refused("a take holds exactly one stream, a video one")
    info = streams[0]
    fourcc = CODECS.get((info["codec_name"], None)) or CODECS.get((info["codec_name"], info.get("pix_fmt")))
    if fourcc is None:
        raise Refused(f"{info['codec_name']} {info.get('pix_fmt')} is not a card format")
    listed = subprocess.run(
        [
            *argv[:3],
            "-select_streams",
            "v:0",
            "-show_entries",
            "packet=pts,size,flags",
            "-of",
            "csv=p=0",
            str(path),
        ],
        capture_output=True,
        check=True,
        text=True,
    ).stdout.split()
    rows = [line.split(",") for line in listed]
    if not all("K" in r[2] for r in rows):
        raise Refused("a take of other than keyframes")
    pts = np.array([int(r[0]) for r in rows], dtype=np.int64)
    if np.any(np.diff(pts) <= 0):
        raise Refused("packet timestamps do not rise")
    layout = Layout(fourcc, int(info["width"]), int(info["height"]), Fraction(info["time_base"]))
    return Packets(pts, np.array([int(r[1]) for r in rows], dtype=np.int64), layout)


def originals(path: str | Path, sizes: np.ndarray, *, ffmpeg: str = "ffmpeg") -> Iterator[bytes]:
    """Each packet of a take's video, untouched, in file order."""
    argv = [ffmpeg, "-nostdin", "-loglevel", "error", "-i", str(path), "-map", "0:v:0", "-c", "copy"]
    proc = subprocess.Popen(  # pylint: disable=consider-using-with
        [*argv, "-f", "rawvideo", "pipe:1"], stdin=subprocess.DEVNULL, stdout=subprocess.PIPE
    )
    try:
        for size in sizes:
            data = proc.stdout.read(int(size))
            if len(data) != size:
                raise Refused("the take ends before its packets do")
            yield data
    finally:
        proc.kill()
        proc.stdout.close()
        proc.wait()


def deduplicate(rows: np.ndarray, found: Packets, datas: Iterable[bytes], sink=None) -> np.ndarray:
    """The take's :data:`~.v4l2rec.STORED` rows, each run stored once into ``sink`` as IVF if given.

    A capture repeats the frame stored before it only when the bytes match, not the digest alone; its
    bytes must have the digest its row logged. Rows past the file's last packet are captures it never
    stored, each numbered as a frame of its own after the last and timed a tick apart.
    """
    count = found.sizes.size
    out = np.zeros(rows.size, dtype=STORED)
    for name in rows.dtype.names:
        out[name] = rows[name]
    held, stored = None, -1
    for capture, data in enumerate(track(datas, desc="deduplicating", total=count, unit="capture")):
        if digest_of(data) != int(rows["digest"][capture]):
            raise Refused(f"capture {capture}'s bytes are not those its sidecar logged")
        if data != held:
            held, stored = data, stored + 1
            if sink is not None:
                sink.write(ivf_frame(data, int(found.pts[capture])))
                sink.write(data)
        out["stored"][capture] = stored
    out["pts"][:count] = found.pts
    tail = np.arange(1, rows.size - count + 1)
    out["stored"][count:] = stored + tail
    out["pts"][count:] = (found.pts[-1] if count else 0) + tail
    return out


def framemd5s(source: Sequence[str], *, ffmpeg: str = "ffmpeg") -> list[str]:
    """Each decoded frame's MD5, in order, from ffmpeg reading ``source`` input arguments."""
    argv = [ffmpeg, "-nostdin", "-loglevel", "error", *source, "-map", "0:v:0", "-fps_mode", "passthrough"]
    out = subprocess.run(
        [*argv, "-f", "framemd5", "pipe:1"], capture_output=True, check=True, text=True
    ).stdout
    return [line.rsplit(",", 1)[-1].strip() for line in out.splitlines() if not line.startswith("#")]


def verify(take: Path, new: Path, found: Packets, *, ffmpeg: str = "ffmpeg") -> None:
    """Refuse ``new`` unless every capture of ``take`` comes back from it byte for byte, decoding the same."""
    decoded: dict[str, list[str]] = {}
    jobs = {"old": ["-i", str(take)], "new": ["-i", str(new)]}

    def decode(name: str) -> None:
        try:
            decoded[name] = framemd5s(jobs[name], ffmpeg=ffmpeg)
        except subprocess.CalledProcessError:
            decoded[name] = []

    threads = [threading.Thread(target=decode, args=(name,), daemon=True) for name in jobs]
    for thread in threads:
        thread.start()
    stored = Stored.of(new)
    count = 0
    pairs = zip(originals(take, found.sizes, ffmpeg=ffmpeg), stored.frames(ffmpeg=ffmpeg))
    for count, (data, (pts, again)) in enumerate(track(pairs, desc="verifying", total=found.sizes.size), 1):
        if again != data or pts != found.pts[count - 1]:
            raise Refused(f"capture {count - 1} does not come back from the rewrite as it was")
    if count != found.sizes.size:
        raise Refused(f"the rewrite gives back {count} of {found.sizes.size} captures")
    for thread in threads:
        thread.join()
    index = stored.index[: found.sizes.size]
    if len(decoded.get("old", ())) != found.sizes.size or len(decoded.get("new", ())) != index[-1] + 1:
        raise Refused("the take or its rewrite decodes to other than one frame per packet")
    if any(a != decoded["new"][j] for a, j in zip(decoded["old"], index)):
        raise Refused("a capture decodes differently from the rewrite")


def names_of(path: Path, roots: Sequence[str | Path]) -> list[Path]:
    """Every name under ``roots`` for the file at ``path``, ``path`` first."""
    held = path.stat()
    found = {path.absolute()}
    for root in roots:
        for folder, _, files in os.walk(root):
            for name in files:
                candidate = Path(folder, name)
                try:
                    there = candidate.stat(follow_symlinks=False)
                except OSError:
                    continue
                if (there.st_ino, there.st_dev) == (held.st_ino, held.st_dev):
                    found.add(candidate.absolute())
    return [path.absolute(), *sorted(found - {path.absolute()})]


def _publish(source: Path, target: Path, mode: int) -> None:
    """Hard-link ``source`` over ``target`` atomically, with ``mode``."""
    tmp = target.with_name(f".{target.name}.{os.getpid()}.dedupe")
    tmp.unlink(missing_ok=True)
    os.link(source, tmp)
    os.chmod(tmp, mode)
    os.replace(tmp, target)


def _names(take: Path, roots: Sequence[str | Path]) -> list[Path]:
    """Every name the take goes by, each beside the take's sidecar or none; else :class:`Refused`."""
    held = take.stat()
    names = names_of(take, roots) if held.st_nlink > 1 else [take.absolute()]
    if len(names) != held.st_nlink:
        raise Refused(
            f"hard-linked {held.st_nlink} times, {len(names)} found{'' if roots else '; pass --links'}"
        )
    locked = [n for n in names if not os.access(n.parent, os.W_OK) or not n.parent.stat().st_mode & 0o222]
    if locked:
        raise Refused(f"{locked[0].parent} is read-only")
    own = sidecar(take).stat()
    for name in names:
        side = sidecar(name)
        if side.exists() and not os.path.samestat(side.stat(), own):
            raise Refused(f"{side} is not the take's sidecar")
    return names


def dedupe(
    take: str | Path, *, links: Sequence[str | Path] = (), dry_run: bool = False, ffmpeg: str = "ffmpeg"
) -> Outcome:
    """Rewrite one take deduplicated and put it in place under every name it has, or say why not.

    A dry run reads the take and counts what the rewrite would store, writing nothing; its size after
    is the bytes of the captures it would keep. A take that would be refused is left untouched.
    """
    take = Path(take)
    try:
        before = take.stat().st_size + (sidecar(take).stat().st_size if sidecar(take).is_file() else 0)
    except OSError as err:
        return Outcome(str(take), "refused", why=str(err))
    outcome = Outcome(str(take), "refused", before=before)
    tmp = take.with_name(f".{take.name}.dedupe{take.suffix}")
    if tmp.exists():
        return Outcome(
            str(take), "refused", before=before, why=f"an interrupted rewrite lies beside it at {tmp}"
        )
    try:
        if not sidecar(take).is_file():
            raise Refused("no sidecar")
        rows, layout = read_sidecar(take)
        if layout is not None:
            count = int(rows["stored"][-1]) + 1 if rows.size else 0
            return Outcome(str(take), "already", rows.size, count, before, before)
        found = packets(take, ffmpeg=ffmpeg)
        if found.sizes.size > rows.size or not np.array_equal(rows["bytes"][: found.sizes.size], found.sizes):
            raise Refused("its packets are not the captures its sidecar logged")
        outcome.captures = int(rows.size)
        names: list[Path] = []
        if not dry_run:
            try:
                names = _names(take, links)
            except Refused as err:
                dry_run, outcome.why = True, str(err)
        datas = originals(take, found.sizes, ffmpeg=ffmpeg)
        if dry_run:
            stored = deduplicate(rows, found, datas)
            firsts = np.flatnonzero(np.r_[True, np.diff(stored["stored"][: found.sizes.size]) != 0])
            outcome.stored = int(firsts.size)
            outcome.after = before - int(found.sizes.sum() - found.sizes[firsts].sum())
            outcome.status = "skipped" if outcome.why else "dry-run"
            return outcome
        with subprocess.Popen(  # pylint: disable=consider-using-with
            encoder_argv(["-map", "0", "-c", "copy"], str(tmp), ffmpeg=ffmpeg), stdin=subprocess.PIPE
        ) as enc:
            enc.stdin.write(found.layout.ivf_header())
            stored = deduplicate(rows, found, datas, enc.stdin)
            enc.stdin.close()
            if enc.wait():
                raise Refused(f"the rewrite's encoder exited {enc.returncode}")
        sidecar(tmp).write_bytes(found.layout.sidecar_header() + stored.tobytes())
        verify(take, tmp, found, ffmpeg=ffmpeg)
        mode = stat.S_IMODE(take.stat().st_mode)
        for name in names:
            _publish(sidecar(tmp), sidecar(name), mode)
            _publish(tmp, name, mode)
        outcome.status, outcome.stored = "deduplicated", int(stored["stored"][found.sizes.size - 1]) + 1
        outcome.after = take.stat().st_size + sidecar(take).stat().st_size
        return outcome
    except (Refused, subprocess.CalledProcessError, OSError) as err:
        outcome.why = str(err)
        LOG.warning("%s: left as it is: %s", take, err)
        return outcome
    finally:
        tmp.unlink(missing_ok=True)
        sidecar(tmp).unlink(missing_ok=True)


def dedupe_all(
    takes: Sequence[str | Path],
    *,
    links: Sequence[str | Path] = (),
    dry_run: bool = False,
    ffmpeg: str = "ffmpeg",
) -> list[Outcome]:
    """:func:`dedupe` each take in turn, printing a row for each as it finishes and the totals last."""
    print(HEADING, flush=True)
    done = []
    for take in track(takes, desc="takes", total=len(takes), unit="take"):
        done.append(dedupe(take, links=links, dry_run=dry_run, ffmpeg=ffmpeg))
        print(done[-1].line(), flush=True)
    total = Outcome(
        "total",
        "",
        sum(o.captures for o in done),
        sum(o.stored for o in done),
        sum(o.before for o in done if o.after),
        sum(o.after for o in done),
    )
    print(total.line(), flush=True)
    return done


def add_parser(probe_sub) -> None:
    """The ``probe dedupe`` command."""
    found = probe_sub.add_parser("dedupe", help="rewrite takes to store each run of identical captures once")
    found.add_argument("takes", nargs="+", help="takes, each with its .v4l2 sidecar")
    found.add_argument("--dry-run", action="store_true", help="report what each would save, writing nothing")
    found.add_argument(
        "--links",
        action="append",
        default=[],
        metavar="DIR",
        help="where a hard-linked take's other names may be, all rewritten together (repeatable)",
    )
    found.add_argument("--ffmpeg", default="ffmpeg")
    found.set_defaults(func=_dedupe_cmd)


def _dedupe_cmd(args) -> int:
    done = dedupe_all(args.takes, links=args.links, dry_run=args.dry_run, ffmpeg=args.ffmpeg)
    return 0 if all(o.status in ("deduplicated", "already", "dry-run") for o in done) else 1

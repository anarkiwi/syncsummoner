"""Vsync-locked clip player for a DRM/KMS display, run on the stimulus source host.

One atomic flip per vblank, each clip frame held for exactly ``refresh / fps`` vblanks, every
flip-complete event checked against the vblank it was committed for; a late one is logged as
:data:`MISSED` and play goes on on the same grid.
Imports only the standard library at module level: it runs under the host's system python.
"""

from __future__ import annotations

import argparse
import gc
import glob
import json
import os
import queue
import select
import signal
import struct
import subprocess
import sys
import threading
import time
from fractions import Fraction
from typing import Any, Callable, Iterable, Iterator

#: One flip: clip vblank, display vblank sequence, clip frame shown, event time (ns), flags.
RECORD = struct.Struct("<qqqqq")
#: Flag on the first flip of a segment, after the start or after the link came back up.
SEGMENT = 1
#: Flag on a flip that landed on a later vblank than it was committed for.
MISSED = 2
#: ``DRM_MODE_FLAG_INTERLACE``: such a mode has no single frame per vblank.
INTERLACE = 0x10
#: The framebuffer layout bgr565le words land in, as fbdev and the device chain expect.
FORMAT = "RG16"
BYTES_PER_PIXEL = 2
#: Displayed, committed and being filled.
BUFFERS = 3
#: ``SCHED_FIFO`` priority of the pacing thread, above every normal thread and below the kernel's own.
REALTIME_PRIORITY = 10


class PlayoutFault(RuntimeError):
    """The display did not show the frame due on a vblank, or could not be made to."""


def refresh(mode: Any) -> Fraction:
    """A mode's exact vertical refresh in Hz, from its pixel clock (kHz) and totals."""
    return Fraction(int(mode.clock) * 1000, int(mode.htotal) * int(mode.vtotal))


def parse_rate(text: str) -> Fraction:
    """A frame rate given as an integer, a decimal or ``num/den``."""
    rate = Fraction(text)
    if rate <= 0:
        raise ValueError(f"frame rate must be positive, got {text}")
    return rate


def pick_mode(modes: Iterable[Any], width: int, height: int, fps: Fraction) -> tuple[Any, int]:
    """The first progressive mode of this size refreshing at a whole multiple of ``fps``, and that multiple.

    ``modes`` is in preference order, the current mode first, so a mode already driven is
    kept and the link is not retrained.
    """
    seen = []
    for mode in modes:
        if mode is None or (int(mode.hdisplay), int(mode.vdisplay)) != (width, height):
            continue
        if int(mode.flags) & INTERLACE:
            continue
        rate = refresh(mode)
        seen.append(rate)
        hold = rate / fps
        if hold.denominator == 1:
            return mode, int(hold)
    said = ", ".join(f"{float(r):.4f}" for r in dict.fromkeys(seen)) or "none"
    raise PlayoutFault(
        f"no {width}x{height} progressive mode refreshes at a whole multiple of {float(fps):g} fps; "
        f"offered {said} Hz"
    )


class Prerolled:
    """Frames from a source, read ahead into memory on a thread.

    ``get(index)`` returns clip frame ``index``, discarding any before it; a frame not yet
    decoded when it is due is an underrun, which is a fault.
    """

    def __init__(self, frames: Iterator[bytes], depth: int):
        if depth < 1:
            raise ValueError(f"preroll depth must be at least one frame, got {depth}")
        self._queue: queue.Queue = queue.Queue(maxsize=depth)
        self._next = 0
        self._ended = False
        self.depth = depth
        self._error: list[BaseException] = []
        self._thread = threading.Thread(target=self._fill, args=(frames,), daemon=True)
        self._thread.start()

    def _fill(self, frames: Iterator[bytes]) -> None:
        """Read the source into the queue, ending with a sentinel."""
        try:
            for frame in frames:
                self._queue.put(frame)
        except Exception as err:  # pylint: disable=broad-exception-caught
            self._error.append(err)
        self._queue.put(None)

    def wait_full(self, timeout_s: float, *, clock: Callable[[], float] = time.monotonic) -> None:
        """Block until the read-ahead is full or the source has ended."""
        deadline = clock() + timeout_s
        while not self._queue.full() and self._thread.is_alive():
            if clock() >= deadline:
                raise PlayoutFault(
                    f"preroll held {self._queue.qsize()} of {self.depth} frames after {timeout_s}s"
                )
            time.sleep(0.01)

    @property
    def ended(self) -> bool:
        """Whether the source has no more frames."""
        return self._ended

    def get(self, index: int) -> bytes | None:
        """Clip frame ``index``, or None once the source has ended; raises on underrun."""
        while not self._ended:
            try:
                frame = self._queue.get_nowait()
            except queue.Empty as err:
                raise PlayoutFault(f"frame {index} was not decoded when it was due") from err
            if frame is None:
                self._ended = True
                if self._error:
                    raise PlayoutFault(f"the frame source failed: {self._error[0]}")
                return None
            self._next += 1
            if self._next - 1 == index:
                return frame
            if self._next - 1 > index:
                raise PlayoutFault(f"frame {index} was asked for after frame {self._next - 1}")
        return None


class Timeline:
    """Maps the display's vblanks to the clip's, and checks every flip against it.

    Vblank ``v`` counts from the first flip at ``t0`` and shows frame ``v // hold``. Within a
    segment each flip lands on its own vblank of the segment's grid, within half a vblank;
    ``phases_ns`` keeps where each segment's grid sits against ``t0``'s.
    """

    def __init__(self, hold: int, period_ns: Fraction):
        self.hold = int(hold)
        self.period_ns = Fraction(period_ns)
        self.t0: int | None = None
        self.base: int | None = None
        self.start: tuple[int, int] = (0, 0)
        self.phases_ns: list[int] = []
        self.worst_ns = 0

    def at(self, t_ns: int) -> int:
        """The clip vblank nearest a time, once the timeline is anchored."""
        return round((t_ns - self.t0) / self.period_ns)

    def anchor(self, seq: int, t_ns: int) -> int:
        """Start the first segment on this flip, or a later one at its time's vblank; returns that vblank."""
        if self.t0 is None:
            self.t0 = t_ns
        v = self.at(t_ns)
        self.base, self.start = seq - v, (v, t_ns)
        self.phases_ns.append(int(t_ns - self.t0 - v * self.period_ns))
        return v

    def check(self, seq: int, t_ns: int, v: int) -> int:
        """The vblank the flip committed for vblank ``v`` landed on, ``v`` or later, on its segment's grid."""
        landed = seq - self.base
        if landed < v:
            raise PlayoutFault(
                f"flip for vblank {v} (frame {v // self.hold}) landed early, on vblank {landed}"
            )
        late = abs(t_ns - self.start[1] - (landed - self.start[0]) * self.period_ns)
        self.worst_ns = max(self.worst_ns, int(late))
        if 2 * late >= self.period_ns:
            raise PlayoutFault(
                f"vblank {landed} came {float(late) / 1e6:.3f} ms off a grid of "
                f"{float(self.period_ns) / 1e6:.3f} ms "
                "vblanks: the display clock and the event clock disagree"
            )
        return landed


class Player:
    """Paces prerolled frames onto a display, one atomic flip per vblank.

    ``signals`` holds pending ``stop``, ``down`` and ``up`` requests set by signal handlers.
    While the link is down the timeline runs on the event clock and due frames are discarded.
    A flip landing late is logged :data:`MISSED` on the vblank it landed on, which is counted in ``missed``,
    and the next commit is for the vblank after it, so frames due meanwhile are skipped.
    """

    def __init__(
        self,
        display: Any,
        frames: Prerolled,
        *,
        hold: int,
        period_ns: Fraction,
        record: Callable[[int, int, int, int, int], Any],
        signals: dict,
        clock: Callable[[], int] = time.monotonic_ns,
        sleep: Callable[[float], None] = time.sleep,
        event_timeout_s: float = 1.0,
    ):
        self.display = display
        self.frames = frames
        self.timeline = Timeline(hold, period_ns)
        self.record = record
        self.signals = signals
        self.clock = clock
        self.sleep = sleep
        self.event_timeout_s = float(event_timeout_s)
        self.flips = 0
        self.segments = 0
        self.missed = 0
        self.ended = False
        self.shown: tuple[int, int] | None = None
        self._ready: tuple[int, int] | None = None

    @staticmethod
    def _free(*busy: int) -> int:
        """A buffer neither displayed nor committed."""
        return next(b for b in range(BUFFERS) if b not in busy)

    def _prepare(self, frame: int, buf: int) -> tuple[int, int] | None:
        """Fill ``buf`` with clip frame ``frame``; None once the clip has ended."""
        data = self.frames.get(frame)
        if data is None:
            return None
        self.display.fill(buf, data)
        return frame, buf

    def _event(self) -> tuple[int, int, int]:
        """The next flip-complete event as ``(seq, t_ns, vblank)``; a fault when none arrives."""
        event = self.display.wait(self.event_timeout_s)
        if event is None:
            raise PlayoutFault(f"no flip completed within {self.event_timeout_s}s")
        return event

    def _land(self, seq: int, t_ns: int, v: int, flags: int = 0) -> None:
        """Account for a flip that completed, and record it."""
        self.flips += 1
        self.record(v, seq, self.shown[0], t_ns, flags)

    def _commit(self, v: int) -> None:
        """Commit the frame due on vblank ``v``, preparing the one after into a free buffer."""
        frame = v // self.timeline.hold
        if frame == self.shown[0]:
            self.display.flip(self.shown[1], v)
            return
        while self._ready is not None and self._ready[0] < frame:
            self._ready = self._prepare(frame, self._ready[1])
        if self._ready is None:
            self.ended = True
            return
        pending = self._ready
        self.display.flip(pending[1], v)
        self._ready = self._prepare(frame + 1, self._free(self.shown[1], pending[1]))
        self.shown = pending

    def _skip_to(self, frame: int) -> None:
        """Discard read-ahead frames due before ``frame``."""
        while self._ready is not None and self._ready[0] < frame:
            self._ready = self._prepare(self._ready[0] + 1, self._ready[1])

    def _down(self) -> None:
        """Hold the link down until asked up, then resume on the frame due."""
        self.display.set_active(False, self.shown[1])
        while not self.signals.pop("up", False) and not self.signals.get("stop"):
            self.sleep(float(self.timeline.period_ns) / 1e9)
            self._skip_to(self.timeline.at(self.clock()) // self.timeline.hold)
        if self.signals.get("stop"):
            return
        self.display.set_active(True, self.shown[1])
        self.display.flip(self.shown[1], -1)
        seq, t_ns, _ = self._event()
        v = self.timeline.anchor(seq, t_ns)
        self.segments += 1
        self._land(seq, t_ns, v, SEGMENT)
        self._skip_to(v // self.timeline.hold)
        self._commit(v + 1)

    def _hold(self) -> None:
        """With the clip ended, keep its last frame up, still honouring link requests."""
        if self.signals.pop("down", False):
            self.display.set_active(False, self.shown[1])
        if self.signals.pop("up", False):
            self.display.set_active(True, self.shown[1])
        self.sleep(float(self.timeline.period_ns) / 1e9)

    def run(self) -> None:
        """Play until a stop is asked for, holding the last frame once a clip ends."""
        self.shown = self._prepare(0, 0)
        if self.shown is None:
            raise PlayoutFault("the clip holds no frames")
        self._ready = self._prepare(1, 1)
        self.display.start(0)
        self.display.flip(0, 0)
        seq, t_ns, _ = self._event()
        self.timeline.anchor(seq, t_ns)
        self.segments = 1
        self._land(seq, t_ns, 0, SEGMENT)
        self._commit(1)
        while not self.signals.get("stop"):
            if self.ended:
                self._hold()
                continue
            seq, t_ns, v = self._event()
            landed = self.timeline.check(seq, t_ns, v)
            self.missed += landed != v
            self._land(seq, t_ns, landed, MISSED if landed != v else 0)
            if self.signals.pop("down", False):
                self._down()
            else:
                self._commit(landed + 1)


class KmsDisplay:  # pragma: no cover - needs a DRM device and pykms
    """One connector's CRTC and primary plane, driven by atomic commits through pykms."""

    def __init__(self, device: str, connector: str, width: int, height: int, fps: Fraction):
        import pykms  # pylint: disable=import-error,import-outside-toplevel

        self._pykms = pykms
        self.card = pykms.Card(device)
        if not self.card.has_atomic:
            raise PlayoutFault(f"{device} offers no atomic modesetting")
        resources = pykms.ResourceManager(self.card)
        self.conn = resources.reserve_connector(connector)
        if self.conn is None:
            raise PlayoutFault(f"{device} has no connector {connector}")
        self.crtc = resources.reserve_crtc(self.conn)
        self.plane = self.crtc.primary_plane
        current = self.crtc.mode if self.crtc.mode_valid else None
        self.mode, self.hold = pick_mode([current, *self.conn.get_modes()], width, height, fps)
        self.modeset = current is None or self.mode.to_string_long() != current.to_string_long()
        self.blob = self.mode.to_blob(self.card)
        self.buffers = [pykms.DumbFramebuffer(self.card, width, height, FORMAT) for _ in range(BUFFERS)]
        self.maps = [fb.map(0).cast("B") for fb in self.buffers]
        self.frame_bytes = width * height * BYTES_PER_PIXEL
        self._poll = select.poll()
        self._poll.register(self.card.fd, select.POLLIN)

    @property
    def period_ns(self) -> Fraction:
        """The mode's exact vblank period in ns."""
        return Fraction(10**9) / refresh(self.mode)

    def fill(self, buf: int, data: bytes) -> None:
        """Copy one frame into a buffer."""
        if len(data) != self.frame_bytes:
            raise PlayoutFault(f"frame of {len(data)} bytes, buffer of {self.frame_bytes}")
        self.maps[buf][:] = data

    def _request(self, buf: int, *, active: bool | None = None) -> Any:
        """An atomic request putting ``buf`` on the primary plane, with the CRTC state when given."""
        req = self._pykms.AtomicReq(self.card)
        if active is not None:
            req.add_connector(self.conn, self.crtc)
            req.add(self.crtc, {"ACTIVE": int(active), "MODE_ID": self.blob.id})
        req.add_plane(self.plane, self.buffers[buf], self.crtc)
        return req

    def start(self, buf: int) -> None:
        """Put ``buf`` up, setting the mode only when it is not the one already driven."""
        done = self._request(buf, active=True if self.modeset else None).commit_sync(self.modeset)
        if done != 0:
            raise PlayoutFault(
                f"the first commit was refused ({os.strerror(-done)}): is another DRM master running?"
            )

    def set_active(self, active: bool, buf: int) -> None:
        """Turn the CRTC, and so the HDMI link, off or back on with ``buf`` showing."""
        done = self._request(buf, active=active).commit_sync(True)
        if done != 0:
            raise PlayoutFault(
                f"turning the link {'up' if active else 'down'} was refused: {os.strerror(-done)}"
            )

    def flip(self, buf: int, v: int) -> None:
        """Commit ``buf`` for the next vblank, tagged with its clip vblank (offset to stay unsigned)."""
        done = self._request(buf).commit(v + 1)
        if done != 0:
            raise PlayoutFault(f"flip for vblank {v} was refused: {os.strerror(-done)}")

    def wait(self, timeout_s: float) -> tuple[int, int, int] | None:
        """The next flip-complete event as ``(seq, t_ns, vblank)``, or None on timeout."""
        if not self._poll.poll(int(timeout_s * 1000)):
            return None
        for event in self.card.read_events():
            return int(event.seq), int(round(event.time * 1e9)), int(event.data) - 1
        return None

    def frame(self, buf: int) -> bytes:
        """The bytes a buffer holds."""
        return bytes(self.maps[buf])


def read_exactly(stream: Any, size: int) -> bytes | None:
    """One whole record from a pipe, or None when it ends first."""
    buf = bytearray(size)
    view, got = memoryview(buf), 0
    while got < size:
        more = stream.readinto(view[got:])
        if not more:
            return None
        got += more
    return bytes(buf)


def decode_argv(
    path: str,
    width: int,
    height: int,
    *,
    loop: bool,
    span: tuple[int, int] | None = None,
    rate: Fraction = Fraction(1),
    ffmpeg: str = "ffmpeg",
) -> list[str]:
    """ffmpeg decoding a clip to raw frames at the display's geometry and layout, as fast as it goes.

    ``span`` keeps the clip's frames ``a..b-1`` only, counted from its first, for a clip at ``rate``. The
    input is sought half a frame before frame ``a``, so the decoder starts at the keyframe before it and
    drops every frame earlier than ``a`` however its timestamps round, rather than decoding the clip from
    its start; ``trim`` then counts from ``a``.
    """
    seek, trim = [], ""
    if span is not None:
        seek = ["-ss", f"{float((span[0] - Fraction(1, 2)) / rate):.6f}"] if span[0] else []
        trim = f"trim=end_frame={span[1] - span[0]},"
    return [
        ffmpeg,
        "-nostdin",
        "-hide_banner",
        "-loglevel",
        "error",
        *(["-stream_loop", "-1"] if loop else []),
        *seek,
        "-i",
        path,
        "-map",
        "0:v:0",
        "-vf",
        f"{trim}scale={width}:{height}",
        "-fps_mode",
        "passthrough",
        "-pix_fmt",
        "bgr565le",
        "-f",
        "rawvideo",
        "pipe:1",
    ]


def _decoder(argv: list[str]) -> subprocess.Popen:
    """A decoder whose stderr a thread drains as it is written, keeping its last line in ``proc.said``.

    Unread, a decoder's warnings fill the pipe and block it mid-clip, and the player underruns.
    """
    proc = subprocess.Popen(  # pylint: disable=consider-using-with
        argv, stdout=subprocess.PIPE, stderr=subprocess.PIPE
    )
    proc.said = []

    def drain() -> None:
        for line in proc.stderr:
            if line.strip():
                proc.said[:] = [line]

    proc.drained = threading.Thread(target=drain, daemon=True)
    proc.drained.start()
    return proc


def _frames(proc: subprocess.Popen, size: int) -> Iterator[bytes]:
    """Whole frames of ``size`` bytes from a running decoder's stdout; a decoder error is a fault."""
    with proc:
        try:
            while (frame := read_exactly(proc.stdout, size)) is not None:
                yield frame
        finally:
            proc.kill()
            proc.drained.join()
    said = b"".join(proc.said).decode(errors="replace").strip()
    if proc.returncode not in (0, -signal.SIGKILL) and said:
        raise PlayoutFault(f"decoding failed: {said}")


def clip_frames(argv: list[str], size: int) -> Iterator[bytes]:
    """Whole frames of ``size`` bytes from a decoder's stdout; a decoder error is a fault."""
    yield from _frames(_decoder(argv), size)


def parse_spans(text: str) -> list[tuple[int, int]]:
    """Clip frame ranges given as ``a:b,a:b``, each ``a..b-1``."""
    spans = [tuple(int(x) for x in part.split(":")) for part in text.split(",") if part]
    if not spans or any(len(s) != 2 or not 0 <= s[0] < s[1] for s in spans):
        raise ValueError(f"spans must be a:b with 0 <= a < b, got {text!r}")
    return spans


def span_frames(
    path: str, width: int, height: int, spans: Iterable[tuple[int, int]], size: int, rate: Fraction
) -> Iterator[bytes]:
    """The clip's frames over each span in turn, as one stream with no seam, the clip at ``rate``.

    Each span has its own decoder, started while the span before plays, so its decoding up to the
    span's first frame is done by the time that frame is due.
    """
    argvs = [decode_argv(path, width, height, loop=False, span=span, rate=rate) for span in spans]
    upcoming = _decoder(argvs[0]) if argvs else None
    try:
        for index in range(len(argvs)):
            current, upcoming = upcoming, _decoder(argvs[index + 1]) if index + 1 < len(argvs) else None
            yield from _frames(current, size)
    finally:
        if upcoming is not None:
            upcoming.kill()
            upcoming.wait()


def clip_rate(path: str, *, ffprobe: str = "ffprobe") -> Fraction:
    """The clip's frame rate as its container states it."""
    out = subprocess.run(
        [
            ffprobe,
            "-v",
            "error",
            "-select_streams",
            "v:0",
            "-show_entries",
            "stream=r_frame_rate",
            "-of",
            "csv=p=0",
            path,
        ],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    return parse_rate(out)


def dir_frames(directory: str, *, loop: bool) -> Iterator[bytes]:
    """Raw frames from ``frame.*`` files, read into memory once and cycled when looping."""
    frames = []
    for path in sorted(glob.glob(os.path.join(directory, "frame.*"))):
        with open(path, "rb") as handle:
            frames.append(handle.read())
    while frames:
        yield from frames
        if not loop:
            return


def realtime(priority: int = REALTIME_PRIORITY) -> bool:
    """Put the calling thread, the one pacing flips, under ``SCHED_FIFO``; False when not permitted."""
    try:
        os.sched_setscheduler(0, os.SCHED_FIFO, os.sched_param(priority))
    except (PermissionError, OSError):
        return False
    return True


def write_status(path: str, status: dict) -> None:
    """Replace the status file atomically, so a reader never sees half of one."""
    tmp = f"{path}.tmp"
    with open(tmp, "w", encoding="utf-8") as out:
        json.dump(status, out)
    os.replace(tmp, path)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """The player's command line."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--clip", help="clip to decode")
    source.add_argument("--frames", help="directory of raw frame.* files")
    parser.add_argument("--fps", required=True, type=parse_rate)
    parser.add_argument("--loop", action="store_true")
    parser.add_argument("--spans", type=parse_spans, help="play only these clip frame ranges, a:b,a:b")
    parser.add_argument("--width", type=int, default=1920)
    parser.add_argument("--height", type=int, default=1080)
    parser.add_argument("--device", default="/dev/dri/card1")
    parser.add_argument("--connector", default="HDMI-A-1")
    parser.add_argument("--preroll-s", type=float, default=2.0)
    parser.add_argument("--flips", required=True, help="binary log of every flip")
    parser.add_argument("--status", required=True, help="JSON status file")
    parser.add_argument("--fb", default="/dev/fb0", help="fbdev left holding the last frame")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:  # pragma: no cover - needs a DRM device and pykms
    """Play a clip or a frame directory until stopped; exit 2 on any fault."""
    args = parse_args(argv)
    signals: dict = {}
    for sig, name in (
        (signal.SIGTERM, "stop"),
        (signal.SIGINT, "stop"),
        (signal.SIGUSR1, "down"),
        (signal.SIGUSR2, "up"),
    ):
        signal.signal(sig, lambda *_, key=name: signals.__setitem__(key, True))
    status: dict = {"state": "starting", "pid": os.getpid()}
    write_status(args.status, status)
    player = display = None
    try:
        if args.clip and clip_rate(args.clip) != args.fps:
            raise PlayoutFault(f"{args.clip} runs at {clip_rate(args.clip)} fps, not {args.fps}")
        display = KmsDisplay(args.device, args.connector, args.width, args.height, args.fps)
        if args.spans and not args.clip:
            raise PlayoutFault("spans play a clip, not a frame directory")
        if args.spans:
            frames = span_frames(
                args.clip, args.width, args.height, args.spans, display.frame_bytes, args.fps
            )
        elif args.clip:
            frames = clip_frames(
                decode_argv(args.clip, args.width, args.height, loop=args.loop), display.frame_bytes
            )
        else:
            frames = dir_frames(args.frames, loop=args.loop)
        preroll = Prerolled(frames, max(2, round(float(args.preroll_s * args.fps))))
        preroll.wait_full(timeout_s=60.0)
        with open(args.flips, "wb", buffering=0) as log:
            player = Player(
                display,
                preroll,
                hold=display.hold,
                period_ns=display.period_ns,
                record=lambda *fields: log.write(RECORD.pack(*fields)),
                signals=signals,
            )
            status.update(
                state="playing",
                mode=display.mode.to_string_long(),
                modeset=display.modeset,
                hold=display.hold,
                refresh=str(refresh(display.mode)),
                fps=str(args.fps),
                preroll=preroll.depth,
            )
            status["realtime"] = realtime()
            write_status(args.status, status)
            gc.collect()
            gc.freeze()
            gc.disable()
            player.run()
        status["state"] = "stopped"
        return 0
    except Exception as err:  # pylint: disable=broad-exception-caught
        said = str(err) if isinstance(err, PlayoutFault) else f"{type(err).__name__}: {err}"
        print(f"PLAYOUT FAULT: {said}", file=sys.stderr, flush=True)
        status.update(state="failed", error=said)
        return 2
    finally:
        if player is not None:
            status.update(
                flips=player.flips,
                segments=player.segments,
                missed=player.missed,
                ended=player.ended,
                worst_us=player.timeline.worst_ns / 1e3,
                phases_us=[p / 1e3 for p in player.timeline.phases_ns],
                t0_ns=player.timeline.t0,
            )
            if player.shown is not None and os.path.exists(args.fb):
                try:
                    with open(args.fb, "wb") as fb:
                        fb.write(display.frame(player.shown[1]))
                except OSError:
                    pass
        write_status(args.status, status)


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())

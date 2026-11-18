"""Capture recorded by ffmpeg, so nothing on the host competes with the rig.

Reading frames into the host per frame cannot hold a session's rate, while
ffmpeg reading the card in MJPEG can (docs/hardware.md).
"""

from __future__ import annotations

import ctypes
import json
import math
import shutil
import signal
import subprocess
import sys
import tempfile
import time
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Iterator

import numpy as np
from pyvmancer.route import describe_route, route_name

from .capture import QUANTUM
from .transport import transmitting
from .v4l2rec import read_log

__all__ = [
    "FFV1",
    "MJPEG",
    "RAW_422",
    "BlankTakeError",
    "InputRouteError",
    "Recorder",
    "RecorderError",
    "RouteError",
    "SIGMA",
    "SignalLostError",
    "TakeReport",
    "await_transmitting",
    "die_with_parent",
    "held",
    "inspect_luma",
    "inspect_take",
    "require_ffmpeg",
    "require_input",
    "require_route",
    "rig_state",
    "require_signal",
    "settle",
]

#: Compressed by the card; the only mode measured above session rate on this hardware. Clusters
#: are cut short and flushed, so a take can be read while it is still being recorded.
MJPEG = ("mjpeg", "-c:v", "copy", "-flush_packets", "1", "-cluster_time_limit", "200")
#: The card's own samples, unencoded, for a scratch pass that is read once.
RAW_422 = ("yuyv422", "-c:v", "rawvideo")
#: Lossless intra-only at session rate, with the exchanged chroma pair put back.
FFV1 = (
    "yuyv422",
    "-vf",
    "swapuv",
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
    "yuv422p",
)
#: Linux prctl option tying a child's lifetime to its parent's.
PR_SET_PDEATHSIG = 1
#: Size every take is judged at, point-sampled so each sample is a pixel the card
#: delivered: blending would invent mid-tones at edges and average away the
#: noise the motion test is calibrated against.
INSPECT_SIZE = (160, 90)
#: Standard errors within which a statistic counts as equal to its null value:
#: the conventional three-sigma bound, a 0.27% chance of rejecting a true null.
SIGMA = 3.0


class RecorderError(RuntimeError):
    """The recording did not start, or did not produce a file."""


def require_ffmpeg(ffmpeg: str = "ffmpeg") -> str:
    """Resolved path to the ffmpeg binary, or a helpful error before any rig time is spent."""
    found = shutil.which(ffmpeg)
    if found is None:
        raise RecorderError(f"{ffmpeg!r} not found on PATH; install ffmpeg before running this command")
    return found


def die_with_parent() -> None:
    """Ask the kernel to signal this child when its parent dies.

    A recorder holds the capture card, and the harness being killed skips every
    ``finally`` that would have stopped it, so the card stays held by a process
    nobody is waiting on. ``PR_SET_PDEATHSIG`` covers the signal a handler cannot.
    """
    try:
        ctypes.CDLL("libc.so.6", use_errno=True).prctl(PR_SET_PDEATHSIG, signal.SIGTERM)
    except OSError:
        pass


def _diagnostic(errors: Any, *, limit: int = 400) -> str:
    """The tail of what ffmpeg said, for an error that has to be actionable."""
    try:
        errors.seek(0)
        said = errors.read().decode("utf-8", "replace").strip()
    except (OSError, ValueError, AttributeError):
        return "no diagnostic"
    return said[-limit:] if said else "no diagnostic"


class BlankTakeError(RuntimeError):
    """The device never returned a picture, or the take held none that moved."""


class SignalLostError(BlankTakeError):
    """The Videomancer's transmitter was not driving the session timing when it had to be."""


def _lost(program: str, timing: str, status: Any, when: str) -> SignalLostError:
    """The error for a transmitter not driving ``timing``, carrying the status that said so."""
    known = transmitting(status, timing) is not None
    why = f"the transmitter is not driving {timing}" if known else "the firmware reports no transmitter state"
    return SignalLostError(f"{program}: {why} {when}; last video status {json.dumps(status.raw)}")


class InputRouteError(RuntimeError):
    """The device has selected an input other than the one the session feeds."""


def require_input(transport: Any, *, program: str, source: str | None) -> Any:
    """The video status, when the device reads from ``source``; :class:`InputRouteError` if not.

    Under Vid Route Mode Auto a dropped source link can leave the device on
    another input, black, with nothing else amiss. None skips the check.
    """
    status = transport.video_status()
    if source is not None and str(status.input_source).lower() != source.lower():
        raise InputRouteError(
            f"{program}: the device reads input {status.input_source!r}, the session feeds {source!r}; "
            "check front panel SETTINGS > Vid Route Mode"
        )
    return status


class RouteError(InputRouteError):
    """Front panel Vid Route Mode reads other than the value the session needs."""


def require_route(transport: Any, *, program: str, route: int | None) -> int | None:
    """Vid Route Mode's settings value, when it reads ``route``; :class:`RouteError` if not.

    Under another route the device can report the session's input locked and transmit while
    every program outputs black, so nothing in ``video status`` shows it. None skips the check.
    """
    if route is None:
        return None
    value = transport.route()
    if value != route:
        raise RouteError(
            f"{program}: Vid Route Mode reads {describe_route(value)}, the session needs "
            f"{describe_route(route)}; set SETTINGS > Vid Route Mode to {route_name(route)} "
            "on the front panel"
        )
    return value


def rig_state(transport: Any) -> str:
    """Vid Route Mode, the selected input and its lock, and the transmitter, as one line; never raises."""
    try:
        status = transport.video_status()
        sent = status.raw.get("hdmi_tx") or {}
        lock = "locked" if status.source_locked else "unlocked"
        return (
            f"Vid Route Mode {describe_route(transport.route())}, input {status.input_source} {lock}, "
            f"transmitter vic {sent.get('vic')} de {sent.get('de')}"
        )
    except Exception as exc:
        return f"rig state unread: {type(exc).__name__}: {exc}"


def await_transmitting(
    transport: Any,
    *,
    program: str,
    timing: str,
    poll_s: float = 0.5,
    sleep: Callable[[float], None] = time.sleep,
) -> Any:
    """Wait, on liveness, for the transmitter to drive ``timing``; returns that video status.

    The transmitter coming back after a load is the event waited for, so there
    is no deadline: every answered ``video status`` shows the device alive, and
    the transport's watch ends the wait if it leaves USB or, when its opt-in
    stall guard is set, if nothing in the status changes for that long. A
    firmware that reports no transmitter state cannot show the event, so that
    raises :class:`SignalLostError` at once.
    """
    watch = transport.watch(f"waiting for {program} to transmit {timing}")
    status = transport.video_status()
    while not (sending := transmitting(status, timing)):
        if sending is None:
            raise _lost(program, timing, status, "after the load")
        shown = json.dumps([status.raw.get(key) for key in ("output", "hdmi_tx")], sort_keys=True)
        watch.observe("transmitter down", shown)
        sleep(poll_s)
        status = transport.video_status()
    return status


def require_signal(transport: Any, *, program: str, timing: str, when: str) -> Any:
    """The video status, when it shows the transmitter driving ``timing``; :class:`SignalLostError` if not.

    A pass brackets its recording with this, so a transmitter that dropped
    while it ran fails the pass rather than archiving the card's own frames.
    """
    status = transport.video_status()
    if not transmitting(status, timing):
        raise _lost(program, timing, status, when)
    return status


@dataclass(frozen=True)
class TakeReport:
    """What a finished take actually holds, so a dead one is caught here and not on playback.

    ``picture`` counts frames with spatial structure, and the rest are uniform
    fields: a take is recorded only while the transmitter drives the session
    timing, so every frame is the program's own output. ``spread`` is
    the median frame's luma range in 8-bit steps, and ``motion`` the mean
    correlation of consecutive frame differences, -0.5 for a held picture, and
    None when too few distinct frames define one.
    """

    frames: int = 0
    picture: int = 0
    spread: int = 0
    motion: float | None = None
    frozen: bool = True
    luma: float = 0.0

    @property
    def mostly_picture(self) -> bool:
        """Whether most frames carry a picture, moving or not: a render's and a passthrough's load test."""
        return 2 * self.picture > self.frames

    @property
    def usable(self) -> bool:
        """Whether the take carries a picture that is not held still: a render's standard."""
        return self.mostly_picture and not self.frozen

    @property
    def ever_structured(self) -> bool:
        """Whether any frame holds spatial structure: an archive's standard.

        A sweep visits setpoints whose output is a flat field and a generator may
        hold one picture, so neither motion nor a majority is required: only a
        take of uniform fields alone is dark.
        """
        return self.picture > 0

    def __str__(self) -> str:
        verdict = "" if self.usable else "  UNUSABLE: " + ("frozen" if self.mostly_picture else "no picture")
        motion = "n/a" if self.motion is None else f"{self.motion:+.2f}"
        return (
            f"{self.frames} frames: {self.picture} picture, {self.frames - self.picture} uniform; "
            f"luma range {self.spread}, "
            f"motion {motion}, mean luma {self.luma:.3f}{verdict}"
        )


def held(frames: np.ndarray) -> tuple[float | None, bool]:
    """Mean correlation of consecutive frame differences, and whether they show one picture held.

    A held picture is one image plus noise independent from frame to frame, so
    consecutive differences ``n[t+1] - n[t]`` and ``n[t+2] - n[t+1]`` share one
    noise term with opposite sign and correlate at exactly -1/2, whatever the
    noise's size or spatial texture: the take is its own noise floor. Motion
    correlates them positively, flicker towards -1. Snow also gives -1/2, but
    shares no picture across frames, which a one-way analysis of variance over
    pixels tells apart: a held picture's temporal mean varies across the frame
    by more than averaging its residuals could. Exact repeats carry no
    evidence either way and are dropped; fewer than four distinct frames
    cannot show motion.
    """
    if len(frames) < 4:
        return None, True
    flat = frames.reshape(len(frames), -1)
    flat = flat[np.r_[True, np.any(flat[1:] != flat[:-1], axis=1)]]
    if len(flat) < 4:
        return None, True
    step = np.diff(flat, axis=0)
    norm = np.linalg.norm(step, axis=1)
    corr = np.einsum("ij,ij->i", step[:-1], step[1:]) / (norm[:-1] * norm[1:])
    motion = float(corr.mean())
    noise = abs(motion + 0.5) <= SIGMA * corr.std(ddof=1) / math.sqrt(len(corr))
    count, pixels = flat.shape
    mean = flat.mean(axis=0)
    within = float(np.square(flat - mean).sum()) / (pixels * (count - 1))
    ratio = count * float(mean.var(ddof=1)) / within
    shared = ratio > 1.0 + SIGMA * math.sqrt(2.0 / (pixels - 1) + 2.0 / (pixels * (count - 1)))
    return motion, bool(noise and shared)


class Recorder:
    """Records the capture card to a file for the length of a pass.

    ``mode`` is the card format to ask for followed by the output arguments to
    store it with: :data:`MJPEG` for a take that only has to look right,
    :data:`FFV1` for an archive that must not lose a sample.
    """

    def __init__(
        self,
        device: str = "/dev/video0",
        *,
        width: int = 1920,
        height: int = 1080,
        fps: float = 30.0,
        mode: tuple[str, ...] = MJPEG,
        ffmpeg: str = "ffmpeg",
        python: str = sys.executable,
        popen: Callable[..., Any] = subprocess.Popen,
        sleep: Callable[[float], None] = time.sleep,
    ):
        self.device = device
        self.width = int(width)
        self.height = int(height)
        self.fps = float(fps)
        self.mode = tuple(mode)
        self.ffmpeg = ffmpeg
        self.python = python
        self._popen = popen
        self._sleep = sleep

    def command(self, path: str | Path, *, seconds: float | None = None) -> list[str]:
        """Recorder argv: :mod:`~syncsummoner.device.v4l2rec` reading the card, ffmpeg storing it.

        ``seconds`` becomes a frame count at the card's own rate, so a take holds exactly that
        many frames; every frame's V4L2 sequence and timestamp go to the :func:`~.v4l2rec.sidecar`.
        """
        argv = [
            self.python,
            "-m",
            "syncsummoner.device.v4l2rec",
            "--device",
            self.device,
            "--width",
            str(self.width),
            "--height",
            str(self.height),
            "--fps",
            f"{self.fps:g}",
            "--input-format",
            self.mode[0],
            "--ffmpeg",
            self.ffmpeg,
        ]
        if seconds is not None:
            argv += ["--seconds", f"{seconds:.3f}"]
        return argv + [str(path), "--", *self.mode[1:]]

    @contextmanager
    def recording(
        self, path: str | Path, *, seconds: float | None = None, settle_s: float = 1.5
    ) -> Iterator[Any]:
        """Record for the duration of the block, yielding once it is running.

        ffmpeg's diagnostics are kept rather than discarded: a recorder that dies
        silently is a fault nobody can tell from a rig that has gone dark. The
        ``preexec_fn`` warning does not apply: a pass records from one thread.
        """
        errors = tempfile.TemporaryFile()
        try:
            proc = self._popen(  # pylint: disable=subprocess-popen-preexec-fn
                self.command(path, seconds=seconds),
                stdin=subprocess.PIPE,
                stdout=subprocess.DEVNULL,
                stderr=errors,
                preexec_fn=die_with_parent,
            )
            self._sleep(settle_s)
            if proc.poll() is not None:
                raise RecorderError(
                    f"the recorder exited {proc.returncode} for {self.device}: {_diagnostic(errors)}"
                )
            try:
                yield proc
            finally:
                self.stop(proc)
            written = Path(path)
            if not written.exists() or not written.stat().st_size:
                raise RecorderError(f"the recorder wrote nothing to {path}: {_diagnostic(errors)}")
        finally:
            errors.close()

    def roll(self, proc: Any, path: str | Path, opens_ns: int, closes_ns: int) -> None:
        """Send captures from ``opens_ns`` on to ``path``, those before ``closes_ns`` to the file so far too.

        Times are the capture clock's; the card keeps streaming, so both files share the captures between.
        """
        proc.stdin.write(f"roll {int(opens_ns)} {int(closes_ns)} {path}\n".encode())
        proc.stdin.flush()

    def stop(self, proc: Any, *, timeout_s: float = 20.0) -> int:
        """Ask ffmpeg to finish the file, and wait for it to do so."""
        try:
            proc.stdin.write(b"q")
            proc.stdin.flush()
            proc.stdin.close()
        except (BrokenPipeError, OSError, ValueError, AttributeError):
            pass
        try:
            return int(proc.wait(timeout=timeout_s) or 0)
        except subprocess.TimeoutExpired:
            proc.kill()
            return int(proc.wait() or 0)

    def timestamps(self, path: str | Path) -> np.ndarray:
        """When the card delivered each recorded frame, in seconds of ``CLOCK_MONOTONIC``.

        These are the driver's own buffer times from the sidecar, the clock
        :func:`time.monotonic` reads, so the host's schedule and the recording share one timeline.
        """
        return read_log(path)["t_ns"] / 1e9

    def frame_log(self, path: str | Path) -> np.ndarray:
        """Every stored frame's V4L2 sequence, timestamp, size and flags."""
        return read_log(path)


def inspect_take(
    path: str | Path, *, ffmpeg: str = "ffmpeg", samples: int = 240, start_s: float = 0.0
) -> TakeReport:
    """Judge a finished take by structure and motion, never by brightness.

    A frame is picture when its luma ranges over more than one
    :data:`~syncsummoner.device.capture.QUANTUM`, the uniform-field bound
    :func:`~syncsummoner.device.capture.structured` applies; whether the take
    moves is :func:`held`. ``start_s`` skips a lead-in.
    """
    from syncsummoner.aesthetics.levels import LUMA_709  # keeps scipy off the device import path

    from .stored import captures

    width, height = INSPECT_SIZE
    with captures(path, start_s=start_s, ffmpeg=ffmpeg) as source:
        argv = [ffmpeg, "-loglevel", "error", *source, "-vf"]
        argv += [f"scale={width}:{height}:flags=neighbor,fps=source_fps/{max(1, samples // 60)}"]
        argv += ["-f", "rawvideo", "-pix_fmt", "rgb24", "pipe:1"]
        data = subprocess.run(argv, check=False, capture_output=True).stdout
    stride = width * height * 3
    count = len(data) // stride
    if not count:
        return TakeReport()
    rgb = np.frombuffer(data, np.uint8, count * stride).reshape(count, height, width, 3)
    weights = (LUMA_709 / 255.0).astype(np.float32)
    return luma_report(np.einsum("...c,c->...", rgb, weights, dtype=np.float32, casting="unsafe"))


def inspect_luma(frames: np.ndarray, *, start: int = 0, every: int = 4) -> TakeReport:
    """:func:`inspect_take` over decoded 8-bit luma: every ``every``-th frame from ``start``."""
    picked = np.asarray(frames)[max(0, start) :: max(1, every)]
    if not picked.size:
        return TakeReport()
    return luma_report(picked.astype(np.float32) / np.float32(255.0))


def luma_report(luma: np.ndarray) -> TakeReport:
    """A take's :class:`TakeReport` from its sampled luma in ``[0, 1]``."""
    count = len(luma)
    steps = np.rint(np.ptp(luma.reshape(count, -1), axis=1) / QUANTUM)
    motion, frozen = held(luma)
    return TakeReport(
        frames=count,
        picture=int(np.count_nonzero(steps > 1)),
        spread=int(np.median(steps)),
        motion=motion,
        frozen=frozen,
        luma=float(luma.mean()),
    )


def settle(
    recorder: Recorder,
    transport: Any,
    *,
    program: str,
    timing: str,
    probe_timeout_s: float,
    probe_path: str | Path,
    probe_s: float = 2.0,
    poll_s: float = 0.5,
    clock: Callable[[], float] = time.monotonic,
    sleep: Callable[[float], None] = time.sleep,
) -> TakeReport:
    """Wait out a load: the transmitter driving ``timing`` first, then a probe the card delivers.

    The card reports no signal state of its own, so :func:`await_transmitting`
    waits, on liveness, for the Videomancer to drive the session timing before
    anything is recorded. Once it does every frame is the program's, so the
    first probe that holds frames is returned for the caller to judge; a
    recorder that will not start yet is retried until ``probe_timeout_s``, a
    budget for the card and the recorder, not for the device.
    """
    await_transmitting(transport, program=program, timing=timing, poll_s=poll_s, sleep=sleep)
    deadline = clock() + probe_timeout_s
    probe = Path(str(probe_path))
    said = "no recording finished"
    try:
        while clock() < deadline:
            try:
                with recorder.recording(probe, seconds=probe_s):
                    sleep(probe_s)
            except RecorderError as exc:
                said = str(exc)
                continue
            report = inspect_take(probe, ffmpeg=recorder.ffmpeg)
            if report.frames:
                return report
            said = str(report)
    finally:
        probe.unlink(missing_ok=True)
    raise BlankTakeError(
        f"{program}: transmitting {timing} but nothing captured within {probe_timeout_s}s ({said})"
    )

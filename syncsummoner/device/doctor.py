"""Rig diagnostics: source host, capture card and device, as one report of named findings.

Every check is a :class:`pyvmancer.diagnose.Finding`, so the device's own checks
and the rig's land in one uniform report. Every endpoint is injected, so each
check is exercisable with no rig attached.
"""

from __future__ import annotations

import re
import subprocess
import tempfile
import time
from pathlib import Path
from typing import Any, Callable, Iterable, NamedTuple, Sequence

import numpy as np
from pyvmancer.const import ParamKind as Control
from pyvmancer.const import param_kind
from pyvmancer.diagnose import Finding, diagnose, diagnose_latency
from pyvmancer.errors import VmancerError
from pyvmancer.latency import DEFAULT_RATES, measure_latency

from syncsummoner.probe.patterns import generate, read_state_index, state_index_capacity, with_state_index
from syncsummoner.progress import LOG, stage

from .capture import structured
from .host import DEFAULT_HOST, HostCommandError, Runner, ssh_runner
from .link import DEFAULT_BLANK, UNBLANK, Link
from .playout import Playout
from .profile import ParamKind
from .recorder import MJPEG, Recorder, RecorderError, require_ffmpeg
from .transport import transmitting

CURSOR_BLINK = "/sys/class/graphics/fbcon/cursor_blink"
FB_SIZE = "/sys/class/graphics/fb0/virtual_size"
FB_DEPTH = "/sys/class/graphics/fb0/bits_per_pixel"
#: The depth :func:`~syncsummoner.device.playout.to_fb565` packs frames at.
FB_BPP = 16
#: SuperSpeed, below which the card offers only SD uncompressed modes.
USB3_MBPS = 5000
PASSTHRU = "Passthru"
#: Colour of a bar from which of R, G, B it carries.
BAR_NAMES = {
    (1, 1, 1): "white",
    (1, 1, 0): "yellow",
    (0, 1, 1): "cyan",
    (0, 1, 0): "green",
    (1, 0, 1): "magenta",
    (1, 0, 0): "red",
    (0, 0, 1): "blue",
    (0, 0, 0): "black",
}
SMPTE_BARS = 7
KMS_CRTC = re.compile(r"Crtc \d+ \(\d+\) (\d+)x(\d+)@([\d.]+)")
V4L2_FORMAT = re.compile(r"(?:Compressed|Raw)\s*:\s*(\S+)\s*:.* : ([\dx ]+)$")


class Format(NamedTuple):
    """A session format: its name, as the device names its timing, and its raster."""

    name: str
    width: int
    height: int
    fps: float


def _check(check: str, ok: bool, detail: str, fix: str) -> Finding:
    """A finding whose ``fix`` is kept only when it failed."""
    return Finding(check, bool(ok), detail, "" if ok else fix)


def list_formats(node: str, *, ffmpeg: str = "ffmpeg") -> str:
    """What ffmpeg's v4l2 demuxer says the card offers; it reports on stderr."""
    argv = [require_ffmpeg(ffmpeg), "-hide_banner", "-f", "v4l2", "-list_formats", "all", "-i", node]
    return subprocess.run(argv, check=False, capture_output=True, text=True).stderr


def parse_formats(listing: str) -> dict[str, set[str]]:
    """Frame sizes the card offers per pixel format, from a ``-list_formats`` listing."""
    offered: dict[str, set[str]] = {}
    for match in map(V4L2_FORMAT.search, listing.splitlines()):
        if match:
            offered.setdefault(match[1], set()).update(match[2].split())
    return offered


def usb_speed(sysfs: Path, node: str) -> int | None:
    """Link speed in Mbit/s of the USB device carrying a video node's function.

    The nearest ``speed`` above the interface is the video function's own; a hub
    built into the card for its other functions sits further up and is not read.
    """
    path = (sysfs / "class/video4linux" / Path(node).name / "device").resolve()
    for parent in (path, *path.parents):
        speed = parent / "speed"
        if speed.is_file():
            return int(float(speed.read_text().strip()))
    return None


def bar_channels(frame: np.ndarray) -> np.ndarray:
    """Which of R, G, B each SMPTE bar carries, against the bars' own mid level.

    Bar-centre columns through the middle of the band are reduced by median and
    thresholded halfway between the darkest and brightest channel, which no gain
    or pedestal moves.
    """
    height, width = frame.shape[:2]
    centres = np.linspace(0, width, 2 * SMPTE_BARS + 1).astype(int)[1::2]
    levels = np.median(frame[height // 6 : height // 2, centres], axis=0)
    return levels > 0.5 * (levels.max() + levels.min())


def bar_names(channels: np.ndarray) -> list[str]:
    """Colour names for per-bar channel sets, as :func:`bar_channels` returns them."""
    return [BAR_NAMES[tuple(int(v) for v in carried)] for carried in channels]


def last_frame(path: str | Path, width: int, height: int, *, ffmpeg: str = "ffmpeg") -> np.ndarray | None:
    """A frame from the end of an MJPEG take, as RGB float32 in ``[0, 1]``.

    The card's chroma pair arrives exchanged in MJPEG too, so it is put back here
    as the archive recorder does, before any colour is read.
    """
    argv = [ffmpeg, "-loglevel", "error", "-sseof", "-0.5", "-i", str(path), "-vf", "swapuv"]
    argv += ["-frames:v", "1", "-f", "rawvideo", "-pix_fmt", "rgb24", "pipe:1"]
    data = subprocess.run(argv, check=False, capture_output=True).stdout
    size = width * height * 3
    if len(data) < size:
        return None
    return np.frombuffer(data[:size], np.uint8).reshape(height, width, 3).astype(np.float32) / 255.0


def latency_param(info: Any) -> int:
    """A program's first continuous parameter on a knob, short of the crossfader; else P1."""
    return next(
        (
            p.index
            for p in info.sweepable()
            if p.kind is ParamKind.CONTINUOUS and param_kind(p.index) is not Control.TOGGLE
        ),
        1,
    )


def write_latency(transport: Any, *, measure: Callable[..., Any] = measure_latency) -> float | None:
    """Median CC write to readback latency on the loaded program, or ``None`` when unmeasured.

    Single writes only, on :func:`latency_param`: the rate sweep is ``device doctor
    --latency``'s. A refusal, a write that never converges or a missing MIDI link is
    logged and gives ``None``, so the caller keeps its own figure.
    """
    device = transport.device
    if not device.has_midi:
        LOG.warning("write latency not measured: no MIDI link is open")
        return None
    try:
        measured = measure(
            device, params=(latency_param(transport.program_info()),), rates=(), report=LOG.info
        )
    except (ValueError, VmancerError) as err:
        LOG.warning("write latency not measured: %s", err)
        return None
    if not measured.restored:
        LOG.warning(
            "P%s did not read back %s after the latency probe", list(measured.params), list(measured.prior)
        )
    if measured.single_median is None:
        LOG.warning("write latency not measured: a write on P%s never read back", list(measured.params))
    return measured.single_median


class Doctor:
    """Runs the rig's checks; every endpoint is injected, and opened only when used.

    ``runner`` reaches the source host, ``sysfs`` roots the card's USB lookup,
    ``formats`` lists what the card offers and ``open_transport`` opens the device.
    Only the loop check, through ``playout``, ``link`` and ``recorder``, and the
    latency check, through ``measure``, change state.
    """

    def __init__(
        self,
        fmt: Format,
        *,
        host: str = DEFAULT_HOST,
        capture: str = "/dev/video0",
        runner: Runner | None = None,
        sysfs: Path = Path("/sys"),
        formats: Callable[[str], str] = list_formats,
        open_transport: Callable[[], Any] | None = None,
        playout: Any = None,
        link: Any = None,
        recorder: Any = None,
        measure: Callable[..., Any] = measure_latency,
        latency_params: Sequence[int] | None = None,
        latency_rates: Sequence[float] | None = None,
        rng: np.random.Generator | None = None,
        settle_s: float = 40.0,
        probe_s: float = 1.0,
        sleep: Callable[[float], None] = time.sleep,
        clock: Callable[[], float] = time.monotonic,
    ):
        self.fmt = fmt
        self.capture = capture
        self._runner = ssh_runner(host) if runner is None else runner
        self._sysfs = Path(sysfs)
        self._formats = formats
        if open_transport is None:
            from .transport import Transport

            open_transport = Transport.open
        self._open_transport = open_transport
        self._playout = playout or Playout(host, width=fmt.width, height=fmt.height, runner=runner)
        self._link = link or Link(host, runner=runner)
        self._recorder = recorder or Recorder(
            capture, width=fmt.width, height=fmt.height, fps=fmt.fps, mode=MJPEG
        )
        self._measure = measure
        self.latency_params = None if latency_params is None else tuple(latency_params)
        self.latency_rates = DEFAULT_RATES if latency_rates is None else tuple(latency_rates)
        self._rng = np.random.default_rng() if rng is None else rng
        self.settle_s = float(settle_s)
        self.probe_s = float(probe_s)
        self._sleep = sleep
        self._clock = clock

    def source(self) -> list[Finding]:
        """Source host: reachable, link up, no console cursor, raster at the session format."""
        files = (DEFAULT_BLANK, CURSOR_BLINK, FB_SIZE, FB_DEPTH)
        reads = "; ".join(f'echo "=$(cat {path} 2>/dev/null)"' for path in files)
        try:
            out = self._runner(f"{reads}; kmsprint 2>/dev/null || true").splitlines()
        except HostCommandError as err:
            return [
                Finding(
                    "ssh.reachable",
                    False,
                    f"{err}; source host checks skipped",
                    "check the Pi is up and its key is authorized for BatchMode ssh",
                )
            ]
        blank, blink, size, depth = (line.strip().removeprefix("=") for line in out[: len(files)])
        fmt, want = self.fmt, f"{self.fmt.width},{self.fmt.height}"
        found = [
            Finding("ssh.reachable", True, "source host answers"),
            _check(
                "source.fb_unblanked",
                blank == str(UNBLANK),
                f"{DEFAULT_BLANK} reads {blank or 'nothing'}",
                "the framebuffer is powered down, so the device has no source: run with --fix",
            ),
            _check(
                "source.cursor_blink",
                blink == "0",
                f"{CURSOR_BLINK} reads {blink or 'nothing'}",
                "the console cursor flickers into every take: run with --fix",
            ),
            _check(
                "source.fb_geometry",
                size == want and depth == str(FB_BPP),
                f"framebuffer {size.replace(',', 'x')} at {depth} bpp",
                f"set the Pi's framebuffer to {fmt.width}x{fmt.height} at {FB_BPP} bpp",
            ),
        ]
        mode = KMS_CRTC.search("\n".join(out[len(files) :]))
        if mode:
            width, height, rate = int(mode[1]), int(mode[2]), mode[3]
            found.append(
                _check(
                    "source.mode",
                    (width, height, rate) == (fmt.width, fmt.height, f"{fmt.fps:.2f}"),
                    f"HDMI out {width}x{height}@{rate}",
                    f"set the Pi's HDMI mode to {fmt.width}x{fmt.height}@{fmt.fps:g} to match {fmt.name}",
                )
            )
        return found

    def card(self) -> list[Finding]:
        """Capture card: on a SuperSpeed link, and offering the recorder's mode at session size."""
        speed = usb_speed(self._sysfs, self.capture)
        if speed is None:
            return [
                Finding(
                    "capture.present",
                    False,
                    f"{self.capture} is not a USB video node",
                    "plug the capture card in and check it enumerates under uvcvideo",
                )
            ]
        size, want = f"{self.fmt.width}x{self.fmt.height}", self._recorder.mode[0]
        try:
            offered = parse_formats(self._formats(self.capture))
            said = ", ".join(sorted(n for n, sizes in offered.items() if size in sizes)) or "nothing"
        except RecorderError as err:
            offered, said = {}, str(err)
        return [
            _check(
                "capture.usb_speed",
                speed >= USB3_MBPS,
                f"{self.capture} on a {speed} Mbit/s link",
                "plug the card directly into a USB 3 port with a USB 3 cable",
            ),
            _check(
                "capture.mode",
                size in offered.get(want, ()),
                f"{size} offered as {said}",
                f"the card does not offer {want} at {size}: put it on USB 3, and check the source "
                f"is {self.fmt.name}",
            ),
        ]

    def device(self, transport: Any) -> list[Finding]:
        """The device's own checks, and that its output timing is the session format."""
        timing = transport.video_status().output_timing
        device = transport.device
        return diagnose(device.shell, device.midi if device.has_midi else None) + [
            _check(
                "device.format",
                str(timing).lower() == self.fmt.name.lower(),
                f"output timing {timing}, session {self.fmt.name}",
                f"feed the device {self.fmt.name} and clear any front panel timing override "
                "(SETTINGS > Vid Route Mode, SYSTEM > Pref. Timing)",
            )
        ]

    def latency(self, transport: Any) -> list[Finding]:
        """pyvmancer's ``midi.latency`` and ``midi.backlog``; writes, then restores, parameters.

        Without ``latency_params`` the loaded program's :func:`latency_param` is measured.
        """
        device = transport.device
        if not device.has_midi:
            return [
                Finding(
                    "midi.latency",
                    False,
                    "skipped: no MIDI link is open",
                    "check the device's USB MIDI port enumerates (`vmancer list`)",
                )
            ]
        try:
            params = self.latency_params or (latency_param(transport.program_info()),)
            with stage("latency", params=",".join(map(str, params))) as result:
                measured = self._measure(device, params=params, rates=self.latency_rates, report=LOG.info)
                result.update(median_s=measured.single_median, restored=measured.restored)
        except (ValueError, VmancerError) as err:
            return [
                Finding(
                    "midi.latency",
                    False,
                    f"skipped: {err}",
                    "choose unmodulated knobs with headroom with --latency-params",
                )
            ]
        found = diagnose_latency(measured) or [
            Finding("midi.latency", False, f"skipped: output timing {measured.timing} has no frame period")
        ]
        if not measured.restored:
            found.append(
                Finding(
                    "midi.restored",
                    False,
                    f"P{list(measured.params)} did not read back {list(measured.prior)}",
                    "the probed parameters were left offset: reload the program",
                )
            )
        return found

    def _grab(self, index: int) -> np.ndarray | None:
        """Record short probes until the pushed frame comes back, or the budget runs out."""
        deadline, frame = self._clock() + self.settle_s, None
        with tempfile.TemporaryDirectory() as scratch:
            probe = Path(scratch) / "doctor.mkv"
            while self._clock() < deadline:
                try:
                    with self._recorder.recording(probe, seconds=self.probe_s):
                        self._sleep(self.probe_s)
                except RecorderError:
                    self._sleep(self.probe_s)
                    continue
                frame = last_frame(probe, self.fmt.width, self.fmt.height, ffmpeg=self._recorder.ffmpeg)
                if frame is not None and read_state_index(frame) == index:
                    break
        return frame

    def loop(self, transport: Any) -> list[Finding]:
        """Bars with a random state index, pushed through ``Passthru`` and captured back."""
        from .session import Session

        fmt = self.fmt
        index = int(self._rng.integers(state_index_capacity()))
        bars = generate("smpte_bars", width=fmt.width, height=fmt.height, rng=self._rng)
        with stage("loop", program=PASSTHRU, index=index) as result:
            Session(transport, sleep=self._sleep, clock=self._clock).load_program(PASSTHRU, link=self._link)
            self._playout.show(with_state_index(bars, index))
            frame = self._grab(index)
            result["captured"] = frame is not None
            sending = transmitting(transport.video_status(), fmt.name)
            result["transmitting"] = sending
        if not sending:
            state = "reports no transmitter state" if sending is None else f"is not transmitting {fmt.name}"
            said = f"the device {state}"
            hint = "the device sends no video: check `video status` and the device.format check"
        elif frame is None:
            said = "nothing captured"
            hint = "the device transmits but the card delivers nothing: check that HDMI cable and the card"
        elif not structured(frame):
            said = "the device outputs a uniform field"
            hint = "the device sends video but no picture: check its input and that Passthru loaded"
        else:
            said, hint = "picture captured", ""
        found = [_check("loop.signal", not hint, said, hint)]
        if hint:
            return found
        read, want = bar_names(bar_channels(frame)), bar_names(bar_channels(bars))
        return found + [
            _check(
                "loop.index",
                read_state_index(frame) == index,
                f"strip reads {read_state_index(frame)}, pushed {index}",
                "the capture is not the pushed frame: a stale framebuffer, a second writer on it, "
                "or a cropped path",
            ),
            _check(
                "loop.bars",
                read == want,
                f"bars read {' '.join(read)}",
                "colour order is wrong along the chain (a red/blue swap reads cyan for yellow); "
                "check the framebuffer's 565 order, and that the card's chroma exchange is put back",
            ),
        ]

    def fix(self, findings: Iterable[Finding]) -> list[str]:
        """Apply the safe source host remediations for the failed checks; returns what was done."""
        failed = {f.check for f in findings if not f.ok}
        done = []
        if "source.fb_unblanked" in failed:
            done.append(f"link {self._link.up()}")
        if "source.cursor_blink" in failed:
            self._runner(f"echo 0 | sudo -n tee {CURSOR_BLINK} > /dev/null")
            done.append("cursor blink off")
        return done

    def checks(self, *, loop: bool = False, latency: bool = False) -> list[Finding]:
        """Every check, the device opened once for its own checks, the latency and the loop."""
        found = self.source() + self.card()
        try:
            transport = self._open_transport()
        except Exception as err:
            return found + [
                Finding(
                    "device.reachable",
                    False,
                    str(err),
                    "check the Videomancer's USB cable and serial permissions (dialout or a udev rule)",
                )
            ]
        try:
            found += self.device(transport)
            if latency:
                found += self.latency(transport)
            if loop:
                reachable = any(f.check == "ssh.reachable" and f.ok for f in found)
                found += (
                    self.loop(transport)
                    if reachable
                    else [Finding("loop", False, "skipped: the source host is unreachable")]
                )
        finally:
            transport.close()
        return found

    def run(self, *, loop: bool = False, latency: bool = False, fix: bool = False) -> list[Finding]:
        """Run the checks; with ``fix``, remediate what is safe and run them again."""
        found = self.checks(loop=loop, latency=latency)
        applied = self.fix(found) if fix else []
        for action in applied:
            LOG.info("fixed: %s", action)
        return self.checks(loop=loop, latency=latency) if applied else found

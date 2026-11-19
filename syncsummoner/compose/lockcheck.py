"""Acceptance check of the locked frame clock: a timecoded take judged capture by capture.

Every flip must land on its vblank, every clip frame must have its captures in one gap-free
run, every picture change must fall on a frame start of the frame line, every legible strip must
read the frame the line says, and, under Passthru, no capture may show a neighbour in its upper bands.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from syncsummoner.compose import render, timecode
from syncsummoner.compose.lock import signed
from syncsummoner.probe import decoded
from syncsummoner.progress import LOG, stage

#: Bands judged for tearing: the top and middle thirds, as row fractions.
BANDS = ((0.0, 1 / 3), (1 / 3, 2 / 3))


@dataclass(frozen=True)
class LockReport:
    """What a locked take shows: frame starts, drops and tears over its clip frames."""

    frames: int
    attributed: int
    flips: dict
    drops: int
    gaps: int
    off_cadence: int
    read: int
    agreed: int
    judged: int
    torn: int
    line: str = ""

    @property
    def ok(self) -> bool:
        """Whether the take is exact: every frame attributed, nothing missed, dropped, off a start or torn."""
        return (
            self.attributed == self.frames
            and self.flips["missed"] == 0
            and self.flips["out_of_cadence"] == 0
            and self.drops == 0
            and self.gaps == 0
            and self.off_cadence == 0
            and self.agreed == self.read
            and self.torn == 0
        )

    def summary(self) -> str:
        """The verdict and every count behind it."""
        flips = self.flips
        return (
            f"{'LOCKED' if self.ok else 'NOT LOCKED'}: {self.attributed}/{self.frames} clip frames "
            f"attributed; Pi {flips['flips']} flips, {flips['missed']} missed vblanks, "
            f"{flips['out_of_cadence']} off cadence; card {self.drops} lost frames, {self.gaps} gaps and "
            f"{self.off_cadence} picture changes off a frame start in the clip; "
            f"strips {self.agreed}/{self.read} "
            f"agree; {self.torn}/{self.judged} torn"
        )


def band_error(a: np.ndarray, b: np.ndarray, band: slice) -> np.ndarray:
    """Mean absolute difference of two frame stacks over a band of rows, per frame."""
    return np.abs(a[:, band] - b[:, band]).mean(axis=(1, 2))


def tears(captured: np.ndarray, source: np.ndarray) -> tuple[int, int]:
    """Bands judged and bands torn, over captures paired with the source frames they should show.

    A band is judged where its neighbouring source frames differ from it by more than the
    better match's residual, and torn where a neighbour matches the capture better.
    """
    cap = captured.astype(np.float64)
    src = source.astype(np.float64)
    gain, offset = np.polyfit(src[1:-1].ravel()[::7], cap[1:-1].ravel()[::7], 1)
    src = gain * src + offset
    judged = torn = 0
    rows = cap.shape[1]
    for top, bottom in BANDS:
        band = slice(int(round(top * rows)), int(round(bottom * rows)))
        own = band_error(cap[1:-1], src[1:-1], band)
        near = np.minimum(band_error(cap[1:-1], src[:-2], band), band_error(cap[1:-1], src[2:], band))
        apart = np.minimum(band_error(src[1:-1], src[:-2], band), band_error(src[1:-1], src[2:], band))
        seen = apart > np.minimum(own, near)
        judged += int(seen.sum())
        torn += int((seen & (near < own)).sum())
    return judged, torn


def judge(take: str | Path, *, config: Any, clip: str | Path | None = None) -> LockReport:
    """Judge a recorded take; with ``clip`` (a Passthru take's played clip) also check for tears."""
    from syncsummoner.device.playout import flip_summary
    from syncsummoner.device.v4l2rec import read_log

    ffmpeg, spec = getattr(config, "ffmpeg", "ffmpeg"), decoded.Spec.of(config)
    features = decoded.features(take, spec, ffmpeg=ffmpeg)
    codes = features.codes
    found = render.attribute_take(take, config=config, features=features)
    record = np.load(render.lock_path(take))
    frames = read_log(take)
    shown = found.shown
    inside = np.flatnonzero((shown >= 0) & (shown < found.picked.size))
    first, last = int(inside[0]), int(inside[-1])
    span = np.arange(first, last + 1)
    new = np.r_[False, frames["digest"][1:] != frames["digest"][:-1]][span]
    gaps = int(np.count_nonzero(np.diff(frames["sequence"][first : last + 1]) != 1))
    clip_codes = signed(codes[span], config.bits)
    readable = (codes[span] >= 0) & (clip_codes >= 0) & (clip_codes < found.picked.size)
    judged = torn = 0
    if clip is not None:
        with stage("tears", take=str(take)) as said:
            source = decoded.features(clip, spec, ffmpeg=ffmpeg).small[
                found.source_offset : found.source_offset + found.picked.size
            ]
            judged, torn = tears(features.small[found.picked, 0], source[:, 0])
            said.update(judged=judged, torn=torn)
    return LockReport(
        frames=int(found.picked.size),
        attributed=int(np.count_nonzero(found.picked >= 0)),
        flips=flip_summary(record["flips"], int(record["hold"])),
        drops=int(np.clip(np.diff(frames["sequence"][first : last + 1]) - 1, 0, None).sum()),
        gaps=gaps,
        off_cadence=int(np.count_nonzero(new & ~found.starts[span])),
        read=int(readable.sum()),
        agreed=int(np.count_nonzero(readable & (clip_codes == found.shown[span]))),
        judged=judged,
        torn=torn,
        line=found.summary(),
    )


def clip_seconds(clip: str | Path, config: Any) -> float:
    """A prepared clip's length between its lead-in and lead-out, from its layout."""
    return timecode.clip_layout(clip).frames / config.fps


def check_frames(fps: float, lead_in_s: float | None = None) -> tuple[int, int]:
    """A lockcheck clip's lead-in and lead-out: ``lead_in_s`` each, else what a load inside each needs.

    A program loaded in the lead-in is a session with no dry control, sized by the rig's longest load.
    """
    from syncsummoner.compose import session  # pylint: disable=import-outside-toplevel
    from syncsummoner.device import loadtimes  # pylint: disable=import-outside-toplevel

    if lead_in_s is not None:
        return (session.frames_of(lead_in_s, fps),) * 2
    loads, lead_in = loadtimes.longest(), session.LeadIn()
    return (
        session.lead_in_frames(lead_in, fps, dry=False, load_s=loads),
        session.lead_out_frames(lead_in, fps, dry=False, load_s=loads),
    )


def lockcheck(
    take: str | Path,
    *,
    clip: str | Path,
    config: Any,
    program: str | None = None,
    rig: Any = None,
    recorder: Any = None,
    seconds: float | None = None,
) -> LockReport:
    """Play ``clip`` through Passthru, or ``program`` loaded in the lead-in; record it and judge it."""
    seconds = clip_seconds(clip, config) if seconds is None else seconds
    owned = rig is None
    rig = render.open_rig(config, capture=False) if owned else rig
    if recorder is None:
        from syncsummoner.device.recorder import MJPEG, Recorder

        recorder = Recorder(width=config.width, height=config.height, fps=config.fps, mode=MJPEG)
    try:
        with stage("upload", clip=str(clip)) as sent:
            sent["bytes"] = rig.playout.upload(str(clip))
        render.play_take(
            rig,
            recorder,
            take,
            program=program,
            seconds=seconds,
            config=config,
            clip=timecode.clip_layout(clip),
        )
    finally:
        if owned:
            rig.transport.close()
    report = judge(take, config=config, clip=clip if program is None else None)
    LOG.info("%s: %s", take, report.summary())
    return report

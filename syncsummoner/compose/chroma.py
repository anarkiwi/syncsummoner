"""Which channel order a take shows its clip in, judged on the stretch of its lead-in held dry.

A dry picture is the clip itself, so its captures are judged against the clip frames their strips name
(:func:`~syncsummoner.probe.chroma.order_of`). A program whose 4:2:2 packing came back exchanged for its
session is read back through :data:`SWAP`, so every session of it reads alike (docs/hardware.md).
"""

from __future__ import annotations

import json
import subprocess
from pathlib import Path
from typing import Any, Mapping

import numpy as np

from syncsummoner.compose.lock import ANCHOR_FRAMES, signed
from syncsummoner.compose.locktake import LeadIn, wet_at
from syncsummoner.device.v4l2rec import read_log
from syncsummoner.device.wire import load_wire
from syncsummoner.probe import decoded
from syncsummoner.probe.chroma import SWAP, ChromaOrder, order_of

__all__ = ["SWAP", "UNREADABLE", "dry_codes", "exchanged", "judge", "lead_in_order", "record_path", "swapped"]


#: What reading a take or its clip raises when either cannot be judged.
UNREADABLE = (ValueError, RuntimeError, OSError, KeyError, subprocess.CalledProcessError)
#: Captures past the last dry code's flip that are decoded, so a late capture of it is still read.
SLACK_NS = 500_000_000


def dry_codes(wire: Mapping[str, np.ndarray], fps: float, lead_in: LeadIn = LeadIn()) -> np.ndarray:
    """The clip codes a take's lead-in showed dry, from its wire record.

    With a dry control, the ``bound_s`` before the wet stretch, once the start state has settled; else the
    :data:`~syncsummoner.compose.lock.ANCHOR_FRAMES` shown under Passthru before the program's load began.
    """
    if json.loads(str(wire["dry"])) is not None:
        wet = wet_at(lead_in, fps)
        return np.arange(wet - int(round(lead_in.bound_s * fps)), wet)
    loads = json.loads(str(wire["loads"]))
    shown = wire["shown"]
    before = shown["frame"][shown["t_ns"] < loads[0]["t0_ns"]] if loads else shown["frame"][:0]
    return np.asarray(before[-ANCHOR_FRAMES - 1 : -1], dtype=np.int64)


def record_path(take: str | Path) -> Path:
    """Where a take's channel-order verdict is kept."""
    return Path(f"{take}.chroma.json")


def judge(take: str | Path, clip: str | Path, config: Any, *, lead: int | None = None) -> dict:
    """Judge a take's dry lead-in on ``clip`` and keep the verdict beside it; why not, when it cannot be.

    ``config`` gives the strip, rate, width and ffmpeg; ``lead`` the clip's lead-in, else it is read.
    """
    from syncsummoner.compose.timecode import clip_layout  # pylint: disable=import-outside-toplevel

    ffmpeg = getattr(config, "ffmpeg", "ffmpeg")
    try:
        if lead is None:
            lead = clip_layout(clip, ffprobe=decoded.ffprobe_for(ffmpeg)).lead
        found = lead_in_order(
            take,
            clip,
            spec=decoded.Spec.of(config),
            fps=config.fps,
            width=config.width,
            lead=lead,
            ffmpeg=ffmpeg,
        )
        said = {**found.as_dict(), "dry": json.loads(str(load_wire(take)["dry"])) is not None}
    except UNREADABLE as err:
        said = {"error": f"{type(err).__name__}: {err}"}
    record_path(take).write_text(json.dumps(said), encoding="utf-8")
    return said


def exchanged(take: str | Path) -> bool:
    """Whether a take's kept verdict judged its chroma pair exchanged; False with none kept."""
    try:
        said = json.loads(record_path(take).read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return False
    return bool(said.get("judged")) and not said.get("ok", True)


def swapped(take: str | Path) -> list[str]:
    """The filters reading ``take`` in its clip's channel order: :data:`SWAP` when it came back exchanged."""
    return [SWAP] if exchanged(take) else []


def head(path: str | Path, spec: decoded.Spec, count: int, width: int, ffmpeg: str) -> dict[str, np.ndarray]:
    """The first ``count`` captures of ``path``, decoded to ``spec``."""
    return decoded.segment(
        path, spec, start_s=None, count=count, width=width, ffmpeg=ffmpeg, done=lambda _n: None
    )


def lead_in_order(
    take: str | Path,
    clip: str | Path,
    *,
    spec: decoded.Spec,
    fps: float,
    width: int,
    lead: int,
    ffmpeg: str = "ffmpeg",
) -> ChromaOrder:
    """The order ``take``'s dry lead-in shows ``clip`` in; ``lead`` is the clip's own lead-in length."""
    wire = load_wire(take)
    codes = dry_codes(wire, fps)
    shown = wire["shown"]
    flips = shown["t_ns"][np.isin(shown["frame"], codes)]
    if not flips.size:
        raise ValueError(f"{take}: its lead-in showed none of its dry frames")
    count = int(np.searchsorted(read_log(take)["t_ns"], flips.max() + SLACK_NS)) + 1
    got = head(take, spec, count, width, ffmpeg)
    played = head(clip, spec, lead, width, ffmpeg)
    at = signed(got["codes"], spec.bits)
    source = dict(zip(signed(played["codes"], spec.bits).tolist(), played["small"]))
    keep = np.isin(at, codes) & np.isin(at, list(source))
    if not keep.any():
        raise ValueError(f"{take}: no capture of its dry lead-in carries a legible strip")
    return order_of(got["small"][keep], np.stack([source[int(c)] for c in at[keep]]))

"""Which channel order a take shows its clip in, judged on the stretch of its lead-in held dry.

A dry picture is the clip itself, so its captures are judged against the clip frames their strips name
(:func:`~syncsummoner.probe.chroma.order_of`).
"""

from __future__ import annotations

import json
import subprocess
from pathlib import Path
from typing import Mapping

import numpy as np

from syncsummoner.compose.lock import ANCHOR_FRAMES, signed
from syncsummoner.compose.locktake import LeadIn, wet_at
from syncsummoner.device.v4l2rec import read_log
from syncsummoner.device.wire import load_wire
from syncsummoner.probe import decoded
from syncsummoner.probe.chroma import ChromaOrder, order_of

__all__ = ["UNREADABLE", "dry_codes", "lead_in_order"]

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

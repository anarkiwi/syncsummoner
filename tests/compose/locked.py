"""Synthetic locked takes: the Pi's flip log and the card's frame log for known captured slots."""

import contextlib
import dataclasses
import json
import types
from pathlib import Path

import numpy as np

from syncsummoner.compose import lock as L
from syncsummoner.compose import judge as J
from syncsummoner.compose import locktake
from syncsummoner.compose import render as R
from syncsummoner.compose.score import Layer, Score, Section
from syncsummoner.device.playout import FLIP
from syncsummoner.device.v4l2rec import FRAME, sidecar
from syncsummoner.probe import decoded
from tests.compose import PROBED, rig_transport

VBLANK_NS = 16_666_667
PI_T0 = 1_000_000_000
#: Capture clock minus Pi clock.
CARD_NS = 7_000_000_000
WRAP = 0xFFFF


@dataclasses.dataclass
class Take:
    """A take's logs, and the played frame each capture truly shows."""

    frame: np.ndarray
    frames: np.ndarray
    flips: np.ndarray
    codes: np.ndarray
    lead: int
    count: int
    per_frame: int
    hold: int
    tail: int = 0

    @property
    def kw(self) -> dict:
        """What the lock record says was played."""
        return {"hold": self.hold, "lead": self.lead, "count": self.count, "tail": self.tail}

    def shown(self) -> np.ndarray:
        """The clip frame each capture shows, negative in the lead-in."""
        return self.frame - self.lead

    def shows(self, picked: np.ndarray) -> bool:
        """Whether every pick is a capture of its own clip frame."""
        picked = np.asarray(picked)
        return bool((picked >= 0).all()) and bool((self.shown()[picked] == np.arange(picked.size)).all())


def flip_log(frames: int, hold: int) -> np.ndarray:
    """A flip every vblank, each frame held ``hold`` vblanks, one segment, on the Pi's clock."""
    v = np.arange(frames * hold)
    out = np.zeros(v.size, FLIP)
    out["v"] = out["seq"] = v
    out["frame"] = v // hold
    out["t_ns"] = PI_T0 + v * VBLANK_NS
    out["flags"][0] = 1
    return out


def take(
    runs,
    *,
    lead=8,
    count=6,
    per_frame=2,
    hold=2,
    strips=None,
    misread=None,
    ppm=0.0,
    jitter_ns=0,
    static=None,
    wet=None,
    tail=0,
    kink=None,
    seed=0,
):
    """Captures ``first..last`` of the card's clock per run ``(first, last, late, lost)``, ``late`` in captures.

    The device's frames, locked to the Pi, run ``ppm`` off ``per_frame`` captures from midway between
    the first run's; ``lost`` sequence numbers go before a run and timestamps up to ``jitter_ns`` late.
    Runs in ``strips`` (all by default) read their frame outside the played frames ``wet``, frames in
    ``static`` show one picture, ``misread`` is ``(capture, code)`` and ``kink`` ``(frame, ns)`` bends the line.
    """
    capture = hold * VBLANK_NS / per_frame
    period = per_frame * capture * (1 + ppm * 1e-6)
    origin = PI_T0 + CARD_NS + (runs[0][2] - 0.5) * capture
    times, sequence, legible = [], [], []
    seq = 0
    for index, (first, last, delay, *lost) in enumerate(runs):
        span = np.arange(first, last + 1)
        seq += lost[0] if lost else 0
        times.append(PI_T0 + CARD_NS + (span + delay) * capture)
        sequence.append(seq + np.arange(span.size))
        seq += span.size
        legible.append(np.full(span.size, strips is None or index in strips))
    tau = np.concatenate(times)
    frame = np.floor((tau - origin) / period).astype(np.int64)
    if kink is not None:
        frame = np.floor((tau - origin - np.where(frame >= kink[0], kink[1], 0)) / period).astype(np.int64)
    rng = np.random.default_rng(seed)
    frames = np.zeros(tau.size, FRAME)
    frames["sequence"] = np.concatenate(sequence)
    frames["t_ns"] = np.round(tau + jitter_ns * rng.random(tau.size)).astype(np.int64)
    frames["bytes"] = 1
    picture = frame if static is None else np.where((frame >= static[0]) & (frame < static[1]), static[0], frame)
    frames["digest"] = picture + 1
    readable = np.concatenate(legible) & (frame >= 0)
    if wet is not None:
        readable &= (frame < wet[0]) | (frame >= wet[1])
    codes = np.where(readable, (frame - lead) & WRAP, -1)
    if misread is not None:
        codes[misread[0]] = misread[1]
    flips = flip_log(lead + count + tail + 1, hold)
    return Take(frame, frames, flips, codes, lead, count, per_frame, hold, tail)


#: Lead-in with strips, a load's outage, a run on another card phase, a lost frame, the rest of the clip.
STANDARD = ((0, 7, 0.0), (13, 19, 0.3), (21, 27, 0.3, 1))
#: The same take with nothing lost after the outage.
CLEAN = ((0, 7, 0.0), (13, 27, 0.3))


class Clock:
    """Monotonic clock advanced only by the sleeps handed to it."""

    def __init__(self, now=0.0):
        self.now = now
        self.slept = []

    def __call__(self):
        return self.now

    def sleep(self, seconds):
        """Sleep."""
        self.slept.append(seconds)
        self.now += seconds


class Played:
    """A rig and recorder playing ``played`` on ``clock``, logging each step; the card's log written on stop.

    ``skew_s`` is how long after the recording starts the first lead-in frame reaches the card,
    and ``load_s`` how long each program's load takes.
    """

    def __init__(self, played, transport=None, *, clock=None, load_s=None, skew_s=0.0):
        self.take, self.clock, self.skew_s = played, clock or Clock(), skew_s
        self.load_s, self.log = dict(load_s or {}), []
        session = types.SimpleNamespace(
            load_program=self.load,
            working_point=lambda info: {1: 0.5},
            set_params=lambda values: self.log.append(("params", dict(values))),
        )
        self.rig = R.Rig(
            session=session, capture=None, playout=self, transport=transport or rig_transport(), link="link"
        )

    @property
    def steps(self) -> list:
        """The step names, in order."""
        return [step[0] for step in self.log]

    def load(self, program, *, link=None, park=True):
        """Load a program, taking its time on the clock."""
        del park
        self.log.append(("load", program, link))
        self.clock.now += self.load_s.get(program, 0.0)

    def upload(self, path):
        """Upload."""
        self.log.append(("upload", str(path)))
        return 1

    @contextlib.contextmanager
    def playing(self, *, fps):
        """Play for the block."""
        self.log.append(("play", fps, self.clock.now))
        yield self
        self.log.append(("stop",))

    def status(self):
        """The player's status."""
        return {"hold": self.take.hold}

    def flips(self):
        """The flip log of the last play."""
        self.log.append(("flips",))
        return self.take.flips

    @contextlib.contextmanager
    def recording(self, path, seconds=None):
        """Record for the block; the frame log is on the clock the recording started on."""
        del seconds
        self.log.append(("record", str(path)))
        base = int(round((self.clock.now + self.skew_s) * 1e9)) - PI_T0 - CARD_NS
        yield
        frames = self.take.frames.copy()
        frames["t_ns"] += base
        Path(sidecar(path)).write_bytes(frames.tobytes())
        self.log.append(("recorded",))


#: A session the standard take was played at: its lead-in and clip in frames, and a short tail.
LOCKED = R.RenderConfig(width=160, height=24, fps=30.0, strip_px=8, bits=16, lead_in_s=8 / 30, tail_s=0.05)


@dataclasses.dataclass
class Proof:
    """A stand-in for a take's evidence: whether it holds, and what it says."""

    ok: bool = True

    def lines(self) -> list:
        """The report as log lines."""
        return [f"evidence {'holds' if self.ok else 'fails'}"]

    def to_json(self) -> str:
        """The report as JSON."""
        return json.dumps({"ok": self.ok})

    def schedule(self) -> list:
        """Undetermined parameters to sweep; none."""
        return []


def locked_pass(monkeypatch, played=None, transport=None, *, proof=None, **kwargs):
    """A :class:`Played` rig on its own clock, its frame-locked take and evidence logged rather than run.

    ``proof`` is what each take's evidence says, one per take and the last held; it holds by default.
    """
    rig = Played(take(STANDARD) if played is None else played, transport, **kwargs)
    monkeypatch.setattr(R, "time", types.SimpleNamespace(monotonic=rig.clock, sleep=rig.clock.sleep))
    monkeypatch.setattr(
        decoded,
        "features",
        lambda *a, **k: decoded.Features({"codes": rig.take.codes, "luma": np.zeros(0)}, ""),
    )
    monkeypatch.setattr(R, "plan_automation", lambda *a, **k: {})
    proofs = list(proof or [Proof()])

    def locked_take(rig_, recorder, out, *, program, seconds, config, plan, working, load):
        lead, count = R.lead_frames(config), int(round(seconds * config.fps))
        working(load(program))
        with recorder.recording(out), rig_.playout.playing(fps=config.fps) as player:
            rig.log.append(("drive", program, plan.size))
            rig.clock.sleep(lead / config.fps + seconds + config.tail_s)
            hold = int(player.status()["hold"])
        L.save_lock(out, rig_.playout.flips(), hold=hold, lead=lead, count=count)

    def take_evidence(take_path, *, program, clip, config, **kw):
        del config, kw
        rig.log.append(("evidence", program, str(clip), Path(take_path).name))
        return proofs.pop(0) if len(proofs) > 1 else proofs[0]

    monkeypatch.setattr(locktake, "locked_take", locked_take)
    monkeypatch.setattr(locktake, "take_evidence", take_evidence)
    monkeypatch.setattr(J, "inspect_luma", lambda luma, start: rig.log.append(("inspect", start)) or PROBED)
    return rig


def render_locked(rig, out, config=LOCKED, program="Teletext"):
    """Render a one-layer score the length of the standard take's clip on ``rig``."""
    score = Score(duration=0.2, fps=30.0, sections=[Section(0.0, 0.2, "A")], layers=[Layer(program, 0, [])])
    return R.render_played(
        score,
        "clip.mkv",
        out,
        profiles={},
        rig=rig.rig,
        config=config,
        scratch="tc.mkv",
        prepared=True,
        recorder=rig,
    )

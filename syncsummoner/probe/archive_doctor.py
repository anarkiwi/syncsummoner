"""``probe doctor``: an offline check of a sweep archive directory before rig time is spent on it.

Each check is a function of what the archive holds, returning a :class:`pyvmancer.diagnose.Finding`.
"""

from __future__ import annotations

import argparse
import json
import shutil
from collections import Counter
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np
from pyvmancer.diagnose import Finding, format_finding, report

from syncsummoner.device.kmsplay import SEGMENT
from syncsummoner.device.v4l2rec import read_sidecar, sidecar
from syncsummoner.progress import LOG, track

#: Characters of a fault message kept in a finding's detail.
MESSAGE = 120
#: Distinct fault messages a finding's detail names.
TOP = 3
#: Files a finding names before counting the rest.
NAMED = 8


def finding(check: str, ok: bool, detail: str, fix: str = "") -> Finding:
    """A finding whose ``fix`` is kept only when the check fails."""
    return Finding(check, bool(ok), detail, "" if ok else fix)


def named(paths: Sequence[str | Path]) -> str:
    """The first :data:`NAMED` files by name, then how many more there are."""
    shown = ", ".join(Path(p).name for p in paths[:NAMED])
    return shown + (f" and {len(paths) - NAMED} more" if len(paths) > NAMED else "")


def plan_slug(path: str | Path) -> str:
    """The slug a plan's takes, hold record and done marker are named by."""
    return Path(path).stem.removeprefix("sweep-")


def read_plans(paths: Iterable[str | Path]) -> tuple[dict[str, dict], dict[str, str]]:
    """Each listed plan's record by path, and why each one that could not be read was not."""
    plans, errors = {}, {}
    for path in map(str, paths):
        try:
            plans[path] = json.loads(Path(path).read_text(encoding="utf-8"))
        except (OSError, ValueError) as err:
            errors[path] = f"{type(err).__name__}: {err}"
    return plans, errors


def plans_readable(plans: dict[str, dict], errors: dict[str, str]) -> Finding:
    """Every listed plan exists and parses."""
    bad = "; ".join(f"{Path(p).name}: {why}" for p, why in list(errors.items())[:NAMED])
    return finding(
        "plans.readable",
        bool(plans) and not errors,
        f"{len(plans)} of {len(plans) + len(errors)} plans read" + (f"; {bad}" if bad else ""),
        "list one existing sweep plan JSON per line" if not plans else f"fix or drop: {named(list(errors))}",
    )


def clip_of(plans: dict[str, dict]) -> str | None:
    """The clip most plans play."""
    sources = Counter(str(p.get("source")) for p in plans.values())
    return sources.most_common(1)[0][0] if sources else None


def one_clip(plans: dict[str, dict]) -> Finding:
    """Every plan plays the same, existing clip."""
    clip = clip_of(plans)
    differ = [p for p, said in plans.items() if str(said.get("source")) != clip]
    exists = clip is not None and Path(clip).is_file()
    detail = f"{len(plans) - len(differ)} of {len(plans)} plans play {clip}" + ("" if exists else ", missing")
    fix = f"re-plan against {clip}: {named(differ)}" if differ else f"restore the clip {clip}"
    return finding("plans.one_clip", exists and not differ, detail, fix)


def stimulus_key(clip: str | Path) -> dict | None:
    """A synthetic clip's stimulus spec, from the key beside it; None for footage."""
    key = Path(f"{clip}.stimulus.json")
    return json.loads(key.read_text(encoding="utf-8"))["stimulus"] if key.is_file() else None


def shot_aligned(plans: dict[str, dict]) -> Finding:
    """Every plan against a synthetic clip puts its settings on that clip's shots."""
    keys = {c: stimulus_key(c) for c in {str(p.get("source")) for p in plans.values()}}
    synthetic = {c: k["shot_frames"] for c, k in keys.items() if k is not None}
    if not synthetic:
        return finding("plans.shot_aligned", True, "footage clip")
    off = [
        p
        for p, said in plans.items()
        if str(said.get("source")) in synthetic
        and (said.get("regions") or {}).get("shot_frames") != synthetic[str(said["source"])]
    ]
    shots = ", ".join(f"{Path(c).name} {n}" for c, n in sorted(synthetic.items()))
    return finding(
        "plans.shot_aligned",
        not off,
        f"{len(plans) - len(off)} of {len(plans)} plans on the clip's shots ({shots} frames)",
        f"re-plan with `probe sweep regions`: {named(off)}",
    )


#: Fields a hold record may name the clip it was measured on by.
HOLD_CLIP = ("source", "clip")


def holds_clip(holds: Sequence[str | Path], clip: str | None) -> Finding:
    """Each hold record was measured on the plans' clip."""
    if not holds:
        return finding("holds.clip", True, "no hold records")
    records = {str(h): json.loads(Path(h).read_text(encoding="utf-8")) for h in holds}
    clips = {h: next((r[f] for f in HOLD_CLIP if f in r), None) for h, r in records.items()}
    unnamed = [h for h, c in clips.items() if c is None]
    wrong = [h for h, c in clips.items() if c is not None and Path(c).resolve() != Path(str(clip)).resolve()]
    detail = f"{len(holds) - len(unnamed) - len(wrong)} of {len(holds)} hold records measured on {clip}"
    if unnamed:
        detail += f"; {len(unnamed)} name no clip, so cannot be verified"
    return finding(
        "holds.clip",
        not unnamed and not wrong,
        detail,
        f"measure again with `probe hold --source {clip}`: {named(wrong + unnamed)}",
    )


def remaining(archive: Path, plans: dict[str, dict]) -> dict[str, dict]:
    """Plans with no ``logs/done-<slug>`` marker."""
    return {p: s for p, s in plans.items() if not (archive / "logs" / f"done-{plan_slug(p)}").exists()}


def rig_seconds(plans: dict[str, dict]) -> float:
    """Rig seconds the plans' takes take."""
    return float(sum(float((s.get("regions") or {}).get("rig_s") or 0.0) for s in plans.values()))


def takes_of(archive: Path) -> list[Path]:
    """The archive's takes."""
    return sorted((archive / "takes").glob("take-*.mkv"))


def capture_times(take: Path) -> np.ndarray:
    """A take's capture times (ns), empty when it has no sidecar."""
    if not sidecar(take).is_file():
        return np.empty(0, dtype=np.int64)
    return read_sidecar(take)[0]["t_ns"]


def write_rate(takes: Iterable[Path]) -> float | None:
    """Bytes a second the takes were written at, over their captured seconds; None with none captured."""
    size = seconds = 0.0
    for take in track(list(takes), desc="write rate", unit="take"):
        t = capture_times(take)
        if t.size > 1:
            size += take.stat().st_size
            seconds += (t[-1] - t[0]) / 1e9
    return size / seconds if seconds > 0 else None


def disk_budget(rig_s: float, rate: float | None, free: float, floor: float) -> Finding:
    """The remaining takes fit in the free disk above the floor."""
    if rate is None:
        return finding(
            "disk.budget", False, "no take captured to measure a write rate", "pass --rate-bytes-s"
        )
    projected, room = rig_s * rate, free - floor
    hour = max(room, 0.0) / rate / 3600
    detail = (
        f"{rate / 1e6:.1f} MB/s over {rig_s / 3600:.1f} rig h projects {projected / 1e9:.1f} GB; "
        f"{free / 1e9:.1f} GB free, floor reached at rig hour {hour:.1f}"
    )
    return finding(
        "disk.budget",
        projected <= room,
        detail,
        f"free {(projected - room) / 1e9:.1f} GB or plan fewer rig seconds",
    )


def rig_time(rig_s: float, left: int, total: int, paused: bool) -> Finding:
    """The rig hours left: detail only."""
    detail = f"{rig_s / 3600:.1f} rig h left over {left} of {total} plans" + (", paused" if paused else "")
    return finding("rig.time", True, detail)


def attempts(take: Path) -> dict[str, dict]:
    """A take's capture attempts, from its ``.meta.json``."""
    meta = Path(f"{take}.meta.json")
    return json.loads(meta.read_text(encoding="utf-8")) if meta.is_file() else {}


def logged_faults(path: Path) -> list[str]:
    """The fault messages a ``faults.tsv`` log filed: its last field after time, program and take."""
    if not path.is_file():
        return []
    rows = (line.split("\t", 3) for line in path.read_text(encoding="utf-8").splitlines())
    return [r[3] for r in rows if len(r) == 4]


def take_faults(messages: Sequence[str]) -> Finding:
    """No take faulted, else the commonest fault messages and how often each was seen."""
    counts = Counter(m[:MESSAGE] for m in messages)
    top = "; ".join(f"{n}x {m}" for m, n in counts.most_common(TOP))
    return finding(
        "takes.faults",
        not counts,
        f"{len(messages)} faults, {len(counts)} distinct" + (f": {top}" if top else ""),
        "fix the commonest fault before resuming",
    )


def capture_period(rows: np.ndarray) -> float:
    """Nanoseconds between captures: the slope of capture time over driver sequence."""
    return float(np.polyfit(rows["sequence"] - rows["sequence"][0], rows["t_ns"] - rows["t_ns"][0], 1)[0])


def flip_period(flips: np.ndarray) -> float:
    """Nanoseconds between the Pi's consecutive flips, each one vblank and one frame after the last."""
    step = (np.diff(flips["seq"]) == 1) & (np.diff(flips["v"]) == 1)
    return float(np.median(np.diff(flips["t_ns"])[step])) if step.any() else float("nan")


def lock_ppm(rows: np.ndarray, flips: np.ndarray) -> tuple[float, float] | None:
    """Two captures' period against one flip's, in ppm, and the ppm one capture period amounts to over the
    take; None when the flips hold more than one segment or the take too few captures."""
    if rows.size < 2 or flips.size < 2 or np.count_nonzero(flips["flags"][1:] & SEGMENT):
        return None
    period, flip = capture_period(rows), flip_period(flips)
    duration = float(rows["t_ns"][-1] - rows["t_ns"][0])
    return (2 * period - flip) / flip * 1e6, period / duration * 1e6


def rate_lock(locks: dict[str, tuple[float, float] | None]) -> Finding:
    """Every single-segment take's capture rate is locked to the Pi's flips within a capture period."""
    if not locks:
        return finding("takes.rate_lock", True, "no take with both a lock record and a sidecar")
    measured = {t: v for t, v in locks.items() if v is not None}
    loose = [Path(t).name for t, (ppm, bound) in measured.items() if abs(ppm) > bound]
    detail = f"{len(measured)} takes measured, {len(locks) - len(measured)} multi-segment skipped"
    if measured:
        ppm, bound = np.array(list(measured.values())).T
        detail += f"; ppm {ppm.min():+.1f} to {ppm.max():+.1f}, the tightest bound {bound.min():.1f}"
    if loose:
        detail += f"; drifting: {named(loose)}"
    return finding("takes.rate_lock", not loose, detail, "re-lock the capture card to the Pi and recapture")


def takes_rate_lock(takes: Iterable[Path]) -> dict[str, tuple[float, float] | None]:
    """:func:`lock_ppm` of each take with both a ``.lock.npz`` and a sidecar."""
    found = {}
    for take in track(list(takes), desc="rate lock", unit="take"):
        lock = Path(f"{take}.lock.npz")
        if lock.is_file() and sidecar(take).is_file():
            with np.load(lock) as record:
                found[str(take)] = lock_ppm(read_sidecar(take)[0], record["flips"])
    return found


def captured(take: Path) -> bool:
    """Whether a take was captured: it exists and an attempt finished with no fault."""
    return take.is_file() and any("fault" not in a for a in attempts(take).values())


def collected(takes: Iterable[Path]) -> Finding:
    """Every captured take has its collected rows."""
    done = [t for t in takes if captured(t)]
    missing = [t.name for t in done if not Path(f"{t}.rows.parquet").is_file()]
    return finding(
        "takes.collected",
        not missing,
        f"{len(done) - len(missing)} of {len(done)} captured takes collected"
        + (f"; not: {named(missing)}" if missing else ""),
        "run `probe sweep collect` on them and read its log for why it failed",
    )


def effect_summaries(profiles: Path) -> dict[str, dict[str, dict]]:
    """Each profiled program's ``probe effects`` summary by clip colour, empty where none was measured."""
    from syncsummoner.probe.fit import load_profile  # pylint: disable=import-outside-toplevel

    found = {}
    for path in sorted(Path(profiles).glob("*.yaml")):
        profile = load_profile(path)
        found[profile.program] = {kind: entry["summary"] for kind, entry in profile.effects.items()}
    return found


def _range(span: Sequence[float] | None) -> str:
    return "none" if not span else f"{span[0]:.3f}-{span[1]:.3f}"


def effects_measured(summaries: dict[str, dict[str, dict]]) -> Finding:
    """Every profile carries its settings' effects; each program's safe intensity range and flag counts."""
    missing = [name for name, kinds in summaries.items() if not kinds]
    lines = [
        f"{name} {kind}: {s['safe']}/{s['settings']} safe, intensity {_range(s['safe_intensity'])}, "
        f"blank {s['blank']} static {s['static']} flashing {s['flashing']} red {s['red_flashing']}"
        for name, kinds in sorted(summaries.items())
        for kind, s in sorted(kinds.items())
    ]
    return finding(
        "effects.measured",
        bool(summaries) and not missing,
        f"{len(summaries) - len(missing)} of {len(summaries)} profiles carry effects"
        + (f"; not: {', '.join(missing[:NAMED])}" if missing else "")
        + "".join(f"; {line}" for line in lines),
        "run `probe effects --archive ARCHIVE/archive --profiles PROFILES` on them",
    )


def diagnose(
    archive: str | Path,
    plans_file: str | Path | None = None,
    *,
    rate: float | None = None,
    floor: float = 0.0,
    profiles: str | Path | None = None,
) -> list[Finding]:
    """Every check over the archive at ``archive``, its plans listed in ``plans_file``.

    Profiles in ``profiles``, default ``ARCHIVE/profiles-each``, are checked for their effects when there.
    """
    archive = Path(archive)
    listing = Path(plans_file) if plans_file else archive / "plans.txt"
    paths = listing.read_text(encoding="utf-8").split() if listing.is_file() else []
    plans, errors = read_plans(paths)
    if not listing.is_file():
        errors[str(listing)] = "no plan listing"
    clip = clip_of(plans)
    shelf = Path(profiles) if profiles else archive / "profiles-each"
    left = remaining(archive, plans)
    takes, rig_s = takes_of(archive), rig_seconds(left)
    messages = logged_faults(archive / "logs" / "faults.tsv") + [
        a["fault"] for t in takes for a in attempts(t).values() if "fault" in a
    ]
    return [
        plans_readable(plans, errors),
        one_clip(plans),
        shot_aligned(plans),
        holds_clip(sorted((archive / "holds").glob("hold-*.json")), clip),
        disk_budget(rig_s, write_rate(takes) or rate, shutil.disk_usage(archive).free, floor),
        rig_time(rig_s, len(left), len(plans), (archive / "pause").exists()),
        take_faults(messages),
        rate_lock(takes_rate_lock(takes)),
        collected(takes),
    ] + ([effects_measured(effect_summaries(shelf))] if shelf.is_dir() else [])


def _doctor_cmd(args: argparse.Namespace) -> int:
    findings = diagnose(
        args.archive, args.plans, rate=args.rate_bytes_s, floor=args.floor_bytes, profiles=args.profiles
    )
    if not args.json:
        for found in findings:
            (LOG.info if found.ok else LOG.error)("%s", format_finding(found))
    summary = report(findings)
    shelf = Path(args.profiles) if args.profiles else Path(args.archive) / "profiles-each"
    if shelf.is_dir():
        summary["effects"] = effect_summaries(shelf)
    print(json.dumps(summary, indent=2))
    return 0 if summary["ok"] else 1


def add_parser(probe_sub) -> None:
    """The ``probe doctor`` command."""
    doctor = probe_sub.add_parser(
        "doctor", help="check a sweep archive offline before starting or resuming it"
    )
    doctor.add_argument("archive", help="the archive directory")
    doctor.add_argument(
        "--plans", help="the plan listing, one sweep plan path a line; default ARCHIVE/plans.txt"
    )
    doctor.add_argument(
        "--rate-bytes-s", type=float, help="take bytes written a rig second, when no take measures it"
    )
    doctor.add_argument("--floor-bytes", type=float, default=0.0, help="free bytes the disk must keep")
    doctor.add_argument(
        "--profiles", help="the profiles whose effects to report; default ARCHIVE/profiles-each"
    )
    doctor.add_argument("--json", action="store_true", help="print only the JSON report")
    doctor.set_defaults(func=_doctor_cmd)

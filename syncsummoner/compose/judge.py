"""A finished take judged from its recording and sidecars alone, with no device attached.

The take and the clip it was fed are each decoded once (:mod:`syncsummoner.probe.decoded`), and
one attribution is handed to the evidence, the picture check and the take check (docs/judging.md).
"""

from __future__ import annotations

import json
import multiprocessing
import os
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

from syncsummoner.compose import locktake
from syncsummoner.compose.cut import Cut
from syncsummoner.compose.evidence import Evidence
from syncsummoner.compose.lock import Attribution, attribute_take
from syncsummoner.compose.score import Layer, Score
from syncsummoner.compose.takecheck import CheckReport, check_takes
from syncsummoner.device.recorder import TakeReport, inspect_luma
from syncsummoner.probe import decoded
from syncsummoner.progress import LOG, configure

#: How judging processes start: fresh interpreters, inheriting none of the decoders' threads.
START_METHOD = "spawn"

__all__ = ["Judged", "check_path", "judge_take", "judge_takes", "one_program", "recorded"]


@dataclass(frozen=True)
class Judged:
    """Everything a take was judged on and by."""

    take: Path
    program: str
    found: Attribution
    features: decoded.Features
    proof: Evidence
    report: TakeReport
    check: CheckReport | None = None

    def lines(self) -> list[str]:
        """The attribution, the evidence lines, the picture and any take check, one per line."""
        said = [self.found.summary(), *self.proof.lines(), f"{self.program}: picture {self.report}"]
        return said + ([] if self.check is None else self.check.lines())


def check_path(take: str | Path) -> Path:
    """Where a take's check against the score sits."""
    return Path(f"{take}.check.json")


def one_program(score: Score, program: str) -> tuple[Score, list[Cut]]:
    """The score as one program's pass plays, its own layer or else the first, cut to it at every section."""
    layer = next((item for item in score.layers if item.program == program), None)
    layer = layer or (score.layers[0] if score.layers else None)
    gestures = list(layer.gestures) if layer is not None else []
    spans = [(s.start, s.end) for s in score.sections] or [(0.0, score.duration)]
    played = replace(score, layers=[Layer(index=0, program=program, gestures=gestures)])
    return played, [Cut(start, end, program) for start, end in spans]


def judge_take(
    take: str | Path,
    *,
    clip: str | Path,
    config: Any,
    program: str | None = None,
    score: Score | None = None,
    source: decoded.Features | None = None,
    features: decoded.Features | None = None,
    predict: bool = False,
    workers: int | None = None,
) -> Judged:
    """Attribute a take once, prove its parameters, check its picture and, given a ``score``, its bars.

    ``<take>.evidence.json`` is written, and ``<take>.check.json`` with a score; a take that does
    not lock raises :class:`~syncsummoner.compose.lock.LockError`.
    """
    from syncsummoner.device.wire import load_wire  # pylint: disable=import-outside-toplevel

    ffmpeg = getattr(config, "ffmpeg", "ffmpeg")
    spec = decoded.Spec.of(config)
    if features is None:
        features = decoded.features(take, spec, workers=workers, ffmpeg=ffmpeg)
    if source is None:
        source = decoded.features(clip, spec, workers=workers, ffmpeg=ffmpeg)
    program = program or str(load_wire(take).get("program", Path(take).stem))
    found = attribute_take(take, config=config, features=features)
    proof = locktake.take_evidence(
        take,
        program=program,
        clip=clip,
        config=config,
        predict=predict,
        found=found,
        features=features,
        source=source,
    )
    locktake.evidence_path(take).write_text(proof.to_json(), encoding="utf-8")
    report = inspect_luma(features.luma, start=int(found.picked[0]) - found.per_frame // 2)
    check = None
    if score is not None:
        played, cuts = one_program(score, program)
        check = check_takes(
            played,
            cuts,
            {program: take},
            clip,
            strip_px=config.strip_px,
            bits=config.bits,
            attributions={program: found.picked},
            features={program: features},
            source=source,
            ffmpeg=ffmpeg,
        )
        check_path(take).write_text(
            json.dumps({"lines": check.lines(), "failed": [str(s) for s in check.failed]}, indent=1),
            encoding="utf-8",
        )
    return Judged(Path(take), program, found, features, proof, report, check)


def recorded(take: str | Path, options: dict) -> dict:
    """Judge one take as ``options`` ask, its outcome kept under ``judged`` in ``<take>.meta.json``.

    ``options`` holds the render ``config``, ``source``, ``program``, ``score``, ``predict``, ``workers``
    and ``threads``, the BLAS threads each process may use; undetermined parameters go to
    ``<take>.schedule.json``.
    """
    from threadpoolctl import threadpool_limits  # pylint: disable=import-outside-toplevel

    from syncsummoner.compose.lock import LockError  # pylint: disable=import-outside-toplevel

    config = options["config"]
    score = Score.load(Path(options["score"])) if options["score"] else None
    meta_path = Path(f"{take}.meta.json")
    meta = json.loads(meta_path.read_text(encoding="utf-8")) if meta_path.is_file() else {}
    try:
        with threadpool_limits(options["threads"]):
            judged = judge_take(
                take,
                clip=options["source"],
                config=config,
                program=options["program"],
                score=score,
                predict=options["predict"],
                workers=options["workers"],
            )
    except LockError as err:
        LOG.error("%s: %s", take, err)
        meta["judged"] = {"lock": f"{type(err).__name__}: {err}"}
    else:
        Path(f"{take}.schedule.json").write_text(
            json.dumps(judged.proof.schedule(), indent=1), encoding="utf-8"
        )
        for line in judged.lines():
            LOG.info("%s", line)
        meta["judged"] = {
            "lock": "ok",
            "per_frame": int(judged.found.per_frame),
            "clip_frames": int(judged.found.picked.size),
            "evidence_ok": bool(judged.proof.ok),
            "picture": str(judged.report),
            "usable": bool(judged.report.usable),
            **({} if judged.check is None else {"check_failed": [str(s) for s in judged.check.failed]}),
        }
    meta_path.write_text(json.dumps(meta, indent=1, default=str), encoding="utf-8")
    return meta["judged"]


def judge_takes(
    takes: list[str], options: dict, *, jobs: int = 1, workers: int | None = None, verbosity: int = 0
) -> list[dict]:
    """:func:`recorded` over every take, ``jobs`` processes at once sharing the cores between their decoders.

    The clip is decoded first, once, so no two jobs decode it side by side.
    """
    jobs = max(1, min(jobs, len(takes)))
    workers = workers or max(1, len(os.sched_getaffinity(0)) // jobs)
    config = options["config"]
    ffmpeg = getattr(config, "ffmpeg", "ffmpeg")
    decoded.features(options["source"], decoded.Spec.of(config), workers=workers, ffmpeg=ffmpeg)
    asked = {**options, "workers": workers, "threads": 2 if jobs > 1 else None}
    if jobs == 1:
        return [recorded(take, asked) for take in takes]
    context = multiprocessing.get_context(START_METHOD)
    with ProcessPoolExecutor(jobs, mp_context=context, initializer=configure, initargs=(verbosity,)) as pool:
        return list(pool.map(recorded, takes, [asked] * len(takes)))

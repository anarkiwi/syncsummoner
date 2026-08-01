"""Analog or digital style: value remapping versus spatial re-addressing.

The device documents only its SD-installed programs, so descriptions label the
part of the library that has them and the measured classifier covers the rest.
Labels exist to validate the measurement, not to replace it.
"""

from __future__ import annotations

import re
from typing import Any, Iterable

from syncsummoner.device.profile import ProgramStyle

#: Author wording for operations that move or re-address samples.
DIGITAL_TERMS = (
    "displac|coordinate|remap|scanline|scroll|mirror|reflect|kaleidosc|tile|"
    "transform|skew|perspective|warp|fold|shift|offset|delay|buffer|address|"
    "position|scan|sample|decimat|pixelat|block|zoom|rotat|paralla|particle|"
    "sprite|bounc|orbit|trail|feedback"
)
#: Author wording for operations that remap values where they already are.
ANALOG_TERMS = (
    "colou?r|saturat|hue|tint|luma|chroma|contrast|brightness|gamma|curve|"
    "level|posteris|posteriz|solaris|threshold|invert|desaturat|palette|"
    "grade|bleach|sepia|tone|gain|balance|clip"
)

DIGITAL_RE = re.compile(DIGITAL_TERMS, re.IGNORECASE)
ANALOG_RE = re.compile(ANALOG_TERMS, re.IGNORECASE)


def label_from_description(text: str) -> ProgramStyle:
    """Style implied by an author's own description of what a program does.

    A weak label for validating the measurement, not a substitute: it reads
    intent from prose and says UNKNOWN whenever the prose does not commit.
    """
    if not text:
        return ProgramStyle.UNKNOWN
    digital = len(DIGITAL_RE.findall(text))
    analog = len(ANALOG_RE.findall(text))
    if digital == analog:
        return ProgramStyle.MIXED if digital else ProgramStyle.UNKNOWN
    return ProgramStyle.DIGITAL if digital > analog else ProgramStyle.ANALOG


def labels_from_manifest(manifest: Any) -> dict[str, ProgramStyle]:
    """Description-derived labels for every program the manifest describes."""
    if manifest is None:
        return {}
    entries = getattr(manifest, "entries", None) or {}
    values = entries.values() if isinstance(entries, dict) else entries
    return {
        entry.program_name: label_from_description(entry.description or "")
        for entry in values
        if getattr(entry, "program_name", None)
    }


def classify(pointwise: float, *, analog_above: float, digital_below: float) -> ProgramStyle:
    """Style from measured pointwise fit, with an explicit undecided band."""
    if pointwise >= analog_above:
        return ProgramStyle.ANALOG
    if pointwise <= digital_below:
        return ProgramStyle.DIGITAL
    return ProgramStyle.MIXED


def agreement(measured: dict[str, ProgramStyle], labelled: dict[str, ProgramStyle]) -> dict[str, Any]:
    """Compare measurement against description labels where both commit.

    Reports the shared support so a high score on three programs cannot be
    mistaken for a validated classifier.
    """
    decided = [ProgramStyle.ANALOG, ProgramStyle.DIGITAL]
    shared = [name for name, style in labelled.items() if style in decided and measured.get(name) in decided]
    hits = [name for name in shared if measured[name] is labelled[name]]
    return {
        "support": len(shared),
        "agree": len(hits),
        "accuracy": (len(hits) / len(shared)) if shared else 0.0,
        "disagree": sorted(set(shared) - set(hits)),
    }


def thresholds_from_labels(
    scores: dict[str, float], labelled: dict[str, ProgramStyle]
) -> tuple[float, float]:
    """Derive the decision band from labelled programs rather than assert one.

    Returns the midpoints between the two classes' medians and their nearer
    quartiles, so the undecided band is where the labelled populations overlap.
    """
    import numpy as np

    def side(style: ProgramStyle) -> list[float]:
        return [scores[n] for n, s in labelled.items() if s is style and n in scores]

    analog, digital = side(ProgramStyle.ANALOG), side(ProgramStyle.DIGITAL)
    if not analog or not digital:
        return 0.9, 0.7
    low = float(np.percentile(analog, 25))
    high = float(np.percentile(digital, 75))
    if low <= high:
        middle = 0.5 * (float(np.median(analog)) + float(np.median(digital)))
        return middle, middle
    return low, high


def measured_styles(
    scores: dict[str, float], labelled: dict[str, ProgramStyle]
) -> tuple[dict[str, ProgramStyle], tuple[float, float]]:
    """Classify every measured program using label-derived thresholds."""
    analog_above, digital_below = thresholds_from_labels(scores, labelled)
    return (
        {n: classify(v, analog_above=analog_above, digital_below=digital_below) for n, v in scores.items()},
        (analog_above, digital_below),
    )


def unlabelled(scores: Iterable[str], labelled: dict[str, ProgramStyle]) -> list[str]:
    """Programs the manifest does not describe, which only measurement covers."""
    decided = {ProgramStyle.ANALOG, ProgramStyle.DIGITAL, ProgramStyle.MIXED}
    return sorted(n for n in scores if labelled.get(n) not in decided)

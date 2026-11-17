"""A span of the timeline and the program whose take shows it, and the plan of them a score is cut by."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Sequence


@dataclass(frozen=True)
class Cut:
    """One span of the timeline and the program that renders it."""

    start: float
    end: float
    program: str

    @property
    def duration(self) -> float:
        """Span length in seconds."""
        return max(0.0, self.end - self.start)


def cut_plan(score: Any, programs: Sequence[str] = ()) -> list[Cut]:
    """The cuts a score is spliced by: those it was composed against, else its sections in rotation.

    Cutting inside a pass is not open to us: a program change blacks this device out for
    seconds, so each program renders the whole take and the spans are taken from those
    afterwards. Sections are where the music already changes. A score composed against a plan
    carries it, its layers planned for those spans alone, so ``programs`` may only name the
    programs that plan shows.
    """
    if score.cuts:
        shown = list(dict.fromkeys(cut.program for cut in score.cuts))
        if programs and set(programs) != set(shown):
            raise ValueError(
                f"the score was composed for cuts over {', '.join(shown)}; "
                f"compose again to cut over {', '.join(programs)}"
            )
        return list(score.cuts)
    if not programs:
        raise ValueError("cutting needs at least one program")
    spans = [(s.start, s.end) for s in score.sections] or [(0.0, score.duration)]
    return [Cut(start, end, programs[i % len(programs)]) for i, (start, end) in enumerate(spans)]


def spans_of(cuts: Sequence[Cut], program: str) -> list[tuple[float, float]]:
    """The spans of a plan that show ``program``, in time order."""
    return sorted((cut.start, cut.end) for cut in cuts if cut.program == program)

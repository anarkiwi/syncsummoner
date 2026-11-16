"""A span of the timeline and the program whose take shows it."""

from __future__ import annotations

from dataclasses import dataclass


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

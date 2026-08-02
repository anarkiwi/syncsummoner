"""A durable record of what was asked of the device, and its health when asked.

The input drops intermittently and a killed process leaves no trace, so events
append to disk as they happen. When the fault appears, only a written trail can
say what preceded it and how long ago the device was last good.
"""

from __future__ import annotations

import json
import os
import time
from pathlib import Path
from typing import Any, Callable, Iterable

#: Actions kept in memory for a report; the file keeps everything.
WINDOW = 400


class Journal:
    """Append-only log of device actions and health probes.

    ``clock`` is injected for tests; a ``path`` of None keeps the log in memory
    only, which is what unit tests need and a dry run does not.
    """

    def __init__(
        self,
        path: str | os.PathLike[str] | None = None,
        *,
        window: int = WINDOW,
        clock: Callable[[], float] = time.time,
    ):
        self.path = Path(path) if path else None
        self.window = int(window)
        self._clock = clock
        self._events: list[dict[str, Any]] = []
        self._last_good: dict[str, Any] | None = None
        if self.path:
            self.path.parent.mkdir(parents=True, exist_ok=True)

    def record(self, kind: str, **detail: Any) -> dict[str, Any]:
        """Append one event, flushing it to disk before returning."""
        event = {"t": round(self._clock(), 3), "kind": kind, **detail}
        self._events.append(event)
        del self._events[: max(0, len(self._events) - self.window)]
        if kind == "health" and detail.get("ok"):
            self._last_good = event
        if self.path:
            with self.path.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(event, default=str) + "\n")
                handle.flush()
                os.fsync(handle.fileno())
        return event

    @property
    def events(self) -> list[dict[str, Any]]:
        """Copy of the in-memory window, oldest first."""
        return list(self._events)

    def since_last_good(self) -> list[dict[str, Any]]:
        """Everything done since the device was last observed healthy."""
        if self._last_good is None:
            return self.events
        mark = self._last_good["t"]
        return [e for e in self._events if e["t"] > mark]

    def report(self, *, limit: int = 30) -> str:
        """Human-readable account of what preceded the current state."""
        suspects = self.since_last_good()
        now = self._clock()
        lines = []
        if self._last_good is None:
            lines.append("device was never observed healthy in this run")
        else:
            gap = now - self._last_good["t"]
            lines.append(f"last healthy {gap:.1f}s ago; {len(suspects)} actions since")
        for event in suspects[-limit:]:
            age = now - event["t"]
            detail = " ".join(f"{k}={v}" for k, v in event.items() if k not in ("t", "kind"))
            lines.append(f"  -{age:7.1f}s {event['kind']:<16} {detail}")
        return "\n".join(lines)


def probe_health(transport: Any, capture: Any = None, *, frames: int = 6) -> dict[str, Any]:
    """Snapshot of whether the device is carrying video, and what it claims.

    ``hdmi.connected`` reads false on this rig even immediately after a power
    cycle with a working link, so only frames decide ``ok``.
    """
    out: dict[str, Any] = {}
    try:
        status = transport.video_status()
        out.update(
            timing=status.timing,
            source=status.input_source,
            locked=bool(status.locked),
            source_locked=bool(status.source_locked),
        )
    except Exception as err:
        out["status_error"] = str(err)
    try:
        out["program"] = transport.current_program()
    except Exception as err:
        out["program_error"] = str(err)
    if capture is None:
        out["ok"] = bool(out.get("source_locked"))
        return out
    try:
        got = capture.frames(frames, timeout_s=8.0, settle=4)
        out["frames"] = len(got)
        if got:
            import numpy as np

            stack = np.stack(got)
            out["mean"] = round(float(stack.mean()), 4)
            out["chroma"] = round(float(capture.chroma_fraction(got[-1])), 4)
            out["motion"] = round(float(np.abs(np.diff(stack, axis=0)).mean()), 5) if len(got) > 1 else 0.0
        out["ok"] = bool(got) and float(out.get("mean", 0.0)) > 0.002
    except Exception as err:
        out["capture_error"] = str(err)
        out["ok"] = False
    return out


def watched(transport: Any, journal: Journal, verbs: Iterable[str]) -> Any:
    """Wrap a transport so the named verbs are journalled as they are called."""
    names = set(verbs)

    class _Watched:
        def __getattr__(self, name: str) -> Any:
            attr = getattr(transport, name)
            if name not in names or not callable(attr):
                return attr

            def call(*args: Any, **kwargs: Any) -> Any:
                journal.record("call", verb=name, args=[str(a) for a in args])
                return attr(*args, **kwargs)

            return call

    return _Watched()

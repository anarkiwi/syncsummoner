"""Stage logging and progress bars, so a long run says what it is doing.

Everything here degrades: with no tqdm installed, or output that is not a
terminal, a tracked loop is the loop and a stage is two log lines.
"""

from __future__ import annotations

import logging
import sys
import time
from contextlib import contextmanager
from typing import Any, Iterable, Iterator

try:
    from tqdm.auto import tqdm
except ImportError:  # pragma: no cover - exercised by the aesthetics-only install
    tqdm = None

LOG = logging.getLogger("syncsummoner")


def configure(verbosity: int = 0, *, stream: Any = None) -> None:
    """Send logs to stderr, at a level chosen by how many times ``-v`` was given."""
    level = {0: logging.INFO, 1: logging.DEBUG}.get(verbosity, logging.DEBUG)
    handler = logging.StreamHandler(sys.stderr if stream is None else stream)
    handler.setFormatter(logging.Formatter("%(asctime)s %(message)s", datefmt="%H:%M:%S"))
    root = logging.getLogger("syncsummoner")
    root.handlers[:] = [handler]
    root.setLevel(logging.WARNING if verbosity < 0 else level)
    root.propagate = False


def human(seconds: float) -> str:
    """Duration as the largest unit that keeps it readable."""
    if seconds < 90:
        return f"{seconds:.0f}s"
    if seconds < 5400:
        return f"{seconds / 60:.1f} min"
    return f"{seconds / 3600:.1f} h"


@contextmanager
def stage(message: str, **fields: Any) -> Iterator[dict]:
    """Log a stage starting, and how long it took on the way out.

    The yielded dict is the stage's result line: whatever a caller puts in it is
    reported when the block ends, so the outcome lands beside the elapsed time.
    """
    detail = " ".join(f"{k}={v}" for k, v in fields.items())
    LOG.info("%s%s", message, f" ({detail})" if detail else "")
    result: dict = {}
    start = time.monotonic()
    try:
        yield result
    except BaseException as err:
        LOG.error("%s failed after %s: %s", message, human(time.monotonic() - start), err)
        raise
    done = " ".join(f"{k}={v}" for k, v in result.items())
    LOG.info("%s done in %s%s", message, human(time.monotonic() - start), f": {done}" if done else "")


def track(iterable: Iterable, *, desc: str, total: int | None = None, unit: str = "it") -> Iterable:
    """Wrap a loop in a progress bar when there is a terminal and a tqdm to draw it."""
    if tqdm is None or not sys.stderr.isatty() or not LOG.isEnabledFor(logging.INFO):
        return iterable
    return tqdm(iterable, desc=desc, total=total, unit=unit, leave=False, file=sys.stderr)

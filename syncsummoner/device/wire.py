"""What crosses the wire during a take: every CC sent, every readback, every receipt.

A :class:`Sender` writes CCs at scheduled host ``CLOCK_MONOTONIC`` instants; a :class:`Poller`
keeps the serial shell busy with one ``modulation status`` at a time and collects the receipts
``midi monitor`` logs. Their records are numpy arrays in the dtypes below, saved beside the take.
"""

from __future__ import annotations

import queue
import re
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Sequence

import numpy as np

#: One CC write: send interval, parameter (1-based), value on the wire, target clip frame, when it was due.
SEND = np.dtype(
    [
        ("t0_ns", "<i8"),
        ("t1_ns", "<i8"),
        ("param", "<i8"),
        ("value", "<i8"),
        ("target", "<i8"),
        ("due_ns", "<i8"),
    ]
)
#: One ``modulation status`` reply: its request interval and the 12 combined values.
READ = np.dtype([("t0_ns", "<i8"), ("t1_ns", "<i8"), ("o", "<i8", (12,))])
#: One CC message the firmware logged receiving, at the latest host time it can have arrived.
RECEIPT = np.dtype([("t_ns", "<i8"), ("cc", "<i8"), ("value", "<i8")])
#: ``midi monitor on`` logs each channel message it receives in this form.
MONITOR = re.compile(r"\[MIDI USB\] CC\s+ch=(\d+)\s+cc=(\d+)\s+val=(\d+)")
#: Combined value keys a ``modulation status`` modulator carries, in order of preference.
COMBINED_KEYS = ("o", "out", "output", "combined")


def wire_path(take: str | Path) -> Path:
    """Where a take's wire record sits: sends, readbacks and receipts."""
    return Path(f"{take}.wire.npz")


def save_wire(take: str | Path, **arrays: Any) -> Path:
    """Keep a take's wire record beside it."""
    path = wire_path(take)
    np.savez(path, **arrays)
    return path


def load_wire(take: str | Path) -> dict[str, np.ndarray]:
    """A take's wire record, every array it holds."""
    with np.load(wire_path(take), allow_pickle=False) as record:
        return {key: record[key] for key in record.files}


def combined(status: dict) -> list[int]:
    """The 12 combined values of a ``modulation status`` reply."""
    return [int(next(slot[key] for key in COMBINED_KEYS if key in slot)) for slot in status["modulators"]]


def receipts(lines: Sequence[str], t_ns: int) -> list[tuple[int, int, int]]:
    """``(t_ns, cc, value)`` for every monitor line among ``lines``."""
    return [(t_ns, int(m.group(2)), int(m.group(3))) for line in lines if (m := MONITOR.search(line))]


@dataclass
class Asked:
    """A shell command run between polls, with when it was sent and answered."""

    words: tuple[str, ...]
    t0_ns: int = 0
    t1_ns: int = 0
    payload: str = ""
    error: str = ""
    done: threading.Event = field(default_factory=threading.Event)


class Poller:
    """Polls ``modulation status`` from one thread, the only client the shell has meanwhile.

    One command is ever in flight, so the device never holds more than one queued line.
    :meth:`ask` runs another command between polls; monitor lines are stamped with the reply they came in on.
    """

    def __init__(
        self,
        shell: Any,
        *,
        interval_s: float,
        clock: Callable[[], int] = time.monotonic_ns,
        sleep: Callable[[float], None] = time.sleep,
    ):
        self.shell = shell
        self.interval_s = float(interval_s)
        self.clock = clock
        self.sleep = sleep
        self.reads: list[tuple] = []
        self.receipts: list[tuple[int, int, int]] = []
        self.asked: list[Asked] = []
        self.errors: list[tuple[int, str]] = []
        self._pending: queue.Queue = queue.Queue()
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None

    def _drain_log(self, t_ns: int) -> None:
        lines = getattr(self.shell, "log_lines", None)
        if lines:
            self.receipts.extend(receipts(lines, t_ns))
            del lines[:]

    def _run_asked(self) -> None:
        while True:
            try:
                item = self._pending.get_nowait()
            except queue.Empty:
                return
            item.t0_ns = self.clock()
            try:
                item.payload = self.shell.command(*item.words).payload
            except Exception as err:  # pylint: disable=broad-exception-caught
                item.error = f"{type(err).__name__}: {err}"
            item.t1_ns = self.clock()
            self._drain_log(item.t1_ns)
            self.asked.append(item)
            item.done.set()

    def poll_once(self) -> None:
        """One readback, any asked-for commands, and the monitor lines that came with them."""
        self._run_asked()
        t0 = self.clock()
        try:
            values = combined(self.shell.modulation_status())
        except Exception as err:  # pylint: disable=broad-exception-caught
            t1 = self.clock()
            self.errors.append((t1, f"{type(err).__name__}: {err}"))
            self._drain_log(t1)
            return
        t1 = self.clock()
        self.reads.append((t0, t1, values))
        self._drain_log(t1)

    def _loop(self) -> None:
        while not self._stop.is_set():
            began = self.clock()
            self.poll_once()
            left = self.interval_s - (self.clock() - began) / 1e9
            if left > 0:
                self.sleep(left)

    def ask(self, *words: str, timeout_s: float | None = None) -> Asked:
        """Run ``words`` on the polling thread, between two polls; returns it answered."""
        item = Asked(tuple(words))
        self._pending.put(item)
        if self._thread is None:
            self._run_asked()
        item.done.wait(timeout_s)
        return item

    def start(self) -> "Poller":
        """Begin polling on a thread of its own."""
        self._stop.clear()
        self._thread = threading.Thread(target=self._loop, name="poller", daemon=True)
        self._thread.start()
        return self

    def stop(self) -> None:
        """Stop polling and wait for the thread; anything still asked for is run first."""
        self._stop.set()
        if self._thread is not None:
            self._thread.join()
            self._thread = None
        self._run_asked()

    def arrays(self) -> dict[str, np.ndarray]:
        """Readbacks, receipts and the times of failed reads, as arrays."""
        return {
            "reads": np.array([(a, b, tuple(o)) for a, b, o in self.reads], dtype=READ),
            "receipts": np.array(self.receipts, dtype=RECEIPT),
            "read_errors": np.array([t for t, _ in self.errors], dtype=np.int64),
        }


class Sender:
    """Writes CCs at scheduled host instants from a thread of its own, recording each.

    ``write(param, value)`` puts one parameter on the wire and returns the value it encoded.
    Rows are ``(target, param, value)``, sent in order at ``when(target)``, asked as each comes up.
    """

    def __init__(
        self,
        write: Callable[[int, int], int],
        *,
        when: Callable[[int], int] | None = None,
        clock: Callable[[], int] = time.monotonic_ns,
        sleep: Callable[[float], Any] | None = None,
        spin_ns: int = 2_000_000,
    ):
        self._stop = threading.Event()
        self.write = write
        self.when = when
        self.clock = clock
        self.sleep = self._stop.wait if sleep is None else sleep
        self.spin_ns = int(spin_ns)
        self.sent: list[tuple[int, int, int, int, int, int]] = []
        self.queued = 0
        self._queue: queue.Queue = queue.Queue()
        self._thread: threading.Thread | None = None

    def _wait(self, t_ns: int) -> bool:
        while not self._stop.is_set():
            left = t_ns - self.clock()
            if left <= 0:
                return True
            if left > self.spin_ns:
                self.sleep((left - self.spin_ns) / 1e9)
        return False

    def send(self, param: int, value: int, target: int, due: int | None = None) -> None:
        """One write now, recorded with its send interval and when it was due (now, when not given)."""
        t0 = self.clock()
        wire = self.write(int(param), int(value))
        t1 = self.clock()
        self.sent.append((t0, t1, int(param), int(wire), int(target), t0 if due is None else int(due)))

    def _loop(self) -> None:
        while not self._stop.is_set():
            try:
                target, param, value = self._queue.get(timeout=0.05)
            except queue.Empty:
                continue
            due = None if target is None else int(self.when(target))
            if due is None or not self._wait(due):
                return
            self.send(param, value, target, due)

    def plan(self, rows: Sequence[tuple[int, int, int]]) -> None:
        """Queue ``(target, param, value)`` rows, in send order."""
        for row in rows:
            self._queue.put(tuple(int(v) for v in row))
            self.queued += 1

    def start(self) -> "Sender":
        """Begin sending on a thread of its own."""
        self._stop.clear()
        self._thread = threading.Thread(target=self._loop, name="sender", daemon=True)
        self._thread.start()
        return self

    def drain(self, *, timeout_s: float, poll_s: float = 0.01) -> bool:
        """Wait until every queued row is sent; False if ``timeout_s`` passed first."""
        deadline = time.monotonic() + timeout_s
        while len(self.sent) < self.queued and time.monotonic() < deadline:
            time.sleep(poll_s)
        return len(self.sent) >= self.queued

    def stop(self) -> None:
        """Stop sending; rows not yet due are dropped, and show as missing."""
        self._stop.set()
        self._queue.put((None, 0, 0))
        if self._thread is not None:
            self._thread.join()
            self._thread = None

    def array(self) -> np.ndarray:
        """Every write so far as :data:`SEND` records."""
        return np.array(self.sent, dtype=SEND)

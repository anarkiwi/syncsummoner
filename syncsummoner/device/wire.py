"""What crosses the wire during a take: every CC sent, every readback, every receipt.

A :class:`Sender` writes each CC once the Pi shows its target frame, stamped with the frame shown; a :class:`Poller`
keeps the serial shell busy with one ``modulation status`` at a time and collects the receipts
``midi monitor`` logs. :class:`Echoes` turns the receipts into acknowledgements, so the sender
keeps one write in flight. Their records are numpy arrays in the dtypes below, saved beside the take.
"""

from __future__ import annotations

import difflib
import json
import queue
import re
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Iterable, Sequence

import numpy as np

from syncsummoner.progress import LOG

#: One CC write: send interval, parameter (1-based), value on the wire, target clip frame, when it was due,
#: and the floor: the latest clip frame the host had seen the Pi show before the write left it.
SEND = np.dtype(
    [
        ("t0_ns", "<i8"),
        ("t1_ns", "<i8"),
        ("param", "<i8"),
        ("value", "<i8"),
        ("target", "<i8"),
        ("due_ns", "<i8"),
        ("floor", "<i8"),
    ]
)
#: The floor of a write sent before the host had seen any frame shown.
NO_FLOOR = int(np.iinfo(np.int64).min)
#: One ``modulation status`` reply: its request interval and the 12 combined values.
READ = np.dtype([("t0_ns", "<i8"), ("t1_ns", "<i8"), ("o", "<i8", (12,))])
#: One ``transport status`` reply: its request interval and the device's ``field_count``, its frame tick.
TICK = np.dtype([("t0_ns", "<i8"), ("t1_ns", "<i8"), ("field", "<i8")])
#: One CC message the firmware logged receiving, at the latest host time it can have arrived.
RECEIPT = np.dtype([("t_ns", "<i8"), ("cc", "<i8"), ("value", "<i8")])
#: ``midi monitor on`` logs each channel message it receives in this form.
MONITOR = re.compile(r"\[MIDI USB\] CC\s+ch=(\d+)\s+cc=(\d+)\s+val=(\d+)")
#: Combined value keys a ``modulation status`` modulator carries, in order of preference.
COMBINED_KEYS = ("o", "out", "output", "combined")
#: Controllers below this carry an MSB whose LSB partner is 32 higher (MIDI 1.0 14-bit pairs).
LSB_OFFSET = 32
#: One CC message put on the wire: controller, 7-bit value, and the index of the send that carried it.
MESSAGE = np.dtype([("cc", "<i8"), ("value", "<i8"), ("send", "<i8")])


def wire_messages(cc_map: Sequence[int], *, high_resolution: bool = True) -> np.ndarray:
    """CC messages one write of each slot puts on the wire: an MSB/LSB pair, or one 7-bit message."""
    cc = np.asarray(cc_map, dtype=np.int64)
    return np.where(high_resolution & (cc < LSB_OFFSET), 2, 1)


def expected_messages(
    sends: np.ndarray, cc_map: Sequence[int], *, high_resolution: bool = True
) -> np.ndarray:
    """The :data:`MESSAGE` records the sends put on the wire, in order, as pyvmancer encodes them."""
    cc = np.asarray(cc_map, dtype=np.int64)
    per = wire_messages(cc, high_resolution=high_resolution)
    param = np.asarray(sends["param"], dtype=np.int64) - 1
    value = np.asarray(sends["value"], dtype=np.int64)
    pair = per[param] == 2
    word = np.rint(value * 16383 / 1023).astype(np.int64)
    out = np.zeros(int(per[param].sum()), dtype=MESSAGE)
    out["send"] = np.repeat(np.arange(value.size), per[param])
    head = np.r_[0, np.cumsum(per[param])[:-1]].astype(np.int64)
    out["cc"][head] = cc[param]
    out["value"][head] = np.where(pair, word >> 7, np.rint(value * 127 / 1023).astype(np.int64))
    tail = head[pair] + 1
    out["cc"][tail] = cc[param][pair] + LSB_OFFSET
    out["value"][tail] = word[pair] & 0x7F
    return out


def aligned(expected: np.ndarray, got: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Indices of the expected messages the receipts answer, and of the receipts answering them.

    The alignment is difflib's matching blocks over ``(cc, value)``, a common subsequence in which
    every receipt answers at most one message, in order, so a lost run costs only its own messages.
    """
    exp = np.asarray(expected["cc"], dtype=np.int64) * 128 + np.asarray(expected["value"], dtype=np.int64)
    rec = np.asarray(got["cc"], dtype=np.int64) * 128 + np.asarray(got["value"], dtype=np.int64)
    blocks = difflib.SequenceMatcher(None, exp.tolist(), rec.tolist(), autojunk=False).get_matching_blocks()
    runs = [(np.arange(b.a, b.a + b.size), np.arange(b.b, b.b + b.size)) for b in blocks]
    return tuple(np.concatenate([r[side] for r in runs]).astype(np.int64) for side in (0, 1))


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


#: A sent message no receipt answers: the send's index, interval start, parameter, target and the message.
MISSING = np.dtype(
    [("send", "<i8"), ("t0_ns", "<i8"), ("param", "<i8"), ("target", "<i8"), ("cc", "<i8"), ("value", "<i8")]
)


@dataclass(frozen=True)
class Received:
    """Sent against received: the firmware's monitor receipts aligned to the messages put on the wire."""

    sent: int
    received: int
    matched: int
    missing: np.ndarray = field(default_factory=lambda: np.zeros(0, dtype=MISSING))

    @property
    def ok(self) -> bool:
        """Every message received, in order, and nothing else."""
        return self.sent == self.received == self.matched

    @property
    def extra(self) -> int:
        """Receipts no sent message answers."""
        return self.received - self.matched

    def gaps(self) -> list[dict]:
        """Runs of consecutive missing messages: how many, the sends that carried them and when."""
        miss = self.missing
        if not miss.size:
            return []
        cut = np.flatnonzero(np.diff(miss["send"]) > 1) + 1
        return [
            {
                "messages": int(run.size),
                "sends": [int(run["send"][0]), int(run["send"][-1])],
                "params": sorted({int(p) for p in run["param"]}),
                "targets": sorted({int(t) for t in run["target"]}),
                "t0_ns": [int(run["t0_ns"][0]), int(run["t0_ns"][-1])],
            }
            for run in np.split(miss, cut)
        ]

    def summary(self) -> str:
        """One line of counts, and each run of missing messages."""
        said = (
            f"{self.sent} messages sent, {self.received} received, {self.matched} aligned, "
            f"{self.sent - self.matched} missing, {self.extra} unexpected"
        )
        for gap in self.gaps():
            span = (gap["t0_ns"][1] - gap["t0_ns"][0]) / 1e6
            said += (
                f"; {gap['messages']} missing from sends {gap['sends'][0]}..{gap['sends'][1]} "
                f"(P{', P'.join(map(str, gap['params']))} for frames "
                f"{gap['targets'][0]}..{gap['targets'][-1]}, sent over {span:.2f} ms)"
            )
        return said


def received(expected: np.ndarray, got: np.ndarray, sends: np.ndarray | None = None) -> Received:
    """Receipts aligned against the expected messages, so a gap costs only the messages it lost.

    The alignment is :func:`aligned`. ``sends`` names the missing messages' writes.
    """
    hit = np.zeros(len(expected), dtype=bool)
    hit[aligned(expected, got)[0]] = True
    lost = np.flatnonzero(~hit)
    missing = np.zeros(lost.size, dtype=MISSING)
    missing["cc"], missing["value"] = expected["cc"][lost], expected["value"][lost]
    missing["send"] = expected["send"][lost]
    if sends is not None and lost.size:
        carried = sends[missing["send"]]
        missing["t0_ns"], missing["param"], missing["target"] = (
            carried["t0_ns"],
            carried["param"],
            carried["target"],
        )
    return Received(sent=int(hit.size), received=len(got), matched=int(hit.sum()), missing=missing)


def echoed(
    sends: np.ndarray,
    got: np.ndarray,
    reads: np.ndarray,
    cc_map: Sequence[int],
    *,
    read_errors: np.ndarray | None = None,
    high_resolution: bool = True,
) -> np.ndarray:
    """When the device took each write, as ``(n, 2)`` brackets: NaN for one not wholly echoed.

    A message's echo comes with the first reply written after the device took it: after the
    previous reply's request left and the write began, by its own reply's return (docs/evidence.md).
    A write's bracket is its last message's; a resent copy's echo aligns to the first copy, the earlier.
    """
    expected = expected_messages(sends, cc_map, high_resolution=high_resolution)
    exp, rec = aligned(expected, got)
    stamp = got["t_ns"].astype(np.float64)
    errors = np.zeros(0, np.int64) if read_errors is None else np.asarray(read_errors, np.int64)
    replies = np.unique(np.r_[reads["t1_ns"], errors, got["t_ns"]]).astype(np.float64)
    asked = np.r_[-np.inf, replies[:-1]]
    read = np.isin(replies, reads["t1_ns"])
    asked[read] = reads["t0_ns"][np.searchsorted(reads["t1_ns"], replies[read])]
    k = np.searchsorted(replies, stamp)
    after = np.where(k > 0, asked[np.maximum(k - 1, 0)], -np.inf)
    took = np.full((len(expected), 2), np.nan)
    took[exp] = np.c_[after[rec], stamp[rec]]
    per = np.bincount(expected["send"], minlength=len(sends))
    whole = np.bincount(expected["send"][exp], minlength=len(sends)) == per
    out = np.where(whole[:, None], took[np.cumsum(per) - 1], np.nan)
    out[:, 0] = np.where(whole, np.fmax(out[:, 0], sends["t0_ns"]), np.nan)
    return out


def encoder(cc_map: Sequence[int], *, high_resolution: bool = True) -> Callable[[int, int], list]:
    """``(param, value) -> [(cc, value), ...]``: the messages one write puts on the wire."""

    def encode(param: int, value: int) -> list[tuple[int, int]]:
        row = np.array([(0, 0, param, value, 0, 0)], dtype=SEND)
        out = expected_messages(row, cc_map, high_resolution=high_resolution)
        return [(int(c), int(v)) for c, v in zip(out["cc"], out["value"])]

    return encode


class Echoes:
    """``midi monitor`` echoes as acknowledgements: fed by the poller, awaited by the sender.

    A write is acknowledged when its messages come back in order among the receipts not yet
    claimed. It is overdue once the longest echo this take has acknowledged has passed since the
    write returned and a reply asked for after that has come back without it: the take's own echo
    delays and poll cadence set the wait, and a stalled poll never makes a write overdue. Until
    one echo is acknowledged nothing is overdue, so the first write waits for its echo. ``idle``,
    when given, is called in place of blocking, for a caller that polls on the waiting thread.
    """

    def __init__(self, *, idle: Callable[[], None] | None = None) -> None:
        self._idle = idle
        self._cond = threading.Condition()
        self._heard: list[tuple[int, int, int]] = []
        self._cursor = 0
        self._asked_ns = -1
        self._closed = False
        #: Longest delay, write return to echo, acknowledged so far; None before the first.
        self.longest_ns: int | None = None
        #: Set while a write waits, so the poller polls back to back.
        self.pending = threading.Event()

    def hear(self, got: Iterable[tuple[int, int, int]], asked_ns: int) -> None:
        """Receipts that came with a reply whose request left at ``asked_ns``."""
        with self._cond:
            self._heard.extend(got)
            self._asked_ns = max(self._asked_ns, int(asked_ns))
            self._cond.notify_all()

    def close(self) -> None:
        """No more replies are coming: every wait ends."""
        with self._cond:
            self._closed = True
            self._cond.notify_all()

    def _match(self, want: Sequence[tuple[int, int]]) -> int | None:
        k = 0
        for i in range(self._cursor, len(self._heard)):
            if self._heard[i][1:] == want[k]:
                k += 1
                if k == len(want):
                    self._cursor = i + 1
                    return self._heard[i][0]
        return None

    def wait(self, want: Sequence[tuple[int, int]], since_ns: int, began_ns: int | None = None) -> int | None:
        """When the last of ``want`` came back, for a write returned at ``since_ns``; None if overdue.

        ``began_ns`` is when the write first returned, for a copy written again, so an echo of
        either copy measures the delay from the first.
        """
        want = [tuple(m) for m in want]
        self.pending.set()
        try:
            with self._cond:
                while (got := self._match(want)) is None:
                    late = self.longest_ns is not None and self._asked_ns > since_ns + self.longest_ns
                    if self._closed or late:
                        self._cursor = len(self._heard)
                        return None
                    if self._idle is None:
                        self._cond.wait()
                    else:
                        self._idle()
                delay = got - int(since_ns if began_ns is None else began_ns)
                self.longest_ns = delay if self.longest_ns is None else max(self.longest_ns, delay)
                return got
        finally:
            self.pending.clear()


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
    """Polls ``field_count`` and ``modulation status`` in turn from one thread, the shell's only client.

    One command is ever in flight, so the device never holds more than one queued line, and every
    readback lies between the frame ticks read either side of it (docs/evidence.md).
    :meth:`ask` runs another command between polls; monitor lines are stamped with the reply they came in on.
    """

    def __init__(
        self,
        shell: Any,
        *,
        interval_s: float,
        clock: Callable[[], int] = time.monotonic_ns,
        sleep: Callable[[float], None] = time.sleep,
        echoes: Echoes | None = None,
    ):
        self.shell = shell
        self.echoes = echoes
        self.interval_s = float(interval_s)
        self.clock = clock
        self.sleep = sleep
        self.reads: list[tuple] = []
        self.ticks: list[tuple[int, int, int]] = []
        self.receipts: list[tuple[int, int, int]] = []
        self.asked: list[Asked] = []
        self.errors: list[tuple[int, str]] = []
        self._pending: queue.Queue = queue.Queue()
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None

    def _drain_log(self, t_ns: int, asked_ns: int) -> None:
        lines = getattr(self.shell, "log_lines", None)
        got = receipts(lines, t_ns) if lines else []
        if lines:
            self.receipts.extend(got)
            del lines[:]
        if self.echoes is not None:
            self.echoes.hear(got, asked_ns)

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
            self._drain_log(item.t1_ns, item.t0_ns)
            self.asked.append(item)
            item.done.set()

    def tick_once(self) -> None:
        """One ``field_count``, and the monitor lines that came with it."""
        t0 = self.clock()
        try:
            count = int(json.loads(self.shell.command("transport", "status").payload)["field_count"])
        except Exception as err:  # pylint: disable=broad-exception-caught
            t1 = self.clock()
            self.errors.append((t1, f"{type(err).__name__}: {err}"))
            self._drain_log(t1, t0)
            return
        t1 = self.clock()
        self.ticks.append((t0, t1, count))
        self._drain_log(t1, t0)

    def poll_once(self) -> None:
        """Any asked-for commands, one frame tick, one readback, and the monitor lines that came with them."""
        self._run_asked()
        self.tick_once()
        t0 = self.clock()
        try:
            values = combined(self.shell.modulation_status())
        except Exception as err:  # pylint: disable=broad-exception-caught
            t1 = self.clock()
            self.errors.append((t1, f"{type(err).__name__}: {err}"))
            self._drain_log(t1, t0)
            return
        t1 = self.clock()
        self.reads.append((t0, t1, values))
        self._drain_log(t1, t0)

    def _loop(self) -> None:
        while not self._stop.is_set():
            began = self.clock()
            self.poll_once()
            left = self.interval_s - (self.clock() - began) / 1e9
            if left > 0 and not (self.echoes is not None and self.echoes.pending.is_set()):
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
        if self.echoes is not None:
            self.echoes.close()

    def arrays(self) -> dict[str, np.ndarray]:
        """Readbacks, frame ticks, receipts and the times of failed reads, as arrays."""
        return {
            "reads": np.array([(a, b, tuple(o)) for a, b, o in self.reads], dtype=READ),
            "ticks": np.array(self.ticks, dtype=TICK),
            "receipts": np.array(self.receipts, dtype=RECEIPT),
            "read_errors": np.array([t for t, _ in self.errors], dtype=np.int64),
        }


class Sender:
    """Writes CCs from a thread of its own as their frames come up, recording each with its floor.

    ``write(param, value)`` puts one parameter on the wire and returns the value it encoded. ``floor()``
    is the latest clip frame seen shown, read before each write leaves, so a write can only show on
    that frame or later. Rows are ``(target, param, value)``, sent in order once ``due(target, stop)``
    returns, the host time the target was seen shown (None to stop). With ``echoes``, one write is in
    flight: each waits for its messages, ``encode(param, value)``, to come back, and an overdue one is
    written once more, the copy's row listed in :meth:`resent`.
    """

    def __init__(
        self,
        write: Callable[[int, int], int],
        *,
        floor: Callable[[], int] = lambda: NO_FLOOR,
        due: Callable[[int, threading.Event], int | None] | None = None,
        clock: Callable[[], int] = time.monotonic_ns,
        echoes: Echoes | None = None,
        encode: Callable[[int, int], Sequence[tuple[int, int]]] | None = None,
    ):
        if (echoes is None) != (encode is None):
            raise ValueError("echoes and encode come together")
        self.echoes, self.encode = echoes, encode
        self.resends: list[int] = []
        self._stop = threading.Event()
        self.write = write
        self.floor = floor
        self.due = due
        self.clock = clock
        self.sent: list[tuple[int, int, int, int, int, int, int]] = []
        self.queued = 0
        self._queue: queue.Queue = queue.Queue()
        self._thread: threading.Thread | None = None

    def _put(self, param: int, value: int, target: int, due: int | None) -> tuple[int, int]:
        floor = int(self.floor())
        t0 = self.clock()
        wire = self.write(int(param), int(value))
        t1 = self.clock()
        self.sent.append((t0, t1, int(param), int(wire), int(target), t0 if due is None else int(due), floor))
        return int(wire), t1

    def send(self, param: int, value: int, target: int, due: int | None = None) -> None:
        """One write now, recorded with its send interval and when it was due (now, when not given).

        With echoes it returns once the write is acknowledged, or written twice and still unechoed.
        """
        wire, first = self._put(param, value, target, due)
        if self.echoes is None or self.encode is None:
            return
        want = self.encode(int(param), wire)
        if self.echoes.wait(want, first) is not None:
            return
        self.resends.append(len(self.sent))
        _, again = self._put(param, value, target, due)
        if self.echoes.wait(want, again, first) is None:
            LOG.warning("P%d=%d for frame %d: no echo after a resend", param, value, target)

    def _loop(self) -> None:
        while not self._stop.is_set():
            try:
                target, param, value = self._queue.get(timeout=0.05)
            except queue.Empty:
                continue
            due = None if target is None or self.due is None else self.due(target, self._stop)
            if due is None or self._stop.is_set():
                return
            self.send(param, value, target, int(due))

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

    def resent(self) -> np.ndarray:
        """Rows of :meth:`array` that wrote an overdue write again, each just after its first copy."""
        return np.asarray(self.resends, dtype=np.int64)

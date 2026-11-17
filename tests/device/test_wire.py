"""The wire record: CCs sent at scheduled instants, readbacks polled, monitor receipts parsed."""

# pylint: disable=missing-function-docstring,protected-access

import threading
import time
import types

import numpy as np
import pytest

from syncsummoner.device import wire as W


class Ticker:
    """A clock that advances a fixed step each time it is read."""

    def __init__(self, step=1000, start=0):
        self.now, self.step = start, step

    def __call__(self):
        self.now += self.step
        return self.now


class Shell:
    """A serial shell answering ``modulation status`` and logging monitor lines as replies arrive."""

    def __init__(self, statuses, lines=None, fail_on=()):
        self.statuses = list(statuses)
        self.lines = list(lines or [])
        self.fail_on = set(fail_on)
        self.log_lines = []
        self.commands = []
        self.polls = 0

    def modulation_status(self):
        self.polls += 1
        if self.lines:
            self.log_lines.append(self.lines.pop(0))
        if self.polls in self.fail_on:
            raise TimeoutError("no reply")
        return self.statuses[(self.polls - 1) % len(self.statuses)]

    def command(self, *words):
        self.commands.append(words)
        self.log_lines.append("[MIDI USB] CC  ch=1 cc=7 val=99")
        if words[0] == "bad":
            raise ValueError("unknown command")
        return types.SimpleNamespace(payload=f"ran {' '.join(words)}")


def status(values, key="o"):
    return {"modulators": [{key: v, "src": 0} for v in values]}


def test_combined_takes_the_first_key_a_modulator_carries():
    reply = {"modulators": [{"out": 5, "combined": 9}, {"o": 1, "out": 2}, {"output": 7}, {"combined": 3}]}
    assert W.combined(reply) == [5, 1, 7, 3]


def test_receipts_parse_only_monitor_lines_stamped_with_their_reply():
    lines = ["[MIDI USB] CC  ch=1 cc=0 val=12", "ok", "[MIDI USB] CC ch=16 cc=32 val=127", "[MIDI USB] NOTE"]
    assert W.receipts(lines, 55) == [(55, 0, 12), (55, 32, 127)]


def test_a_wire_record_round_trips_beside_its_take(tmp_path):
    take = tmp_path / "take.mkv"
    sends = np.array([(1, 2, 3, 4, 5)], dtype=W.SEND)
    path = W.save_wire(take, sends=sends, program=np.array("Edge"))
    assert path == tmp_path / "take.mkv.wire.npz" == W.wire_path(take)
    back = W.load_wire(take)
    assert back["sends"].dtype == W.SEND and back["sends"].tolist() == sends.tolist()
    assert str(back["program"]) == "Edge"


def test_a_poll_records_its_interval_values_and_the_receipts_that_came_with_it():
    shell = Shell([status(range(12))], lines=["[MIDI USB] CC  ch=1 cc=0 val=12"])
    poller = W.Poller(shell, interval_s=0.01, clock=Ticker())
    poller.poll_once()
    assert poller.reads == [(1000, 2000, list(range(12)))]
    assert poller.receipts == [(2000, 0, 12)] and not shell.log_lines
    arrays = poller.arrays()
    assert arrays["reads"]["o"][0].tolist() == list(range(12))
    assert arrays["receipts"].tolist() == [(2000, 0, 12)] and arrays["read_errors"].size == 0


def test_a_failed_poll_is_an_error_at_its_reply_time_and_still_drains_the_log():
    shell = Shell([status([0] * 12)], lines=["x", "[MIDI USB] CC  ch=1 cc=3 val=4"], fail_on={2})
    poller = W.Poller(shell, interval_s=0.01, clock=Ticker())
    poller.poll_once()
    poller.poll_once()
    assert len(poller.reads) == 1
    assert poller.errors == [(4000, "TimeoutError: no reply")]
    assert poller.receipts == [(4000, 3, 4)]
    assert poller.arrays()["read_errors"].tolist() == [4000]


def test_asking_without_a_thread_runs_the_command_at_once_and_records_failures():
    shell = Shell([status([0] * 12)])
    poller = W.Poller(shell, interval_s=0.01, clock=Ticker())
    good = poller.ask("fpga", "status")
    bad = poller.ask("bad")
    assert good.done.is_set() and good.payload == "ran fpga status" and not good.error
    assert (good.t0_ns, good.t1_ns) == (1000, 2000)
    assert bad.error == "ValueError: unknown command" and bad.payload == ""
    assert poller.asked == [good, bad]
    assert poller.receipts == [(2000, 7, 99), (4000, 7, 99)]


def test_asked_commands_run_before_the_next_poll():
    shell = Shell([status([1] * 12)])
    poller = W.Poller(shell, interval_s=0.01, clock=Ticker())
    item = W.Asked(("fpga", "status"))
    poller._pending.put(item)
    poller.poll_once()
    assert item.t1_ns < poller.reads[0][0]


def test_a_threaded_poller_polls_until_stopped_and_serves_asks_between_polls():
    shell = Shell([status(range(12))])
    poller = W.Poller(shell, interval_s=0.001).start()
    item = poller.ask("fpga", "status", timeout_s=5.0)
    deadline = time.monotonic() + 5.0
    while len(poller.reads) < 3 and time.monotonic() < deadline:
        time.sleep(0.001)
    poller.stop()
    assert item.done.is_set() and item.payload == "ran fpga status"
    assert len(poller.reads) >= 3 and poller._thread is None
    t = np.array([(a, b) for a, b, _ in poller.reads])
    assert np.all(t[:, 1] >= t[:, 0]) and np.all(t[1:, 0] >= t[:-1, 1])


def test_a_send_records_its_interval_and_the_value_the_wire_encoded():
    written = []

    def write(param, value):
        written.append((param, value))
        return value - 1

    sender = W.Sender(write, clock=Ticker(step=10))
    sender.send(3, 500, 7)
    assert written == [(3, 500)]
    assert sender.array().tolist() == [(10, 20, 3, 499, 7)]


def test_waiting_sleeps_to_within_the_spin_window_then_spins():
    now = [0]
    sleeps = []

    def clock():
        now[0] += 100_000
        return now[0]

    def sleep(seconds):
        sleeps.append(seconds)
        now[0] += int(seconds * 1e9)

    sender = W.Sender(lambda p, v: v, clock=clock, sleep=sleep, spin_ns=1_000_000)
    assert sender._wait(10_000_000)
    assert sleeps and sleeps[0] == pytest.approx((10_000_000 - 100_000 - 1_000_000) / 1e9)
    assert now[0] >= 10_000_000
    sender._stop.set()
    assert not sender._wait(10**12)


def test_a_threaded_sender_writes_each_row_in_order_no_earlier_than_due():
    lock = threading.Lock()
    order = []

    def write(param, value):
        with lock:
            order.append((param, value))
        return value

    base = time.monotonic_ns() + 20_000_000
    sender = W.Sender(write, when=lambda target: base + target * 2_000_000, spin_ns=500_000)
    sender.plan([(0, 1, 10), (2, 2, 20), (1, 3, 30)])
    sender.start()
    assert sender.drain(timeout_s=5.0)
    sender.stop()
    sent = sender.array()
    assert order == [(1, 10), (2, 20), (3, 30)]
    assert np.all(sent["t0_ns"] >= base + sent["target"] * 2_000_000)
    assert sent["target"].tolist() == [0, 2, 1] and sender._thread is None


def test_drain_times_out_while_rows_are_not_yet_due():
    sender = W.Sender(lambda p, v: v, when=lambda target: time.monotonic_ns() + 200_000_000)
    sender.plan([(0, 1, 1)])
    sender.start()
    assert not sender.drain(timeout_s=0.02, poll_s=0.005)
    sender.stop()
    assert sender.array().size == 0 and sender.queued == 1


def test_stopping_does_not_wait_for_a_row_far_in_the_future():
    sender = W.Sender(lambda p, v: v, when=lambda target: time.monotonic_ns() + 3_000_000_000)
    sender.plan([(0, 1, 1)])
    sender.start()
    time.sleep(0.05)
    stopper = threading.Thread(target=sender.stop, daemon=True)
    stopper.start()
    stopper.join(0.5)
    assert not stopper.is_alive()

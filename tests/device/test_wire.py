"""The wire record: CCs sent at scheduled instants, readbacks polled, monitor receipts parsed."""

# pylint: disable=missing-function-docstring,protected-access

import json
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
        self.fields = 0

    def modulation_status(self):
        self.polls += 1
        if self.lines:
            self.log_lines.append(self.lines.pop(0))
        if self.polls in self.fail_on:
            raise TimeoutError("no reply")
        return self.statuses[(self.polls - 1) % len(self.statuses)]

    def command(self, *words):
        if words == ("transport", "status"):
            self.fields += 1
            said = {"field_count": self.fields, "phase": 1343 * self.fields, "bpm_x100": 3689, "state": "playing"}
            return types.SimpleNamespace(payload=json.dumps(said))
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
    sends = np.array([(1, 2, 3, 4, 5, 0, 7)], dtype=W.SEND)
    path = W.save_wire(take, sends=sends, program=np.array("Edge"))
    assert path == tmp_path / "take.mkv.wire.npz" == W.wire_path(take)
    back = W.load_wire(take)
    assert back["sends"].dtype == W.SEND and back["sends"].tolist() == sends.tolist()
    assert str(back["program"]) == "Edge"


def test_a_poll_records_its_interval_values_and_the_receipts_that_came_with_it():
    shell = Shell([status(range(12))], lines=["[MIDI USB] CC  ch=1 cc=0 val=12"])
    poller = W.Poller(shell, interval_s=0.01, clock=Ticker())
    poller.poll_once()
    assert poller.ticks == [(1000, 2000, 1, 1343, 3689, True)], "the frame tick is read before the readback"
    assert poller.reads == [(3000, 4000, list(range(12)))]
    assert poller.receipts == [(4000, 0, 12)] and not shell.log_lines
    arrays = poller.arrays()
    assert arrays["reads"]["o"][0].tolist() == list(range(12))
    assert arrays["ticks"].tolist() == [(1000, 2000, 1, 1343, 3689, True)]
    assert arrays["receipts"].tolist() == [(4000, 0, 12)] and arrays["read_errors"].size == 0


def test_a_failed_poll_is_an_error_at_its_reply_time_and_still_drains_the_log():
    shell = Shell([status([0] * 12)], lines=["x", "[MIDI USB] CC  ch=1 cc=3 val=4"], fail_on={2})
    poller = W.Poller(shell, interval_s=0.01, clock=Ticker())
    poller.poll_once()
    poller.poll_once()
    assert len(poller.reads) == 1 and len(poller.ticks) == 2
    assert poller.errors == [(8000, "TimeoutError: no reply")]
    assert poller.receipts == [(8000, 3, 4)]
    assert poller.arrays()["read_errors"].tolist() == [8000]


def test_a_failed_tick_is_an_error_and_the_readback_still_runs():
    shell = Shell([status([0] * 12)])
    shell.command = lambda *words: types.SimpleNamespace(payload="not json")
    poller = W.Poller(shell, interval_s=0.01, clock=Ticker())
    poller.poll_once()
    assert not poller.ticks and len(poller.reads) == 1
    assert poller.errors[0][0] == 2000 and poller.errors[0][1].startswith("JSONDecodeError")


def test_each_tick_read_calls_its_hook_after_the_reply():
    shell = Shell([status([0] * 12)])
    poller = W.Poller(shell, interval_s=0.01, clock=Ticker())
    seen = []
    poller.on_tick = lambda: seen.append(len(poller.ticks))
    poller.poll_once()
    poller.poll_once()
    assert seen == [1, 2]


def test_waiting_for_a_count_returns_the_reply_that_reached_it_or_none_once_stopped():
    shell = Shell([status([0] * 12)])
    poller = W.Poller(shell, interval_s=0.001).start()
    try:
        assert poller.wait_count(3, threading.Event()) >= poller.ticks[2][1]
    finally:
        poller.stop()
    stop = threading.Event()
    stop.set()
    assert poller.wait_count(10**9, stop) is None


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
    sender.send(4, 9, 8, due=5)
    assert written == [(3, 500), (4, 9)]
    none = W.NO_FLOOR
    assert sender.array().tolist() == [(10, 20, 3, 499, 7, 10, none), (30, 40, 4, 8, 8, 5, none)]


def test_each_send_carries_the_floor_read_before_it_left():
    shown = iter([4, 9])
    order = []

    def floor():
        order.append("floor")
        return next(shown)

    sender = W.Sender(lambda p, v: order.append("write") or v, floor=floor)
    sender.send(1, 2, 5)
    sender.send(1, 3, 10)
    assert order == ["floor", "write", "floor", "write"]
    assert sender.array()["floor"].tolist() == [4, 9]


def test_a_threaded_sender_writes_each_row_in_order_no_earlier_than_due():
    lock = threading.Lock()
    order = []

    def write(param, value):
        with lock:
            order.append((param, value))
        return value

    base = time.monotonic_ns() + 20_000_000

    def due(target, stop):
        at = base + target * 2_000_000
        while time.monotonic_ns() < at and not stop.is_set():
            time.sleep(0.0005)
        return at

    sender = W.Sender(write, due=due)
    sender.plan([(0, 1, 10), (2, 2, 20), (1, 3, 30)])
    sender.start()
    assert sender.drain(timeout_s=5.0)
    sender.stop()
    sent = sender.array()
    assert order == [(1, 10), (2, 20), (3, 30)]
    assert sent["due_ns"].tolist() == (base + sent["target"] * 2_000_000).tolist()
    assert np.all(sent["t0_ns"] >= sent["due_ns"])
    assert sent["target"].tolist() == [0, 2, 1] and sender._thread is None


def test_drain_times_out_while_rows_are_not_yet_due():
    sender = W.Sender(lambda p, v: v, due=lambda target, stop: None if stop.wait(0.2) else time.monotonic_ns())
    sender.plan([(0, 1, 1)])
    sender.start()
    assert not sender.drain(timeout_s=0.02, poll_s=0.005)
    sender.stop()
    assert sender.array().size == 0 and sender.queued == 1


def test_stopping_does_not_wait_for_a_row_far_in_the_future():
    sender = W.Sender(lambda p, v: v, due=lambda target, stop: stop.wait(3.0) and None)
    sender.plan([(0, 1, 1)])
    sender.start()
    time.sleep(0.05)
    stopper = threading.Thread(target=sender.stop, daemon=True)
    stopper.start()
    stopper.join(0.5)
    assert not stopper.is_alive()


def test_a_write_takes_a_pair_only_where_its_msb_has_an_lsb_partner():
    """Controllers 0..31 pair with one 32 higher at high resolution; any other is one message."""
    assert W.wire_messages([0, 31, 32, 64]).tolist() == [2, 2, 1, 1]
    assert W.wire_messages([0, 31], high_resolution=False).tolist() == [1, 1]


def test_expected_messages_follow_the_cc_encoding():
    sends = np.zeros(3, dtype=W.SEND)
    sends["param"], sends["value"] = [1, 2, 2], [1023, 0, 512]
    msgs = W.expected_messages(sends, [16, 40])
    assert msgs[["cc", "value"]].tolist() == [(16, 127), (48, 127), (40, 0), (40, 64)]
    assert msgs["send"].tolist() == [0, 0, 1, 2]
    lo = W.expected_messages(sends[:1], [16], high_resolution=False)
    assert lo[["cc", "value", "send"]].tolist() == [(16, 127, 0)]
    word = W.expected_messages(sends[[2]], [3, 3])["value"]
    assert abs(((word[0] << 7) | word[1]) / 16383 * 1023 - 512) < 0.5
    assert W.encoder([16, 40])(1, 1023) == [(16, 127), (48, 127)]
    assert W.encoder([16], high_resolution=False)(1, 1023) == [(16, 127)]


def test_alignment_pairs_each_receipt_with_at_most_one_message_in_order():
    sent = np.zeros(4, dtype=W.MESSAGE)
    sent["cc"], sent["value"] = [1, 33, 2, 34], [5, 6, 7, 8]
    got = np.zeros(3, dtype=W.RECEIPT)
    got["cc"], got["value"] = [9, 1, 2], [9, 5, 7]
    exp, rec = W.aligned(sent, got)
    assert exp.tolist() == [0, 2] and rec.tolist() == [1, 2]


def test_an_echo_acknowledges_its_messages_in_order_past_stray_receipts():
    echoes = W.Echoes()
    echoes.hear([(10, 9, 9), (11, 1, 5), (12, 9, 9), (13, 33, 6)], asked_ns=0)
    assert echoes.wait([(1, 5), (33, 6)], since_ns=8) == 13
    assert echoes.longest_ns == 5
    echoes.hear([(20, 1, 5)], asked_ns=1)
    assert echoes._match([(1, 5), (33, 6)]) is None, "claimed receipts are never matched again"


def test_nothing_is_overdue_before_one_echo_then_after_the_longest_and_a_later_reply():
    script = [([], 50), ([(70, 1, 5)], 60)]
    echoes = W.Echoes(idle=lambda: echoes.hear(*script.pop(0)))
    assert echoes.wait([(1, 5)], since_ns=10) == 70, "with no echo yet, a reply past the write is no timeout"
    assert echoes.longest_ns == 60
    script[:] = [([(80, 9, 9)], 100), ([], 140)]
    assert echoes.wait([(2, 5)], since_ns=79) is None, "a reply asked at 140 is past 79 + 60"
    assert not script and not echoes.pending.is_set()
    script[:] = [([(300, 2, 5)], 200)]
    assert echoes.wait([(2, 5)], since_ns=250, began_ns=100) == 300 and echoes.longest_ns == 200
    echoes.close()
    assert echoes.wait([(1, 1)], since_ns=0) is None


def test_the_poller_polls_back_to_back_while_a_write_waits_and_rests_otherwise():
    shell = Shell([status(range(12))])
    echoes = W.Echoes()
    slept = []
    poller = W.Poller(shell, interval_s=10.0, sleep=slept.append, echoes=echoes)

    def poll():
        if shell.polls == 3:
            echoes.pending.clear()
        if shell.polls == 4:
            poller._stop.set()
        return status(range(12))

    shell.modulation_status = lambda: (setattr(shell, "polls", shell.polls + 1), poll())[1]
    echoes.pending.set()
    poller._loop()
    assert shell.polls == 4 and len(slept) == 2, "no rest after the two polls a write waited on"


def test_a_write_waiting_on_a_threaded_poller_is_acknowledged_and_a_stop_closes_its_echoes():
    shell = Shell([status(range(12))], lines=["", "[MIDI USB] CC  ch=1 cc=0 val=12"])
    echoes = W.Echoes()
    poller = W.Poller(shell, interval_s=0.001, echoes=echoes).start()
    try:
        assert echoes.wait([(0, 12)], since_ns=0) is not None
    finally:
        poller.stop()
    assert echoes.wait([(0, 13)], since_ns=0) is None


def test_every_reply_feeds_the_echoes_with_its_request_time():
    shell = Shell([status(range(12))], lines=["[MIDI USB] CC  ch=1 cc=0 val=12"], fail_on={2})
    echoes = W.Echoes()
    poller = W.Poller(shell, interval_s=0.01, clock=Ticker(), echoes=echoes)
    poller.poll_once()
    poller.poll_once()
    poller.ask("fpga", "status")
    assert echoes._asked_ns == 9000 and len(echoes._heard) == 2


class Acks:
    """Echoes that answer each wait from a script: a stamp, or None for overdue."""

    def __init__(self, *answers):
        self.answers, self.waits = list(answers), []

    def wait(self, want, since_ns, began_ns=None):
        self.waits.append((want, since_ns, began_ns))
        return self.answers.pop(0)


def test_an_acknowledged_send_returns_after_one_write():
    acks, wrote = Acks(99), []
    sender = W.Sender(
        lambda p, v: wrote.append((p, v)) or v, clock=Ticker(), echoes=acks, encode=W.encoder([0])
    )
    sender.send(1, 1023, 4)
    assert wrote == [(1, 1023)] and sender.resent().size == 0
    assert acks.waits == [([(0, 127), (32, 127)], 2000, None)]


def test_an_overdue_write_is_written_once_more_and_the_copy_recorded(caplog):
    acks, wrote = Acks(None, None), []
    sender = W.Sender(
        lambda p, v: wrote.append((p, v)) or v, clock=Ticker(), echoes=acks, encode=W.encoder([0])
    )
    sender.send(1, 0, 4)
    assert wrote == [(1, 0), (1, 0)] and sender.resent().tolist() == [1]
    assert acks.waits[1][2] == acks.waits[0][1], "the copy's echo is timed from the first write"
    assert "no echo after a resend" in caplog.text
    with pytest.raises(ValueError, match="come together"):
        W.Sender(lambda p, v: v, echoes=acks)

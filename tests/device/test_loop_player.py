"""The host's control of the Pi-side KMS player: upload, start, watch and stop by pidfile."""

# pylint: disable=missing-function-docstring,redefined-outer-name

import base64
import gzip
import json
import logging
import os
import re
import signal
import subprocess
import sys
import time
from fractions import Fraction

import pytest

from syncsummoner.device import host, playout as po
from syncsummoner.device.playout import ClipPlayer, LoopPlayer, PlayoutError

from .conftest import FakeClock
from .test_playout import frame

FAKE_PLAYER = """\
import json, os, signal, struct, sys, time
args = sys.argv[2:]
status, flips = args[args.index("--status") + 1], args[args.index("--flips") + 1]
mode = open(os.path.join(os.path.dirname(status), "mode")).read().split()
def put(**state):
    with open(status + ".tmp", "w") as out:
        json.dump(state, out)
    os.replace(status + ".tmp", status)
print("argv", *args, flush=True)
if "fail" in mode:
    put(state="failed", error="no 1920x1080 progressive mode")
    sys.exit(2)
with open(flips, "wb") as log:
    log.write(struct.pack("<qqqqq", 0, 1000, 0, 5, 1) + struct.pack("<qqqqq", 1, 1001, 0, 9, 0))
signal.signal(signal.SIGTERM, lambda *_: (put(state="stopped"), sys.exit(0)))
put(state="playing", realtime="rt" in mode, pid=os.getpid())
if "die" in mode:
    os._exit(1)
while True:
    time.sleep(1)
"""


class Pi:
    """A local shell standing in for the Pi, with ``python3`` and ``sudo`` replaced by stubs.

    ``setsid`` waits on the player it starts, so a player that exits is reaped without an init.
    """

    def __init__(self, root):
        self.root = root
        self.directory = root / "pi"
        bin_dir = root / "bin"
        bin_dir.mkdir()
        (bin_dir / "python3").write_text(f"#!{sys.executable}\n{FAKE_PLAYER}")
        (bin_dir / "sudo").write_text('#!/bin/sh\n[ "$1" = -n ] && shift\nexec "$@"\n')
        (bin_dir / "setsid").write_text('#!/bin/sh\n/usr/bin/setsid "$@" &\nwait\n')
        for stub in bin_dir.iterdir():
            stub.chmod(0o755)
        self.env = {"PATH": f"{bin_dir}:/usr/bin:/bin"}
        self.mode("")

    def mode(self, mode):
        self.directory.mkdir(exist_ok=True)
        (self.directory / "mode").write_text(mode)

    def __call__(self, command, data=None):
        done = subprocess.run(
            ["sh", "-c", command], input=data, env=self.env, capture_output=True, check=True
        )
        return done.stdout.decode().strip()

    def argv(self):
        said = (self.directory / "player.log").read_text()
        return re.search(r"^argv (.*)$", said, re.M)[1].split()


@pytest.fixture
def pi(tmp_path):
    shell = Pi(tmp_path)
    yield shell
    pidfile = shell.directory / "player.pid"
    if pidfile.exists() and pidfile.read_text().strip():
        try:
            os.kill(int(pidfile.read_text()), signal.SIGKILL)
        except ProcessLookupError:
            pass


def on(pi, cls=LoopPlayer, **kwargs):
    return cls(
        runner=pi, directory=str(pi.directory), width=4, height=2, poll_s=0.01, timeout_s=10.0, **kwargs
    )


def alive(pid):
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    return True


def test_show_holds_one_frame_by_its_own_pid_until_stopped(pi):
    player = on(pi)
    still = frame((1.0, 0.5, 0.0), 2, 4)
    pid = player.show(still, fps=29.97)
    assert (pi.directory / "frames" / "frame.000").read_bytes() == player.encode(still)
    assert player.frame_count() == 1
    assert pid == player.pid() == int((pi.directory / "player.pid").read_text()) and alive(pid)
    assert player.check()["state"] == "playing"
    argv = pi.argv()
    assert argv[argv.index("--fps") + 1] == "30000/1001" and "--loop" in argv
    assert argv[argv.index("--frames") + 1] == player.frames_dir
    assert (pi.directory / "kmsplay.py").read_bytes() == po.player_source()
    assert player.stop() == {"state": "stopped"}
    assert not alive(pid) and player.pid() is None and not player.is_running()


def test_flips_are_read_back_whole(pi):
    player = on(pi)
    player.show(frame((0, 0, 0), 2, 4))
    got = player.flips()
    player.stop()
    assert got.tolist() == [(0, 1000, 0, 5, po.SEGMENT), (1, 1001, 0, 9, 0)]
    assert po.flip_summary(got, 2) == {"flips": 2, "segments": 1, "missed": 0, "out_of_cadence": 0}
    assert po.flip_summary(got, 1)["out_of_cadence"] == 1


def test_restarting_leaves_one_player(pi):
    player = on(pi)
    first = player.show(frame((0, 0, 0), 2, 4))
    second = player.start(fps=10)
    assert second != first and not alive(first) and alive(second)
    player.stop()


def test_a_player_that_faults_on_start_is_raised_with_its_reason(pi):
    pi.mode("fail")
    with pytest.raises(PlayoutError, match="faulted: no 1920x1080 progressive mode"):
        on(pi).show(frame((0, 0, 0), 2, 4))
    assert not (pi.directory / "player.pid").exists()


def test_a_player_that_dies_while_playing_fails_the_check(pi):
    pi.mode("die")
    player = on(pi)
    pid = player.show(frame((0, 0, 0), 2, 4))
    deadline = time.monotonic() + 10.0
    while alive(pid) and time.monotonic() < deadline:
        time.sleep(0.01)
    with pytest.raises(PlayoutError, match="died while playing"):
        player.check()


@pytest.mark.parametrize(("mode", "warned"), [("rt", False), ("", True)])
def test_running_without_realtime_is_warned(pi, caplog, mode, warned):
    pi.mode(mode)
    player = on(pi)
    with caplog.at_level(logging.WARNING, logger="syncsummoner"):
        player.show(frame((0, 0, 0), 2, 4))
    player.stop()
    assert ("without SCHED_FIFO" in caplog.text) is warned


@pytest.mark.parametrize("loop", [False, True])
def test_a_clip_goes_up_once_and_plays_from_tmpfs(pi, tmp_path, loop):
    clip = tmp_path / "clip.mkv"
    clip.write_bytes(b"x" * 2048)
    player = on(pi, ClipPlayer, loop=loop)
    assert player.upload(str(clip)) == 2048
    with player.playing(fps="25"):
        argv = pi.argv()
    assert (pi.directory / "clip.mkv").read_bytes() == clip.read_bytes()
    assert argv[argv.index("--clip") + 1] == player.clip_path and ("--loop" in argv) is loop
    assert (
        argv[argv.index("--fps") + 1] == "25" and "--width" in argv and argv[argv.index("--width") + 1] == "4"
    )


class FakePi:
    """Pidfile, status and liveness modelled without a shell, for the timing paths."""

    def __init__(self, *, settles="playing", stuck=False, status=None, alive=False, flips=b""):
        self.commands = []
        self.settles, self.stuck, self.flips = settles, stuck, flips
        self.state = {} if status is None else dict(status)
        self.pid = 4242 if alive or status else None
        self.alive = alive
        self.frames = []

    def __call__(self, command, data=None):
        self.commands.append(command)
        if "split -b" in command:
            size = int(re.search(r"split -b (\d+)", command)[1])
            blob = gzip.decompress(data)
            self.frames = [blob[i : i + size] for i in range(0, len(blob), size)]
        elif "setsid" in command:
            self.pid, self.alive = 4243, True
            self.state = {"state": self.settles, "realtime": True} if self.settles else {}
        elif command.startswith("kill -0"):
            return po.RUNNING if self.alive else po.STOPPED
        elif command.startswith("kill "):
            self.alive = self.stuck
        elif command.startswith("rm -f") and "player.pid" in command:
            self.pid = None
        elif command.startswith("cat ") and "player.pid" in command:
            return "" if self.pid is None else str(self.pid)
        elif command.startswith("cat ") and "status.json" in command:
            return json.dumps(self.state) if self.state else ""
        elif command.startswith("base64"):
            return base64.b64encode(self.flips).decode()
        elif "wc -l" in command:
            return str(len(self.frames))
        return ""


def make_player(pi=None, cls=LoopPlayer, **kwargs):
    pi = FakePi() if pi is None else pi
    clock = FakeClock()
    kwargs.setdefault("width", 4)
    kwargs.setdefault("height", 2)
    return cls(runner=pi, sleep=clock.sleep, clock=clock, **kwargs), pi, clock


def test_upload_sends_one_compressed_blob_split_into_frames():
    player, pi, _ = make_player()
    frames = [frame((k / 24, 0.25, 0.5), 2, 4) for k in range(24)]
    sent = player.upload(frames)
    assert pi.frames == [player.encode(f) for f in frames]
    assert sent < 24 * player.frame_bytes and player.frame_count() == 24


def test_upload_clears_any_previous_loop_and_orders_frames_lexically():
    player, pi, _ = make_player(directory="/dev/shm/loop/")
    player.upload([frame((0, 0, 0), 2, 4)] * 1001)
    assert player.directory == "/dev/shm/loop"
    assert "rm -f /dev/shm/loop/frames/frame.*" in pi.commands[0] and "-d -a 4" in pi.commands[0]


@pytest.mark.parametrize(
    ("frames", "match"), [([], "at least one frame"), ([frame((0, 0, 0), 8, 8)], r"\(2, 4\)")]
)
def test_upload_rejects_an_empty_or_misshapen_loop(frames, match):
    with pytest.raises(ValueError, match=match):
        make_player()[0].upload(frames)


def test_a_player_that_never_settles_times_out_with_its_log():
    player, _, clock = make_player(FakePi(settles=None), timeout_s=1.0, poll_s=0.5)
    with pytest.raises(PlayoutError, match="did not start at 30000/1001 fps after 1.0s"):
        player.start(fps=29.97)
    assert clock.slept == [0.5, 0.5]


def test_stop_waits_for_the_process_to_go_and_gives_up_at_the_timeout():
    player, _, clock = make_player(FakePi(stuck=True), timeout_s=1.0, poll_s=0.25)
    player.start()
    with pytest.raises(PlayoutError, match="pid 4243 did not exit after 1.0s"):
        player.stop()
    assert clock.now == pytest.approx(1.0)


def test_stop_without_a_player_is_a_no_op():
    player, pi, _ = make_player()
    assert player.stop() == {} and not any(c.startswith("kill") for c in pi.commands)


def test_stop_raises_the_fault_a_dead_player_left_behind():
    player, pi, _ = make_player(FakePi(status={"state": "failed", "error": "vblank 7 missed"}))
    with pytest.raises(PlayoutError, match="vblank 7 missed"):
        player.stop()
    assert pi.pid is None and not any(c.startswith("kill 4242") for c in pi.commands)


def test_playing_raises_a_fault_on_the_way_out_but_never_masks_the_blocks_own_error():
    player, pi, _ = make_player()
    with pytest.raises(PlayoutError, match="underrun"):
        with player.playing():
            pi.alive, pi.state = False, {"state": "failed", "error": "underrun"}
    with pytest.raises(ZeroDivisionError):
        with player.playing():
            pi.alive, pi.state = False, {"state": "failed", "error": "underrun"}
            raise ZeroDivisionError


def test_check_raises_a_reported_fault():
    with pytest.raises(PlayoutError, match="faulted: frame 3 was not decoded"):
        make_player(FakePi(status={"state": "failed", "error": "frame 3 was not decoded"}))[0].check()


def test_check_passes_a_stopped_player_and_a_live_one():
    assert make_player(FakePi(status={"state": "stopped"}))[0].check() == {"state": "stopped"}
    assert make_player(FakePi(status={"state": "playing"}, alive=True))[0].check()["state"] == "playing"


def test_flips_drop_a_partial_trailing_record():
    record = po.RECORD.pack(3, 1003, 1, 77, 0)
    player = make_player(FakePi(flips=record + record[:7]))[0]
    assert player.flips().tolist() == [(3, 1003, 1, 77, 0)]
    assert make_player()[0].flips().size == 0


@pytest.mark.parametrize(("fps", "poll"), [(0, 0.25), (Fraction(-1), 0.25), (10, 0)])
def test_rates_and_polls_must_be_positive(fps, poll):
    with pytest.raises(ValueError, match="positive"):
        make_player(poll_s=poll)[0].start(fps=fps)


def test_a_bare_player_names_no_source():
    with pytest.raises(NotImplementedError):
        po.KmsPlayer(runner=FakePi()).source_args()


def test_default_runner_is_ssh_with_the_upload_timeout(monkeypatch):
    seen = {}

    def fake_run(argv, **kwargs):
        seen.update(argv=argv, kwargs=kwargs)
        return subprocess.CompletedProcess(argv, 0, stdout=b"", stderr=b"")

    monkeypatch.setattr(host.subprocess, "run", fake_run)
    assert LoopPlayer(width=2, height=2).pid() is None
    assert seen["argv"][:4] == ["ssh", "-o", "BatchMode=yes", po.DEFAULT_HOST]
    assert seen["kwargs"]["timeout"] == po.UPLOAD_TIMEOUT

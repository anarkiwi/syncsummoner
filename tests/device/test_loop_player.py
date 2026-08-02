"""Pi-side looping player: upload once, cycle locally, with no Pi and no network."""

# pylint: disable=missing-function-docstring,too-many-return-statements  ; the fake is a command dispatch

import gzip
import re
import subprocess
import sys
import time
from contextlib import contextmanager

import numpy as np
import pytest

from syncsummoner.device import host, playout as po
from syncsummoner.device.playout import LoopPlayer, PlayoutError

from .conftest import FakeClock
from .test_playout import frame

KILL_RE = re.compile(r"^kill (?:-0 )?(\d+)")


class FakePi:
    """Models the Pi's tmpfs, pidfile and process liveness, without any shell."""

    def __init__(self, *, spawns=True, stuck=False):
        self.commands = []
        self.frames = []
        self.script = None
        self.pidfile = None
        self.alive = set()
        self.next_pid = 4242
        self.launched = None
        self.spawns = spawns
        self.stuck = stuck
        self.log = ""

    def __call__(self, command, data=None):
        self.commands.append((command, data))
        if "split -b" in command:
            size = int(re.search(r"split -b (\d+)", command).group(1))
            blob = gzip.decompress(data)
            self.frames = [blob[i : i + size] for i in range(0, len(blob), size)]
            return ""
        if command.endswith("player.py"):
            self.script = data.decode()
            return ""
        if command.startswith("setsid "):
            self.launched = command.split("player.py", 1)[1].split("'", 1)[0].split()
            if not self.spawns:
                self.log = "python3: command not found"
                return ""
            self.pidfile = self.next_pid
            self.alive.add(self.pidfile)
            self.next_pid += 1
            return ""
        if command.startswith("kill -0 "):
            return po.RUNNING if int(KILL_RE.match(command).group(1)) in self.alive else po.STOPPED
        if command.startswith("kill "):
            if not self.stuck:
                self.alive.discard(int(KILL_RE.match(command).group(1)))
            return ""
        if command.startswith("cat ") and "player.pid" in command:
            return "" if self.pidfile is None else f"{self.pidfile}\n"
        if command.startswith("rm -f ") and "player.pid" in command:
            self.pidfile = None
            return ""
        if "wc -l" in command:
            return str(len(self.frames))
        if command.startswith("tail "):
            return self.log
        raise AssertionError(f"unmodelled command: {command}")

    @property
    def texts(self):
        return [c for c, _ in self.commands]


def make_player(pi=None, **kwargs):
    """Player wired to a fake Pi and a virtual clock, at a small geometry."""
    pi = FakePi() if pi is None else pi
    clock = FakeClock()
    kwargs.setdefault("width", 4)
    kwargs.setdefault("height", 2)
    return LoopPlayer(runner=pi, sleep=clock.sleep, clock=clock, **kwargs), pi, clock


def loop_frames(count, width=4, height=2):
    return [frame((k / count, 0.25, 0.5), height, width) for k in range(count)]


def test_upload_sends_one_compressed_blob_split_into_frames():
    player, pi, _ = make_player()
    frames = loop_frames(24)
    sent = player.upload(frames)
    assert len(pi.frames) == 24
    assert pi.frames == [player.encode(f) for f in frames]
    assert sent == len(pi.commands[0][1]) < 24 * player.frame_bytes


def test_upload_lands_in_tmpfs_and_clears_any_previous_loop():
    player, pi, _ = make_player(directory="/dev/shm/loop/")
    player.upload(loop_frames(2))
    command = pi.texts[0]
    assert player.directory == "/dev/shm/loop"
    assert "mkdir -p /dev/shm/loop" in command
    assert "rm -f /dev/shm/loop/frame.*" in command
    assert command.endswith("- /dev/shm/loop/frame.")


def test_upload_names_frames_so_lexical_order_is_loop_order():
    player, pi, _ = make_player()
    player.upload(loop_frames(1001))
    assert "-d -a 4" in pi.texts[0]
    assert len(pi.frames) == 1001


def test_upload_rejects_an_empty_loop():
    player, _, _ = make_player()
    with pytest.raises(ValueError, match="at least one frame"):
        player.upload([])


def test_upload_rejects_wrong_geometry():
    player, _, _ = make_player()
    with pytest.raises(ValueError, match=r"\(2, 4\)"):
        player.upload(loop_frames(2, width=8, height=8))


def test_frame_count_reports_what_is_uploaded():
    player, _, _ = make_player()
    assert player.frame_count() == 0
    player.upload(loop_frames(3))
    assert player.frame_count() == 3


def test_start_detaches_the_player_and_records_its_own_pid():
    player, pi, _ = make_player()
    pid = player.start(fps=15.0)
    assert pid == 4242 and player.is_running()
    launch = next(c for c in pi.texts if c.startswith("setsid "))
    assert launch.startswith("setsid sh -c 'echo $$ > /dev/shm/syncsummoner/player.pid;")
    assert "exec python3" in launch and launch.endswith("&")
    assert "< /dev/null" in launch
    assert pi.launched == ["/dev/shm/syncsummoner", "/dev/fb0", "15.0"]


def test_start_uploads_the_player_program():
    player, pi, _ = make_player()
    player.start()
    assert pi.script == po.PLAYER_SOURCE
    assert any(c.endswith(f"cat > {player.script_path}") for c in pi.texts)


def test_rate_is_a_parameter_with_a_capture_compatible_default():
    player, pi, _ = make_player()
    player.start()
    assert pi.launched[-1] == str(po.DEFAULT_LOOP_FPS)
    assert 10.0 <= po.DEFAULT_LOOP_FPS <= 15.0


def test_start_is_idempotent_and_leaves_one_player_running():
    player, pi, _ = make_player()
    first = player.start()
    second = player.start()
    assert second != first
    assert pi.alive == {second}
    assert f"kill {first} 2>/dev/null || true" in pi.texts


def test_liveness_is_never_decided_by_matching_process_text():
    player, pi, _ = make_player()
    player.upload(loop_frames(2))
    player.start()
    player.is_running()
    player.stop()
    assert not any(re.search(r"\bp(kill|s)\b|pgrep|killall", c) for c in pi.texts)
    assert any(c.startswith("kill -0 ") for c in pi.texts)


def test_stop_kills_by_pid_clears_the_pidfile_and_reports_it_was_running():
    player, pi, _ = make_player()
    pid = player.start()
    assert player.stop() is True
    assert not pi.alive and pi.pidfile is None
    assert f"kill {pid} 2>/dev/null || true" in pi.texts
    assert pi.texts[-1] == f"rm -f {player.pid_path}"
    assert not player.is_running()


def test_stop_without_a_player_is_a_no_op():
    player, pi, _ = make_player()
    assert player.stop() is False
    assert player.pid() is None
    assert not any(c.startswith("kill ") for c in pi.texts)


def test_stop_never_writes_to_the_framebuffer():
    player, pi, _ = make_player()
    player.start()
    mark = len(pi.commands)
    player.stop()
    assert not any(player.framebuffer in c for c in pi.texts[mark:])


def test_stop_waits_for_the_process_to_actually_go_away():
    pi = FakePi(stuck=True)
    player, _, clock = make_player(pi, timeout_s=1.0, poll_s=0.25)
    player.start()
    with pytest.raises(PlayoutError, match="did not exit"):
        player.stop()
    assert clock.now == pytest.approx(1.0)


def test_failed_start_raises_with_the_players_own_output():
    pi = FakePi(spawns=False)
    player, _, clock = make_player(pi, timeout_s=1.0, poll_s=0.5)
    with pytest.raises(PlayoutError, match="did not start.*command not found"):
        player.start()
    assert clock.slept == [0.5, 0.5]


def test_start_rejects_a_non_positive_rate():
    player, _, _ = make_player()
    with pytest.raises(ValueError, match="positive"):
        player.start(fps=0)


def test_rejects_non_positive_poll():
    with pytest.raises(ValueError, match="positive"):
        LoopPlayer(runner=FakePi(), poll_s=0)


def test_playing_stops_the_loop_on_the_way_out():
    player, pi, _ = make_player()
    with player.playing(fps=10.0) as held:
        assert held.is_running()
        assert pi.launched[-1] == "10.0"
    assert not player.is_running()


def test_playing_stops_the_loop_on_error():
    player, _, _ = make_player()
    with pytest.raises(ZeroDivisionError):
        with player.playing():
            raise ZeroDivisionError
    assert not player.is_running()


def test_default_runner_is_ssh_with_the_upload_timeout(monkeypatch):
    seen = {}

    def fake_run(argv, **kwargs):
        seen.update(argv=argv, kwargs=kwargs)
        return subprocess.CompletedProcess(argv, 0, stdout=b"", stderr=b"")

    monkeypatch.setattr(host.subprocess, "run", fake_run)
    assert LoopPlayer(width=2, height=2).pid() is None
    assert seen["argv"][:4] == ["ssh", "-o", "BatchMode=yes", po.DEFAULT_HOST]
    assert seen["kwargs"]["timeout"] == po.UPLOAD_TIMEOUT


@contextmanager
def running_player(tmp_path, frames, fps):
    """Run the uploaded program itself against a file standing in for fb0."""
    for index, data in enumerate(frames):
        (tmp_path / f"frame.{index:03d}").write_bytes(data)
    (tmp_path / "player.py").write_text(po.PLAYER_SOURCE)
    fb = tmp_path / "fb0"
    fb.write_bytes(b"")
    argv = [sys.executable, str(tmp_path / "player.py"), str(tmp_path), str(fb), str(fps)]
    proc = subprocess.Popen(argv)
    try:
        yield proc, fb
    finally:
        proc.terminate()
        proc.wait(timeout=5.0)


def test_player_program_cycles_frames_and_exits_cleanly_on_sigterm(tmp_path):
    frames = [bytes([k]) * 64 for k in range(3)]
    seen = set()
    with running_player(tmp_path, frames, 200) as (proc, fb):
        deadline = time.monotonic() + 5.0
        while len(seen) < len(frames) and time.monotonic() < deadline:
            data = fb.read_bytes()
            if len(data) == 64:
                seen.add(data)
    assert proc.returncode == 0
    assert seen == set(frames)
    assert fb.read_bytes() in frames


def test_player_program_leaves_a_whole_frame_when_stopped_mid_stride(tmp_path):
    """A signal landing inside a multi-megabyte write must not tear the framebuffer."""
    frames = [np.full(1 << 22, k, dtype=np.uint8).tobytes() for k in (1, 2)]
    with running_player(tmp_path, frames, 200) as (proc, fb):
        deadline = time.monotonic() + 5.0
        while len(fb.read_bytes()) < len(frames[0]) and time.monotonic() < deadline:
            pass
    assert proc.returncode == 0
    assert fb.read_bytes() in frames

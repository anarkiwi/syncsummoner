"""The bitstream path: IceStorm decompilation and SPI-driven identification of ``registers_in``."""

# pylint: disable=missing-function-docstring

import os
import subprocess
import threading
import time
from concurrent.futures import ThreadPoolExecutor

from syncsummoner.program import hdl
from syncsummoner.program.library import Program
from syncsummoner.program.model import ProgramModel, model
from syncsummoner.program.vmprog import read_package
from tests.program.conftest import FIXTURES, needs_hdl, needs_sdk


@needs_hdl
@needs_sdk
def test_quilt_bitstream_and_source_agree_on_every_bit_read(quilt_bytes, library, tmp_path):
    package = read_package(quilt_bytes)
    program = Program(package, "quilt", "czartroniks", (FIXTURES / "quilt" / "quilt.vhd",))
    rtl = ProgramModel(program, library, cache=tmp_path, source="rtl")
    chip = ProgramModel(program, library, cache=tmp_path, source="bitstream")
    want = {1: (8, 9), 2: tuple(range(10)), 3: tuple(range(10)), 4: tuple(range(2, 10)), 5: (6, 7, 8, 9)}
    want |= {6: (8, 9), 12: tuple(range(2, 10))} | {p: (p - 7,) for p in range(7, 12)}
    assert rtl.resolution() == want
    assert chip.resolution() == want
    assert chip.latched() == want
    picked = model(program, library, cache=tmp_path)
    assert picked.source == "rtl" and not picked.note
    assert chip.live(chip.initial(), gates=False)[10].live


@needs_hdl
def test_identical_designs_are_proved_equivalent(tiny):
    folded = tiny._folded(tiny.initial(), None)  # pylint: disable=protected-access
    assert hdl.compare(tiny.tools, folded, folded, "clk").differs is False


def test_missing_tools_are_reported(tmp_path):
    tools = hdl.Tools(bin_dir=tmp_path)
    assert not tools.available()
    try:
        tools.path("yosys")
    except hdl.ToolError as err:
        assert "OSS CAD Suite" in str(err)


@needs_hdl
@needs_sdk
def test_a_register_derived_from_a_control_is_not_taken_for_the_control(library, tmp_path):
    package = read_package((FIXTURES / "fray" / "fray.vmprog").read_bytes())
    chip = ProgramModel(Program(package, "fray", "czartroniks"), library, cache=tmp_path, source="bitstream")
    assert chip.latched()[1] == tuple(range(10))
    assert chip.resolution()[1] == tuple(range(10)) and chip.resolution()[5] == (7, 8, 9)


def test_tool_processes_never_exceed_the_cpus_the_process_may_use(monkeypatch, tmp_path):
    allowed = len(os.sched_getaffinity(0))
    quota = tmp_path / "cpu.max"
    quota.write_text("150000 100000\n")
    assert hdl.cpus(quota) == min(2, allowed)
    quota.write_text("max 100000\n")
    assert hdl.cpus(quota) == allowed == hdl.cpus(tmp_path / "absent")
    monkeypatch.setattr(hdl, "_SLOTS", [threading.BoundedSemaphore(2)])
    running, peak, lock = [0], [0], threading.Lock()

    def fake_run(*_args, **_kwargs):
        with lock:
            running[0] += 1
            peak[0] = max(peak[0], running[0])
        time.sleep(0.01)
        with lock:
            running[0] -= 1
        return subprocess.CompletedProcess([], 0, "ok", "")

    monkeypatch.setattr(hdl.subprocess, "run", fake_run)
    tools = hdl.Tools()
    monkeypatch.setattr(tools, "path", lambda name: name)
    with ThreadPoolExecutor(max_workers=16) as pool:
        assert set(pool.map(lambda _: tools.run(["yosys"], tmp_path), range(64))) == {"ok"}
    assert peak[0] == 2

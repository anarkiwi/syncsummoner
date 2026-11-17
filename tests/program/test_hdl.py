"""The bitstream path: IceStorm decompilation and SPI-driven identification of ``registers_in``."""

# pylint: disable=missing-function-docstring

from syncsummoner.program import hdl
from syncsummoner.program.library import Program
from syncsummoner.program.model import ProgramModel
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
    assert (chip.cache / "register-map.json").exists()
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

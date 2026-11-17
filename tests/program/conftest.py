"""Fixtures: the SDK's own tools, the GPL Quilt package and a tiny VHDL design with known gating."""

import os
import subprocess
import sys
from pathlib import Path

import pytest

from syncsummoner.program import hdl
from syncsummoner.program.library import Library, Program
from syncsummoner.program.model import ProgramModel
from syncsummoner.program.vmprog import read_package

FIXTURES = Path(__file__).resolve().parent.parent / "fixtures" / "program"
SDK = Path(os.environ.get("SYNCSUMMONER_SDK", "captures/upstream/videomancer-sdk")).resolve()

needs_sdk = pytest.mark.skipif(not (SDK / "tools").is_dir(), reason="Videomancer SDK not installed")
needs_hdl = pytest.mark.skipif(not hdl.Tools().available(), reason="HDL tools not installed")


def sdk_tool(*args, cwd):
    """Run one of the SDK's own Python tools."""
    return subprocess.run(
        [sys.executable, *map(str, args)], cwd=cwd, check=True, capture_output=True, text=True
    )


def pack(toml: Path, work: Path, bitstreams: dict[str, bytes]) -> bytes:
    """A package built by the SDK's converter and packer from ``toml`` and raw bitstreams."""
    (work / "bitstreams").mkdir(parents=True, exist_ok=True)
    sdk_tool(
        SDK / "tools/toml-converter/toml_to_config_binary.py",
        toml,
        work / "program_config.bin",
        "-q",
        cwd=work,
    )
    for name, data in bitstreams.items():
        (work / "bitstreams" / name).write_bytes(data)
    sdk_tool(SDK / "tools/vmprog-packer/vmprog_pack.py", work, work / "out.vmprog", "--no-sign", cwd=work)
    return (work / "out.vmprog").read_bytes()


@pytest.fixture(scope="session", name="quilt_bytes")
def quilt_bytes_fixture():
    return (FIXTURES / "quilt" / "quilt.vmprog").read_bytes()


@pytest.fixture(scope="session", name="library")
def library_fixture():
    return Library(sdk=SDK)


@pytest.fixture(scope="session", name="tiny")
def tiny_fixture(tmp_path_factory, library):
    if not (SDK / "tools").is_dir() or not hdl.Tools().available():
        pytest.skip("SDK and HDL tools needed")
    work = tmp_path_factory.mktemp("tiny")
    package = read_package(pack(FIXTURES / "tiny" / "tiny.toml", work, {"hd_hdmi.bin": b"\0" * 64}))
    program = Program(package, "tiny", "test", (FIXTURES / "tiny" / "tiny.vhd",))
    return ProgramModel(program, library, cache=tmp_path_factory.mktemp("cache"))

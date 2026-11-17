"""Finding programs by device name across a card copy and firmware images, with their sources."""

# pylint: disable=missing-function-docstring

import struct

import pytest

from syncsummoner.program.library import ENV, Library
from tests.program.conftest import FIXTURES


def uf2(image: bytes) -> bytes:
    blocks = b""
    for i in range(0, len(image), 256):
        chunk = image[i : i + 256]
        head = struct.pack("<8I", 0x0A324655, 0x9E5D5157, 0, 0x10000000 + i, len(chunk), i // 256, 0, 0)
        blocks += (head + chunk).ljust(512, b"\0")
    return blocks


@pytest.fixture(name="lib")
def lib_fixture(tmp_path, quilt_bytes):
    card = tmp_path / "card"
    (card / "czartroniks").mkdir(parents=True)
    (card / "czartroniks" / "quilt.vmprog").write_bytes(quilt_bytes)
    (card / "czartroniks" / "broken.vmprog").write_bytes(b"VMPG")
    fw = tmp_path / "fw"
    fw.mkdir()
    (fw / "image.uf2").write_bytes(uf2(b"\xff" * 300 + quilt_bytes))
    community = tmp_path / "community" / "programs" / "czartroniks" / "quilt"
    community.mkdir(parents=True)
    for name in ("quilt.toml", "quilt.vhd"):
        (community / name).write_bytes((FIXTURES / "quilt" / name).read_bytes())
    (community.parent / "bad.toml").write_text("not = [toml")
    return Library(card=card, firmware=fw, community=tmp_path / "community")


def test_card_packages_are_found_by_device_name_or_id(lib):
    assert lib.find("czartroniks/Quilt").origin.endswith("quilt.vmprog")
    assert lib.find("Quilt").folder == "czartroniks"
    assert lib.find("com.czartroniks.quilt").sources[0].name == "quilt.vhd"
    assert len(lib.programs) == 1
    with pytest.raises(KeyError, match="0 programs"):
        lib.find("someone/Quilt")


def test_firmware_packages_carry_their_image_and_offset(lib, tmp_path):
    lib = Library(firmware=tmp_path / "fw" / "image.uf2")
    (prog,) = lib.programs
    assert prog.origin.endswith("image.uf2@0x12c") and prog.name == "czartroniks/Quilt" and not prog.sources


def test_defaults_follow_the_environment(monkeypatch, tmp_path):
    for var in ENV.values():
        monkeypatch.delenv(var, raising=False)
    monkeypatch.setenv("SYNCSUMMONER_CARD", str(tmp_path))
    lib = Library.default(tmp_path / "nowhere")
    assert lib.card == tmp_path and lib.sdk is None and lib.programs == []

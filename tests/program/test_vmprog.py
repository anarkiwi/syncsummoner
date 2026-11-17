"""The parser against packages the SDK's own converter and packer build, and the real Quilt package."""

# pylint: disable=missing-function-docstring

import struct
import tomllib

import pytest

from syncsummoner.program import vmprog as V
from tests.program.conftest import FIXTURES, needs_sdk, pack


@needs_sdk
def test_a_package_the_sdk_packs_reads_back_exactly(tmp_path):
    toml = FIXTURES / "quilt" / "quilt.toml"
    bitstream = bytes(range(256)) * 300
    package = V.read_package(pack(toml, tmp_path, {"hd_hdmi.bin": bitstream, "sd_hdmi.bin": b"\1" * 999}))
    declared = tomllib.loads(toml.read_text())
    cfg = package.config
    assert (cfg.program_id, cfg.version_text) == ("com.czartroniks.quilt", "1.0.3")
    assert [p.name for p in cfg.params] == [p["name_label"] for p in declared["parameter"]]
    assert [p.value_labels for p in cfg.params][:2] == [("Large", "Big", "Medium", "Small"), ()]
    assert [p.name for p in cfg.presets] == [p["name"] for p in declared["preset"]]
    assert cfg.processing_delay_clks == declared["program"]["processing_delay_clks"]
    streams = package.bitstreams()
    assert streams[V.SectionType.HD_HDMI] == bitstream and streams[V.SectionType.SD_HDMI] == b"\1" * 999
    assert any(s.flags & V.TOC_COMPRESSED for s in package.sections)


def test_the_quilt_package_declares_its_own_parameters(quilt_bytes):
    cfg = V.read_package(quilt_bytes).config
    assert cfg.param(12).name == "Density" and cfg.param(12).control_mode == V.ControlMode.LINEAR
    assert cfg.param(7).control_mode == V.ControlMode.BOOLEAN and cfg.param(7).value_labels == ("Off", "On")
    assert cfg.preset("luma mod").values[3] == 512
    with pytest.raises(KeyError):
        cfg.preset("nope")
    assert cfg.param(13) is None and cfg.param(1).id_name == "rotary_potentiometer_1"


@pytest.mark.parametrize(
    "damage,why",
    [
        (lambda d: d[:-1], "header says"),
        (lambda d: b"XXXX" + d[4:], "not a vmprog"),
        (lambda d: d[:-1] + bytes([d[-1] ^ 1]), "fails its hash"),
        (lambda d: d[:24] + struct.pack("<I", 7) + d[28:], "table of contents"),
    ],
)
def test_damage_is_refused_with_the_reason(quilt_bytes, damage, why):
    with pytest.raises(V.PackageError, match=why):
        V.read_package(damage(quilt_bytes))


def test_a_corrupt_compressed_size_is_refused(quilt_bytes):
    head, secs = V.sections(quilt_bytes)
    bad = next(s for s in secs if s.flags & V.TOC_COMPRESSED)
    with pytest.raises(V.PackageError, match="inflates"):
        _ = V.Section(bad.kind, bad.flags, bad.payload, bad.uncompressed_size + 1).data
    assert head["toc_count"] == len(secs)


def test_config_fields_are_bounds_checked(quilt_bytes):
    raw = bytearray(V.read_config(quilt_bytes))
    with pytest.raises(V.PackageError, match="bytes"):
        V.parse_config(bytes(raw[:-1]))
    raw[7916] = 7
    with pytest.raises(V.PackageError, match="program type"):
        V.parse_config(bytes(raw))
    raw[7916], raw[728] = 0, 9
    with pytest.raises(V.PackageError, match="count"):
        V.parse_config(bytes(raw))


def test_packages_inside_a_firmware_image_are_found(quilt_bytes):
    image = b"\xff" * 100 + b"VMPG" + b"\0" * 40 + quilt_bytes + b"\0" * 7 + quilt_bytes[:500]
    found = list(V.embedded_packages(image))
    assert [o for o, _ in found] == [144] and found[0][1].config.name == "Quilt"


def test_a_uf2_flattens_to_the_flash_it_programs():
    payload = bytes(range(256)) * 3
    blocks = b""
    for i in range(3):
        head = struct.pack("<8I", 0x0A324655, 0x9E5D5157, 0, 0x10000000 + 256 * i, 256, i, 3, 0)
        blocks += (head + payload[256 * i : 256 * (i + 1)]).ljust(512, b"\0")
    assert V.uf2_image(blocks) == payload

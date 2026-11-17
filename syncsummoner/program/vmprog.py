"""Read a ``.vmprog`` package: header, table of contents, config, presets and bitstreams.

Layout: the Videomancer SDK's ``src/lzx/videomancer/vmprog_format.hpp``. Section
hashes are BLAKE2b-256 over the stored bytes; bitstreams flagged
``compressed_deflate`` are raw DEFLATE with a 1 KiB window.
"""

from __future__ import annotations

import enum
import hashlib
import struct
import zlib
from dataclasses import dataclass
from typing import Iterator

import numpy as np

MAGIC = b"VMPG"
HEADER_BYTES = 64
TOC_BYTES = 64
CONFIG_BYTES = 7936
MAX_FILE_BYTES = 1 << 20
NUM_PARAMETERS = 12
MAX_PRESETS = 8
DEFLATE_WBITS = -10
TOC_COMPRESSED = 0x1
PROGRAM_TYPES = ("processing", "synthesis")
CORES = {1: "yuv444_30b", 2: "yuv422_20b", 3: "gbr444_30b", 4: "gbr422_20b"}


class PackageError(ValueError):
    """A ``.vmprog`` that the device would not be able to load."""


class SectionType(enum.IntEnum):
    """TOC entry types (``vmprog_toc_entry_type_v1_0``)."""

    CONFIG = 1
    SIGNED_DESCRIPTOR = 2
    SIGNATURE = 3
    FPGA_BITSTREAM = 4
    SD_ANALOG = 5
    SD_HDMI = 6
    SD_DUAL = 7
    HD_ANALOG = 8
    HD_HDMI = 9
    HD_DUAL = 10
    SD_STANDALONE = 11
    HD_STANDALONE = 12


class ControlMode(enum.IntEnum):
    """``vmprog_parameter_control_mode_v1_0``."""

    LINEAR = 0
    LINEAR_HALF = 1
    LINEAR_QUARTER = 2
    LINEAR_DOUBLE = 3
    BOOLEAN = 4
    STEPS_4 = 5
    STEPS_8 = 6
    STEPS_16 = 7
    STEPS_32 = 8
    STEPS_64 = 9
    STEPS_128 = 10
    STEPS_256 = 11
    POLAR_DEGS_90 = 12
    POLAR_DEGS_180 = 13
    POLAR_DEGS_360 = 14
    POLAR_DEGS_720 = 15
    POLAR_DEGS_1440 = 16
    POLAR_DEGS_2880 = 17
    QUAD_IN = 18
    QUAD_OUT = 19
    QUAD_IN_OUT = 20
    SINE_IN = 21
    SINE_OUT = 22
    SINE_IN_OUT = 23
    CIRC_IN = 24
    CIRC_OUT = 25
    CIRC_IN_OUT = 26
    QUINT_IN = 27
    QUINT_OUT = 28
    QUINT_IN_OUT = 29
    QUART_IN = 30
    QUART_OUT = 31
    QUART_IN_OUT = 32
    EXPO_IN = 33
    EXPO_OUT = 34
    EXPO_IN_OUT = 35


PARAMETER_IDS = (
    "none",
    *(f"rotary_potentiometer_{i}" for i in range(1, 7)),
    *(f"toggle_switch_{i}" for i in range(7, 12)),
    "linear_potentiometer_12",
)

_HEADER = np.dtype(
    {
        "names": ["magic", "major", "minor", "header_size", "file_size", "flags", "toc_offset", "toc_bytes"]
        + ["toc_count"],
        "formats": ["S4", "<u2", "<u2", "<u2", "<u4", "<u4", "<u4", "<u4", "<u4"],
        "offsets": [0, 4, 6, 8, 12, 16, 20, 24, 28],
        "itemsize": HEADER_BYTES,
    }
)
_TOC = np.dtype(
    {
        "names": ["kind", "flags", "offset", "size", "digest", "uncompressed"],
        "formats": ["<u4", "<u4", "<u4", "<u4", ("u1", 32), "<u4"],
        "offsets": [0, 4, 8, 12, 16, 48],
        "itemsize": TOC_BYTES,
    }
)
_PARAM = np.dtype(
    {
        "names": [
            "pid",
            "mode",
            "min",
            "max",
            "initial",
            "dmin",
            "dmax",
            "digits",
            "nlabels",
            "name",
            "labels",
        ]
        + ["suffix"],
        "formats": ["<u4", "<u4", "<u2", "<u2", "<u2", "<i2", "<i2", "u1", "u1", "S32", ("S32", 16), "S4"],
        "offsets": [0, 4, 8, 10, 12, 14, 16, 18, 19, 22, 54, 566],
        "itemsize": 572,
    }
)
_PRESET = np.dtype({"names": ["name", "values"], "formats": ["S16", ("<u2", NUM_PARAMETERS)], "itemsize": 40})
_CONFIG = np.dtype(
    {
        "names": ["program_id", "version", "abi", "hw_mask", "core_id", "name", "author", "license"]
        + ["categories", "description", "url", "nparams", "npresets", "ncategories", "params", "presets"]
        + ["timings", "program_type", "delay"],
        "formats": ["S64", ("<u2", 3), ("<u2", 4), "<u4", "<u4", "S32", "S64", "S32", ("S32", 8), "S128"]
        + ["S128", "<u2", "u1", "u1", (_PARAM, NUM_PARAMETERS), (_PRESET, MAX_PRESETS), "<u2", "u1", "<u2"],
        "offsets": [0, 64, 70, 78, 82, 86, 118, 182, 214, 470, 598, 726, 728, 729, 730, 7594, 7914, 7916]
        + [7917],
        "itemsize": CONFIG_BYTES,
    }
)


def _text(raw: bytes) -> str:
    return bytes(raw).split(b"\0", 1)[0].decode("utf-8")


@dataclass(frozen=True)
class ParamDecl:
    """One parameter exactly as its program declares it (``vmprog_parameter_config_v1_0``)."""

    parameter_id: int
    control_mode: int
    min_value: int
    max_value: int
    initial_value: int
    display_min: int
    display_max: int
    float_digits: int
    name: str
    value_labels: tuple[str, ...]
    suffix: str

    @property
    def id_name(self) -> str:
        """The SDK's name for the control: ``rotary_potentiometer_1`` .. ``linear_potentiometer_12``."""
        return PARAMETER_IDS[self.parameter_id]


@dataclass(frozen=True)
class Preset:
    """A factory preset: raw manual values, indexed by ``parameter_id - 1``."""

    name: str
    values: tuple[int, ...]


@dataclass(frozen=True)
class ProgramConfig:
    """The program's own metadata and parameter declarations (``vmprog_program_config_v1_0``)."""

    program_id: str
    version: tuple[int, int, int]
    core: str
    name: str
    author: str
    license: str
    categories: tuple[str, ...]
    description: str
    params: tuple[ParamDecl, ...]
    presets: tuple[Preset, ...]
    program_type: str
    hw_mask: int
    supported_timings: int
    processing_delay_clks: int

    @property
    def version_text(self) -> str:
        """``major.minor.patch``."""
        return ".".join(map(str, self.version))

    def param(self, parameter_id: int) -> ParamDecl | None:
        """The declaration bound to a control, or None when the program leaves it undeclared."""
        return next((p for p in self.params if p.parameter_id == parameter_id), None)

    def preset(self, name: str) -> Preset:
        """A factory preset by name, case-insensitively."""
        for preset in self.presets:
            if preset.name.lower() == name.lower():
                return preset
        raise KeyError(f"{self.name} has no preset {name!r}; it has {[p.name for p in self.presets]}")


def parse_config(raw: bytes) -> ProgramConfig:
    """Decode a 7936-byte config block."""
    if len(raw) != CONFIG_BYTES:
        raise PackageError(f"config is {len(raw)} bytes, not {CONFIG_BYTES}")
    rec = np.frombuffer(raw, dtype=_CONFIG)[0]
    if rec["program_type"] >= len(PROGRAM_TYPES):
        raise PackageError(f"unknown program type {rec['program_type']}")
    if rec["nparams"] > NUM_PARAMETERS or rec["npresets"] > MAX_PRESETS:
        raise PackageError("parameter or preset count out of range")
    params = tuple(
        ParamDecl(
            parameter_id=int(p["pid"]),
            control_mode=int(p["mode"]),
            min_value=int(p["min"]),
            max_value=int(p["max"]),
            initial_value=int(p["initial"]),
            display_min=int(p["dmin"]),
            display_max=int(p["dmax"]),
            float_digits=int(p["digits"]),
            name=_text(p["name"]),
            value_labels=tuple(_text(label) for label in p["labels"][: p["nlabels"]]),
            suffix=_text(p["suffix"]),
        )
        for p in rec["params"][: rec["nparams"]]
        if 0 < p["pid"] <= NUM_PARAMETERS
    )
    return ProgramConfig(
        program_id=_text(rec["program_id"]),
        version=tuple(int(v) for v in rec["version"]),
        core=CORES.get(int(rec["core_id"]), f"core{int(rec['core_id'])}"),
        name=_text(rec["name"]),
        author=_text(rec["author"]),
        license=_text(rec["license"]),
        categories=tuple(_text(c) for c in rec["categories"][: rec["ncategories"]]),
        description=_text(rec["description"]),
        params=params,
        presets=tuple(
            Preset(_text(p["name"]), tuple(int(v) for v in p["values"]))
            for p in rec["presets"][: rec["npresets"]]
        ),
        program_type=PROGRAM_TYPES[rec["program_type"]],
        hw_mask=int(rec["hw_mask"]),
        supported_timings=int(rec["timings"]),
        processing_delay_clks=int(rec["delay"]),
    )


@dataclass(frozen=True)
class Section:
    """One TOC entry and the bytes it points at, as stored."""

    kind: int
    flags: int
    payload: bytes
    uncompressed_size: int

    @property
    def data(self) -> bytes:
        """The payload, inflated when the entry is flagged compressed."""
        if not self.flags & TOC_COMPRESSED:
            return self.payload
        out = zlib.decompress(self.payload, DEFLATE_WBITS)
        if len(out) != self.uncompressed_size:
            raise PackageError(f"section {self.kind} inflates to {len(out)}, not {self.uncompressed_size}")
        return out


@dataclass(frozen=True)
class Package:
    """A parsed ``.vmprog``."""

    version: tuple[int, int]
    flags: int
    sections: tuple[Section, ...]
    config: ProgramConfig
    digest: str

    def bitstreams(self) -> dict[SectionType, bytes]:
        """Every FPGA bitstream in the package, inflated, by its TOC type."""
        return {
            SectionType(s.kind): s.data
            for s in self.sections
            if SectionType.FPGA_BITSTREAM <= s.kind <= SectionType.HD_STANDALONE
        }


def sections(data: bytes, *, verify: bool = True) -> tuple[np.ndarray, tuple[Section, ...]]:
    """The header and every TOC section of a package, hash-checked when ``verify``."""
    if len(data) < HEADER_BYTES or data[:4] != MAGIC:
        raise PackageError("not a vmprog")
    head = np.frombuffer(data[:HEADER_BYTES], dtype=_HEADER)[0]
    if head["file_size"] != len(data):
        raise PackageError(f"header says {head['file_size']} bytes, file has {len(data)}")
    count, offset = int(head["toc_count"]), int(head["toc_offset"])
    if head["toc_bytes"] != TOC_BYTES * count or offset + TOC_BYTES * count > len(data):
        raise PackageError("table of contents outside the file")
    out = []
    for i, entry in enumerate(np.frombuffer(data, dtype=_TOC, count=count, offset=offset)):
        start, size = int(entry["offset"]), int(entry["size"])
        if start + size > len(data):
            raise PackageError(f"section {i} runs past the end of the file")
        payload = data[start : start + size]
        if verify and hashlib.blake2b(payload, digest_size=32).digest() != entry["digest"].tobytes():
            raise PackageError(f"section type {int(entry['kind'])} fails its hash")
        out.append(Section(int(entry["kind"]), int(entry["flags"]), payload, int(entry["uncompressed"])))
    return head, tuple(out)


def read_config(data: bytes) -> bytes:
    """The verified raw config block of a package."""
    found = [s.payload for s in sections(data)[1] if s.kind == SectionType.CONFIG]
    if not found or len(found[0]) != CONFIG_BYTES:
        raise PackageError("no config section")
    return found[0]


def read_package(data: bytes, *, verify: bool = True) -> Package:
    """Parse and verify a whole package."""
    head, secs = sections(data, verify=verify)
    configs = [s.payload for s in secs if s.kind == SectionType.CONFIG]
    if not configs:
        raise PackageError("no config section")
    return Package(
        version=(int(head["major"]), int(head["minor"])),
        flags=int(head["flags"]),
        sections=secs,
        config=parse_config(configs[0]),
        digest=hashlib.blake2b(data, digest_size=32).hexdigest(),
    )


def uf2_image(data: bytes) -> bytes:
    """Flatten a UF2 firmware file into the contiguous flash image it programs."""
    blocks = np.frombuffer(data, dtype=np.uint8).reshape(-1, 512)
    head = blocks[:, :32].copy().view("<u4")
    addr, size = head[:, 3].astype(np.int64), head[:, 4].astype(np.int64)
    base = int(addr.min())
    image = np.zeros(int((addr + size).max()) - base, dtype=np.uint8)
    for row, start, length in zip(blocks, addr - base, size):
        image[start : start + length] = row[32 : 32 + length]
    return image.tobytes()


def embedded_packages(image: bytes) -> Iterator[tuple[int, Package]]:
    """Every intact package laid out whole inside a flash image, with its offset."""
    start = image.find(MAGIC)
    while start >= 0:
        size = struct.unpack_from("<I", image, start + 12)[0] if start + 16 <= len(image) else 0
        if HEADER_BYTES < size <= MAX_FILE_BYTES and start + size <= len(image):
            try:
                yield start, read_package(image[start : start + size])
                start = image.find(MAGIC, start + size)
                continue
            except (PackageError, ValueError, struct.error, UnicodeDecodeError):
                pass
        start = image.find(MAGIC, start + 1)

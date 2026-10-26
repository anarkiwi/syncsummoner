"""Manifest rebuild against synthetic packages laid out as the SDK specifies."""

# pylint: disable=missing-function-docstring

import hashlib
import json
import struct
from datetime import datetime, timezone

import pytest

from syncsummoner import cli
from syncsummoner.device import manifest as M

NOW = datetime(2026, 1, 2, 3, 4, 5, tzinfo=timezone.utc)


def blake(data):
    return hashlib.blake2b(data, digest_size=32).digest()


def fixed(text, size):
    return text.encode().ljust(size, b"\0")


def config(pid, name, author, *, version=(1, 2, 3), categories=("Glitch", "Retro"), kind=0, desc="d"):
    buf = bytearray(M.CONFIG_BYTES)
    buf[0:64] = fixed(pid, 64)
    struct.pack_into("<3H", buf, 64, *version)
    buf[86:118] = fixed(name, 32)
    buf[118:182] = fixed(author, 64)
    for i, cat in enumerate(categories):
        buf[214 + 32 * i : 246 + 32 * i] = fixed(cat, 32)
    buf[470:598] = fixed(desc, 128)
    buf[729] = len(categories)
    buf[7916] = kind
    return bytes(buf)


def package(cfg, extra=b"bitstream"):
    sections = [(M.TOC_CONFIG, cfg), (4, extra)]
    offset = M.HEADER_BYTES + M.TOC_BYTES * len(sections)
    toc, body = b"", b""
    for kind, payload in sections:
        toc += struct.pack("<4I", kind, 0, offset + len(body), len(payload)) + blake(payload)
        toc += b"\0" * 16
        body += payload
    size = offset + len(body)
    head = b"VMPG" + struct.pack("<HHHH", 1, 0, M.HEADER_BYTES, 0)
    head += struct.pack("<4I", size, 0, M.HEADER_BYTES, len(toc)) + struct.pack("<I", len(sections))
    return head.ljust(M.HEADER_BYTES, b"\0") + toc + body


@pytest.fixture(name="card")
def make_card(tmp_path):
    def add(author, stem, pid, name, **kw):
        directory = tmp_path / author
        directory.mkdir(exist_ok=True)
        (directory / f"{stem}.vmprog").write_bytes(package(config(pid, name, author, **kw)))

    add("lzx", "teletext", "com.lzx.teletext", "Teletext")
    add("czart", "teletext", "com.czart.teletext", "Teletext", kind=1)
    add("czart", "aurora", "com.czart.aurora", "Aurora", version=(0, 1, 0))
    return tmp_path, add


def test_entries_come_from_the_config_block(card):
    root, _ = card
    report = M.rebuild(root, NOW)
    assert report.ok
    row = next(r for r in report.manifest["programs"] if r["file"] == "czart/aurora.vmprog")
    assert row == {
        "name": "aurora",
        "program_id": "com.czart.aurora",
        "program_name": "Aurora",
        "program_version": "0.1.0",
        "categories": ["Glitch", "Retro"],
        "program_type": "processing",
        "description": "d",
        "author": "czart",
        "file": "czart/aurora.vmprog",
    }
    types = {r["file"]: r["program_type"] for r in report.manifest["programs"]}
    assert types["czart/teletext.vmprog"] == "synthesis"


def test_a_name_shared_across_authors_is_listed_and_reported(card):
    root, _ = card
    report = M.rebuild(root, NOW)
    assert len(report.manifest["programs"]) == 3
    assert report.shared_names == {"teletext": ["czart/teletext.vmprog", "lzx/teletext.vmprog"]}


def test_header_fields_survive_and_created_is_refreshed(card):
    root, _ = card
    (root / "manifest.json").write_text(
        json.dumps({"format_version": "1.0", "version": "9.9.9", "programs": []})
    )
    manifest = M.rebuild(root, NOW).manifest
    assert (manifest["version"], manifest["created"]) == ("9.9.9", "2026-01-02T03:04:05Z")
    assert manifest["product"] == "videomancer"


def test_duplicate_program_ids_are_all_excluded(card):
    root, add = card
    add("lzx", "twin", "com.czart.aurora", "Twin")
    report = M.rebuild(root, NOW)
    assert set(report.errors) == {"czart/aurora.vmprog", "lzx/twin.vmprog"}
    assert not report.ok
    assert len(report.manifest["programs"]) == 2


def test_the_same_program_twice_under_one_author_is_excluded(card):
    root, add = card
    add("czart", "aurora2", "com.czart.aurora2", "aurora")
    assert set(M.rebuild(root, NOW).errors) == {"czart/aurora.vmprog", "czart/aurora2.vmprog"}


@pytest.mark.parametrize(
    "damage,why",
    [
        (lambda d: d[:-1], "header says"),
        (lambda d: b"XXXX" + d[4:], "not a vmprog"),
        (lambda d: d[:-1] + bytes([d[-1] ^ 1]), "fails its hash"),
        (lambda d: d[:24] + struct.pack("<I", 7) + d[28:], "table of contents"),
    ],
)
def test_a_damaged_package_is_excluded_with_the_reason(card, damage, why):
    root, _ = card
    path = root / "czart" / "aurora.vmprog"
    path.write_bytes(damage(path.read_bytes()))
    report = M.rebuild(root, NOW)
    assert why in report.errors["czart/aurora.vmprog"]
    assert len(report.manifest["programs"]) == 2


def test_an_unknown_program_type_is_excluded(card):
    root, add = card
    add("lzx", "odd", "com.lzx.odd", "Odd", kind=9)
    assert "program type" in M.rebuild(root, NOW).errors["lzx/odd.vmprog"]


def test_a_section_running_past_the_file_is_excluded(card):
    root, _ = card
    path = root / "czart" / "aurora.vmprog"
    data = bytearray(path.read_bytes())
    struct.pack_into("<I", data, M.HEADER_BYTES + 12, 10**6)
    path.write_bytes(bytes(data))
    assert "past the end" in M.rebuild(root, NOW).errors["czart/aurora.vmprog"]


def test_a_package_without_a_config_section_is_excluded(card):
    root, _ = card
    path = root / "czart" / "aurora.vmprog"
    data = bytearray(path.read_bytes())
    struct.pack_into("<I", data, M.HEADER_BYTES, 9)
    path.write_bytes(bytes(data))
    assert "no config" in M.rebuild(root, NOW).errors["czart/aurora.vmprog"]


def test_write_replaces_the_manifest_atomically(card):
    root, _ = card
    path = M.write(root, M.rebuild(root, NOW))
    assert len(json.loads(path.read_text())["programs"]) == 3
    assert [p.name for p in root.iterdir() if p.name.startswith(".")] == []


def test_cli_checks_without_writing_and_fails_on_exclusions(card, capsys):
    root, add = card
    assert cli.main(["manifest", str(root)]) == 0
    assert json.loads(capsys.readouterr().out)["listed"] == 3
    assert not (root / "manifest.json").exists()
    add("lzx", "twin", "com.czart.aurora", "Twin")
    assert cli.main(["manifest", str(root), "--write"]) == 1
    assert json.loads(capsys.readouterr().out)["excluded"].keys() == {
        "czart/aurora.vmprog",
        "lzx/twin.vmprog",
    }
    assert len(json.loads((root / "manifest.json").read_text())["programs"]) == 2

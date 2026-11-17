"""Where programs come from: card copies, firmware images, and the sources they were built from."""

from __future__ import annotations

import os
import re
import tomllib
from dataclasses import dataclass, field
from functools import cached_property
from pathlib import Path

from .vmprog import Package, PackageError, embedded_packages, read_package, uf2_image

ENV = {
    "card": "SYNCSUMMONER_CARD",
    "firmware": "SYNCSUMMONER_FIRMWARE",
    "sdk": "SYNCSUMMONER_SDK",
    "community": "SYNCSUMMONER_COMMUNITY",
}
DEFAULTS = {
    "card": "captures/card/programs",
    "firmware": "captures/firmware",
    "sdk": "captures/upstream/videomancer-sdk",
    "community": "captures/upstream/videomancer-community-programs",
}
#: Firmware programs under this id prefix are listed by the device without a folder.
BUILTIN_PREFIX = "com.lzxindustries."


def _key(text: str) -> str:
    return re.sub(r"[^a-z0-9]", "", text.lower())


@dataclass(frozen=True)
class Program:
    """One installed program and, where published, the VHDL it was built from."""

    package: Package
    origin: str
    folder: str
    sources: tuple[Path, ...] = ()

    @property
    def name(self) -> str:
        """``folder/Program Name``, or the bare name for firmware built-ins, as the device lists it."""
        return f"{self.folder}/{self.package.config.name}" if self.folder else self.package.config.name

    @property
    def key(self) -> str:
        """Cache key: program id, version and package digest."""
        cfg = self.package.config
        return f"{cfg.program_id}-{cfg.version_text}-{self.package.digest[:12]}"


@dataclass
class Library:
    """Every program found under the card copy and firmware images, with sources matched by id and version."""

    card: Path | None = None
    firmware: Path | None = None
    sdk: Path | None = None
    community: Path | None = None
    _programs: list[Program] = field(default_factory=list, init=False, repr=False)

    @classmethod
    def default(cls, root: Path | None = None) -> "Library":
        """Paths from ``SYNCSUMMONER_*`` variables, else the gitignored ``captures/`` layout."""
        root = Path.cwd() if root is None else root
        paths = {k: Path(os.environ.get(var) or root / DEFAULTS[k]) for k, var in ENV.items()}
        return cls(**{k: p if p.exists() else None for k, p in paths.items()})

    @cached_property
    def sources(self) -> dict[tuple[str, str], tuple[Path, ...]]:
        """VHDL files per (program_id, version) from every TOML under the SDK and community trees."""
        found = {}
        for tree in filter(
            None, (self.sdk and self.sdk / "programs", self.community and self.community / "programs")
        ):
            for toml in sorted(tree.rglob("*.toml")):
                try:
                    prog = tomllib.loads(toml.read_text()).get("program", {})
                except (tomllib.TOMLDecodeError, UnicodeDecodeError):
                    continue
                if "program_id" in prog:
                    vhd = tuple(sorted(toml.parent.glob("*.vhd")))
                    found.setdefault((prog["program_id"], str(prog.get("program_version", ""))), vhd)
        return found

    @property
    def programs(self) -> list[Program]:
        """Card packages first, then firmware built-ins not already on the card."""
        if not self._programs:
            seen = set()
            for package, origin, folder in self._scan():
                cfg = package.config
                if (cfg.program_id, cfg.version_text) in seen:
                    continue
                seen.add((cfg.program_id, cfg.version_text))
                src = self.sources.get((cfg.program_id, cfg.version_text), ())
                self._programs.append(Program(package, origin, folder, src))
        return self._programs

    def _scan(self):
        if self.card:
            for path in sorted(self.card.glob("*/*.vmprog")):
                try:
                    yield read_package(path.read_bytes()), str(path), path.parent.name
                except PackageError:
                    continue
        if self.firmware:
            images = [self.firmware] if self.firmware.is_file() else sorted(self.firmware.glob("*.uf2"))
            for image in images:
                for offset, package in embedded_packages(uf2_image(image.read_bytes())):
                    builtin = package.config.program_id.startswith(BUILTIN_PREFIX)
                    folder = "" if builtin else _key(package.config.author)
                    yield package, f"{image}@{offset:#x}", folder

    def find(self, name: str) -> Program:
        """A program by device name (``author/Name`` or ``Name``) or by program id."""
        folder, _, title = name.rpartition("/")
        hits = [
            p
            for p in self.programs
            if p.package.config.program_id == name
            or (
                _key(p.package.config.name) == _key(title)
                and (not folder or _key(folder) in (_key(p.folder), _key(p.package.config.author)))
            )
        ]
        if len(hits) > 1 and not folder:
            hits = [p for p in hits if not p.folder] or hits
        if len(hits) != 1:
            raise KeyError(f"{name!r} matches {len(hits)} programs: {[p.name for p in hits]}")
        return hits[0]

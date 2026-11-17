"""Parameter specs from a program's own ``.vmprog`` declaration, resolved by device name."""

from __future__ import annotations

from functools import cache

from syncsummoner.program import declared
from syncsummoner.program.library import DEFAULTS, ENV, Library
from syncsummoner.program.vmprog import ParamDecl, ProgramConfig

from .profile import PARAM_COUNT, PARAM_MAX, ParamKind, ParamSpec

__all__ = [
    "UndeclaredProgramError",
    "declaration",
    "default_library",
    "initial_value",
    "spec_from_decl",
    "specs",
]


class UndeclaredProgramError(LookupError):
    """No ``.vmprog`` declaration was found for a program the device runs."""


@cache
def default_library() -> Library:
    """The card copies and firmware images under ``SYNCSUMMONER_*`` or ``captures/``, scanned once."""
    return Library.default()


def declaration(name: str, library: Library | None = None) -> ProgramConfig:
    """The declaration of the program the device names ``name``, raising when none is installed."""
    try:
        return (library or default_library()).find(name).package.config
    except KeyError as err:
        raise UndeclaredProgramError(
            f"no single .vmprog declares {name!r} ({err}): copy the card's programs/<author>/*.vmprog under "
            f"${ENV['card']} (default {DEFAULTS['card']}) or the firmware .uf2 under ${ENV['firmware']} "
            f"(default {DEFAULTS['firmware']})"
        ) from err


def initial_value(decl: ParamDecl) -> int:
    """A declaration's initial value in device units: full-scale ranges pass through, others scale."""
    span = decl.max_value - decl.min_value
    if span >= PARAM_MAX:
        return int(decl.initial_value)
    return round((decl.initial_value - decl.min_value) * PARAM_MAX / max(span, 1))


def spec_from_decl(decl: ParamDecl | None, index: int, name: str | None = None) -> ParamSpec:
    """The spec of slot ``index``: kind and step count as declared, UNUSED where nothing is declared."""
    found, levels = declared.kind(decl)
    return ParamSpec(
        index=index,
        name=name if name is not None else decl.name if decl else "",
        native_min=float(decl.display_min) if decl else 0.0,
        native_max=float(decl.display_max) if decl else 0.0,
        kind=ParamKind(found.value),
        steps=levels,
    )


def specs(config: ProgramConfig | None, names: dict[int, str] | None = None) -> list[ParamSpec]:
    """All twelve slot specs of ``config``, labelled by ``names`` where given."""
    names = names or {}
    return [
        spec_from_decl(config.param(i) if config else None, i, names.get(i))
        for i in range(1, PARAM_COUNT + 1)
    ]

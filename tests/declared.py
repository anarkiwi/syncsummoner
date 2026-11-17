"""Program declarations built in memory, and a library that serves them by device name."""

from types import SimpleNamespace

from syncsummoner.program.vmprog import ControlMode, ParamDecl, ProgramConfig


def decl(pid, name="", *, mode=ControlMode.LINEAR, low=0, high=1023, initial=0, labels=(), display=(0, 100)):
    """One ``ParamDecl`` with the SDK's field layout."""
    return ParamDecl(
        parameter_id=pid,
        control_mode=int(mode),
        min_value=low,
        max_value=high,
        initial_value=initial,
        display_min=display[0],
        display_max=display[1],
        float_digits=0,
        name=name,
        value_labels=tuple(labels),
        suffix="",
    )


def toggle(pid, name="", *, initial=0):
    """A two-state switch declared as the SDK declares one."""
    return decl(
        pid, name, mode=ControlMode.BOOLEAN, high=1, initial=initial, labels=("Off", "On"), display=(0, 1)
    )


def config(name, *params, program_id=None):
    """A ``ProgramConfig`` carrying only ``params``."""
    return ProgramConfig(
        program_id=program_id or f"test.{name.lower().replace(' ', '_')}",
        version=(1, 0, 0),
        core="yuv444_30b",
        name=name,
        author="test",
        license="",
        categories=(),
        description="",
        params=tuple(params),
        presets=(),
        program_type="processing",
        hw_mask=0,
        supported_timings=0,
        processing_delay_clks=0,
    )


class Declarations:
    """``Library`` stand-in: ``find`` by device name (``author/Name`` or ``Name``), KeyError otherwise."""

    def __init__(self, *configs):
        self.configs = {c.name: c for c in configs}
        self.asked = []

    def find(self, name):
        """The program whose plain name ends ``name``, as ``Library.find`` returns it."""
        self.asked.append(name)
        plain = name.rpartition("/")[2]
        if plain not in self.configs:
            raise KeyError(f"{name!r} matches 0 programs: []")
        return SimpleNamespace(package=SimpleNamespace(config=self.configs[plain]))

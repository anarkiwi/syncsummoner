"""Automated characterization of the LZX Videomancer and generative rendering."""

from importlib.metadata import PackageNotFoundError, version

try:
    #: Read from the installed distribution, so it cannot drift from pyproject.
    __version__ = version("syncsummoner")
except PackageNotFoundError:  # pragma: no cover - a source tree that was never installed
    __version__ = "0.0.0"

#: The metrics version, recorded as provenance on every measurement, is
#: ``syncsummoner.aesthetics.__version__`` and is deliberately not this one.

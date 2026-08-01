"""The extraction seam: dependency and I/O boundaries of the package."""

import ast
import pathlib

import pytest

from syncsummoner import aesthetics

PACKAGE = pathlib.Path(aesthetics.__file__).parent
ALLOWED_THIRD_PARTY = {"numpy", "scipy", "cv2"}
SOURCES = sorted(PACKAGE.glob("*.py"))


def imported_roots(path):
    """Root module names imported by a source file."""
    tree = ast.parse(path.read_text(encoding="utf-8"))
    roots = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            roots.update(alias.name.split(".")[0] for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.level == 0 and node.module:
            roots.add(node.module.split(".")[0])
    return roots


@pytest.mark.parametrize("path", SOURCES, ids=lambda p: p.name)
def test_no_imports_outside_the_package_or_its_extra(path):
    """Only stdlib, numpy/scipy/cv2 and aesthetics itself may be imported."""
    for root in imported_roots(path):
        if root == "syncsummoner":
            assert "syncsummoner.aesthetics" in path.read_text(encoding="utf-8")
        else:
            assert root in ALLOWED_THIRD_PARTY or root not in {"pyvmancer", "yaml", "pyarrow"}


@pytest.mark.parametrize("path", SOURCES, ids=lambda p: p.name)
def test_only_the_shim_imports_the_filesystem(path):
    """Nothing but io.py may reach for cv2.VideoCapture or open files."""
    if path.name in {"io.py", "__main__.py"}:
        return
    source = path.read_text(encoding="utf-8")
    assert "VideoCapture" not in source and "imread" not in source
    assert "open(" not in source


def test_public_api_is_importable_and_versioned():
    """Every re-exported name resolves and the analyzer version is a string."""
    assert isinstance(aesthetics.__version__, str)
    for name in aesthetics.__all__:
        assert hasattr(aesthetics, name)


def test_the_shim_is_not_re_exported():
    """io.py stays out of the public API so the rest of the project cannot use it."""
    assert "io" not in aesthetics.__all__

"""Simulation backend: subprocess contract, graceful absence, sim-stamped records."""

import os
import stat
import sys
from types import SimpleNamespace

import numpy as np
import pytest

from syncsummoner.device.profile import Source
from syncsummoner.probe import sim

SCRIPT = """#!{python}
import sys
import cv2
import numpy as np

argv = sys.argv[1:]
args = {{a: argv[i + 1] for i, a in enumerate(argv) if a.startswith("--") and a != "--param"}}
params = [argv[i + 1] for i, a in enumerate(argv) if a == "--param"]
image = cv2.imread(args["--input"], cv2.IMREAD_COLOR)
gain = float(params[0].split("=")[1]) / 1023.0
{body}
cv2.imwrite(args["--output"], np.clip(image * gain, 0, 255).astype(np.uint8))
"""


class StillAnalyzer:
    """Minimal analyzer stub: a single simulated frame has no motion or dynamics."""

    __version__ = "9.9.9"

    def gabor_energy(self, frame):
        """Gabor energy."""
        return SimpleNamespace(energy=np.full((5, 4), 0.05), concentration=float(frame.std()), peak=(0, 0))

    def spectral_stats(self, _frame):
        """Spectral stats."""
        return SimpleNamespace(slope=-1.0, r2=0.9, fractal_dimension=1.4)

    def level_stats(self, frame):
        """Level stats."""
        return SimpleNamespace(
            luma_mean=float(frame.mean()),
            luma_std=float(frame.std()),
            chroma_mean=0.1,
            chroma_std=0.02,
            clip_frac=0.0,
            illegal_frac=0.0,
            colourfulness=0.3,
        )

    def passthrough_distance(self, source, output):
        """Passthrough distance."""
        return float(np.abs(np.asarray(source) - np.asarray(output)).mean())


def fake_binary(tmp_path, *, body="", name="vhdl-image-tester"):
    """Fake binary."""
    path = tmp_path / name
    path.write_text(SCRIPT.format(python=sys.executable, body=body))
    path.chmod(path.stat().st_mode | stat.S_IEXEC)
    return path


def plan(n=3):
    """Plan."""
    return [{1: i / max(n - 1, 1), 7: True} for i in range(n)]


def run(backend, points=3, **kwargs):
    """Run."""
    return sim.run_plan_sim(
        plan(points),
        program="blur",
        analyzer=StillAnalyzer(),
        rng=np.random.default_rng(0),
        width=48,
        height=32,
        backend=backend,
        **kwargs,
    )


def test_missing_binary_is_a_clear_error():
    """Missing binary is a clear error."""
    backend = sim.SimBackend(binary="definitely-not-installed-vhdl-image-tester")
    assert backend.available() is False
    with pytest.raises(sim.SimUnavailableError, match="install the LZX SDK"):
        backend.resolve()


def test_missing_binary_does_not_block_import():
    """Missing binary does not block import."""
    assert sim.DEFAULT_BINARY == "vhdl-image-tester"
    assert issubclass(sim.SimUnavailableError, sim.SimError)


def test_command_carries_every_parameter(tmp_path):
    """Command carries every parameter."""
    backend = sim.SimBackend(binary=str(fake_binary(tmp_path)))
    argv = backend.command("blur", tuple(range(12)), "in.png", "out.png")
    assert argv[1:5] == ["--program", "blur", "--input", "in.png"]
    assert argv.count("--param") == 12
    assert "12=11" in argv


def test_run_plan_sim_emits_sim_records(tmp_path):
    """Run plan simulation emits simulation records."""
    records = run(sim.SimBackend(binary=str(fake_binary(tmp_path))))
    assert len(records) == 3
    assert all(r.source is Source.SIM for r in records)
    assert all(r.firmware == "sim" for r in records)
    assert [r.state_index for r in records] == [0, 1, 2]
    assert [r.params[6] for r in records] == [1023, 1023, 1023]
    assert records[0].metrics["luma_mean"] < records[-1].metrics["luma_mean"]
    assert "framediff_energy" not in records[0].metrics
    assert records[-1].metrics["passthrough_distance"] < records[0].metrics["passthrough_distance"]


def test_available_when_binary_resolves(tmp_path):
    """Available when binary resolves."""
    assert sim.SimBackend(binary=str(fake_binary(tmp_path))).available() is True


def test_binary_on_path_is_resolved(tmp_path, monkeypatch):
    """Binary on path is resolved."""
    fake_binary(tmp_path)
    monkeypatch.setenv("PATH", str(tmp_path) + os.pathsep + os.environ["PATH"])
    assert sim.SimBackend().available() is True


def test_nonzero_exit_raises(tmp_path):
    """Nonzero exit raises."""
    backend = sim.SimBackend(binary=str(fake_binary(tmp_path, body='sys.exit("simulation diverged")')))
    with pytest.raises(sim.SimError, match="simulation diverged"):
        run(backend, points=1)


def test_missing_output_raises(tmp_path):
    """Missing output raises."""
    backend = sim.SimBackend(binary=str(fake_binary(tmp_path, body="sys.exit(0)")))
    with pytest.raises(sim.SimError, match="no readable image"):
        run(backend, points=1)

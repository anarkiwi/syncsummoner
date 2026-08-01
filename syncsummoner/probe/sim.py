"""GHDL ``vhdl-image-tester`` backend: still-image pre-screening with no hardware.

Simulation gives static parameter response only, never temporal or feedback
behaviour, so records are stamped ``source=sim`` and the planner discounts them.
Absence of the SDK binary is an error at call time, never at import time.
"""

from __future__ import annotations

import importlib
import shutil
import subprocess
import tempfile
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from syncsummoner.device.profile import MeasurementRecord, Source
from syncsummoner.probe import patterns
from syncsummoner.probe.runner import raw_params, frame_metrics

__all__ = ["SimError", "SimUnavailableError", "SimBackend", "run_plan_sim"]

DEFAULT_BINARY = "vhdl-image-tester"


class SimError(RuntimeError):
    """The simulation backend ran but failed."""


class SimUnavailableError(SimError):
    """The SDK's ``vhdl-image-tester`` binary is not installed or not executable."""


def _codec():
    """Image codec, imported on demand: only the simulation backend touches files."""
    return importlib.import_module("cv2")


def _write(path, frame):
    _codec().imwrite(str(path), np.round(np.clip(frame, 0.0, 1.0)[:, :, ::-1] * 255).astype(np.uint8))


def _read(path):
    cv2 = _codec()
    image = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if image is None:
        raise SimError(f"simulation produced no readable image at {path}")
    return (image[:, :, ::-1].astype(np.float32) / 255.0).copy()


@dataclass
class SimBackend:
    """Subprocess driver for the SDK simulator, one still image per invocation."""

    binary: str = DEFAULT_BINARY
    timeout_s: float = 120.0

    def resolve(self):
        """Absolute path to the simulator binary, or raise :class:`SimUnavailableError`."""
        found = shutil.which(self.binary) or (str(Path(self.binary)) if Path(self.binary).is_file() else None)
        if found is None:
            raise SimUnavailableError(
                f"{self.binary!r} not found on PATH; install the LZX SDK or pass "
                "SimBackend(binary=...) to use the simulation backend"
            )
        return found

    def available(self):
        """True when the simulator binary can be resolved."""
        try:
            self.resolve()
        except SimUnavailableError:
            return False
        return True

    def command(self, program, raw, in_path, out_path):
        """Argument vector for one simulation run; override for a different SDK CLI."""
        argv = [self.resolve(), "--program", program, "--input", str(in_path), "--output", str(out_path)]
        for index, value in enumerate(raw, start=1):
            argv += ["--param", f"{index}={value}"]
        return argv

    def render(self, program, raw, frame, *, workdir):
        """Run one program over one still frame and return the simulated output."""
        in_path, out_path = Path(workdir) / "in.png", Path(workdir) / "out.png"
        _write(in_path, frame)
        out_path.unlink(missing_ok=True)
        result = subprocess.run(
            self.command(program, raw, in_path, out_path),
            capture_output=True,
            text=True,
            check=False,
            timeout=self.timeout_s,
        )
        if result.returncode != 0:
            raise SimError(f"{self.binary} exited {result.returncode}: {result.stderr.strip()}")
        return _read(out_path)


def run_plan_sim(
    plan,
    *,
    program,
    analyzer,
    rng,
    stimulus="zoneplate",
    width=256,
    height=256,
    backend=None,
    firmware="sim",
):
    """Run a plan as GHDL simulations against a still stimulus, one record per point."""
    backend = backend or SimBackend()
    source_frame = patterns.generate(stimulus, width=width, height=height, rng=rng)
    analyzer_version = f"aesthetics {getattr(analyzer, '__version__', 'unknown')}"
    records = []
    with tempfile.TemporaryDirectory(prefix="syncsummoner-sim-") as workdir:
        for step, vector in enumerate(plan):
            raw = raw_params(vector)
            output = backend.render(program, raw, source_frame, workdir=workdir)
            records.append(
                MeasurementRecord(
                    program=program,
                    firmware=firmware,
                    analyzer=analyzer_version,
                    source=Source.SIM,
                    params=raw,
                    state_index=step,
                    stimulus=stimulus,
                    metrics=frame_metrics([output], analyzer, reference=source_frame),
                )
            )
    return records

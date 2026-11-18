"""A program's survival at any setting, kriged jointly from the settings its archive measured.

Each response is a Gaussian process over every parameter the archive varied, its kernel fitted by
marginal likelihood, so a setting is predicted from the records it resembles in the parameters that
matter there rather than as a sum of one-parameter averages, which cannot carry one parameter gating another.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Mapping

import numpy as np
from scipy.linalg import cho_solve, solve_triangular
from sklearn.exceptions import ConvergenceWarning
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel, WhiteKernel

from syncsummoner.device.profile import PARAM_COUNT, PARAM_MAX

#: Each survival response's kriging space: ratios perceived logarithmically, a correlation as its Fisher z.
TRANSFORMS: dict[str, Callable[[np.ndarray], np.ndarray]] = {
    "si_ratio": np.log,
    "ti_ratio": np.log,
    "motion_follow": np.arctanh,
}
INVERSE: dict[str, Callable[[np.ndarray], np.ndarray]] = {
    "si_ratio": np.exp,
    "ti_ratio": np.exp,
    "motion_follow": np.tanh,
}


def transformed(name: str, values: np.ndarray) -> np.ndarray:
    """``values`` in kriging space, clipped inside the transform's domain at the nearest measured value."""
    values = np.asarray(values, dtype=np.float64)
    inside = np.abs(values) < 1 if name == "motion_follow" else values > 0
    if not inside.any():
        return np.full(values.shape, np.nan)
    if name == "motion_follow":
        edge = float(np.abs(values[inside]).max())
        return TRANSFORMS[name](np.clip(values, -edge, edge))
    return TRANSFORMS[name](np.maximum(values, float(values[inside].min())))


@dataclass(frozen=True)
class Kriged:
    """One response at a run of settings, in kriging space.

    Per setting, ``pull`` is what the records add to their mean ``centre``, ``weight`` the share of
    a prior mean they replace, and ``sd`` the posterior spread.
    """

    name: str
    centre: float
    pull: np.ndarray
    weight: np.ndarray
    sd: np.ndarray

    @property
    def mean(self) -> np.ndarray:
        """Posterior mean per setting, in the response's own units."""
        return INVERSE[self.name](self.centre + self.pull)

    def margin(self, mask: np.ndarray, bound: float) -> float:
        """How far the settings under ``mask`` are kriged above ``bound`` taken as the prior mean.

        A setting no record informs reverts to that prior, so it sits on the boundary and not above it.
        """
        return float(self.pull[mask].mean() + (self.centre - bound) * self.weight[mask].mean())


def boundary(name: str, value: float) -> float:
    """A bound in a response's units, in its kriging space; one outside the response's range is infinite."""
    low, high = (-1.0, 1.0) if name == "motion_follow" else (0.0, np.inf)
    if value <= low or value >= high:
        return -np.inf if value <= low else np.inf
    return float(TRANSFORMS[name](np.float64(value)))


@dataclass(frozen=True)
class Response:
    """One survival response kriged over the parameters an archive varied."""

    name: str
    columns: np.ndarray
    centre: float
    gp: GaussianProcessRegressor
    ones: np.ndarray

    @classmethod
    def fit(cls, name: str, settings: np.ndarray, values: np.ndarray) -> Response | None:
        """The response from raw ``settings`` (one row each) and their ``values``; None if it never varied."""
        y = transformed(name, values)
        ok = np.isfinite(y)
        x = np.asarray(settings, dtype=np.float64)[ok]
        columns = np.flatnonzero(np.ptp(x, axis=0) > 0) if ok.sum() > 1 else np.zeros(0, np.intp)
        if columns.size == 0 or np.ptp(y[ok]) == 0:
            return None
        x, centre = x[:, columns] / PARAM_MAX, float(y[ok].mean())
        kernel = ConstantKernel() * RBF(np.ones(columns.size)) + WhiteKernel()
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", ConvergenceWarning)
            gp = GaussianProcessRegressor(kernel, normalize_y=False).fit(x, y[ok] - centre)
        return cls(name, columns, centre, gp, cho_solve((gp.L_, True), np.ones(x.shape[0])))

    def at(self, raw: np.ndarray) -> Kriged:
        """The response at each row of raw settings ``(n, PARAM_COUNT)``."""
        x = np.asarray(raw, dtype=np.float64)[:, self.columns] / PARAM_MAX
        unique, inverse = np.unique(x, axis=0, return_inverse=True)
        signal = self.gp.kernel_.k1
        cross = signal(unique, self.gp.X_train_)
        v = solve_triangular(self.gp.L_, cross.T, lower=True)
        sd = np.sqrt(np.maximum(signal.diag(unique) - np.square(v).sum(axis=0), 0.0))
        inverse = inverse.ravel()
        return Kriged(
            self.name,
            self.centre,
            (cross @ self.gp.alpha_)[inverse],
            (cross @ self.ones)[inverse],
            sd[inverse],
        )


def fit(settings: np.ndarray, metrics: Mapping[str, np.ndarray]) -> dict[str, Response]:
    """Every survival response in ``metrics``, measured at ``settings``, that varied."""
    found = {name: Response.fit(name, settings, metrics[name]) for name in TRANSFORMS if name in metrics}
    return {name: model for name, model in found.items() if model is not None}


def from_rows(rows: list[Mapping[str, Any]]) -> dict[str, dict[str, Response]]:
    """Survival models by program from measurement rows, as an archive's Parquet holds them."""
    by_program: dict[str, list[Mapping[str, Any]]] = {}
    for row in rows:
        by_program.setdefault(str(row["program"]), []).append(row)
    out = {}
    for program, found in by_program.items():
        settings = np.array([[row[f"p{i + 1}"] for i in range(PARAM_COUNT)] for row in found], np.float64)
        metrics = {
            name: np.array([np.nan if row.get(name) is None else row[name] for row in found], np.float64)
            for name in TRANSFORMS
        }
        out[program] = fit(settings, metrics)
    return out


def from_directory(directory: str | Path | None) -> dict[str, dict[str, Response]]:
    """Survival models by program from every measurement Parquet in a profiles directory."""
    from syncsummoner.probe.fit import load_measurements  # pylint: disable=import-outside-toplevel

    if directory is None:
        return {}
    return from_rows(
        [row for path in sorted(Path(directory).glob("*.parquet")) for row in load_measurements(path)]
    )

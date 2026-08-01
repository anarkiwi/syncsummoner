"""Information content of a scalar series under a fixed-order Markov model.

The series is quantized to ``n_bins`` levels and an order-``order`` transition
table is fitted with Laplace smoothing; surprisal is ``-log2 P(x_t | context)``.
Contexts are stored sparsely, so cost is O(T) in series length, not O(n_bins**order).
"""

import numpy as np

LAPLACE_ALPHA = 1.0


class SurprisalModel:
    """Order-N Markov predictor over a quantized scalar series."""

    def __init__(self, *, order: int = 3, n_bins: int = 16, rng: np.random.Generator):
        if order < 1 or n_bins < 2:
            raise ValueError("need order >= 1 and n_bins >= 2")
        self.order = int(order)
        self.n_bins = int(n_bins)
        self.rng = rng
        self._edges: np.ndarray | None = None
        self._contexts = np.empty(0, dtype=np.int64)
        self._conditional = np.empty((0, self.n_bins), dtype=np.float64)
        self._marginal = np.full(self.n_bins, 1.0 / self.n_bins)

    def _quantize(self, series: np.ndarray) -> np.ndarray:
        arr = np.asarray(series, dtype=np.float64).ravel()
        if self._edges is None:
            lo, hi = float(arr.min()), float(arr.max())
            span = hi - lo
            self._edges = np.linspace(lo, hi, self.n_bins + 1)[1:-1] if span > 0.0 else np.empty(0)
        if self._edges.size == 0:
            return np.zeros(arr.size, dtype=np.int64)
        return np.searchsorted(self._edges, arr, side="right").astype(np.int64)

    def _context_ids(self, symbols: np.ndarray) -> np.ndarray:
        windows = np.lib.stride_tricks.sliding_window_view(symbols[:-1], self.order)
        return windows @ (self.n_bins ** np.arange(self.order, dtype=np.int64))

    def fit(self, series: np.ndarray) -> "SurprisalModel":
        """Estimate the transition table from a series; returns self."""
        symbols = self._quantize(series)
        counts = np.bincount(symbols, minlength=self.n_bins) + LAPLACE_ALPHA
        self._marginal = counts / counts.sum()
        if symbols.size <= self.order:
            return self
        ids = self._context_ids(symbols)
        self._contexts, inverse = np.unique(ids, return_inverse=True)
        flat = inverse.ravel() * self.n_bins + symbols[self.order :]
        table = np.bincount(flat, minlength=self._contexts.size * self.n_bins).astype(np.float64)
        table = table.reshape(self._contexts.size, self.n_bins) + LAPLACE_ALPHA
        self._conditional = table / table.sum(axis=1, keepdims=True)
        return self

    def information_content(self, series: np.ndarray) -> np.ndarray:
        """Surprisal ``-log2 P(x_t | context)`` per sample, ``(T,) float32``."""
        symbols = self._quantize(series)
        probs = self._marginal[symbols].copy()
        if symbols.size > self.order and self._contexts.size:
            ids = self._context_ids(symbols)
            slot = np.searchsorted(self._contexts, ids)
            slot = np.clip(slot, 0, self._contexts.size - 1)
            known = self._contexts[slot] == ids
            rows = self._conditional[slot[known]]
            probs[self.order :][known] = rows[np.arange(rows.shape[0]), symbols[self.order :][known]]
        return (-np.log2(probs)).astype(np.float32)

    def sample(self, n_samples: int) -> np.ndarray:
        """Draw a symbol sequence from the fitted model using the caller's generator."""
        out = np.empty(int(n_samples), dtype=np.int64)
        powers = self.n_bins ** np.arange(self.order, dtype=np.int64)
        for t in range(out.size):
            dist = self._marginal
            if t >= self.order and self._contexts.size:
                ctx = int(out[t - self.order : t] @ powers)
                slot = int(np.searchsorted(self._contexts, ctx))
                if slot < self._contexts.size and self._contexts[slot] == ctx:
                    dist = self._conditional[slot]
            out[t] = self.rng.choice(self.n_bins, p=dist)
        return out


def information_content(
    series: np.ndarray, *, rng: np.random.Generator, order: int = 3, n_bins: int = 16
) -> np.ndarray:
    """Per-sample surprisal of a series under a model fitted to that series."""
    model = SurprisalModel(order=order, n_bins=n_bins, rng=rng).fit(series)
    return model.information_content(series)


def entropy_rate(series: np.ndarray, *, rng: np.random.Generator, order: int = 3, n_bins: int = 16) -> float:
    """Mean information content in bits per sample."""
    return float(np.mean(information_content(series, rng=rng, order=order, n_bins=n_bins)))

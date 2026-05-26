"""Canonical synthetic inputs for regression tests.

Single registry every regression test and candidate draws from, so candidates
are comparable. Inputs regenerate from a generator+seed (matching the seeded
style of the unit tests); only the *golden output* is committed. Keep tiny.
"""

from __future__ import annotations

import numpy as np

DEFAULT_SEED = 20260526


def roi_timeseries(n_rois: int = 20, n_tps: int = 200, seed: int = DEFAULT_SEED) -> np.ndarray:
    """ROI × time matrix (functional-connectivity inputs)."""
    rng = np.random.default_rng(seed)
    return rng.standard_normal((n_rois, n_tps))


def correlation_matrix(n: int = 20, seed: int = DEFAULT_SEED) -> np.ndarray:
    """Symmetric correlation matrix, unit diagonal, off-diagonals in (-1, 1)."""
    rng = np.random.default_rng(seed)
    a = rng.uniform(-0.9, 0.9, size=(n, n))
    a = np.triu(a, k=1)
    a = a + a.T
    np.fill_diagonal(a, 1.0)
    return a


def connectome(n: int = 20, density: float = 0.3, seed: int = 7) -> np.ndarray:
    """Symmetric weighted adjacency matrix, zero diagonal (graph metrics)."""
    rng = np.random.default_rng(seed)
    w = np.triu(rng.uniform(0.0, 1.0, size=(n, n)), k=1)
    mask = np.triu(rng.uniform(0.0, 1.0, size=(n, n)) < density, k=1)
    w = w * mask
    return w + w.T

"""Gate-tier regression: graph metrics (deterministic, hermetic)."""

from __future__ import annotations

import numpy as np
import pytest

from neurovrai.connectome.graph_metrics import (
    compute_global_efficiency,
    compute_node_degree,
)
from tests.regression import fixtures
from tests.regression._equivalence import assert_array_matches_golden

pytestmark = pytest.mark.regression


def test_node_degree_preserved():
    result = compute_node_degree(fixtures.connectome())
    assert_array_matches_golden(np.asarray(result), "connectivity_node_degree")


def test_global_efficiency_preserved():
    result = compute_global_efficiency(fixtures.connectome())
    assert_array_matches_golden(np.asarray(result, dtype=float), "connectivity_global_efficiency")

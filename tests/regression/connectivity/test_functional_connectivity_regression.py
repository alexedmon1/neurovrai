"""Gate-tier regression: functional-connectivity math (deterministic, hermetic)."""

from __future__ import annotations

import pytest

from neurovrai.connectome.functional_connectivity import fisher_z_transform
from tests.regression import fixtures
from tests.regression._equivalence import assert_array_matches_golden

pytestmark = pytest.mark.regression


def test_fisher_z_transform_preserved():
    result = fisher_z_transform(fixtures.correlation_matrix())
    assert_array_matches_golden(result, "connectivity_fisher_z")

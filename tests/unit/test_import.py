"""Baseline unit test: the package imports and exposes a version.

Keeps `make test` green from the start. Add real unit tests for pure
functions (config parsing, path resolution, metrics) over time.
"""

import pytest

pytestmark = pytest.mark.unit


def test_package_imports_with_version():
    import neurovrai

    assert isinstance(neurovrai.__version__, str)
    assert neurovrai.__version__

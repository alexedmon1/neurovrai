"""The installed entry point starts.

`neurovrai --help` was broken for an unknown stretch: cli.py imported
`neurovrai.config_generator`, which lives at `neurovrai.preprocess.config_generator`,
so the console script died at import. `make check` stayed green throughout, because
nothing here invoked the CLI. Importing the module is enough to catch it -- the
failure was at import time, not in any command.
"""

import pytest

pytestmark = pytest.mark.unit


def test_cli_module_imports():
    from neurovrai.preprocess.cli import main

    assert callable(main)


def test_help_exits_cleanly():
    from click.testing import CliRunner

    from neurovrai.preprocess.cli import main

    result = CliRunner().invoke(main, ["--help"])
    assert result.exit_code == 0, result.output
    assert "Commands:" in result.output

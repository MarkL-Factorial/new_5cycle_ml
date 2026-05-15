"""CLI parsing smoke."""

import pytest

from cell_lifetime.cli import main


def test_run_help():
    with pytest.raises(SystemExit) as e:
        main(["run", "--help"])
    assert e.value.code == 0


def test_unknown_subcommand_errors():
    with pytest.raises(SystemExit):
        main(["badcmd"])

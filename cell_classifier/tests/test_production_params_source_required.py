"""--production-params-source is required for --mode production; no silent default."""

from pathlib import Path

import pytest

from cell_classifier.cli import main


_TEMPLATE = Path(__file__).resolve().parents[1] / "configs" / "rf.yaml"


def test_missing_production_params_source_exits(tmp_path: Path):
    with pytest.raises(SystemExit, match="production-params-source"):
        main([
            "run", "--mode", "production",
            "--model-config", str(_TEMPLATE),
            "--N", "300", "--db-version", "A2.2",
            "--baseline-cycle", "1", "--feature-subset", "fs_cv",
            "--seeds", "1,2",
            "--out-root", str(tmp_path),
        ])


def test_error_message_lists_choices(tmp_path: Path, capsys):
    """The error message should name both choices so the user knows what to pass."""
    with pytest.raises(SystemExit) as excinfo:
        main([
            "run", "--mode", "production",
            "--model-config", str(_TEMPLATE),
            "--N", "300", "--db-version", "A2.2",
            "--baseline-cycle", "1", "--feature-subset", "fs_cv",
            "--seeds", "1,2",
            "--out-root", str(tmp_path),
        ])
    msg = str(excinfo.value)
    assert "from_validation_run" in msg
    assert "retune" in msg

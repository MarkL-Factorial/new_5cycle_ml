"""CLI: arg parsing + idempotency + mode dispatch (no actual training)."""

import json
from pathlib import Path

import pytest

from cell_classifier.cli import main


def test_run_validation_requires_tuning_protocol(monkeypatch, tmp_path):
    """Missing --tuning-protocol for validation should fail cleanly."""
    with pytest.raises(SystemExit, match="tuning-protocol"):
        main([
            "run", "--mode", "validation",
            "--model-config", str(_template()),
            "--N", "300", "--db-version", "A2.2",
            "--baseline-cycle", "1", "--feature-subset", "fs_cv",
            "--out-root", str(tmp_path),
            "--seeds", "1",
        ])


def test_unknown_subcommand():
    with pytest.raises(SystemExit):
        main(["nope"])


def _template() -> Path:
    return Path(__file__).resolve().parents[1] / "configs" / "rf.yaml"

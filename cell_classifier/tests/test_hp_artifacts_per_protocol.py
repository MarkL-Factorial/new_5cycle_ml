"""Each tuning protocol emits its own HP artifact filename, and the wrong
filename for a given protocol is an error when production tries to load it.

  - tune_inner_cv → best_params_per_seed.csv
  - nested_cv     → best_params_per_fold.csv
  - both          → best_params_summary.json
  - never         → best_params.json (v1 ambiguous filename)
"""

from pathlib import Path

import pandas as pd
import pytest

from cell_classifier.cli import main


_TEMPLATE = Path(__file__).resolve().parents[1] / "configs" / "rf.yaml"


def _run_validation(tmp_path: Path, protocol: str, extra_args: list[str]) -> Path:
    rc = main([
        "run", "--mode", "validation",
        "--model-config", str(_TEMPLATE),
        "--N", "300", "--db-version", "A2.2",
        "--baseline-cycle", "1", "--feature-subset", "fs_cv",
        "--tuning-protocol", protocol,
        "--tune.n-trials", "3", "--tune.inner-cv", "3",
        "--seeds", "1,2",
        "--out-root", str(tmp_path),
        *extra_args,
    ])
    assert rc == 0
    return tmp_path / "runs" / "validation" / "rf__N300__A2.2_b1__fs_cv"


def test_tune_inner_cv_writes_per_seed_csv(tmp_path: Path):
    run = _run_validation(tmp_path, "tune_inner_cv", [])
    assert (run / "best_params_per_seed.csv").exists()
    assert (run / "best_params_summary.json").exists()
    assert not (run / "best_params_per_fold.csv").exists()
    assert not (run / "best_params.json").exists()
    df = pd.read_csv(run / "best_params_per_seed.csv")
    assert "seed" in df.columns
    assert "fold" not in df.columns
    assert len(df) == 2  # 2 seeds


def test_nested_cv_writes_per_fold_csv(tmp_path: Path):
    run = _run_validation(tmp_path, "nested_cv", ["--outer-k", "3"])
    assert (run / "best_params_per_fold.csv").exists()
    assert (run / "best_params_summary.json").exists()
    assert not (run / "best_params_per_seed.csv").exists()
    assert not (run / "best_params.json").exists()
    df = pd.read_csv(run / "best_params_per_fold.csv")
    assert "seed" in df.columns
    assert "fold" in df.columns
    assert len(df) == 2 * 3  # 2 seeds × 3 folds


def test_production_errors_on_protocol_mismatch(tmp_path: Path):
    """If the validation run is tune_inner_cv but its CSV is removed,
    production from_validation_run should fail with a clear message."""
    run = _run_validation(tmp_path, "tune_inner_cv", [])
    (run / "best_params_per_seed.csv").unlink()
    with pytest.raises(FileNotFoundError, match="best_params_per_seed.csv"):
        main([
            "run", "--mode", "production",
            "--model-config", str(_TEMPLATE),
            "--N", "300", "--db-version", "A2.2",
            "--baseline-cycle", "1", "--feature-subset", "fs_cv",
            "--production-params-source", "from_validation_run",
            "--seeds", "1,2",
            "--out-root", str(tmp_path),
        ])

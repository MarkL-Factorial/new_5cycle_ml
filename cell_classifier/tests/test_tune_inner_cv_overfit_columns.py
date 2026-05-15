"""tune_inner_cv emits both train_* and test_* metrics so overfit_* is computable."""

from pathlib import Path

import pandas as pd
import pytest

from cell_classifier.cli import main


_TEMPLATE = Path(__file__).resolve().parents[1] / "configs" / "rf.yaml"


@pytest.fixture
def tune_inner_cv_run(tmp_path: Path) -> Path:
    rc = main([
        "run", "--mode", "validation",
        "--model-config", str(_TEMPLATE),
        "--N", "300", "--db-version", "A2.2",
        "--baseline-cycle", "1", "--feature-subset", "fs_cv",
        "--tuning-protocol", "tune_inner_cv",
        "--tune.n-trials", "3", "--tune.inner-cv", "3",
        "--seeds", "1,2",
        "--out-root", str(tmp_path),
    ])
    assert rc == 0
    return tmp_path / "runs" / "validation" / "rf__N300__A2.2_b1__fs_cv"


def test_train_and_test_metrics_present(tune_inner_cv_run: Path):
    df = pd.read_csv(tune_inner_cv_run / "per_seed_metrics.csv")
    for metric in ("f1", "roc_auc", "accuracy", "precision", "recall"):
        assert f"train_{metric}" in df.columns, f"missing train_{metric}"
        assert f"test_{metric}" in df.columns, f"missing test_{metric}"


def test_overfit_columns_computable(tune_inner_cv_run: Path):
    df = pd.read_csv(tune_inner_cv_run / "per_seed_metrics.csv")
    assert "overfit_f1" in df.columns
    assert "overfit_auc" in df.columns
    # And they should be consistent with train - test
    for _, row in df.iterrows():
        if pd.notna(row["train_f1"]) and pd.notna(row["test_f1"]):
            assert row["overfit_f1"] == pytest.approx(
                row["train_f1"] - row["test_f1"], abs=1e-9,
            )

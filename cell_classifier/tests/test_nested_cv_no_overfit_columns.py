"""nested_cv emits only test_* metrics — no train_* or overfit_* columns.

Rationale (from the design note): in nested CV every cell is held out in
exactly one fold, so there is no single fixed training set to compute
train_* metrics against. Fabricating a train_* mean across K outer-train
slices would be misleading; the columns are omitted entirely.
"""

from pathlib import Path

import pandas as pd
import pytest

from cell_classifier.cli import main


_TEMPLATE = Path(__file__).resolve().parents[1] / "configs" / "rf.yaml"


@pytest.fixture
def nested_cv_run(tmp_path: Path) -> Path:
    rc = main([
        "run", "--mode", "validation",
        "--model-config", str(_TEMPLATE),
        "--N", "300", "--db-version", "A2.2",
        "--baseline-cycle", "1", "--feature-subset", "fs_cv",
        "--tuning-protocol", "nested_cv", "--outer-k", "3",
        "--tune.n-trials", "3", "--tune.inner-cv", "3",
        "--seeds", "1,2",
        "--out-root", str(tmp_path),
    ])
    assert rc == 0
    return tmp_path / "runs" / "validation" / "rf__N300__A2.2_b1__fs_cv"


def test_no_train_columns(nested_cv_run: Path):
    df = pd.read_csv(nested_cv_run / "per_seed_metrics.csv")
    train_cols = [c for c in df.columns if c.startswith("train_")]
    assert train_cols == [], f"nested_cv must not emit train_* cols, got {train_cols}"


def test_no_overfit_columns(nested_cv_run: Path):
    df = pd.read_csv(nested_cv_run / "per_seed_metrics.csv")
    overfit_cols = [c for c in df.columns if c.startswith("overfit_")]
    assert overfit_cols == [], (
        f"nested_cv must not emit overfit_* cols, got {overfit_cols}"
    )


def test_test_columns_still_present(nested_cv_run: Path):
    df = pd.read_csv(nested_cv_run / "per_seed_metrics.csv")
    for metric in ("f1", "roc_auc", "accuracy", "precision", "recall"):
        assert f"test_{metric}" in df.columns, f"missing test_{metric}"

"""Smoke test for the end-to-end pipeline.

Uses N=300 with a tiny seed list + few Optuna trials so the test runs in seconds.
"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest

from ml_classification.models import EBMModelSpec, get_model_spec
from ml_classification.pipeline import run_experiment


def test_get_model_spec_unknown():
    with pytest.raises(KeyError):
        get_model_spec("xgboost")


def test_ebm_stub_raises():
    spec = EBMModelSpec()
    with pytest.raises(NotImplementedError, match="Stage 2"):
        spec.build({})


def test_smoke_rf_n300(tmp_path):
    config = {
        "experiment_name": "rf_n300_smoke",
        "model": "random_forest",
        "N": 300,
        "feature_subset": "fs_cv",
        "seeds": [42, 1729],
        "split": {"train_frac": 0.8, "test_frac": 0.2},
        "tune": {"n_trials": 5, "inner_cv": 3, "optimize": "roc_auc"},
        "out_dir": str(tmp_path / "rf_n300_smoke"),
    }
    summary = run_experiment(config)

    assert summary["n_features"] == 12
    assert summary["test_roc_auc_mean"] > 0.7  # smoke bar; production bar is > 0.75

    out = Path(config["out_dir"])
    assert (out / "per_seed_metrics.csv").exists()
    assert (out / "feature_importance.csv").exists()
    assert (out / "optuna_history.csv").exists()
    assert (out / "best_params.json").exists()
    assert (out / "summary.json").exists()
    assert (out / "model_best.joblib").exists()

    per_seed = pd.read_csv(out / "per_seed_metrics.csv")
    assert len(per_seed) == 2
    assert "val_roc_auc" not in per_seed.columns  # val slice removed
    assert "inner_cv_roc_auc" in per_seed.columns  # added in this round
    for _, row in per_seed.iterrows():
        if not pd.isna(row["test_n_AR"]) and not pd.isna(row["test_n_0MC"]):
            assert row["test_n_AR"] + row["test_n_0MC"] == row["test_n"]

    saved = json.loads((out / "summary.json").read_text())
    assert saved["model"] == "random_forest"
    assert saved["N"] == 300
    assert saved["best_seed_selection"] == "max_inner_cv_roc_auc"
    assert "sklearn_version" in saved
    assert "optuna_version" in saved

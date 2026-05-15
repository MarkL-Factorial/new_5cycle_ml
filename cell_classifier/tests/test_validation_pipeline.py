"""Smoke run: validation tune_inner_cv produces all expected files."""

import json
from pathlib import Path

import pytest

from cell_classifier.cli import main


_TEMPLATE = Path(__file__).resolve().parents[1] / "configs" / "rf.yaml"


@pytest.fixture
def smoke_validation(tmp_path: Path) -> Path:
    rc = main([
        "run", "--mode", "validation",
        "--model-config", str(_TEMPLATE),
        "--N", "300", "--db-version", "A2.2",
        "--baseline-cycle", "1", "--feature-subset", "fs_cv",
        "--tuning-protocol", "tune_inner_cv",
        "--tune.n-trials", "3", "--tune.inner-cv", "3",
        "--seeds", "1,2,3",
        "--out-root", str(tmp_path),
    ])
    assert rc == 0
    return tmp_path / "runs" / "validation" / "rf__N300__A2.2_b1__fs_cv"


def test_smoke_artifacts_present(smoke_validation: Path):
    for f in [
        "per_seed_metrics.csv", "summary.json", "manifest.json",
        "feature_importance.csv", "shap_per_seed.parquet", "shap_summary.csv",
        "optuna_history.csv",
        "best_params_per_seed.csv", "best_params_summary.json",
        "plots/perm_importance.png", "plots/shap_summary.png",
    ]:
        assert (smoke_validation / f).exists(), f"missing {f}"
    # And the *wrong*-protocol artifact must NOT exist
    assert not (smoke_validation / "best_params_per_fold.csv").exists()
    assert not (smoke_validation / "best_params.json").exists()


def test_summary_has_all_metrics(smoke_validation: Path):
    s = json.loads((smoke_validation / "summary.json").read_text())
    for metric in ("f1", "accuracy", "precision", "recall", "roc_auc"):
        assert f"test_{metric}_mean" in s
        assert f"test_{metric}_std" in s


def test_manifest_has_hash(smoke_validation: Path):
    m = json.loads((smoke_validation / "manifest.json").read_text())
    assert "resolved_config_sha256" in m
    assert m["mode"] == "validation"
    assert m["tuning"]["protocol"] == "tune_inner_cv"
    assert m["tuning"]["test_frac"] == 0.2
    assert m["tuning"]["outer_cv_folds"] is None
    assert m["hp_provenance"]["source"] is None
    assert len(m["seeds"]) == 3


def test_idempotency_skip(smoke_validation: Path, tmp_path: Path):
    """Rerunning same axes is a skip."""
    rc = main([
        "run", "--mode", "validation",
        "--model-config", str(_TEMPLATE),
        "--N", "300", "--db-version", "A2.2",
        "--baseline-cycle", "1", "--feature-subset", "fs_cv",
        "--tuning-protocol", "tune_inner_cv",
        "--tune.n-trials", "3", "--tune.inner-cv", "3",
        "--seeds", "1,2,3",
        "--out-root", str(tmp_path),
    ])
    assert rc == 0   # skipped, exit 0


def test_force_overwrites(smoke_validation: Path, tmp_path: Path):
    """--force with different n_trials should overwrite."""
    rc = main([
        "run", "--mode", "validation",
        "--model-config", str(_TEMPLATE),
        "--N", "300", "--db-version", "A2.2",
        "--baseline-cycle", "1", "--feature-subset", "fs_cv",
        "--tuning-protocol", "tune_inner_cv",
        "--tune.n-trials", "5", "--tune.inner-cv", "3",
        "--seeds", "1,2,3",
        "--out-root", str(tmp_path),
        "--force",
    ])
    assert rc == 0

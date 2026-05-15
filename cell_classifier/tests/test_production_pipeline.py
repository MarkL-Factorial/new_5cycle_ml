"""Smoke run: production from_validation_run + retune."""

import json
from pathlib import Path

import pytest

from cell_classifier.cli import main


_TEMPLATE = Path(__file__).resolve().parents[1] / "configs" / "rf.yaml"


def test_production_requires_matching_validation(tmp_path: Path):
    """production with from_validation_run errors when validation is missing."""
    with pytest.raises(FileNotFoundError, match="no matching validation run"):
        main([
            "run", "--mode", "production",
            "--model-config", str(_TEMPLATE),
            "--N", "300", "--db-version", "A2.2",
            "--baseline-cycle", "1", "--feature-subset", "fs_cv",
            "--production-params-source", "from_validation_run",
            "--seeds", "1,2,3",
            "--out-root", str(tmp_path),
        ])


def test_production_with_retune(tmp_path: Path):
    rc = main([
        "run", "--mode", "production",
        "--model-config", str(_TEMPLATE),
        "--N", "300", "--db-version", "A2.2",
        "--baseline-cycle", "1", "--feature-subset", "fs_cv",
        "--production-params-source", "retune",
        "--tune.n-trials", "3", "--tune.inner-cv", "3",
        "--seeds", "1,2,3",
        "--out-root", str(tmp_path),
    ])
    assert rc == 0
    run = tmp_path / "runs" / "production" / "rf__N300__A2.2_b1__fs_cv"
    for f in [
        "predictions.csv", "predictions_per_seed.parquet",
        "feature_importance.csv", "manifest.json",
        "best_params_production.json", "optuna_history.csv",
        "plots/feature_importance.png",
    ]:
        assert (run / f).exists(), f"missing {f}"

    m = json.loads((run / "manifest.json").read_text())
    assert m["mode"] == "production"
    assert m["hp_provenance"]["source"] == "retune"
    assert m["hp_provenance"]["source_run_slug"] is None
    assert m["hp_provenance"]["representative_strategy"] is None
    # No summary.json (production has no metrics)
    assert not (run / "summary.json").exists()
    # No legacy per-seed best_params (production uses one representative set)
    assert not (run / "best_params.json").exists()
    # retune populates optuna_history.csv with the single study's trials
    text = (run / "optuna_history.csv").read_text()
    assert text.strip(), "retune should write trials to optuna_history.csv"


def test_production_from_validation_run(tmp_path: Path):
    """Run validation, then production with from_validation_run."""
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
    rc = main([
        "run", "--mode", "production",
        "--model-config", str(_TEMPLATE),
        "--N", "300", "--db-version", "A2.2",
        "--baseline-cycle", "1", "--feature-subset", "fs_cv",
        "--production-params-source", "from_validation_run",
        "--seeds", "1,2,3",
        "--out-root", str(tmp_path),
    ])
    assert rc == 0
    run = tmp_path / "runs" / "production" / "rf__N300__A2.2_b1__fs_cv"
    m = json.loads((run / "manifest.json").read_text())
    assert m["mode"] == "production"
    assert m["hp_provenance"]["source"] == "from_validation_run"
    assert m["hp_provenance"]["source_run_slug"] == "rf__N300__A2.2_b1__fs_cv"
    assert m["hp_provenance"]["representative_strategy"] == "mode_or_median_per_hp"
    # from_validation_run does NOT re-run Optuna → optuna_history.csv empty
    assert (run / "optuna_history.csv").read_text().strip() == ""
    # The representative HP set is a single dict, not per-seed
    rep = json.loads((run / "best_params_production.json").read_text())
    assert isinstance(rep, dict)
    assert "n_estimators" in rep

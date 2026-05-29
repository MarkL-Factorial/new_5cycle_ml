"""Persistence + predict-pipeline tests.

Covers:
  1. EBMClassifierModel fit -> joblib.dump -> joblib.load -> predict_proba
     is bit-identical (model state survives the roundtrip).
  2. RSFModel fit -> joblib.dump -> joblib.load -> predict +
     predict_survival_curve are bit-identical.
  3. run_predict integration: build a synthetic run dir with persisted
     EBM + RSF + manifest, score a fake cell_features.parquet, assert
     the output CSV has the expected schema and row count.
"""

from __future__ import annotations

import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import polars as pl
import pytest

pytest.importorskip("sksurv")

from cell_lifetime.data.synthetic import make_synthetic_dataset
from cell_lifetime.models.ebm_classifier import EBMClassifierModel
from cell_lifetime.models.rsf import RSFModel


_EBM_PARAMS = {
    "max_bins": 64,
    "max_interaction_bins": 8,
    "interactions": 0,
    "learning_rate": 0.1,
    "min_samples_leaf": 4,
    "max_leaves": 3,
}

_RSF_PARAMS = {
    "n_estimators": 25,
    "max_depth": 6,
    "min_samples_split": 5,
    "min_samples_leaf": 5,
    "max_features": "sqrt",
}


def test_ebm_joblib_roundtrip(tmp_path: Path) -> None:
    ds = make_synthetic_dataset(n_faded=30, n_censored=30, seed=0)
    view = ds.view_for_task("classification")
    model = EBMClassifierModel(_EBM_PARAMS).fit(view.X, view.y_class)
    proba_before = model.predict_proba(view.X)

    dump_path = tmp_path / "ebm.joblib"
    joblib.dump(model, dump_path)
    loaded = joblib.load(dump_path)
    proba_after = loaded.predict_proba(view.X)

    np.testing.assert_array_equal(proba_before, proba_after)


def test_rsf_joblib_roundtrip(tmp_path: Path) -> None:
    ds = make_synthetic_dataset(n_faded=40, n_censored=40, seed=0)
    view = ds.view_for_task("survival")
    model = RSFModel({**_RSF_PARAMS, "low_memory": False, "random_state": 0}).fit(
        view.X, time=view.time, event=view.event,
    )
    risk_before = model.predict(view.X)
    sfs_before = model.predict_survival_curve(view.X)

    dump_path = tmp_path / "rsf.joblib"
    joblib.dump(model, dump_path)
    loaded = joblib.load(dump_path)
    risk_after = loaded.predict(view.X)
    sfs_after = loaded.predict_survival_curve(view.X)

    # RSF parallel n_jobs=10 introduces floating-point dust (~1e-14) on
    # the reduce step. Roundtrip must match within numerical tolerance,
    # not bit-identically.
    np.testing.assert_allclose(risk_before, risk_after, atol=1e-12)
    assert len(sfs_before) == len(sfs_after)
    for sf_b, sf_a in zip(sfs_before, sfs_after):
        np.testing.assert_array_equal(sf_b.x, sf_a.x)
        np.testing.assert_allclose(sf_b.y, sf_a.y, atol=1e-12)


def test_predict_pipeline_smoke(tmp_path: Path) -> None:
    """Build a fake run dir + manifest, score the synthetic features, check schema."""
    ds = make_synthetic_dataset(n_faded=40, n_censored=40, seed=0)
    view_c = ds.view_for_task("classification")
    view_s = ds.view_for_task("survival")
    feature_cols = list(view_c.X.columns)
    n_cells = len(view_c.X)

    ebm = EBMClassifierModel(_EBM_PARAMS).fit(view_c.X, view_c.y_class)
    rsf = RSFModel({**_RSF_PARAMS, "low_memory": False, "random_state": 0}).fit(
        view_s.X, time=view_s.time, event=view_s.event,
    )

    run_dir = tmp_path / "fake_run"
    run_dir.mkdir()
    models_dir = run_dir / "models_fake_run"
    models_dir.mkdir()
    joblib.dump(ebm, models_dir / "ebm_classifier_n300_seed0.joblib")
    joblib.dump(rsf, models_dir / "rsf_seed0.joblib")

    cf = view_c.X.copy()
    cf.insert(0, "cell_name", ds.cell_names[:n_cells])
    cf_path = tmp_path / "cell_features.parquet"
    pl.from_pandas(cf).write_parquet(cf_path)

    manifest = {
        "schema_version": 1,
        "ensemble_seeds": 1,
        "feature_columns": feature_cols,
        "feature_base": "synthetic",
        "extra_feature_blocks": [],
        "n_features": len(feature_cols),
        "horizons": [300],
        "rsf_t_cap": float(view_s.time.max()),
        "inputs": {
            "bundle_name": "synthetic",
            "cell_features_path": str(cf_path),
            "dqdv_v1_path": "",
            "dop_path": "",
        },
        "models": [
            {"head": "ebm_classifier_n300", "horizon": 300, "seed": 0,
             "path": "ebm_classifier_n300_seed0.joblib"},
            {"head": "rsf", "horizon": None, "seed": 0, "path": "rsf_seed0.joblib"},
        ],
    }
    (models_dir / "predict_manifest.json").write_text(json.dumps(manifest))

    from cell_lifetime.pipelines.predict import run_predict
    out_csv = tmp_path / "predictions.csv"
    summary = run_predict(run_dir=run_dir, out_csv=out_csv)

    assert out_csv.exists()
    df = pd.read_csv(out_csv)
    assert len(df) == n_cells
    for col in ("cell_name", "prob_pass_n300", "prob_pass_n300_std",
                "pred_pass_n300", "rsf_median_cycle", "rsf_median_cycle_std"):
        assert col in df.columns, f"missing {col}"
    assert summary["n_cells"] == n_cells
    assert summary["horizons"] == [300]
    assert 0.0 <= summary["pass_rates"][300] <= 1.0

    # Direct sanity: predict.py output must match raw model output on the
    # same X (no v1/dop join in this test → X is just the 12-col view).
    direct_proba = ebm.predict_proba(view_c.X)[:, 1]
    np.testing.assert_allclose(df["prob_pass_n300"].to_numpy(), direct_proba,
                               atol=1e-12)

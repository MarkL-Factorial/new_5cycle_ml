"""End-to-end experiment orchestration.

`run_experiment(config)` is the single entry point used by `run.py`.
For each seed it:
  1. Splits 80/20 train/test (stratified by target).
  2. Tunes hyperparameters with Optuna on the train slice (5-fold inner CV).
  3. Fits the model with best params on the train slice.
  4. Evaluates on train and test (overall + per-cohort).
  5. Computes feature importance on the test slice (30 permutation repeats).

Then aggregates across seeds, persists artifacts, and returns a summary dict.

The "best seed" model is the one whose Optuna study reports the highest
mean inner-CV ROC-AUC — NOT the highest test AUC. Picking by test AUC would
leak the test set into the model-selection step. Inner CV is the proper
criterion: it's computed on train-only data and should be positively but
imperfectly correlated with test AUC.
"""

from __future__ import annotations

import json
import platform
import sys
import time
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import optuna
import pandas as pd
import sklearn
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline

from .data import load_dataset
from .importance import compute_importance
from .metrics import evaluate, prefix
from .models import get_model_spec
from .splits import stratified_split
from .tuning import tune


def _final_estimator(model_spec, best_params: dict, seed: int):
    final = {**model_spec.fixed_params, **best_params, "random_state": seed}
    return Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("model", model_spec.build(final)),
        ]
    )


def _study_to_df(study: optuna.Study, seed: int) -> pd.DataFrame:
    rows = []
    for t in study.trials:
        rows.append(
            {
                "seed": seed,
                "trial_number": t.number,
                "value": t.value,
                "state": t.state.name,
                **{f"param_{k}": v for k, v in t.params.items()},
            }
        )
    return pd.DataFrame(rows)


def run_experiment(config: dict[str, Any]) -> dict[str, Any]:
    out_dir = Path(config["out_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)

    t0 = time.time()

    ds = load_dataset(N=config["N"], feature_subset=config["feature_subset"])
    print(
        f"[pipeline] loaded N={ds.N} subset={config['feature_subset']}: "
        f"{len(ds)} cells, {len(ds.feature_names)} features, "
        f"pass={int(ds.y.sum())} bad={int((1 - ds.y).sum())}"
    )

    model_spec = get_model_spec(config["model"])

    optuna.logging.set_verbosity(optuna.logging.WARNING)

    rows: list[dict[str, Any]] = []
    importance_frames: list[pd.DataFrame] = []
    trial_frames: list[pd.DataFrame] = []
    best_params_per_seed: dict[int, dict] = {}
    inner_cv_score_per_seed: dict[int, float] = {}
    best_inner_cv = -np.inf
    best_model_blob: tuple[int, Any, dict, float] | None = None

    for seed in config["seeds"]:
        seed_t0 = time.time()
        i_tr, i_te = stratified_split(
            ds.y,
            test_frac=config["split"]["test_frac"],
            seed=seed,
        )

        best_params, study = tune(
            model_spec,
            ds.X.iloc[i_tr],
            ds.y[i_tr],
            n_trials=config["tune"]["n_trials"],
            inner_cv=config["tune"]["inner_cv"],
            seed=seed,
            optimize=config["tune"]["optimize"],
        )
        best_params_per_seed[int(seed)] = best_params
        inner_cv_score_per_seed[int(seed)] = float(study.best_value)
        trial_frames.append(_study_to_df(study, seed))

        model = _final_estimator(model_spec, best_params, seed=seed)
        model.fit(ds.X.iloc[i_tr], ds.y[i_tr])

        m_tr = evaluate(model, ds.X.iloc[i_tr], ds.y[i_tr], ds.cohorts[i_tr])
        m_te = evaluate(model, ds.X.iloc[i_te], ds.y[i_te], ds.cohorts[i_te])

        fitted_inner = model.named_steps["model"]
        X_te_imp = pd.DataFrame(
            model.named_steps["imputer"].transform(ds.X.iloc[i_te]),
            columns=ds.feature_names,
            index=ds.X.iloc[i_te].index,
        )
        importance_frames.append(
            compute_importance(
                model_spec, fitted_inner, X_te_imp, ds.y[i_te], ds.feature_names,
                seed=seed,
            ).assign(seed=seed)
        )

        row = {
            "seed": int(seed),
            "inner_cv_roc_auc": float(study.best_value),
            **prefix(m_tr, "train_"),
            **prefix(m_te, "test_"),
            "overfit_auc": m_tr["roc_auc"] - m_te["roc_auc"]
            if not np.isnan(m_tr["roc_auc"]) and not np.isnan(m_te["roc_auc"])
            else float("nan"),
            "best_params": json.dumps(best_params),
        }
        rows.append(row)

        if study.best_value > best_inner_cv:
            best_inner_cv = study.best_value
            best_model_blob = (seed, model, best_params, m_te["roc_auc"])

        print(
            f"[pipeline] seed={seed:>6} "
            f"test_auc={m_te['roc_auc']:.3f} train_auc={m_tr['roc_auc']:.3f} "
            f"inner_cv_auc={study.best_value:.3f} "
            f"({time.time() - seed_t0:.1f}s)"
        )

    per_seed_df = pd.DataFrame(rows)
    per_seed_df.to_csv(out_dir / "per_seed_metrics.csv", index=False)

    importance_all = pd.concat(importance_frames, ignore_index=True)
    importance_agg = (
        importance_all.groupby("feature")
        .agg(
            native_mean=("native_importance", "mean"),
            native_std=("native_importance", "std"),
            perm_mean=("perm_importance_mean", "mean"),
            perm_std=("perm_importance_mean", "std"),
        )
        .reset_index()
        .sort_values("perm_mean", ascending=False)
    )
    importance_agg.to_csv(out_dir / "feature_importance.csv", index=False)

    pd.concat(trial_frames, ignore_index=True).to_csv(
        out_dir / "optuna_history.csv", index=False
    )

    (out_dir / "best_params.json").write_text(
        json.dumps(best_params_per_seed, indent=2)
    )

    if best_model_blob is not None:
        seed, model, params, test_auc_of_best = best_model_blob
        joblib.dump(
            {"model": model, "best_params": params, "best_seed": seed,
             "feature_names": ds.feature_names, "N": ds.N},
            out_dir / "model_best.joblib",
        )

    summary = {
        "experiment_name": config["experiment_name"],
        "model": config["model"],
        "N": config["N"],
        "feature_subset": config["feature_subset"],
        "n_features": len(ds.feature_names),
        "n_cells_trainable": len(ds),
        "n_pass": int(ds.y.sum()),
        "n_bad": int((1 - ds.y).sum()),
        "test_roc_auc_mean": float(per_seed_df["test_roc_auc"].mean()),
        "test_roc_auc_std": float(per_seed_df["test_roc_auc"].std()),
        "test_auc_AR_mean": float(per_seed_df["test_auc_AR"].mean(skipna=True)),
        "test_auc_0MC_mean": float(per_seed_df["test_auc_0MC"].mean(skipna=True)),
        "test_f1_mean": float(per_seed_df["test_f1"].mean()),
        "test_accuracy_mean": float(per_seed_df["test_accuracy"].mean()),
        "overfit_auc_mean": float(per_seed_df["overfit_auc"].mean(skipna=True)),
        "best_seed": int(best_model_blob[0]) if best_model_blob else None,
        "best_seed_selection": "max_inner_cv_roc_auc",
        "best_seed_inner_cv_auc": float(best_inner_cv) if np.isfinite(best_inner_cv) else None,
        "best_seed_test_auc": float(best_model_blob[3]) if best_model_blob else None,
        "runtime_seconds": round(time.time() - t0, 1),
        "sklearn_version": sklearn.__version__,
        "optuna_version": optuna.__version__,
        "python_version": platform.python_version(),
        "config_snapshot": config,
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2, default=str))

    print(
        f"[pipeline] DONE — mean test ROC-AUC = "
        f"{summary['test_roc_auc_mean']:.3f} ± {summary['test_roc_auc_std']:.3f} "
        f"({summary['runtime_seconds']}s)"
    )

    return summary

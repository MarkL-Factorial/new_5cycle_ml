"""Validation-mode pipeline orchestrator.

For each seed:
  - tune_inner_cv protocol: stratified 80/20 split → inner-CV tune → fit →
    evaluate on the held-out test set. Emits both `train_*` and `test_*`
    metrics so `overfit_*` columns can be computed.
  - nested_cv protocol: K-fold outer CV; for each fold tune (inner CV) on
    outer-train, fit, predict on outer-test; concatenate predictions across
    folds so every cell appears exactly once in the test set per seed. Emits
    only `test_*` — there is no fixed training set to compute `train_*` from.

Either way, persists per_seed_metrics.csv, summary.json, feature_importance.csv,
shap_per_seed.parquet, shap_summary.csv, optuna_history.csv (populated for
both protocols), best_params_per_seed.csv OR best_params_per_fold.csv,
best_params_summary.json, manifest.json, plots/.
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any

import numpy as np
import optuna
import pandas as pd

from cell_classifier.data.loader import column_roles_path, load_dataset
from cell_classifier.data.splits import split_validation_tune_inner_cv
from cell_classifier.evaluation.importance import compute_importance
from cell_classifier.evaluation.metrics import (
    evaluate, metrics_from_predictions, prefix,
)
from cell_classifier.evaluation.plots import plot_perm_importance, plot_shap_summary
from cell_classifier.evaluation.shap import aggregate_shap, compute_seed_shap
from cell_classifier.models.registry import get_model_class
from cell_classifier.training.core import train
from cell_classifier.training.nested import nested_cv
from cell_classifier.training.representative import hp_summary, write_hp_csv
from cell_classifier.training.tuning import tune
from cell_classifier.utils.manifest import build_manifest, write_manifest
from cell_classifier.utils.reproducibility import (
    snapshot_inputs, write_resolved_config,
)


def run_validation(config: dict[str, Any], *, out_dir: Path) -> dict[str, Any]:
    t0 = time.time()
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    plots_dir = out_dir / "plots"

    optuna.logging.set_verbosity(optuna.logging.WARNING)

    preprocess_root_override = config.get("data", {}).get("preprocess_root")
    ds = load_dataset(
        N=config["N"],
        feature_subset=config["feature_subset"],
        baseline_cycle=config["baseline_cycle"],
        db_version=config["db_version"],
        preprocess_root=preprocess_root_override,
    )
    snapshot_inputs(
        out_dir,
        source_bundle=ds.source_dir,
        column_roles_yaml=column_roles_path(preprocess_root_override),
    )
    write_resolved_config(out_dir, config)
    labeled = ds.labeled_view()

    tuning = config["tuning"]
    protocol = tuning["protocol"]
    n_trials = int(tuning["n_trials"])
    inner_cv = int(tuning["inner_cv_folds"])
    optimize = str(tuning["optimize_metric"])

    print(
        f"[validation] N={ds.N} db={ds.db_version} b={ds.baseline_cycle} "
        f"subset={config['feature_subset']} protocol={protocol}"
    )
    print(
        f"[validation] {len(labeled)} trainable cells "
        f"(pass={int(labeled.y.sum())} bad={int((1 - labeled.y).sum())})"
    )

    ModelClass = get_model_class(config["model"])
    imputer_strategy = config["preprocessing"]["imputer_strategy"]

    rows: list[dict[str, Any]] = []
    importance_frames: list[pd.DataFrame] = []
    shap_frames: list[pd.DataFrame] = []
    trial_frames: list[pd.DataFrame] = []

    # Per-row HP records. tune_inner_cv: one row per seed. nested_cv: one
    # row per (seed, fold). `hp_template` is captured from the first emitted
    # row and used for the diagnostic summary.
    hp_rows: list[dict[str, Any]] = []
    hp_template: dict[str, Any] | None = None

    seeds = list(config["seeds"])
    for idx, seed in enumerate(seeds, 1):
        seed = int(seed)
        seed_t0 = time.time()
        print(
            f"[validation] seed={seed} ({idx}/{len(seeds)}) starting...",
            flush=True,
        )

        if protocol == "tune_inner_cv":
            test_frac = float(tuning["test_frac"])
            train_idx, test_idx = split_validation_tune_inner_cv(
                labeled.y, test_frac=test_frac, seed=seed,
            )
            X_tr, X_te = labeled.X.iloc[train_idx], labeled.X.iloc[test_idx]
            y_tr, y_te = labeled.y[train_idx], labeled.y[test_idx]
            cohorts_tr, cohorts_te = labeled.cohorts[train_idx], labeled.cohorts[test_idx]
            cells_te = labeled.cell_names[test_idx]

            best_params, study = tune(
                ModelClass, X_tr, y_tr,
                n_trials=n_trials, inner_cv=inner_cv,
                seed=seed, optimize=optimize,
                imputer_strategy=imputer_strategy,
            )
            hp_rows.append({"seed": seed, **best_params})
            if hp_template is None:
                hp_template = dict(best_params)
            trial_frames.append(_study_to_df(study, seed=seed, fold=-1))

            model = train(
                ModelClass, best_params, X_tr, y_tr, seed=seed,
                imputer_strategy=imputer_strategy,
            )

            m_tr = evaluate(model, X_tr, y_tr, cohorts_tr)
            m_te = evaluate(model, X_te, y_te, cohorts_te)

            importance_frames.append(
                compute_importance(model, X_te, y_te, labeled.feature_names, seed=seed)
                .assign(seed=seed)
            )
            shap_frames.append(compute_seed_shap(model, X_te, cells_te, seed=seed))

            row = {
                "seed": seed,
                "fold": -1,
                "inner_cv_score": float(study.best_value),
                "tune_objective": optimize,
                **prefix(m_tr, "train_"),
                **prefix(m_te, "test_"),
                "overfit_auc": (
                    m_tr["roc_auc"] - m_te["roc_auc"]
                    if not (np.isnan(m_tr["roc_auc"]) or np.isnan(m_te["roc_auc"]))
                    else float("nan")
                ),
                "overfit_f1": m_tr["f1"] - m_te["f1"],
            }
            rows.append(row)

        elif protocol == "nested_cv":
            outer_k = int(tuning["outer_cv_folds"])
            result = nested_cv(
                ModelClass, labeled.X, labeled.y,
                outer_k=outer_k, inner_cv=inner_cv,
                n_trials=n_trials, optimize=optimize,
                seed=seed, imputer_strategy=imputer_strategy,
            )
            # one row per outer fold + one aggregate row per seed
            for k in range(outer_k):
                fold_bp = result.per_fold_best_params[k]
                hp_rows.append({"seed": seed, "fold": k, **fold_bp})
                if hp_template is None:
                    hp_template = dict(fold_bp)
                trial_frames.append(
                    _study_to_df(result.per_fold_studies[k], seed=seed, fold=k)
                )
                mask = result.fold_id == k
                m = metrics_from_predictions(
                    result.y_true[mask], result.y_pred[mask],
                    result.y_proba[mask], labeled.cohorts[mask],
                )
                rows.append({
                    "seed": seed, "fold": k,
                    "inner_cv_score": result.per_fold_inner_cv_score[k],
                    "tune_objective": optimize,
                    **prefix(m, "test_"),
                })
            m_overall = metrics_from_predictions(
                result.y_true, result.y_pred, result.y_proba, labeled.cohorts,
            )
            rows.append({
                "seed": seed, "fold": -1,
                "inner_cv_score": float(np.mean(result.per_fold_inner_cv_score)),
                "tune_objective": optimize,
                **prefix(m_overall, "test_"),
            })

            # Diagnostic model on the full labeled set for importance + SHAP.
            # Uses fold 0's HPs purely as a representative — these aggregate
            # frames are summaries, not the per-fold evaluation truth.
            diag_params = result.per_fold_best_params[0]
            diag_model = train(
                ModelClass, diag_params, labeled.X, labeled.y, seed=seed,
                imputer_strategy=imputer_strategy,
            )
            importance_frames.append(
                compute_importance(
                    diag_model, labeled.X, labeled.y, labeled.feature_names,
                    seed=seed,
                ).assign(seed=seed)
            )
            shap_frames.append(
                compute_seed_shap(
                    diag_model, labeled.X, labeled.cell_names, seed=seed,
                )
            )

        else:
            raise ValueError(f"unknown tuning_protocol {protocol!r}")

        print(
            f"[validation] seed={seed} ({idx}/{len(seeds)}) done in "
            f"{time.time() - seed_t0:.1f}s",
            flush=True,
        )

    # --- persist per-seed metrics ---
    per_seed_df = pd.DataFrame(rows)
    per_seed_df.to_csv(out_dir / "per_seed_metrics.csv", index=False)

    # --- aggregate importance ---
    importance_all = pd.concat(importance_frames, ignore_index=True)
    importance_agg = (
        importance_all.groupby("feature").agg(
            native_mean=("native_importance", "mean"),
            native_std=("native_importance", "std"),
            perm_mean=("perm_importance_mean", "mean"),
            perm_std=("perm_importance_mean", "std"),
        ).reset_index().sort_values("perm_mean", ascending=False)
    )
    importance_agg.to_csv(out_dir / "feature_importance.csv", index=False)
    plot_perm_importance(
        importance_agg, plots_dir / "perm_importance.png",
        title=f"Permutation importance — N={ds.N}, {len(config['seeds'])} seeds",
    )

    # --- SHAP ---
    shap_long = pd.concat(shap_frames, ignore_index=True) if shap_frames else pd.DataFrame()
    shap_emitted = not shap_long.empty
    if shap_emitted:
        shap_long.to_parquet(out_dir / "shap_per_seed.parquet", index=False)
        shap_summary = aggregate_shap(shap_long, labeled.feature_names)
        shap_summary.to_csv(out_dir / "shap_summary.csv", index=False)
        plot_shap_summary(
            shap_summary, plots_dir / "shap_summary.png",
            title=f"Mean |SHAP| — N={ds.N}, {len(config['seeds'])} seeds",
        )

    # --- Optuna history (both protocols) + best_params artifacts ---
    if trial_frames:
        pd.concat(trial_frames, ignore_index=True).to_csv(
            out_dir / "optuna_history.csv", index=False,
        )
    else:
        (out_dir / "optuna_history.csv").write_text("")

    if hp_rows and hp_template is not None:
        hp_columns = list(hp_template.keys())
        if protocol == "tune_inner_cv":
            write_hp_csv(
                out_dir / "best_params_per_seed.csv",
                hp_rows, index_columns=["seed"], hp_columns=hp_columns,
            )
        else:
            write_hp_csv(
                out_dir / "best_params_per_fold.csv",
                hp_rows, index_columns=["seed", "fold"], hp_columns=hp_columns,
            )
        hp_only_rows = [{k: r[k] for k in hp_columns} for r in hp_rows]
        summary_dict = hp_summary(hp_only_rows, hp_template)
        (out_dir / "best_params_summary.json").write_text(
            json.dumps(summary_dict, indent=2, default=str) + "\n"
        )

    # --- Summary metrics aggregates ---
    summary = _build_summary(per_seed_df, ds, config, shap_emitted=shap_emitted)
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2, default=str))

    # --- Manifest ---
    preprocess_manifest = _load_preprocess_manifest(ds.source_dir)
    manifest = build_manifest(
        config=config,
        runtime_seconds=time.time() - t0,
        n_cells_labeled_trainable=int(labeled.label_mask.sum()),
        n_cells_scored=int(labeled.label_mask.sum()),
        preprocess_manifest=preprocess_manifest,
        shap_summary_scope="test_set",
    )
    write_manifest(out_dir, manifest)

    print(
        f"[validation] DONE — mean test F1 = {summary['test_f1_mean']:.3f}±{summary['test_f1_std']:.3f}, "
        f"mean test AUC = {summary['test_roc_auc_mean']:.3f}±{summary['test_roc_auc_std']:.3f} "
        f"({manifest['runtime_seconds']}s)"
    )
    return {"status": "ok", "out_dir": str(out_dir), "summary": summary}


def _study_to_df(study, *, seed: int, fold: int) -> pd.DataFrame:
    return pd.DataFrame([
        {"seed": seed, "fold": fold, "trial_number": t.number,
         "value": t.value, "state": t.state.name,
         **{f"param_{k}": v for k, v in t.params.items()}}
        for t in study.trials
    ])


def _build_summary(per_seed_df, ds, config, *, shap_emitted: bool) -> dict[str, Any]:
    """Aggregate per-seed metrics into summary.json keys."""
    tuning = config["tuning"]
    protocol = tuning["protocol"]
    if protocol == "nested_cv":
        agg_df = per_seed_df[per_seed_df["fold"] == -1]
    else:
        agg_df = per_seed_df

    def _mean_std(col: str) -> tuple[float, float]:
        if col not in agg_df.columns:
            return float("nan"), float("nan")
        return float(agg_df[col].mean()), float(agg_df[col].std())

    out: dict[str, Any] = {
        "slug": config["slug"],
        "mode": "validation",
        "tuning_protocol": protocol,
        "model": config["model"],
        "N": ds.N,
        "db_version": ds.db_version,
        "baseline_cycle": ds.baseline_cycle,
        "feature_subset": config["feature_subset"],
        "preprocess_source": str(ds.source_dir),
        "positive_class": "pass (good cell, survived past N cycles)",
        "negative_class": "bad (faded at or before N cycles)",
        "n_features": len(ds.feature_names),
        "n_cells_trainable": int(ds.label_mask.sum()),
        "n_pass": int(ds.y[ds.label_mask].sum()),
        "n_bad": int((1 - ds.y[ds.label_mask]).sum()),
        "n_seeds": len(config["seeds"]),
        "tune_objective": tuning["optimize_metric"],
        "shap_emitted": shap_emitted,
    }
    for metric in ("f1", "accuracy", "precision", "recall", "roc_auc"):
        m, s = _mean_std(f"test_{metric}")
        out[f"test_{metric}_mean"] = m
        out[f"test_{metric}_std"] = s
    # per-cohort AUC + overfit (tune_inner_cv only)
    for col in ("test_auc_AR", "test_auc_0MC"):
        if col in agg_df.columns:
            out[f"{col}_mean"] = float(agg_df[col].mean(skipna=True))
    for col in ("overfit_auc", "overfit_f1"):
        if col in agg_df.columns:
            out[f"{col}_mean"] = float(agg_df[col].mean(skipna=True))
    return out


def _load_preprocess_manifest(bundle_dir: Path) -> dict[str, Any] | None:
    p = bundle_dir / "manifest.json"
    if not p.exists():
        return None
    return json.loads(p.read_text())

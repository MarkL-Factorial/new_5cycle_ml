"""Validation pipeline — task-branched orchestrator.

Phase 1 wires the `classification` and `regression` branches. The
`survival` branch is a stub that Phase 2 / Phase 3 will fill in with
xgb_aft and rsf respectively.

Tuning protocol = `tune_inner_cv` only for Phase 1: stratified 80/20
split → inner-CV Optuna tune → fit → evaluate on held-out test. Nested
CV is a Phase 2+ addition.
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any

import numpy as np
import optuna
import pandas as pd
from sklearn.model_selection import KFold, StratifiedKFold, train_test_split
from sklearn.metrics import (
    f1_score, roc_auc_score, accuracy_score, precision_score, recall_score,
    mean_absolute_error,
)

from cell_lifetime.data.loader import load_dataset, CycleLifeDataset
from cell_lifetime.evaluation.regression_metrics import regression_metrics, prefix
from cell_lifetime.models.registry import get_model_class

try:
    from cell_lifetime.evaluation.survival_metrics import survival_metrics
except ImportError:  # sksurv unavailable
    survival_metrics = None  # type: ignore


def _classification_metrics(
    y_true: np.ndarray, y_pred: np.ndarray, y_proba: np.ndarray | None,
    cohorts: np.ndarray | None,
) -> dict[str, float]:
    out = {
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "n": float(len(y_true)),
    }
    out["roc_auc"] = (
        float(roc_auc_score(y_true, y_proba))
        if y_proba is not None and len(np.unique(y_true)) == 2
        else float("nan")
    )
    if cohorts is not None and y_proba is not None:
        cohorts = np.asarray(cohorts)
        for c in np.unique(cohorts):
            mask = cohorts == c
            if len(np.unique(y_true[mask])) == 2:
                out[f"auc_{c}"] = float(roc_auc_score(y_true[mask], y_proba[mask]))
    return out


def _objective_classification(trial, ModelClass, X, y, inner_cv, optimize, imputer_strategy, target_transform):
    params = ModelClass.suggest_params(trial)
    scores = []
    skf = StratifiedKFold(n_splits=inner_cv, shuffle=True, random_state=trial.number)
    for tr_idx, va_idx in skf.split(X, y):
        m = ModelClass(params, imputer_strategy=imputer_strategy)
        m.fit(X.iloc[tr_idx], y[tr_idx])
        if optimize == "roc_auc":
            proba = m.predict_proba(X.iloc[va_idx])[:, 1]
            scores.append(roc_auc_score(y[va_idx], proba))
        else:
            pred = m.predict(X.iloc[va_idx])
            scores.append(f1_score(y[va_idx], pred, zero_division=0))
    return float(np.mean(scores))


def _objective_regression(trial, ModelClass, X, y, inner_cv, optimize, imputer_strategy, target_transform):
    params = ModelClass.suggest_params(trial)
    scores = []
    kf = KFold(n_splits=inner_cv, shuffle=True, random_state=trial.number)
    for tr_idx, va_idx in kf.split(X):
        m = ModelClass(params, imputer_strategy=imputer_strategy, target_transform=target_transform)
        m.fit(X.iloc[tr_idx], y[tr_idx])
        pred = m.predict(X.iloc[va_idx])
        # optuna maximizes; use -MAE so larger = better
        scores.append(-mean_absolute_error(y[va_idx], pred))
    return float(np.mean(scores))


def _normalize_risk(model, X) -> np.ndarray:
    """Convert a survival model's predict() into risk-positive scores.

    sksurv's concordance_index_censored expects higher score = sooner
    failure. AFT models predict log-time (higher = later failure) — we
    negate. RSF predicts risk directly — we pass through.
    """
    raw = np.asarray(model.predict(X), dtype=float)
    orientation = getattr(model, "risk_orientation", "time_high")
    return -raw if orientation == "time_high" else raw


def _objective_survival(trial, ModelClass, X, time, event, inner_cv, optimize, imputer_strategy):
    """Inner-CV objective for survival models: average C-index across folds."""
    from sklearn.model_selection import StratifiedKFold
    params = ModelClass.suggest_params(trial)
    scores: list[float] = []
    # Stratify by event so each fold has a similar censored/observed mix
    skf = StratifiedKFold(n_splits=inner_cv, shuffle=True, random_state=trial.number)
    for tr_idx, va_idx in skf.split(X, event):
        m = ModelClass(params, imputer_strategy=imputer_strategy)
        m.fit(X.iloc[tr_idx], time[tr_idx], event[tr_idx])
        risk_va = _normalize_risk(m, X.iloc[va_idx])
        # Inline C-index to avoid the survival_metrics import dance
        if survival_metrics is None:
            raise RuntimeError("scikit-survival not installed; cannot run survival task")
        out = survival_metrics(event[va_idx], time[va_idx], risk_va)
        c = out.get("c_index", float("nan"))
        if not np.isnan(c):
            scores.append(c)
    if not scores:
        return float("nan")
    return float(np.mean(scores))


def _tune(task, ModelClass, X, y_or_time, *, n_trials, inner_cv, seed, optimize,
          imputer_strategy, target_transform, event=None):
    sampler = optuna.samplers.TPESampler(seed=seed)
    study = optuna.create_study(direction="maximize", sampler=sampler)
    if task == "classification":
        obj = lambda t: _objective_classification(t, ModelClass, X, y_or_time, inner_cv, optimize, imputer_strategy, target_transform)
    elif task == "regression":
        obj = lambda t: _objective_regression(t, ModelClass, X, y_or_time, inner_cv, optimize, imputer_strategy, target_transform)
    elif task == "survival":
        if event is None:
            raise ValueError("event array required for survival tuning")
        obj = lambda t: _objective_survival(t, ModelClass, X, y_or_time, event, inner_cv, optimize, imputer_strategy)
    else:
        raise ValueError(f"unknown task {task!r}")
    study.optimize(obj, n_trials=n_trials, show_progress_bar=False)
    return study.best_params, study


def run_validation(config: dict[str, Any], *, out_dir: Path) -> dict[str, Any]:
    t0 = time.time()
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    optuna.logging.set_verbosity(optuna.logging.WARNING)

    task = config.get("task", "classification")
    if task == "survival" and survival_metrics is None:
        raise RuntimeError(
            "task='survival' requires scikit-survival; pip install -e .[survival]"
        )

    ds = load_dataset(
        N=config["N"],
        feature_subset=config["feature_subset"],
        baseline_cycle=config["baseline_cycle"],
        db_version=config["db_version"],
        preprocess_root=config.get("data", {}).get("preprocess_root"),
    )
    view = ds.view_for_task(task)
    y, _ = view.task_target(task)

    print(
        f"[validation/{task}] N={view.N} db={view.db_version} b={view.baseline_cycle} "
        f"model={config['model']} subset={config['feature_subset']} "
        f"n_rows={len(view)} n_features={len(view.feature_names)}"
    )

    ModelClass = get_model_class(config["model"])
    if getattr(ModelClass, "task", "classification") != task:
        raise ValueError(
            f"model {config['model']!r} declares task={ModelClass.task!r} but config "
            f"requests task={task!r}"
        )

    tuning = config["tuning"]
    n_trials = int(tuning["n_trials"])
    inner_cv = int(tuning["inner_cv_folds"])
    optimize = str(tuning["optimize_metric"])
    test_frac = float(tuning.get("test_frac", 0.2))
    imputer_strategy = config["preprocessing"]["imputer_strategy"]
    target_transform = config.get("preprocessing", {}).get("target_transform", "log") if task == "regression" else None

    rows: list[dict[str, Any]] = []
    hp_rows: list[dict[str, Any]] = []
    seeds = list(config["seeds"])
    for idx, seed in enumerate(seeds, 1):
        seed = int(seed)
        seed_t0 = time.time()
        print(f"[validation/{task}] seed={seed} ({idx}/{len(seeds)}) starting...", flush=True)

        # Stratify by event for survival, by class for classification, plain otherwise.
        if task == "classification":
            tr_idx, te_idx = train_test_split(
                np.arange(len(view)), test_size=test_frac, random_state=seed,
                stratify=y,
            )
        elif task == "survival":
            tr_idx, te_idx = train_test_split(
                np.arange(len(view)), test_size=test_frac, random_state=seed,
                stratify=view.event,
            )
        else:
            tr_idx, te_idx = train_test_split(
                np.arange(len(view)), test_size=test_frac, random_state=seed,
            )
        X_tr, X_te = view.X.iloc[tr_idx], view.X.iloc[te_idx]
        cohorts_te = view.cohorts[te_idx]

        if task == "survival":
            time_tr, time_te = view.time[tr_idx], view.time[te_idx]
            event_tr, event_te = view.event[tr_idx], view.event[te_idx]
        else:
            y_tr, y_te = y[tr_idx], y[te_idx]

        # Tune
        if task == "survival":
            best_params, study = _tune(
                task, ModelClass, X_tr, time_tr,
                n_trials=n_trials, inner_cv=inner_cv, seed=seed,
                optimize=optimize, imputer_strategy=imputer_strategy,
                target_transform=None, event=event_tr,
            )
        else:
            best_params, study = _tune(
                task, ModelClass, X_tr, y_tr,
                n_trials=n_trials, inner_cv=inner_cv, seed=seed,
                optimize=optimize, imputer_strategy=imputer_strategy,
                target_transform=target_transform,
            )
        hp_rows.append({"seed": seed, **best_params})

        # Final fit + score
        if task == "classification":
            model = ModelClass(best_params, imputer_strategy=imputer_strategy)
            model.fit(X_tr, y_tr)
            pred = model.predict(X_te)
            proba = model.predict_proba(X_te)[:, 1]
            m = _classification_metrics(y_te, pred, proba, cohorts_te)
            head_metric = ("f1", m.get("f1"))
        elif task == "regression":
            model = ModelClass(best_params, imputer_strategy=imputer_strategy, target_transform=target_transform)
            model.fit(X_tr, y_tr)
            pred = model.predict(X_te)
            m = regression_metrics(y_te, pred, cohorts_te)
            head_metric = ("mae", m.get("mae"))
        else:  # survival
            model = ModelClass(best_params, imputer_strategy=imputer_strategy)
            model.fit(X_tr, time_tr, event_tr)
            risk_te = _normalize_risk(model, X_te)
            assert survival_metrics is not None
            m = survival_metrics(event_te, time_te, risk_te, cohorts=cohorts_te)
            head_metric = ("c_index", m.get("c_index"))

        rows.append({
            "seed": seed,
            "inner_cv_score": float(study.best_value) if not np.isnan(study.best_value) else float("nan"),
            "tune_objective": optimize,
            **prefix(m, "test_"),
        })
        print(
            f"[validation/{task}] seed={seed} done in {time.time() - seed_t0:.1f}s — "
            f"test_{head_metric[0]}={head_metric[1]:.3f}",
            flush=True,
        )

    # Persist artifacts
    per_seed_df = pd.DataFrame(rows)
    per_seed_df.to_csv(out_dir / "per_seed_metrics.csv", index=False)

    hp_df = pd.DataFrame(hp_rows)
    hp_df.to_csv(out_dir / "best_params_per_seed.csv", index=False)

    summary = _build_summary(per_seed_df, view, config, task)
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2, default=str))

    manifest = {
        "task": task,
        "model": config["model"],
        "N": view.N,
        "db_version": view.db_version,
        "baseline_cycle": view.baseline_cycle,
        "feature_subset": config["feature_subset"],
        "n_rows": len(view),
        "n_features": len(view.feature_names),
        "n_seeds": len(seeds),
        "tuning_protocol": "tune_inner_cv",
        "n_trials": n_trials,
        "inner_cv_folds": inner_cv,
        "test_frac": test_frac,
        "optimize_metric": optimize,
        "target_transform": target_transform,
        "imputer_strategy": imputer_strategy,
        "runtime_seconds": round(time.time() - t0, 1),
        "preprocess_source": str(view.source_dir),
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }
    (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2))

    print(f"[validation/{task}] DONE — runtime={manifest['runtime_seconds']}s — see {out_dir}")
    return {"status": "ok", "out_dir": str(out_dir), "summary": summary}


def _build_summary(per_seed_df, view, config, task) -> dict[str, Any]:
    out = {
        "slug": config.get("slug", "?"),
        "task": task,
        "mode": "validation",
        "model": config["model"],
        "N": view.N,
        "db_version": view.db_version,
        "baseline_cycle": view.baseline_cycle,
        "feature_subset": config["feature_subset"],
        "n_rows": len(view),
        "n_seeds": int(per_seed_df["seed"].nunique()),
    }
    if task == "classification":
        metric_cols = ("f1", "accuracy", "precision", "recall", "roc_auc")
    elif task == "regression":
        metric_cols = ("mae", "rmse", "r2", "medae")
    else:  # survival
        metric_cols = ("c_index", "auc_at_200", "auc_at_300", "auc_at_400")
    for metric in metric_cols:
        col = f"test_{metric}"
        if col in per_seed_df.columns:
            out[f"{col}_mean"] = float(per_seed_df[col].mean())
            out[f"{col}_std"] = float(per_seed_df[col].std())
    return out

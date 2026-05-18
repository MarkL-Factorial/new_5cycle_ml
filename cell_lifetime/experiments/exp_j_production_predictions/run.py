#!/usr/bin/env python
"""Exp J — production fit & per-cell predictions.

Trains 4 models on the FULL dataset (no train/test split) and emits
per-cell predictions for ALL 415 cells:

  1. ebm_classifier (N=200) — trained on cells where `trainable_n200=True`
                              (= 187 faded ∪ censored with n_regular≥200),
                              fs_a_only (3 features)
  2. ebm_classifier (N=300) — trainable_n300 cells, fs_a_only
  3. ebm_classifier (N=400) — trainable_n400 cells, fs_a_only
  4. rsf                    — trained on all 415 cells, fs_cv (12)

Each model is hyperparameter-tuned via Optuna (30 trials × 5 inner CV).
Out-of-fold (OOF) probabilities for the classifiers are recorded for
all training cells (faded + qualifying censored); cells outside the
training set for that N get OOF=NaN.

Outputs:
  - predictions.csv     — one row per cell (415 rows)
  - best_params.json    — per-model Optuna best params + per-fold CV scores
  - run_logs/full_<UTC>.log — detailed timestamped log

Run from `cell_lifetime/`:
    python experiments/exp_j_production_predictions/run.py
    # or --smoke for a 5-trial × 3-CV smoke test
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time as _time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import optuna
import pandas as pd
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import KFold
from sksurv.metrics import concordance_index_censored

from cell_lifetime.data.loader import load_dataset
from cell_lifetime.models.ebm_classifier import EBMClassifierModel
from cell_lifetime.models.rsf import RSFModel


HERE = Path(__file__).resolve().parent
LOG_DIR = HERE / "run_logs"
LOG_DIR.mkdir(parents=True, exist_ok=True)


# ---------- logging setup ---------------------------------------------------

def setup_logging(level: int = logging.INFO) -> Path:
    """Tee logs to a timestamped file AND stdout. Returns the log path."""
    ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    log_path = LOG_DIR / f"full_{ts}.log"
    root = logging.getLogger()
    root.setLevel(level)
    # Remove any existing handlers (re-runs in the same Python session)
    for h in list(root.handlers):
        root.removeHandler(h)
    fmt = logging.Formatter("%(asctime)s [%(levelname)s] %(message)s",
                            datefmt="%Y-%m-%dT%H:%M:%SZ")
    fh = logging.FileHandler(log_path, mode="w")
    fh.setFormatter(fmt)
    sh = logging.StreamHandler(sys.stdout)
    sh.setFormatter(fmt)
    root.addHandler(fh)
    root.addHandler(sh)
    # Quiet Optuna's per-trial chatter
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    return log_path


# ---------- RSF median-survival extractor (copied from Exp H/I) -------------

def median_survival_from_sf(sf, t_cap: float) -> float:
    times = np.asarray(sf.x, dtype=float)
    surv = np.asarray(sf.y, dtype=float)
    below = np.where(surv <= 0.5)[0]
    if len(below) == 0:
        return float(t_cap)
    return float(times[below[0]])


# ---------- Classifier tune + OOF -------------------------------------------

def _ebm_inner_cv_auc(
    params: dict[str, Any], X: pd.DataFrame, y: np.ndarray,
    inner_cv: int, seed: int,
) -> tuple[float, list[float]]:
    kf = KFold(n_splits=inner_cv, shuffle=True, random_state=seed)
    fold_scores: list[float] = []
    for tr, va in kf.split(X):
        # Guard: a fold may have only one class — skip if so
        if len(np.unique(y[tr])) < 2 or len(np.unique(y[va])) < 2:
            continue
        model = EBMClassifierModel(params)
        model.fit(X.iloc[tr], y[tr])
        prob = model.predict_proba(X.iloc[va])[:, 1]
        fold_scores.append(float(roc_auc_score(y[va], prob)))
    if not fold_scores:
        return float("nan"), []
    return float(np.mean(fold_scores)), fold_scores


def tune_ebm_classifier(
    X: pd.DataFrame, y: np.ndarray, trials: int, inner_cv: int, seed: int,
) -> tuple[dict[str, Any], float, list[float]]:
    sampler = optuna.samplers.TPESampler(seed=seed)
    study = optuna.create_study(direction="maximize", sampler=sampler)

    def objective(trial: optuna.Trial) -> float:
        params = EBMClassifierModel.suggest_params(trial)
        score, _ = _ebm_inner_cv_auc(params, X, y, inner_cv, seed)
        return score

    study.optimize(objective, n_trials=trials, show_progress_bar=False)
    best = dict(study.best_params)
    best_score, fold_scores = _ebm_inner_cv_auc(best, X, y, inner_cv, seed)
    return best, best_score, fold_scores


def oof_probabilities(
    X: pd.DataFrame, y: np.ndarray, params: dict[str, Any],
    inner_cv: int, seed: int,
) -> np.ndarray:
    """Return OOF probabilities aligned with X's row order."""
    kf = KFold(n_splits=inner_cv, shuffle=True, random_state=seed)
    oof = np.full(len(X), np.nan, dtype=float)
    for tr, va in kf.split(X):
        model = EBMClassifierModel(params)
        model.fit(X.iloc[tr], y[tr])
        oof[va] = model.predict_proba(X.iloc[va])[:, 1]
    return oof


# ---------- RSF tune --------------------------------------------------------

def _rsf_inner_cv_cindex(
    params: dict[str, Any],
    X: pd.DataFrame, time_arr: np.ndarray, event_arr: np.ndarray,
    inner_cv: int, seed: int,
) -> tuple[float, list[float]]:
    kf = KFold(n_splits=inner_cv, shuffle=True, random_state=seed)
    fold_scores: list[float] = []
    for tr, va in kf.split(X):
        model = RSFModel(params)
        model.fit(X.iloc[tr], time=time_arr[tr], event=event_arr[tr])
        risk = model.predict(X.iloc[va])
        c, *_ = concordance_index_censored(event_arr[va].astype(bool), time_arr[va], risk)
        fold_scores.append(float(c))
    return float(np.mean(fold_scores)), fold_scores


def tune_rsf(
    X: pd.DataFrame, time_arr: np.ndarray, event_arr: np.ndarray,
    trials: int, inner_cv: int, seed: int,
) -> tuple[dict[str, Any], float, list[float]]:
    sampler = optuna.samplers.TPESampler(seed=seed)
    study = optuna.create_study(direction="maximize", sampler=sampler)

    def objective(trial: optuna.Trial) -> float:
        params = RSFModel.suggest_params(trial)
        try:
            score, _ = _rsf_inner_cv_cindex(
                params, X, time_arr, event_arr, inner_cv, seed
            )
            return score
        except Exception as e:
            logging.warning(f"RSF trial failed: {e}")
            return float("nan")

    study.optimize(objective, n_trials=trials, show_progress_bar=False)
    best = dict(study.best_params)
    best_score, fold_scores = _rsf_inner_cv_cindex(
        best, X, time_arr, event_arr, inner_cv, seed
    )
    return best, best_score, fold_scores


# ---------- Ground-truth pass label per cell -------------------------------

def true_pass_label(event: bool, last_fade: float, n_reg: int, N: int) -> float:
    """Returns 1 (definitively pass), 0 (definitively fail), or NaN (unknown)."""
    if event:
        return 1.0 if last_fade >= N else 0.0
    # censored
    if n_reg >= N:
        return 1.0  # observed to be alive past N → passed
    return float("nan")  # censored before reaching N → unknown


# ---------- Main pipeline ---------------------------------------------------

def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--trials", type=int, default=30)
    parser.add_argument("--inner-cv", type=int, default=5)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--smoke", action="store_true",
                        help="5 trials × 3 inner CV smoke")
    parser.add_argument("--baseline-cycle", type=int, default=1)
    parser.add_argument("--db-version", default="A2.2")
    args = parser.parse_args()

    if args.smoke:
        args.trials = 5
        args.inner_cv = 3

    log_path = setup_logging()
    log = logging.getLogger(__name__)
    log.info(f"Exp J production fit. trials={args.trials}, inner_cv={args.inner_cv}, seed={args.seed}")
    log.info(f"Log path: {log_path}")

    # ---- Load dataset for RSF (fs_cv, N=300 placeholder — N is not used by survival) ----
    log.info("Loading dataset for RSF (fs_cv)…")
    ds_cv = load_dataset(
        N=300, feature_subset="fs_cv",
        baseline_cycle=args.baseline_cycle, db_version=args.db_version,
    )
    n_total = len(ds_cv)
    n_faded = int(ds_cv.event.sum())
    n_censored = int((~ds_cv.event).sum())
    log.info(
        f"Dataset: {n_total} cells = {n_faded} faded + {n_censored} censored; "
        f"fs_cv has {ds_cv.X.shape[1]} cols"
    )

    X_cv_all = ds_cv.X.reset_index(drop=True)
    event_all = ds_cv.event.astype(bool)
    time_all = ds_cv.time.astype(np.int64)
    y_cycle_all = ds_cv.y_cycle.astype(float)
    cell_names = ds_cv.cell_names

    # ---- Train 3 classifiers using trainable_n{N} masks --------------------
    # For each N, the upstream `trainable_n{N}` mask selects cells with
    # definitive labels: all 187 faded cells + censored cells where
    # n_regular >= N (definitively pass). The loader populates
    # `ds_N.label_mask` and `ds_N.y_class` for the requested N directly.
    best_params: dict[str, Any] = {}
    classifier_predictions: dict[int, dict[str, np.ndarray]] = {}
    in_training_set: dict[int, np.ndarray] = {}

    for N in (200, 300, 400):
        log.info(f"=== ebm_classifier × fs_a_only × N={N} ===")
        ds_N = load_dataset(
            N=N, feature_subset="fs_a_only",
            baseline_cycle=args.baseline_cycle, db_version=args.db_version,
        )
        # Cross-load invariants — same cells in same order regardless of N
        assert (ds_N.cell_names == cell_names).all(), \
            f"N={N} loader returned different cell ordering than the fs_cv load"
        assert (ds_N.event == event_all).all(), \
            f"N={N} loader returned different event array than the fs_cv load"

        X_a_all = ds_N.X.reset_index(drop=True)
        if N == 200:
            log.info(f"  fs_a_only has {X_a_all.shape[1]} cols")
        label_mask = ds_N.label_mask.astype(bool)
        y_class_all = ds_N.y_class.astype(np.int8)

        train_idx = np.where(label_mask)[0]
        X_train = X_a_all.iloc[train_idx].reset_index(drop=True)
        y_train = y_class_all[train_idx]
        n_train = len(train_idx)
        n_train_faded = int((event_all[train_idx]).sum())
        n_train_censored = n_train - n_train_faded
        n_pass = int((y_train == 1).sum())
        n_fail = int((y_train == 0).sum())
        log.info(
            f"  trainable_n{N}: {n_train} cells = {n_train_faded} faded + "
            f"{n_train_censored} censored"
        )
        log.info(f"  Labels at N={N}: pass={n_pass}, fail={n_fail}")
        in_training_set[N] = label_mask

        t0 = _time.time()
        log.info(f"  Tuning ({args.trials} trials × {args.inner_cv} inner CV, ROC-AUC objective)…")
        best, auc_best, fold_aucs = tune_ebm_classifier(
            X_train, y_train, args.trials, args.inner_cv, args.seed
        )
        tune_t = _time.time() - t0
        log.info(f"  Best ROC-AUC = {auc_best:.4f} (folds: {[round(s,4) for s in fold_aucs]})")
        log.info(f"  Best params: {best}")
        log.info(f"  Tune time: {tune_t:.1f}s")

        log.info(f"  Computing OOF probabilities ({args.inner_cv}-fold w/ best params)…")
        t0 = _time.time()
        oof_train = oof_probabilities(X_train, y_train, best, args.inner_cv, args.seed)
        log.info(f"  OOF mean prob = {np.nanmean(oof_train):.4f}, OOF time: {_time.time()-t0:.1f}s")

        log.info(f"  Refitting on all {n_train} trainable_n{N} cells…")
        t0 = _time.time()
        model = EBMClassifierModel(best)
        model.fit(X_train, y_train)
        log.info(f"  Refit time: {_time.time()-t0:.1f}s")

        log.info(f"  Predicting on all {n_total} cells…")
        prob = model.predict_proba(X_a_all)[:, 1]
        pred = (prob >= 0.5).astype(np.int8)
        log.info(f"  Predicted pass rate: {pred.mean():.4f}, mean prob: {prob.mean():.4f}")

        # Build OOF aligned to all-cells order: training cells get their fold
        # probability; non-training cells get NaN.
        oof_full = np.full(n_total, np.nan, dtype=float)
        oof_full[train_idx] = oof_train

        classifier_predictions[N] = {"prob": prob, "pred": pred, "oof": oof_full}
        best_params[f"ebm_classifier_n{N}"] = {
            "params": best,
            "inner_cv_auc_mean": auc_best,
            "inner_cv_auc_per_fold": fold_aucs,
            "n_training_cells": n_train,
            "n_training_faded": n_train_faded,
            "n_training_censored": n_train_censored,
            "n_features": int(X_train.shape[1]),
            "label_n_pass": n_pass,
            "label_n_fail": n_fail,
            "tune_time_s": tune_t,
        }

    # ---- Train RSF on all cells -------------------------------------------
    log.info(f"=== rsf × fs_cv (all {n_total} cells) ===")
    t0 = _time.time()
    log.info(f"  Tuning ({args.trials} trials × {args.inner_cv} inner CV, C-index objective)…")
    rsf_best, rsf_cindex, rsf_folds = tune_rsf(
        X_cv_all, time_all, event_all, args.trials, args.inner_cv, args.seed
    )
    tune_t = _time.time() - t0
    log.info(f"  Best C-index = {rsf_cindex:.4f} (folds: {[round(s,4) for s in rsf_folds]})")
    log.info(f"  Best params: {rsf_best}")
    log.info(f"  Tune time: {tune_t:.1f}s")

    log.info(f"  Refitting on all {n_total} cells with low_memory=False…")
    t0 = _time.time()
    rsf_model = RSFModel({**rsf_best, "low_memory": False})
    rsf_model.fit(X_cv_all, time=time_all, event=event_all)
    log.info(f"  Refit time: {_time.time()-t0:.1f}s")

    log.info(f"  Extracting median survival on all {n_total} cells…")
    t0 = _time.time()
    sfs = rsf_model.predict_survival_curve(X_cv_all)
    t_cap = float(time_all.max())
    rsf_median = np.array([median_survival_from_sf(sf, t_cap) for sf in sfs], dtype=float)
    log.info(f"  Median-survival mean: {rsf_median.mean():.2f}, "
             f"range [{rsf_median.min():.0f}, {rsf_median.max():.0f}], "
             f"t_cap={t_cap:.0f}, time: {_time.time()-t0:.1f}s")

    best_params["rsf"] = {
        "params": rsf_best,
        "inner_cv_cindex_mean": rsf_cindex,
        "inner_cv_cindex_per_fold": rsf_folds,
        "n_training_cells": int(n_total),
        "n_features": int(X_cv_all.shape[1]),
        "t_cap": t_cap,
        "tune_time_s": tune_t,
    }

    # ---- Assemble predictions.csv -----------------------------------------
    log.info(f"=== Assembling predictions.csv ({n_total} rows) ===")
    # Per-cell ground-truth label arrays
    true_pass: dict[int, np.ndarray] = {}
    for N in (200, 300, 400):
        true_pass[N] = np.array([
            true_pass_label(
                bool(event_all[i]),
                float(y_cycle_all[i]) if event_all[i] else float("nan"),
                int(time_all[i]),  # for censored, time=n_regular
                N,
            )
            for i in range(n_total)
        ])

    # The OOF arrays from `classifier_predictions[N]["oof"]` are already
    # aligned to all-cells row order (NaN for cells outside training_n{N}).

    df = pd.DataFrame({
        "cell_name": cell_names,
        "status": np.where(event_all, "faded", "censored"),
        "event": event_all,
        "last_fade_cycle": np.where(event_all, y_cycle_all, np.nan),
        "n_regular": time_all,  # = n_regular for censored, = last_fade for faded
        "time": time_all,
        "true_pass_n200": true_pass[200],
        "true_pass_n300": true_pass[300],
        "true_pass_n400": true_pass[400],
        "in_training_set_n200": in_training_set[200],
        "in_training_set_n300": in_training_set[300],
        "in_training_set_n400": in_training_set[400],
        "pred_pass_n200": classifier_predictions[200]["pred"],
        "pred_pass_n300": classifier_predictions[300]["pred"],
        "pred_pass_n400": classifier_predictions[400]["pred"],
        "prob_pass_n200": classifier_predictions[200]["prob"],
        "prob_pass_n300": classifier_predictions[300]["prob"],
        "prob_pass_n400": classifier_predictions[400]["prob"],
        "oof_prob_pass_n200": classifier_predictions[200]["oof"],
        "oof_prob_pass_n300": classifier_predictions[300]["oof"],
        "oof_prob_pass_n400": classifier_predictions[400]["oof"],
        "rsf_median_cycle": rsf_median,
    })

    # ---- Sanity checks -----------------------------------------------------
    # Pass rate should drop as N grows (harder threshold)
    pass_rates = {N: float(df[f"pred_pass_n{N}"].mean()) for N in (200, 300, 400)}
    log.info(f"  Pass rates (predicted): N=200 → {pass_rates[200]:.3f}, "
             f"N=300 → {pass_rates[300]:.3f}, N=400 → {pass_rates[400]:.3f}")
    if not (pass_rates[200] >= pass_rates[300] >= pass_rates[400]):
        log.warning("  Pass rates are NOT monotone decreasing in N — investigate")

    assert (rsf_median > 0).all(), "rsf_median has non-positive values"
    log.info(f"  All {n_total} RSF medians > 0 ✓")

    out_csv = HERE / "predictions.csv"
    df.to_csv(out_csv, index=False)
    log.info(f"  Wrote {out_csv} ({len(df)} rows × {len(df.columns)} cols)")

    # ---- Write best_params.json -------------------------------------------
    out_json = HERE / "best_params.json"
    with out_json.open("w") as f:
        json.dump({
            "experiment": "exp_j_production_predictions",
            "feature_subsets": {"classifier": "fs_a_only", "rsf": "fs_cv"},
            "baseline_cycle": args.baseline_cycle,
            "db_version": args.db_version,
            "trials": args.trials,
            "inner_cv": args.inner_cv,
            "seed": args.seed,
            "n_total_cells": n_total,
            "n_faded": n_faded,
            "n_censored": n_censored,
            "models": best_params,
        }, f, indent=2, default=str)
    log.info(f"  Wrote {out_json}")

    log.info("Exp J complete.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

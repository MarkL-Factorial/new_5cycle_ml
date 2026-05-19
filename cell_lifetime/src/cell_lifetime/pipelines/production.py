"""Production orchestrator: 3 classifiers (N=200/300/400) + 1 RSF.

Called by the `cell-lifetime production` CLI subcommand. Trains the
canonical battery-of-models on the full trainable_n{N} cohort (per
CONVENTIONS.md), runs inference on all 415 cells, and writes a wide
`predictions.csv` plus best-params JSON, log, and 3 plots to a
timestamped run directory.

Default seed strategy is K=5 INDEPENDENT ensembling: each ensemble
member runs its own Optuna study (different inner-CV partition →
different best params), then refits with that seed's params. Per-cell
predictions are the mean across the K models; `_std` columns capture
ensemble dispersion.

Single-fit fallback (K=1) is supported for smoke/debug.

Importable as `run_production(...)`. The CLI builds the kwargs and
calls in.
"""

from __future__ import annotations

import json
import logging
import sys
import time as _time
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
from cell_lifetime.pipelines import production_plots


NS = (200, 300, 400)


# ---------- logging setup ---------------------------------------------------

def setup_logging(out_dir: Path) -> Path:
    log_path = out_dir / "run.log"
    root = logging.getLogger()
    root.setLevel(logging.INFO)
    for h in list(root.handlers):
        root.removeHandler(h)
    fmt = logging.Formatter(
        "%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%Y-%m-%dT%H:%M:%SZ",
    )
    fh = logging.FileHandler(log_path, mode="w")
    fh.setFormatter(fmt)
    sh = logging.StreamHandler(sys.stdout)
    sh.setFormatter(fmt)
    root.addHandler(fh)
    root.addHandler(sh)
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    return log_path


# ---------- RSF median-survival extraction ----------------------------------

def median_survival_from_sf(sf, t_cap: float) -> float:
    times = np.asarray(sf.x, dtype=float)
    surv = np.asarray(sf.y, dtype=float)
    below = np.where(surv <= 0.5)[0]
    if len(below) == 0:
        return float(t_cap)
    return float(times[below[0]])


# ---------- Classifier tune + OOF + refit ----------------------------------

def _ebm_inner_cv_auc(
    params: dict[str, Any], X: pd.DataFrame, y: np.ndarray,
    inner_cv: int, seed: int,
) -> tuple[float, list[float]]:
    kf = KFold(n_splits=inner_cv, shuffle=True, random_state=seed)
    fold_scores: list[float] = []
    for tr, va in kf.split(X):
        if len(np.unique(y[tr])) < 2 or len(np.unique(y[va])) < 2:
            continue
        model = EBMClassifierModel(params)
        model.fit(X.iloc[tr], y[tr])
        prob = model.predict_proba(X.iloc[va])[:, 1]
        fold_scores.append(float(roc_auc_score(y[va], prob)))
    if not fold_scores:
        return float("nan"), []
    return float(np.mean(fold_scores)), fold_scores


def _tune_ebm(
    X: pd.DataFrame, y: np.ndarray, trials: int, inner_cv: int, seed: int,
) -> tuple[dict[str, Any], float, list[float]]:
    sampler = optuna.samplers.TPESampler(seed=seed)
    study = optuna.create_study(direction="maximize", sampler=sampler)
    study.optimize(
        lambda trial: _ebm_inner_cv_auc(
            EBMClassifierModel.suggest_params(trial), X, y, inner_cv, seed,
        )[0],
        n_trials=trials, show_progress_bar=False,
    )
    best = dict(study.best_params)
    best_score, fold_scores = _ebm_inner_cv_auc(best, X, y, inner_cv, seed)
    return best, best_score, fold_scores


def _oof_probabilities(
    X: pd.DataFrame, y: np.ndarray, params: dict[str, Any],
    inner_cv: int, seed: int,
) -> np.ndarray:
    kf = KFold(n_splits=inner_cv, shuffle=True, random_state=seed)
    oof = np.full(len(X), np.nan, dtype=float)
    for tr, va in kf.split(X):
        model = EBMClassifierModel(params)
        model.fit(X.iloc[tr], y[tr])
        oof[va] = model.predict_proba(X.iloc[va])[:, 1]
    return oof


# ---------- RSF tune --------------------------------------------------------

def _rsf_inner_cv_cindex(
    params: dict[str, Any], X: pd.DataFrame,
    time_arr: np.ndarray, event_arr: np.ndarray,
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


def _tune_rsf(
    X: pd.DataFrame, time_arr: np.ndarray, event_arr: np.ndarray,
    trials: int, inner_cv: int, seed: int,
) -> tuple[dict[str, Any], float, list[float]]:
    sampler = optuna.samplers.TPESampler(seed=seed)
    study = optuna.create_study(direction="maximize", sampler=sampler)

    def objective(trial: optuna.Trial) -> float:
        params = RSFModel.suggest_params(trial)
        try:
            return _rsf_inner_cv_cindex(
                params, X, time_arr, event_arr, inner_cv, seed,
            )[0]
        except Exception as e:
            logging.warning(f"RSF trial failed: {e}")
            return float("nan")

    study.optimize(objective, n_trials=trials, show_progress_bar=False)
    best = dict(study.best_params)
    best_score, fold_scores = _rsf_inner_cv_cindex(
        best, X, time_arr, event_arr, inner_cv, seed,
    )
    return best, best_score, fold_scores


# ---------- Overfit diagnostics --------------------------------------------

def _log_overfit_diagnostics(log, df: pd.DataFrame,
                             best_params: dict[str, Any], K: int) -> None:
    """Emit overfit-spotting diagnostics to the log.

    Three signals:
      1. Spread of per-seed inner-CV scores per model — tight clusters
         indicate hyperparameter choice is robust to CV partition.
      2. Hyperparameter agreement across seeds — wildly different best
         params per seed = the optimum is poorly identified.
      3. Per-cell prediction std — high `_std` cells are uncertain.
    """
    if K == 1:
        log.info(
          "  K=1: no ensemble diagnostics (single deterministic fit)"
        )
        return

    # 1. Per-seed inner-CV score spread (classifiers + RSF)
    for N in NS:
        per_seed = best_params[f"ebm_classifier_n{N}"]["inner_cv_auc_per_seed"]
        log.info(
            f"  AUC per seed N={N}: mean={np.mean(per_seed):.4f} ± "
            f"std={np.std(per_seed, ddof=1) if K > 1 else 0:.4f} "
            f"(range {min(per_seed):.4f} - {max(per_seed):.4f})"
        )
    per_seed_c = best_params["rsf"]["inner_cv_cindex_per_seed"]
    log.info(
        f"  C-index per seed (rsf): mean={np.mean(per_seed_c):.4f} ± "
        f"std={np.std(per_seed_c, ddof=1) if K > 1 else 0:.4f} "
        f"(range {min(per_seed_c):.4f} - {max(per_seed_c):.4f})"
    )

    # 2. Hyperparameter spread per model (just key numeric params)
    for key in (f"ebm_classifier_n{N}" for N in NS):
        sets = best_params[key]["best_params_per_seed"]
        # Pick a couple of representative numeric hyperparams
        for pname in ("max_bins", "max_interaction_bins", "interactions",
                      "learning_rate", "max_leaves"):
            vals = [s.get(pname) for s in sets if s.get(pname) is not None]
            if not vals:
                continue
            if isinstance(vals[0], float):
                summary = (
                    f"{min(vals):.4g}-{max(vals):.4g} "
                    f"(med {np.median(vals):.4g})"
                )
            else:
                summary = f"{min(vals)}-{max(vals)} (med {int(np.median(vals))})"
            log.info(f"  {key} hyperparams[{pname}] across seeds: {summary}")
    # RSF hyperparams
    sets = best_params["rsf"]["best_params_per_seed"]
    for pname in ("n_estimators", "max_depth", "min_samples_split",
                  "min_samples_leaf"):
        vals = [s.get(pname) for s in sets if s.get(pname) is not None]
        if not vals:
            continue
        summary = f"{min(vals)}-{max(vals)} (med {int(np.median(vals))})"
        log.info(f"  rsf hyperparams[{pname}] across seeds: {summary}")
    mx = [s.get("max_features") for s in sets]
    log.info(f"  rsf hyperparams[max_features] across seeds: {mx}")

    # 3. Per-cell std distribution: how stable are individual predictions?
    for N in NS:
        col = f"prob_pass_n{N}_std"
        s = df[col].dropna()
        if len(s) == 0:
            continue
        n_high = int((s > 0.10).sum())
        log.info(
            f"  prob_pass_n{N}_std: mean={s.mean():.4f}, "
            f"p50={s.median():.4f}, p90={s.quantile(0.90):.4f}, "
            f"p99={s.quantile(0.99):.4f}; cells with std>0.10: {n_high}/{len(s)}"
        )
    s = df["rsf_median_cycle_std"]
    n_high = int((s > 50).sum())
    log.info(
        f"  rsf_median_cycle_std: mean={s.mean():.1f}, "
        f"p50={s.median():.1f}, p90={s.quantile(0.90):.1f}, "
        f"p99={s.quantile(0.99):.1f}; cells with std>50 cyc: {n_high}/{len(s)}"
    )

    log.info(
        "  Read: tighter inner-CV score spread + smaller per-cell std => "
        "ensemble is more robust to hyperparameter choice (lower overfit risk)."
    )


# ---------- Ground-truth pass labels ---------------------------------------

def _true_pass(event: bool, last_fade: float, n_reg: int, N: int) -> float:
    if event:
        return 1.0 if last_fade >= N else 0.0
    if n_reg >= N:
        return 1.0
    return float("nan")


# ---------- Main entry point ----------------------------------------------

def run_production(
    *,
    out_dir: Path,
    trials: int = 30,
    inner_cv: int = 5,
    ensemble_seeds: int = 5,
    baseline_cycle: int = 1,
    db_version: str = "A2.2",
    classifier_feature_subset: str = "fs_a_only",
    rsf_feature_subset: str = "fs_cv",
    make_plots: bool = True,
) -> dict[str, Any]:
    """Production fit: 3 classifiers + 1 RSF, K-seed independent ensemble.

    Writes:
      - out_dir / "predictions.csv"
      - out_dir / "best_params.json"
      - out_dir / "run.log"
      - (if make_plots) 3 PNGs

    Returns a summary dict with run config + key metrics.
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    log_path = setup_logging(out_dir)
    log = logging.getLogger(__name__)
    log.info(
        f"Production fit. ensemble_seeds={ensemble_seeds}, trials={trials}, "
        f"inner_cv={inner_cv}, baseline_cycle={baseline_cycle}, "
        f"db_version={db_version}, classifier_fs={classifier_feature_subset}, "
        f"rsf_fs={rsf_feature_subset}"
    )
    log.info(f"Out dir: {out_dir}")
    log.info(f"Log path: {log_path}")

    K = max(int(ensemble_seeds), 1)
    seeds = list(range(K))

    # ---- Load RSF data once ------------------------------------------------
    # Production load uses min_n_regular=5 so we PREDICT on cells with
    # n_regular≥5 (per CONVENTIONS.md). Cells with n_regular<6 will be
    # filtered OUT of training masks below — they only get predictions.
    log.info(f"Loading RSF dataset (feature_subset={rsf_feature_subset}, min_n_regular=5)…")
    ds_cv = load_dataset(
        N=300, feature_subset=rsf_feature_subset,
        baseline_cycle=baseline_cycle, db_version=db_version,
        min_n_regular=5,
    )
    n_total = len(ds_cv)
    n_faded = int(ds_cv.event.sum())
    n_censored = int((~ds_cv.event).sum())
    log.info(
        f"Dataset: {n_total} cells = {n_faded} faded + {n_censored} censored; "
        f"RSF features: {ds_cv.X.shape[1]}"
    )
    X_cv_all = ds_cv.X.reset_index(drop=True)
    event_all = ds_cv.event.astype(bool)
    time_all = ds_cv.time.astype(np.int64)
    y_cycle_all = ds_cv.y_cycle.astype(float)
    cell_names = ds_cv.cell_names
    n_regular_all = ds_cv.n_regular.astype(np.int64)

    # Training mask for tasks that require n_regular>=6 (the canonical
    # cell_lifetime training cutoff). Inference happens on the full
    # n_total cells; only training is restricted.
    train_eligible = (n_regular_all >= 6)
    log.info(
        f"Training-eligible cells (n_regular>=6): {int(train_eligible.sum())}/"
        f"{n_total}; inference-only (n_regular=5): {int((~train_eligible).sum())}"
    )

    # ---- Per-N classifier training (independent K-seed ensemble) ----------
    best_params: dict[str, Any] = {}
    classifier_predictions: dict[int, dict[str, np.ndarray]] = {}
    in_training_set: dict[int, np.ndarray] = {}

    for N in NS:
        log.info(f"=== ebm_classifier × {classifier_feature_subset} × N={N} ===")
        ds_N = load_dataset(
            N=N, feature_subset=classifier_feature_subset,
            baseline_cycle=baseline_cycle, db_version=db_version,
            min_n_regular=5,
        )
        assert (ds_N.cell_names == cell_names).all(), \
            f"N={N} loader returned different cell ordering than RSF load"
        assert (ds_N.event == event_all).all(), \
            f"N={N} loader returned different event array than RSF load"
        assert (ds_N.n_regular == n_regular_all).all(), \
            f"N={N} loader returned different n_regular than RSF load"

        X_all_N = ds_N.X.reset_index(drop=True)
        label_mask = ds_N.label_mask.astype(bool)
        y_class = ds_N.y_class.astype(np.int8)

        # Training requires BOTH trainable_n{N} (definitive label) AND
        # n_regular>=6 (stable feature signature). Cells failing either
        # only get predictions, not OOF.
        train_mask = label_mask & train_eligible
        train_idx = np.where(train_mask)[0]
        X_train = X_all_N.iloc[train_idx].reset_index(drop=True)
        y_train = y_class[train_idx]
        n_train = len(train_idx)
        n_train_faded = int((event_all[train_idx]).sum())
        n_train_censored = n_train - n_train_faded
        n_dropped_low_nreg = int(
            (label_mask & ~train_eligible).sum()
        )
        log.info(
            f"  trainable_n{N} ∩ n_reg≥6: {n_train} cells = {n_train_faded} faded + "
            f"{n_train_censored} censored (pass={int((y_train==1).sum())}, "
            f"fail={int((y_train==0).sum())}); "
            f"dropped {n_dropped_low_nreg} labeled cells with n_reg<6"
        )
        in_training_set[N] = train_mask

        # K ensemble members
        member_probs: list[np.ndarray] = []
        member_oofs: list[np.ndarray] = []
        member_bests: list[dict[str, Any]] = []
        member_aucs: list[float] = []
        for k in seeds:
            t0 = _time.time()
            log.info(f"  [k={k}] tune ({trials} trials × {inner_cv} CV)…")
            best, auc_best, fold_aucs = _tune_ebm(
                X_train, y_train, trials, inner_cv, seed=k,
            )
            log.info(
                f"  [k={k}] best AUC = {auc_best:.4f} "
                f"(folds: {[round(s, 4) for s in fold_aucs]}), "
                f"best params: {best}, t={_time.time()-t0:.1f}s"
            )

            oof_train = _oof_probabilities(X_train, y_train, best, inner_cv, seed=k)
            oof_full = np.full(n_total, np.nan, dtype=float)
            oof_full[train_idx] = oof_train
            member_oofs.append(oof_full)

            model = EBMClassifierModel(best)
            model.fit(X_train, y_train)
            prob = model.predict_proba(X_all_N)[:, 1]
            member_probs.append(prob)
            member_bests.append(best)
            member_aucs.append(auc_best)

        prob_stack = np.stack(member_probs, axis=0)  # (K, n_total)
        oof_stack = np.stack(member_oofs, axis=0)     # (K, n_total)
        prob_mean = prob_stack.mean(axis=0)
        prob_std = prob_stack.std(axis=0, ddof=0) if K > 1 else np.zeros(n_total)
        oof_mean = np.nanmean(oof_stack, axis=0)
        oof_std = (
            np.nanstd(oof_stack, axis=0, ddof=0) if K > 1
            else np.zeros(n_total)
        )
        pred = (prob_mean >= 0.5).astype(np.int8)

        log.info(
            f"  ensemble pass rate: {pred.mean():.4f}, mean prob: {prob_mean.mean():.4f}, "
            f"per-cell prob std mean: {np.nanmean(prob_std):.4f}"
        )

        classifier_predictions[N] = {
            "prob_mean": prob_mean,
            "prob_std": prob_std,
            "oof_mean": oof_mean,
            "oof_std": oof_std,
            "pred": pred,
        }
        best_params[f"ebm_classifier_n{N}"] = {
            "ensemble_seeds": K,
            "best_params_per_seed": member_bests,
            "inner_cv_auc_per_seed": member_aucs,
            "inner_cv_auc_mean": float(np.mean(member_aucs)),
            "n_training_cells": n_train,
            "n_training_faded": n_train_faded,
            "n_training_censored": n_train_censored,
            "n_features": int(X_train.shape[1]),
        }

    # ---- RSF training (independent K-seed ensemble) -----------------------
    # Trained on cells with n_regular≥6 only; predicts on all loaded cells.
    rsf_train_idx = np.where(train_eligible)[0]
    n_rsf_train = len(rsf_train_idx)
    X_rsf_train = X_cv_all.iloc[rsf_train_idx].reset_index(drop=True)
    time_rsf_train = time_all[rsf_train_idx]
    event_rsf_train = event_all[rsf_train_idx]
    log.info(
        f"=== rsf × {rsf_feature_subset} (train on {n_rsf_train} n_reg≥6 cells; "
        f"predict on all {n_total}) ==="
    )
    rsf_medians: list[np.ndarray] = []
    rsf_bests: list[dict[str, Any]] = []
    rsf_cindices: list[float] = []
    for k in seeds:
        t0 = _time.time()
        log.info(f"  [k={k}] tune ({trials} trials × {inner_cv} CV)…")
        best, cindex_best, fold_cs = _tune_rsf(
            X_rsf_train, time_rsf_train, event_rsf_train, trials, inner_cv, seed=k,
        )
        log.info(
            f"  [k={k}] best C-index = {cindex_best:.4f} "
            f"(folds: {[round(s, 4) for s in fold_cs]}), "
            f"best params: {best}, t={_time.time()-t0:.1f}s"
        )

        rsf_model = RSFModel({**best, "low_memory": False, "random_state": k})
        rsf_model.fit(X_rsf_train, time=time_rsf_train, event=event_rsf_train)
        sfs = rsf_model.predict_survival_curve(X_cv_all)
        t_cap = float(time_rsf_train.max())  # cap from TRAINING distribution
        median = np.array(
            [median_survival_from_sf(sf, t_cap) for sf in sfs],
            dtype=float,
        )
        rsf_medians.append(median)
        rsf_bests.append(best)
        rsf_cindices.append(cindex_best)

    rsf_stack = np.stack(rsf_medians, axis=0)  # (K, n_total)
    rsf_median_mean = rsf_stack.mean(axis=0)
    rsf_median_std = rsf_stack.std(axis=0, ddof=0) if K > 1 else np.zeros(n_total)
    log.info(
        f"  ensemble median range [{rsf_median_mean.min():.0f}, "
        f"{rsf_median_mean.max():.0f}], mean {rsf_median_mean.mean():.1f}, "
        f"per-cell std mean: {rsf_median_std.mean():.1f}"
    )

    best_params["rsf"] = {
        "ensemble_seeds": K,
        "best_params_per_seed": rsf_bests,
        "inner_cv_cindex_per_seed": rsf_cindices,
        "inner_cv_cindex_mean": float(np.mean(rsf_cindices)),
        "n_training_cells": n_rsf_train,
        "n_inference_cells": n_total,
        "n_features": int(X_cv_all.shape[1]),
        "t_cap": float(time_rsf_train.max()),
    }

    # ---- Assemble predictions.csv -----------------------------------------
    log.info(f"=== Assembling predictions.csv ({n_total} rows) ===")
    true_pass = {
        N: np.array([
            _true_pass(
                bool(event_all[i]),
                float(y_cycle_all[i]) if event_all[i] else float("nan"),
                int(n_regular_all[i]),  # true n_regular per cell (≠ time for faded)
                N,
            )
            for i in range(n_total)
        ])
        for N in NS
    }

    df = pd.DataFrame({
        "cell_name": cell_names,
        "status": np.where(event_all, "faded", "censored"),
        "event": event_all,
        "last_fade_cycle": np.where(event_all, y_cycle_all, np.nan),
        "n_regular": n_regular_all,
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
        "prob_pass_n200": classifier_predictions[200]["prob_mean"],
        "prob_pass_n300": classifier_predictions[300]["prob_mean"],
        "prob_pass_n400": classifier_predictions[400]["prob_mean"],
        "prob_pass_n200_std": classifier_predictions[200]["prob_std"],
        "prob_pass_n300_std": classifier_predictions[300]["prob_std"],
        "prob_pass_n400_std": classifier_predictions[400]["prob_std"],
        "oof_prob_pass_n200": classifier_predictions[200]["oof_mean"],
        "oof_prob_pass_n300": classifier_predictions[300]["oof_mean"],
        "oof_prob_pass_n400": classifier_predictions[400]["oof_mean"],
        "oof_prob_pass_n200_std": classifier_predictions[200]["oof_std"],
        "oof_prob_pass_n300_std": classifier_predictions[300]["oof_std"],
        "oof_prob_pass_n400_std": classifier_predictions[400]["oof_std"],
        "rsf_median_cycle": rsf_median_mean,
        "rsf_median_cycle_std": rsf_median_std,
    })

    # Sanity
    pass_rates = {N: float(df[f"pred_pass_n{N}"].mean()) for N in NS}
    log.info(
        f"  Pass rates (predicted, ensemble mean@0.5): "
        f"N=200 → {pass_rates[200]:.3f}, "
        f"N=300 → {pass_rates[300]:.3f}, "
        f"N=400 → {pass_rates[400]:.3f}"
    )
    if not (pass_rates[200] >= pass_rates[300] >= pass_rates[400]):
        log.warning("  Pass rates are NOT monotone decreasing in N — investigate")

    # Filename includes the timestamp (= the run directory's name) so the
    # CSV is self-identifying even when copied out of its directory.
    timestamp = Path(out_dir).name
    out_csv = out_dir / f"predictions_{timestamp}.csv"
    df.to_csv(out_csv, index=False)
    log.info(f"  Wrote {out_csv} ({len(df)} rows × {len(df.columns)} cols)")

    out_json = out_dir / "best_params.json"
    with out_json.open("w") as f:
        json.dump({
            "ensemble_seeds": K,
            "feature_subsets": {
                "classifier": classifier_feature_subset,
                "rsf": rsf_feature_subset,
            },
            "baseline_cycle": baseline_cycle,
            "db_version": db_version,
            "trials": trials,
            "inner_cv": inner_cv,
            "n_total_cells": n_total,
            "n_faded": n_faded,
            "n_censored": n_censored,
            "models": best_params,
        }, f, indent=2, default=str)
    log.info(f"  Wrote {out_json}")

    if make_plots:
        log.info("Rendering plots…")
        paths = production_plots.render_all(df, out_dir)
        for p in paths:
            log.info(f"  Wrote {p}")

    # ---- Overfit diagnostics ----------------------------------------------
    log.info("=== Overfit diagnostics ===")
    _log_overfit_diagnostics(log, df, best_params, K)

    log.info("Production run complete.")
    return {
        "out_dir": str(out_dir),
        "n_cells": n_total,
        "ensemble_seeds": K,
        "classifier_auc_mean": {
            N: best_params[f"ebm_classifier_n{N}"]["inner_cv_auc_mean"]
            for N in NS
        },
        "rsf_cindex_mean": best_params["rsf"]["inner_cv_cindex_mean"],
        "pass_rates": pass_rates,
    }

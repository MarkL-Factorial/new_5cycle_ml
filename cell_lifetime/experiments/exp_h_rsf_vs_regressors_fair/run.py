#!/usr/bin/env python
"""Exp H: fair 3-way head-to-head — RSF vs XGB regressor vs EBM regressor.

Test set = 20% held-out FADED cells (the cells with ground-truth cycle
counts). RSF training = ALL censored cells + remaining 80% faded. Both
regressors train on the same 80% faded only. Target transform = sqrt
for the regressors; RSF consumes (time, event) directly.

All three models hyperparameter-tuned with identical budget
(30 trials × 5 inner CV, default). 5 seeds. Metrics on the held-out
20% faded test cells: MAE / MAPE / RMSE / R² + per-quartile MAE.

Run from cell_lifetime/:
    python experiments/exp_h_rsf_vs_regressors_fair/run.py \
        --seeds 5 --trials 30 --inner-cv 5

Smoke (1 seed, reduced budget):
    python experiments/exp_h_rsf_vs_regressors_fair/run.py --smoke
"""

from __future__ import annotations

import argparse
import json
import logging
import time as _time
from pathlib import Path
from typing import Any

import numpy as np
import optuna
import pandas as pd
from sklearn.model_selection import KFold, train_test_split
from sksurv.metrics import concordance_index_censored
from sksurv.util import Surv

from cell_lifetime.data.loader import load_dataset
from cell_lifetime.models.ebm_regressor import EBMRegressorModel
from cell_lifetime.models.rsf import RSFModel
from cell_lifetime.models.xgb_regressor import XGBRegressorModel


# ---------- helpers ----------------------------------------------------------

HERE = Path(__file__).resolve().parent
RUNS_DIR = HERE / "runs"
LOG_DIR = HERE / "run_logs"
RUNS_DIR.mkdir(parents=True, exist_ok=True)
LOG_DIR.mkdir(parents=True, exist_ok=True)


def median_survival_from_sf(sf, t_cap: float) -> float:
    """Extract median-survival cycle count from a sksurv StepFunction.

    StepFunction.x: time grid (sorted). StepFunction.y: S(t) at each time.
    Median is the smallest t with S(t) <= 0.5. If S never drops to 0.5,
    clip to t_cap (the max observed time in training).
    """
    times = np.asarray(sf.x, dtype=float)
    surv = np.asarray(sf.y, dtype=float)
    below = np.where(surv <= 0.5)[0]
    if len(below) == 0:
        return float(t_cap)
    return float(times[below[0]])


def mape(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    mask = y_true > 0
    if not mask.any():
        return float("nan")
    return float(np.mean(np.abs(y_pred[mask] - y_true[mask]) / y_true[mask]))


def quartile_mae(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    """Per-quartile MAE (Q1: shortest-lived 25% ... Q4: longest-lived 25%)."""
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    q = np.quantile(y_true, [0.25, 0.5, 0.75])
    out: dict[str, float] = {}
    bins = [
        ("q1", y_true <= q[0]),
        ("q2", (y_true > q[0]) & (y_true <= q[1])),
        ("q3", (y_true > q[1]) & (y_true <= q[2])),
        ("q4", y_true > q[2]),
    ]
    for name, mask in bins:
        if mask.sum() == 0:
            out[f"mae_{name}"] = float("nan")
        else:
            out[f"mae_{name}"] = float(np.mean(np.abs(y_pred[mask] - y_true[mask])))
    return out


# ---------- RSF tune+fit -----------------------------------------------------

def _rsf_inner_cv_score(
    params: dict[str, Any],
    X: pd.DataFrame, time_arr: np.ndarray, event_arr: np.ndarray,
    inner_cv: int, seed: int,
) -> float:
    """Mean C-index across `inner_cv` folds on (X, time, event)."""
    kf = KFold(n_splits=inner_cv, shuffle=True, random_state=seed)
    scores = []
    for tr, va in kf.split(X):
        model = RSFModel(params)
        model.fit(X.iloc[tr], time=time_arr[tr], event=event_arr[tr])
        risk = model.predict(X.iloc[va])
        # concordance_index_censored returns (cindex, concordant, discordant, ...).
        # The risk score from RSF is "higher = sooner failure" → no sign flip.
        c, *_ = concordance_index_censored(event_arr[va].astype(bool), time_arr[va], risk)
        scores.append(float(c))
    return float(np.mean(scores))


def tune_rsf(
    X: pd.DataFrame, time_arr: np.ndarray, event_arr: np.ndarray,
    trials: int, inner_cv: int, seed: int,
) -> dict[str, Any]:
    sampler = optuna.samplers.TPESampler(seed=seed)
    study = optuna.create_study(direction="maximize", sampler=sampler)

    def objective(trial: optuna.Trial) -> float:
        params = RSFModel.suggest_params(trial)
        try:
            return _rsf_inner_cv_score(params, X, time_arr, event_arr, inner_cv, seed)
        except Exception as e:
            logging.warning(f"RSF trial failed: {e}")
            return float("nan")

    optuna.logging.set_verbosity(optuna.logging.WARNING)
    study.optimize(objective, n_trials=trials, show_progress_bar=False)
    return dict(study.best_params)


# ---------- Regressor tune+fit (shared) -------------------------------------

def _reg_inner_cv_mae(
    ModelCls, params: dict[str, Any],
    X: pd.DataFrame, y: np.ndarray,
    inner_cv: int, seed: int,
) -> float:
    """Mean MAE across `inner_cv` folds on inverse-transformed predictions."""
    kf = KFold(n_splits=inner_cv, shuffle=True, random_state=seed)
    scores = []
    for tr, va in kf.split(X):
        model = ModelCls(params, target_transform="sqrt")
        try:
            model.fit(X.iloc[tr], y[tr])
            pred = model.predict(X.iloc[va])
        except Exception as e:
            logging.warning(f"{ModelCls.__name__} trial failed: {e}")
            return float("inf")
        pred = np.clip(pred, 1.0, 5000.0)
        scores.append(float(np.mean(np.abs(pred - y[va]))))
    return float(np.mean(scores))


def tune_regressor(
    ModelCls, X: pd.DataFrame, y: np.ndarray,
    trials: int, inner_cv: int, seed: int,
) -> dict[str, Any]:
    sampler = optuna.samplers.TPESampler(seed=seed)
    study = optuna.create_study(direction="minimize", sampler=sampler)

    def objective(trial: optuna.Trial) -> float:
        params = ModelCls.suggest_params(trial)
        return _reg_inner_cv_mae(ModelCls, params, X, y, inner_cv, seed)

    optuna.logging.set_verbosity(optuna.logging.WARNING)
    study.optimize(objective, n_trials=trials, show_progress_bar=False)
    return dict(study.best_params)


# ---------- Per-seed protocol -----------------------------------------------

def run_seed(
    ds, seed: int, trials: int, inner_cv: int, out_dir: Path,
) -> dict[str, dict[str, float]]:
    """Run one seed: split, tune, fit, score all 3 models."""
    out_dir.mkdir(parents=True, exist_ok=True)

    faded_idx = np.where(ds.event)[0]
    censored_idx = np.where(~ds.event)[0]

    # 80/20 split of faded cells. Same partition for all three models.
    F_train, F_test = train_test_split(
        faded_idx, test_size=0.20, random_state=seed, shuffle=True,
    )

    # Index sets
    rsf_train_idx = np.concatenate([F_train, censored_idx])
    np.random.default_rng(seed).shuffle(rsf_train_idx)  # cosmetic; KFold shuffles too
    reg_train_idx = F_train
    test_idx = F_test

    X_all = ds.X.reset_index(drop=True)
    time_all = ds.time
    event_all = ds.event
    y_cycle_all = ds.y_cycle

    X_test = X_all.iloc[test_idx].reset_index(drop=True)
    y_test = y_cycle_all[test_idx].astype(float)
    cell_names_test = ds.cell_names[test_idx]

    # Sanity: all three models MUST share the same held-out test cells.
    # The test set is the same `test_idx` from line above; the assertions
    # below guard against accidental future divergence (e.g. someone
    # reorders the rows, re-splits per model, etc.).
    assert set(test_idx).isdisjoint(rsf_train_idx), "RSF train leaks into test"
    assert set(test_idx).isdisjoint(reg_train_idx), "Regressor train leaks into test"
    assert len(test_idx) == len(np.unique(test_idx)), "duplicate test cells"

    import hashlib
    test_fingerprint = hashlib.sha256(
        ",".join(sorted(str(c) for c in cell_names_test)).encode()
    ).hexdigest()[:16]

    n_F_train = len(reg_train_idx)
    n_F_test = len(test_idx)
    n_censored = len(censored_idx)
    print(f"  seed={seed}: F_train={n_F_train}, F_test={n_F_test}, censored={n_censored}")
    print(f"  RSF train rows: {len(rsf_train_idx)} (={n_F_train}+{n_censored})")
    print(f"  test cell fingerprint (sha256[:16]): {test_fingerprint}")

    # === RSF ===
    t0 = _time.time()
    X_rsf_train = X_all.iloc[rsf_train_idx].reset_index(drop=True)
    time_rsf_train = time_all[rsf_train_idx]
    event_rsf_train = event_all[rsf_train_idx]
    print(f"  [RSF] tuning ({trials} trials × {inner_cv} CV) …")
    rsf_best = tune_rsf(X_rsf_train, time_rsf_train, event_rsf_train, trials, inner_cv, seed)
    # Final fit needs low_memory=False so we can call predict_survival_function
    # (RSFModel's default fixed_params has low_memory=True). Passing it in
    # params overrides fixed_params via {**fixed, **params} merge.
    rsf_model = RSFModel({**rsf_best, "low_memory": False})
    rsf_model.fit(X_rsf_train, time=time_rsf_train, event=event_rsf_train)
    # Extract median survival on test set
    sfs = rsf_model.predict_survival_curve(X_test)
    t_cap = float(time_rsf_train.max())
    rsf_pred = np.array([median_survival_from_sf(sf, t_cap) for sf in sfs], dtype=float)
    rsf_runtime = _time.time() - t0

    # === XGB regressor ===
    t0 = _time.time()
    X_reg_train = X_all.iloc[reg_train_idx].reset_index(drop=True)
    y_reg_train = y_cycle_all[reg_train_idx].astype(float)
    print(f"  [XGB] tuning ({trials} trials × {inner_cv} CV) …")
    xgb_best = tune_regressor(XGBRegressorModel, X_reg_train, y_reg_train, trials, inner_cv, seed)
    xgb_model = XGBRegressorModel(xgb_best, target_transform="sqrt")
    xgb_model.fit(X_reg_train, y_reg_train)
    xgb_pred = np.clip(xgb_model.predict(X_test), 1.0, 5000.0)
    xgb_runtime = _time.time() - t0

    # === EBM regressor ===
    t0 = _time.time()
    print(f"  [EBM] tuning ({trials} trials × {inner_cv} CV) …")
    ebm_best = tune_regressor(EBMRegressorModel, X_reg_train, y_reg_train, trials, inner_cv, seed)
    ebm_model = EBMRegressorModel(ebm_best, target_transform="sqrt")
    ebm_model.fit(X_reg_train, y_reg_train)
    ebm_pred = np.clip(ebm_model.predict(X_test), 1.0, 5000.0)
    ebm_runtime = _time.time() - t0

    # === Score ===
    results: dict[str, dict[str, float]] = {}
    for name, pred, rt, best in (
        ("rsf", rsf_pred, rsf_runtime, rsf_best),
        ("xgb_regressor", xgb_pred, xgb_runtime, xgb_best),
        ("ebm_regressor", ebm_pred, ebm_runtime, ebm_best),
    ):
        m = {
            "mae": float(np.mean(np.abs(pred - y_test))),
            "rmse": float(np.sqrt(np.mean((pred - y_test) ** 2))),
            "r2": _r2(y_test, pred),
            "medae": float(np.median(np.abs(pred - y_test))),
            "mape": mape(y_test, pred),
            "runtime_s": float(rt),
        }
        m.update(quartile_mae(y_test, pred))
        results[name] = m
        results[f"{name}_best_params"] = best  # type: ignore[assignment]

    # Save per-seed predictions
    pred_df = pd.DataFrame({
        "cell_name": cell_names_test,
        "y_true": y_test,
        "rsf_pred": rsf_pred,
        "xgb_pred": xgb_pred,
        "ebm_pred": ebm_pred,
    })
    pred_df.to_csv(out_dir / "predictions.csv", index=False)
    with (out_dir / "results.json").open("w") as f:
        json.dump({
            "seed": seed,
            "n_F_train": n_F_train,
            "n_F_test": n_F_test,
            "n_censored": n_censored,
            "trials": trials,
            "inner_cv": inner_cv,
            "test_cell_fingerprint": test_fingerprint,
            "test_cell_names": sorted(str(c) for c in cell_names_test),
            "results": results,
        }, f, indent=2, default=str)
    return results


def _r2(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    if len(y_true) < 2:
        return float("nan")
    ss_res = float(np.sum((y_true - y_pred) ** 2))
    ss_tot = float(np.sum((y_true - y_true.mean()) ** 2))
    if ss_tot < 1e-12:
        return float("nan")
    return 1.0 - ss_res / ss_tot


# ---------- Aggregate & write summary ---------------------------------------

def aggregate(per_seed: list[dict[str, dict[str, float]]]) -> dict[str, Any]:
    """Compute mean ± std for each (model, metric) across seeds."""
    summary: dict[str, dict[str, float]] = {}
    for model in ("rsf", "xgb_regressor", "ebm_regressor"):
        per_metric: dict[str, list[float]] = {}
        for ps in per_seed:
            for k, v in ps[model].items():
                per_metric.setdefault(k, []).append(float(v))
        agg = {}
        for k, vals in per_metric.items():
            agg[f"{k}_mean"] = float(np.nanmean(vals))
            agg[f"{k}_std"] = float(np.nanstd(vals, ddof=1)) if len(vals) > 1 else 0.0
        summary[model] = agg
    return summary


# ---------- CLI -------------------------------------------------------------

def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seeds", type=int, default=5)
    parser.add_argument("--trials", type=int, default=30)
    parser.add_argument("--inner-cv", type=int, default=5)
    parser.add_argument("--smoke", action="store_true", help="1 seed × 5 trials × 3 CV smoke test")
    parser.add_argument("--feature-subset", default="fs_cv")
    parser.add_argument("--baseline-cycle", type=int, default=1)
    parser.add_argument("--db-version", default="A2.2")
    args = parser.parse_args()

    if args.smoke:
        args.seeds = 1
        args.trials = 5
        args.inner_cv = 3

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    print(
        f"Exp H: fair RSF vs regressors. seeds={args.seeds}, "
        f"trials={args.trials}, inner_cv={args.inner_cv}, "
        f"fs={args.feature_subset}, baseline={args.baseline_cycle}"
    )

    # Use the lowest supported N (the loader requires N ∈ {200,300,400} but
    # regression+survival don't use it for training; we just need the data
    # to load).
    ds = load_dataset(
        N=300, feature_subset=args.feature_subset,
        baseline_cycle=args.baseline_cycle, db_version=args.db_version,
    )
    n_faded = int(ds.event.sum())
    n_censored = int((~ds.event).sum())
    print(f"Dataset: {len(ds)} cells = {n_faded} faded + {n_censored} censored")

    per_seed: list[dict[str, dict[str, float]]] = []
    for seed in range(args.seeds):
        print(f"\n=== Seed {seed} ===")
        out_dir = RUNS_DIR / f"seed_{seed}"
        res = run_seed(ds, seed, args.trials, args.inner_cv, out_dir)
        per_seed.append(res)

    summary = aggregate(per_seed)
    print("\n=== Summary (mean ± std across seeds) ===")
    headline_metrics = ["mae", "mape", "rmse", "r2", "mae_q1", "mae_q4"]
    for model in ("rsf", "xgb_regressor", "ebm_regressor"):
        bits = [f"{model:>14s}:"]
        for m in headline_metrics:
            mn = summary[model].get(f"{m}_mean", float("nan"))
            sd = summary[model].get(f"{m}_std", float("nan"))
            bits.append(f"  {m}={mn:.3f}±{sd:.3f}")
        print(" ".join(bits))

    # Long-form CSV
    long_rows = []
    for ps_i, ps in enumerate(per_seed):
        for model in ("rsf", "xgb_regressor", "ebm_regressor"):
            for k, v in ps[model].items():
                long_rows.append({"seed": ps_i, "model": model, "metric": k, "value": v})
    pd.DataFrame(long_rows).to_csv(HERE / "metric_long.csv", index=False)
    print(f"\n[exp_h] wrote {HERE / 'metric_long.csv'} ({len(long_rows)} rows)")

    with (HERE / "summary.json").open("w") as f:
        json.dump({
            "experiment": "exp_h_rsf_vs_regressors_fair",
            "feature_subset": args.feature_subset,
            "baseline_cycle": args.baseline_cycle,
            "db_version": args.db_version,
            "n_seeds": args.seeds,
            "trials": args.trials,
            "inner_cv": args.inner_cv,
            "n_faded": n_faded,
            "n_censored": n_censored,
            "summary": summary,
        }, f, indent=2)
    print(f"[exp_h] wrote {HERE / 'summary.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

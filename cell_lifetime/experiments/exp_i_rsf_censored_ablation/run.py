#!/usr/bin/env python
"""Exp I: RSF censored-data ablation.

Tests Exp H's claim that RSF's Q4 (long-life) advantage comes from
training on censored cells. Two variants per seed, same 20% faded
test set as Exp H (sha256 fingerprint asserted):

  - RSF-with-censored  → 80% faded + ALL censored = 377 rows
                         (this is the Exp H baseline; reproduced for
                          a same-seed sanity check)
  - RSF-no-censored    → 80% faded only = 149 rows
                         (all event=1; no censored data)

Both go through identical: 30 trials × 5 inner CV hyperparameter tune
(C-index objective), `low_memory=False` at final fit, median-survival
extraction (`min{t : S(t)≤0.5}`) on the held-out 20% faded cells.

Run from `cell_lifetime/`:
    python experiments/exp_i_rsf_censored_ablation/run.py \
        --seeds 5 --trials 30 --inner-cv 5

Smoke (1 seed, reduced budget):
    python experiments/exp_i_rsf_censored_ablation/run.py --smoke
"""

from __future__ import annotations

import argparse
import hashlib
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

from cell_lifetime.data.loader import load_dataset
from cell_lifetime.models.rsf import RSFModel


HERE = Path(__file__).resolve().parent
RUNS_DIR = HERE / "runs"
LOG_DIR = HERE / "run_logs"
RUNS_DIR.mkdir(parents=True, exist_ok=True)
LOG_DIR.mkdir(parents=True, exist_ok=True)


# ---------- helpers (copied verbatim from Exp H run.py) ---------------------

def median_survival_from_sf(sf, t_cap: float) -> float:
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


# ---------- RSF tune+fit (copied verbatim from Exp H) -----------------------

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


# ---------- Per-seed protocol -----------------------------------------------

def run_seed(
    ds, seed: int, trials: int, inner_cv: int, out_dir: Path,
) -> dict[str, dict[str, float]]:
    """Train BOTH RSF variants on the same 20% faded test cells."""
    out_dir.mkdir(parents=True, exist_ok=True)

    faded_idx = np.where(ds.event)[0]
    censored_idx = np.where(~ds.event)[0]

    # SAME split as Exp H — byte-identical via random_state=seed.
    F_train, F_test = train_test_split(
        faded_idx, test_size=0.20, random_state=seed, shuffle=True,
    )

    # Sanity assertions
    assert set(F_test).isdisjoint(F_train), "F_train ∩ F_test"
    assert len(F_test) == len(np.unique(F_test)), "duplicate test cells"

    X_all = ds.X.reset_index(drop=True)
    time_all = ds.time
    event_all = ds.event
    y_cycle_all = ds.y_cycle

    X_test = X_all.iloc[F_test].reset_index(drop=True)
    y_test = y_cycle_all[F_test].astype(float)
    cell_names_test = ds.cell_names[F_test]

    test_fingerprint = hashlib.sha256(
        ",".join(sorted(str(c) for c in cell_names_test)).encode()
    ).hexdigest()[:16]

    # Cross-check against Exp H's same-seed test cells. Prefer the
    # `test_cell_fingerprint` field in results.json (newer runs), fall
    # back to recomputing the fingerprint from Exp H's predictions.csv
    # (the original Exp H run didn't record the fingerprint).
    exp_h_seed_dir = HERE.parent / "exp_h_rsf_vs_regressors_fair" / "runs" / f"seed_{seed}"
    exp_h_results = exp_h_seed_dir / "results.json"
    exp_h_predictions = exp_h_seed_dir / "predictions.csv"
    recorded = None
    if exp_h_results.exists():
        recorded = json.loads(exp_h_results.read_text()).get("test_cell_fingerprint")
    if recorded is None and exp_h_predictions.exists():
        h_cells = pd.read_csv(exp_h_predictions)["cell_name"].astype(str).tolist()
        recorded = hashlib.sha256(
            ",".join(sorted(h_cells)).encode()
        ).hexdigest()[:16]
    if recorded is not None:
        assert recorded == test_fingerprint, (
            f"seed={seed}: fingerprint drift vs Exp H "
            f"(now={test_fingerprint}, exp_h={recorded})"
        )
        print(f"  seed={seed} fingerprint matches Exp H: {test_fingerprint}")
    else:
        print(
            f"  seed={seed} fingerprint={test_fingerprint} "
            f"(no Exp H data to cross-check)"
        )

    print(
        f"  seed={seed}: F_train={len(F_train)}, F_test={len(F_test)}, "
        f"censored={len(censored_idx)}"
    )

    # === Variant A: RSF-with-censored (Exp H baseline) ===
    print(f"  [RSF-with-censored] tuning ({trials} trials × {inner_cv} CV) …")
    rsf_with_train_idx = np.concatenate([F_train, censored_idx])
    X_w_train = X_all.iloc[rsf_with_train_idx].reset_index(drop=True)
    time_w_train = time_all[rsf_with_train_idx]
    event_w_train = event_all[rsf_with_train_idx]

    t0 = _time.time()
    w_best = tune_rsf(X_w_train, time_w_train, event_w_train, trials, inner_cv, seed)
    w_model = RSFModel({**w_best, "low_memory": False})
    w_model.fit(X_w_train, time=time_w_train, event=event_w_train)
    w_sfs = w_model.predict_survival_curve(X_test)
    t_cap_w = float(time_w_train.max())
    w_pred = np.array([median_survival_from_sf(sf, t_cap_w) for sf in w_sfs], dtype=float)
    w_runtime = _time.time() - t0

    # === Variant B: RSF-no-censored (ablation) ===
    print(f"  [RSF-no-censored]   tuning ({trials} trials × {inner_cv} CV) …")
    # train rows = F_train only; all event=True; time=last_fade_cycle
    X_n_train = X_all.iloc[F_train].reset_index(drop=True)
    time_n_train = time_all[F_train]
    event_n_train = np.ones(len(F_train), dtype=bool)  # all faded

    # Sanity: time should equal last_fade_cycle for faded cells (it does
    # in the loader: time = where(event, last_fade, n_regular)).
    assert (time_n_train == y_cycle_all[F_train].astype(np.int64)).all(), (
        "F_train time != last_fade — loader contract broken"
    )

    t0 = _time.time()
    n_best = tune_rsf(X_n_train, time_n_train, event_n_train, trials, inner_cv, seed)
    n_model = RSFModel({**n_best, "low_memory": False})
    n_model.fit(X_n_train, time=time_n_train, event=event_n_train)
    n_sfs = n_model.predict_survival_curve(X_test)
    t_cap_n = float(time_n_train.max())
    n_pred = np.array([median_survival_from_sf(sf, t_cap_n) for sf in n_sfs], dtype=float)
    n_runtime = _time.time() - t0

    # === Score ===
    results: dict[str, dict[str, float]] = {}
    for variant, pred, rt, best in (
        ("rsf_with_censored", w_pred, w_runtime, w_best),
        ("rsf_no_censored", n_pred, n_runtime, n_best),
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
        results[variant] = m
        results[f"{variant}_best_params"] = best  # type: ignore[assignment]

    # Save per-seed predictions
    pred_df = pd.DataFrame({
        "cell_name": cell_names_test,
        "y_true": y_test,
        "rsf_with_pred": w_pred,
        "rsf_no_pred": n_pred,
    })
    pred_df.to_csv(out_dir / "predictions.csv", index=False)
    with (out_dir / "results.json").open("w") as f:
        json.dump({
            "seed": seed,
            "n_F_train": len(F_train),
            "n_F_test": len(F_test),
            "n_censored": len(censored_idx),
            "trials": trials,
            "inner_cv": inner_cv,
            "test_cell_fingerprint": test_fingerprint,
            "test_cell_names": sorted(str(c) for c in cell_names_test),
            "t_cap_with_censored": t_cap_w,
            "t_cap_no_censored": t_cap_n,
            "results": results,
        }, f, indent=2, default=str)
    return results


# ---------- Aggregate & write summary ---------------------------------------

def aggregate(per_seed: list[dict[str, dict[str, float]]]) -> dict[str, Any]:
    summary: dict[str, dict[str, float]] = {}
    for variant in ("rsf_with_censored", "rsf_no_censored"):
        per_metric: dict[str, list[float]] = {}
        for ps in per_seed:
            for k, v in ps[variant].items():
                per_metric.setdefault(k, []).append(float(v))
        agg = {}
        for k, vals in per_metric.items():
            agg[f"{k}_mean"] = float(np.nanmean(vals))
            agg[f"{k}_std"] = float(np.nanstd(vals, ddof=1)) if len(vals) > 1 else 0.0
        summary[variant] = agg
    return summary


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seeds", type=int, default=5)
    parser.add_argument("--trials", type=int, default=30)
    parser.add_argument("--inner-cv", type=int, default=5)
    parser.add_argument("--smoke", action="store_true",
                        help="1 seed × 5 trials × 3 CV smoke test")
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
        f"Exp I: RSF censored ablation. seeds={args.seeds}, "
        f"trials={args.trials}, inner_cv={args.inner_cv}, "
        f"fs={args.feature_subset}, baseline={args.baseline_cycle}"
    )

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
    headline_metrics = ["mae", "mape", "rmse", "r2", "mae_q1", "mae_q2", "mae_q3", "mae_q4"]
    for variant in ("rsf_with_censored", "rsf_no_censored"):
        bits = [f"{variant:>18s}:"]
        for m in headline_metrics:
            mn = summary[variant].get(f"{m}_mean", float("nan"))
            sd = summary[variant].get(f"{m}_std", float("nan"))
            bits.append(f"  {m}={mn:.3f}±{sd:.3f}")
        print(" ".join(bits))

    # Long-form CSV
    long_rows = []
    for ps_i, ps in enumerate(per_seed):
        for variant in ("rsf_with_censored", "rsf_no_censored"):
            for k, v in ps[variant].items():
                long_rows.append({"seed": ps_i, "variant": variant, "metric": k, "value": v})
    pd.DataFrame(long_rows).to_csv(HERE / "metric_long.csv", index=False)
    print(f"\n[exp_i] wrote {HERE / 'metric_long.csv'} ({len(long_rows)} rows)")

    with (HERE / "summary.json").open("w") as f:
        json.dump({
            "experiment": "exp_i_rsf_censored_ablation",
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
    print(f"[exp_i] wrote {HERE / 'summary.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

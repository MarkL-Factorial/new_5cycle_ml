#!/usr/bin/env python
"""Exp L — Does adding discharge_capacity_c5 to fs_a_only improve
classification accuracy?

Paired comparison: for each of 5 seeds (random_state controls the
stratified 80/20 split), Optuna-tune EBM classifier on (a) fs_a_only
(3 features) and (b) fs_a_plus_cap_c5 (4 features) on the SAME 80%
train, then score both on the SAME 20% test. Report mean±std deltas
per N ∈ {200, 300, 400} and per cohort (AR / 0MC) to distinguish real
signal from a cohort shortcut.

Run from cell_lifetime/:

    python experiments/exp_l_classifier_add_cap_c5/run.py --seeds 5 --trials 30 --inner-cv 5

Smoke (1 seed):

    python experiments/exp_l_classifier_add_cap_c5/run.py --smoke
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time as _time
from pathlib import Path
from typing import Any

import numpy as np
import optuna
import pandas as pd
import polars as pl
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from sklearn.model_selection import KFold, train_test_split

from cell_lifetime.data.loader import load_dataset
from cell_lifetime.models.ebm_classifier import EBMClassifierModel


HERE = Path(__file__).resolve().parent
RUNS_DIR = HERE / "runs"
LOG_DIR = HERE / "run_logs"
NS = (200, 300, 400)
FEATURE_SETS = ("fs_a_only", "fs_a_plus_cap_c5")
FS_A = (
    "coulombic_efficiency_final",
    "discharge_capacity_retention_final",
    "charge_capacity_retention_min",
)
NEW_FEATURE = "discharge_capacity_c5"


def setup_logging(log_path: Path) -> logging.Logger:
    root = logging.getLogger()
    root.setLevel(logging.INFO)
    for h in list(root.handlers):
        root.removeHandler(h)
    fmt = logging.Formatter(
        "%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%Y-%m-%dT%H:%M:%S",
    )
    fh = logging.FileHandler(log_path, mode="w")
    fh.setFormatter(fmt)
    sh = logging.StreamHandler(sys.stdout)
    sh.setFormatter(fmt)
    root.addHandler(fh)
    root.addHandler(sh)
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    return logging.getLogger(__name__)


def load_with_cap_c5(
    N: int, baseline_cycle: int, db_version: str,
) -> tuple[pd.DataFrame, np.ndarray, np.ndarray, np.ndarray]:
    """Load fs_a_only features + derived discharge_capacity_c5 + cohort.

    Returns (X4, y, cell_names, cohorts) where X4 has columns
    [FS_A..., NEW_FEATURE]. Restricted to trainable_n{N} cells with
    n_regular>=6 and status!='excluded' (the canonical training cohort).
    """
    ds = load_dataset(
        N=N, feature_subset="fs_a_only",
        baseline_cycle=baseline_cycle, db_version=db_version,
        min_n_regular=6, drop_excluded=True,
    )
    mask = ds.label_mask.astype(bool)

    # Pull baseline_dis_ah directly from cell_labels.parquet for these cells.
    # `ds.source_dir` is already the resolved snapshot dir.
    labels = pl.read_parquet(
        ds.source_dir / "cell_labels.parquet"
    ).select(["cell_name", "baseline_dis_ah"]).to_pandas()
    cell_to_baseline = dict(zip(labels.cell_name, labels.baseline_dis_ah))

    X3 = ds.X.loc[mask].reset_index(drop=True)
    names = ds.cell_names[mask]
    cohorts = ds.cohorts[mask]
    y = ds.y_class[mask].astype(int)

    baseline_dis = np.array([cell_to_baseline[c] for c in names], dtype=float)
    cap_c5 = baseline_dis * X3["discharge_capacity_retention_final"].to_numpy()
    X4 = X3.copy()
    X4[NEW_FEATURE] = cap_c5

    return X4, y, names, cohorts


def _inner_cv_auc(
    params: dict[str, Any], X: pd.DataFrame, y: np.ndarray,
    inner_cv: int, seed: int,
) -> float:
    kf = KFold(n_splits=inner_cv, shuffle=True, random_state=seed)
    scores: list[float] = []
    for tr, va in kf.split(X):
        if len(np.unique(y[tr])) < 2 or len(np.unique(y[va])) < 2:
            continue
        mdl = EBMClassifierModel(params)
        mdl.fit(X.iloc[tr], y[tr])
        prob = mdl.predict_proba(X.iloc[va])[:, 1]
        scores.append(roc_auc_score(y[va], prob))
    return float(np.mean(scores)) if scores else float("nan")


def tune(
    X: pd.DataFrame, y: np.ndarray, trials: int, inner_cv: int, seed: int,
) -> tuple[dict[str, Any], float]:
    sampler = optuna.samplers.TPESampler(seed=seed)
    study = optuna.create_study(direction="maximize", sampler=sampler)
    study.optimize(
        lambda trial: _inner_cv_auc(
            EBMClassifierModel.suggest_params(trial), X, y, inner_cv, seed,
        ),
        n_trials=trials, show_progress_bar=False,
    )
    return dict(study.best_params), float(study.best_value)


def evaluate(
    X_train: pd.DataFrame, y_train: np.ndarray,
    X_test: pd.DataFrame, y_test: np.ndarray, cohorts_test: np.ndarray,
    params: dict[str, Any],
) -> tuple[dict[str, float], np.ndarray]:
    mdl = EBMClassifierModel(params)
    mdl.fit(X_train, y_train)
    prob = mdl.predict_proba(X_test)[:, 1]
    pred = (prob >= 0.5).astype(int)

    out: dict[str, float] = {
        "auc": float(roc_auc_score(y_test, prob)),
        "f1": float(f1_score(y_test, pred)),
        "acc": float(accuracy_score(y_test, pred)),
        "n_test": int(len(y_test)),
        "pass_rate_test": float(y_test.mean()),
    }
    for ch in ("AR", "0MC"):
        m = (cohorts_test == ch)
        if m.sum() < 2 or len(np.unique(y_test[m])) < 2:
            out[f"auc_{ch}"] = float("nan")
            out[f"f1_{ch}"] = float("nan")
            out[f"acc_{ch}"] = float("nan")
            out[f"n_test_{ch}"] = int(m.sum())
            continue
        out[f"auc_{ch}"] = float(roc_auc_score(y_test[m], prob[m]))
        out[f"f1_{ch}"] = float(f1_score(y_test[m], pred[m]))
        out[f"acc_{ch}"] = float(accuracy_score(y_test[m], pred[m]))
        out[f"n_test_{ch}"] = int(m.sum())
    return out, prob


def run_seed_N(
    N: int, seed: int, trials: int, inner_cv: int,
    baseline_cycle: int, db_version: str,
    log: logging.Logger,
) -> tuple[dict[str, dict[str, float]], pd.DataFrame]:
    """Run paired comparison for one (seed, N). Returns metrics per feature
    set + per-cell test prediction dataframe."""
    X4, y, names, cohorts = load_with_cap_c5(N, baseline_cycle, db_version)
    # Same stratified split used for both feature sets (so we compare apples-to-apples)
    idx_train, idx_test = train_test_split(
        np.arange(len(y)), test_size=0.2,
        stratify=y, random_state=seed,
    )
    X_train_full = X4.iloc[idx_train].reset_index(drop=True)
    X_test_full  = X4.iloc[idx_test].reset_index(drop=True)
    y_train = y[idx_train]
    y_test = y[idx_test]
    names_test = names[idx_test]
    cohorts_test = cohorts[idx_test]

    out: dict[str, dict[str, float]] = {}
    preds: dict[str, np.ndarray] = {}
    for fs in FEATURE_SETS:
        cols = list(FS_A) if fs == "fs_a_only" else (list(FS_A) + [NEW_FEATURE])
        Xt = X_train_full[cols]
        Xe = X_test_full[cols]

        t0 = _time.time()
        best, best_cv = tune(Xt, y_train, trials, inner_cv, seed=seed)
        log.info(
            f"  [seed={seed}, N={N}, fs={fs}] best inner-CV AUC={best_cv:.4f}, "
            f"params={best}, t_tune={_time.time() - t0:.1f}s"
        )
        metrics, prob = evaluate(Xt, y_train, Xe, y_test, cohorts_test, best)
        log.info(
            f"  [seed={seed}, N={N}, fs={fs}] test AUC={metrics['auc']:.4f}, "
            f"F1={metrics['f1']:.4f}, ACC={metrics['acc']:.4f}, "
            f"AR={metrics['auc_AR']:.4f}, 0MC={metrics['auc_0MC']:.4f}"
        )
        metrics["inner_cv_auc"] = best_cv
        metrics["best_params"] = best
        out[fs] = metrics
        preds[fs] = prob

    # Per-cell prediction dataframe (long format: rows = cell × feature_set)
    rows = []
    for fs in FEATURE_SETS:
        for i in range(len(y_test)):
            rows.append({
                "seed": seed, "N": N, "feature_set": fs,
                "cell_name": names_test[i], "cohort": cohorts_test[i],
                "y_true": int(y_test[i]),
                "y_prob": float(preds[fs][i]),
                "y_pred": int(preds[fs][i] >= 0.5),
            })
    return out, pd.DataFrame(rows)


def aggregate(per_seed: list[dict[str, Any]]) -> dict[str, Any]:
    """Build the nested summary dict from per-seed metrics."""
    out: dict[str, Any] = {fs: {} for fs in FEATURE_SETS}
    for fs in FEATURE_SETS:
        for N in NS:
            metric_keys = [k for k in per_seed[0][fs][N] if k != "best_params"]
            n_out: dict[str, Any] = {}
            for k in metric_keys:
                vals = [s[fs][N][k] for s in per_seed]
                vals_clean = [v for v in vals if not (isinstance(v, float) and np.isnan(v))]
                n_out[k] = {
                    "mean": float(np.mean(vals_clean)) if vals_clean else float("nan"),
                    "std": float(np.std(vals_clean, ddof=1)) if len(vals_clean) > 1 else 0.0,
                    "per_seed": vals,
                }
            out[fs][f"n{N}"] = n_out

    # Deltas (fs_a_plus_cap_c5 minus fs_a_only) for headline metrics
    out["delta"] = {}
    for N in NS:
        out["delta"][f"n{N}"] = {
            k: out["fs_a_plus_cap_c5"][f"n{N}"][k]["mean"]
               - out["fs_a_only"][f"n{N}"][k]["mean"]
            for k in ("auc", "f1", "acc", "auc_AR", "auc_0MC", "f1_AR", "f1_0MC")
        }
    return out


def write_cohort_csv(per_seed: list[dict[str, Any]], path: Path) -> None:
    rows = []
    for fs in FEATURE_SETS:
        for N in NS:
            for ch in ("overall", "AR", "0MC"):
                suffix = "" if ch == "overall" else f"_{ch}"
                aucs = [s[fs][N][f"auc{suffix}"] for s in per_seed]
                f1s = [s[fs][N][f"f1{suffix}"] for s in per_seed]
                accs = [s[fs][N][f"acc{suffix}"] for s in per_seed]
                ns = [s[fs][N][f"n_test{suffix}"] for s in per_seed]
                aucs_c = [v for v in aucs if not np.isnan(v)]
                f1s_c = [v for v in f1s if not np.isnan(v)]
                accs_c = [v for v in accs if not np.isnan(v)]
                rows.append({
                    "feature_set": fs, "N": N, "cohort": ch, "n_seeds": len(aucs_c),
                    "n_test_mean": float(np.mean(ns)) if ns else float("nan"),
                    "auc_mean": float(np.mean(aucs_c)) if aucs_c else float("nan"),
                    "auc_std": float(np.std(aucs_c, ddof=1)) if len(aucs_c) > 1 else 0.0,
                    "f1_mean": float(np.mean(f1s_c)) if f1s_c else float("nan"),
                    "f1_std": float(np.std(f1s_c, ddof=1)) if len(f1s_c) > 1 else 0.0,
                    "acc_mean": float(np.mean(accs_c)) if accs_c else float("nan"),
                    "acc_std": float(np.std(accs_c, ddof=1)) if len(accs_c) > 1 else 0.0,
                })
    pd.DataFrame(rows).to_csv(path, index=False)


def write_summary_wide(summary: dict[str, Any], path: Path) -> None:
    rows = []
    for fs in FEATURE_SETS:
        for N in NS:
            for metric in ("auc", "f1", "acc", "auc_AR", "auc_0MC"):
                rows.append({
                    "feature_set": fs, "N": N, "metric": metric,
                    "mean": summary[fs][f"n{N}"][metric]["mean"],
                    "std": summary[fs][f"n{N}"][metric]["std"],
                })
    pd.DataFrame(rows).to_csv(path, index=False)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seeds", type=int, default=5)
    parser.add_argument("--trials", type=int, default=30)
    parser.add_argument("--inner-cv", type=int, default=5)
    parser.add_argument("--db-version", default="A2.2")
    parser.add_argument("--baseline-cycle", type=int, default=1)
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()

    if args.smoke:
        args.seeds = 1
        args.trials = 5
        args.inner_cv = 3

    RUNS_DIR.mkdir(parents=True, exist_ok=True)
    LOG_DIR.mkdir(parents=True, exist_ok=True)

    log_path = LOG_DIR / f"run_{_time.strftime('%Y%m%d_%H%M%S')}.log"
    log = setup_logging(log_path)
    log.info(
        f"Exp L — fs_a_only vs fs_a_plus_cap_c5 paired comparison. "
        f"seeds={args.seeds}, trials={args.trials}, inner_cv={args.inner_cv}, "
        f"db_version={args.db_version}, baseline_cycle={args.baseline_cycle}"
    )

    per_seed: list[dict[str, Any]] = []
    for seed in range(args.seeds):
        log.info(f"=== seed {seed} ===")
        seed_metrics: dict[str, Any] = {fs: {} for fs in FEATURE_SETS}
        seed_preds: list[pd.DataFrame] = []
        for N in NS:
            out, preds_df = run_seed_N(
                N, seed, args.trials, args.inner_cv,
                args.baseline_cycle, args.db_version, log,
            )
            for fs in FEATURE_SETS:
                seed_metrics[fs][N] = out[fs]
            seed_preds.append(preds_df)
        per_seed.append(seed_metrics)
        pd.concat(seed_preds, ignore_index=True).to_csv(
            RUNS_DIR / f"seed_{seed}.csv", index=False,
        )

    summary = aggregate(per_seed)
    (HERE / "summary.json").write_text(json.dumps(summary, indent=2, default=str))
    write_summary_wide(summary, HERE / "summary_wide.csv")
    write_cohort_csv(per_seed, HERE / "comparison_by_cohort.csv")

    log.info("=== Summary (mean ± std across seeds) ===")
    for N in NS:
        a = summary["fs_a_only"][f"n{N}"]
        b = summary["fs_a_plus_cap_c5"][f"n{N}"]
        d = summary["delta"][f"n{N}"]
        log.info(
            f"  N={N}: AUC {a['auc']['mean']:.4f}±{a['auc']['std']:.4f} "
            f"→ {b['auc']['mean']:.4f}±{b['auc']['std']:.4f} (Δ={d['auc']:+.4f}); "
            f"F1 {a['f1']['mean']:.4f} → {b['f1']['mean']:.4f} (Δ={d['f1']:+.4f}); "
            f"AR-AUC Δ={d['auc_AR']:+.4f}; 0MC-AUC Δ={d['auc_0MC']:+.4f}"
        )
    log.info(f"Wrote summary.json + summary_wide.csv + comparison_by_cohort.csv")
    return 0


if __name__ == "__main__":
    sys.exit(main())

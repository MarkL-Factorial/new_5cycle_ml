#!/usr/bin/env python
"""Exp K — Held-out validation of the production K=5 ensemble.

Hyperparameters are FROZEN — loaded from production's best_params.json
(no re-tuning). For each of 5 random 80/20 splits, the 5 production
hyperparameter sets are refit on the 80% and ensemble-averaged on the
20%. Reports test-set metrics with mean±std across seeds, alongside
the production inner-CV value for comparison.

Run from cell_lifetime/:

    python experiments/exp_k_production_validation/run.py --seeds 5

Smoke (1 seed):

    python experiments/exp_k_production_validation/run.py --smoke
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
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from sklearn.model_selection import train_test_split
from sksurv.metrics import concordance_index_censored, cumulative_dynamic_auc
from sksurv.util import Surv

from cell_lifetime.data.loader import load_dataset
from cell_lifetime.models.ebm_classifier import EBMClassifierModel
from cell_lifetime.models.rsf import RSFModel
from cell_lifetime.pipelines.production import median_survival_from_sf


HERE = Path(__file__).resolve().parent
RUNS_DIR = HERE / "runs"
LOG_DIR = HERE / "run_logs"
DEFAULT_PROD_RUN = (
    Path(__file__).resolve().parents[2] / "results" / "run" / "20260519_1145"
)
NS = (200, 300, 400)


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
    return logging.getLogger(__name__)


def load_production_hyperparams(prod_run: Path) -> dict[str, Any]:
    """Read best_params.json and return per-model hyperparameter lists."""
    bp = json.loads((prod_run / "best_params.json").read_text())
    models = bp["models"]
    return {
        "ebm_classifier": {
            N: models[f"ebm_classifier_n{N}"]["best_params_per_seed"]
            for N in NS
        },
        "ebm_classifier_inner_cv_mean": {
            N: models[f"ebm_classifier_n{N}"]["inner_cv_auc_mean"]
            for N in NS
        },
        "rsf": models["rsf"]["best_params_per_seed"],
        "rsf_inner_cv_mean": models["rsf"]["inner_cv_cindex_mean"],
    }


def run_classifier_seed(
    N: int,
    seed: int,
    hp_list: list[dict[str, Any]],
    db_version: str,
    baseline_cycle: int,
    log: logging.Logger,
) -> tuple[dict[str, float], pd.DataFrame]:
    """Fit K=5 ensemble for one N at one seed, return metrics + per-cell preds."""
    ds = load_dataset(
        N=N, feature_subset="fs_a_only",
        baseline_cycle=baseline_cycle, db_version=db_version,
        min_n_regular=6, drop_excluded=True,
    )
    mask = ds.label_mask.astype(bool)
    X = ds.X.loc[mask].reset_index(drop=True)
    y = ds.y_class[mask]
    names = ds.cell_names[mask]

    X_tr, X_te, y_tr, y_te, names_tr, names_te = train_test_split(
        X, y, names, test_size=0.2, stratify=y, random_state=seed,
    )

    t0 = _time.time()
    probs = []
    for k, params in enumerate(hp_list):
        mdl = EBMClassifierModel(params)
        mdl.fit(X_tr, y_tr)
        probs.append(mdl.predict_proba(X_te)[:, 1])
    p_mean = np.mean(probs, axis=0)
    pred = (p_mean >= 0.5).astype(int)

    auc = float(roc_auc_score(y_te, p_mean))
    f1 = float(f1_score(y_te, pred))
    acc = float(accuracy_score(y_te, pred))

    log.info(
        f"  [seed={seed}] classifier N={N}: "
        f"n_train={len(y_tr)}, n_test={len(y_te)}, "
        f"pass_rate_test={float(y_te.mean()):.3f}, "
        f"AUC={auc:.4f}, F1={f1:.4f}, ACC={acc:.4f} "
        f"(t={_time.time() - t0:.1f}s)"
    )

    preds_df = pd.DataFrame({
        "seed": seed,
        "task": f"classifier_n{N}",
        "cell_name": names_te,
        "y_true": y_te,
        "y_pred": pred,
        "y_prob": p_mean,
    })
    metrics = {
        f"auc_n{N}": auc,
        f"f1_n{N}": f1,
        f"acc_n{N}": acc,
        f"n_test_n{N}": len(y_te),
        f"n_train_n{N}": len(y_tr),
        f"pass_rate_test_n{N}": float(y_te.mean()),
    }
    return metrics, preds_df


def run_rsf_seed(
    seed: int,
    hp_list: list[dict[str, Any]],
    db_version: str,
    baseline_cycle: int,
    log: logging.Logger,
) -> tuple[dict[str, float], pd.DataFrame]:
    """Fit K=5 RSF ensemble at one seed, return metrics + per-cell preds."""
    ds = load_dataset(
        N=300, feature_subset="fs_cv",
        baseline_cycle=baseline_cycle, db_version=db_version,
        min_n_regular=6, drop_excluded=True,
    )
    X = ds.X.reset_index(drop=True)
    time = ds.time.astype(np.int64)
    event = ds.event.astype(bool)
    names = ds.cell_names

    idx_tr, idx_te = train_test_split(
        np.arange(len(ds)), test_size=0.2, random_state=seed,
    )

    X_tr, X_te = X.iloc[idx_tr], X.iloc[idx_te]
    time_tr, time_te = time[idx_tr], time[idx_te]
    event_tr, event_te = event[idx_tr], event[idx_te]
    names_te = names[idx_te]

    t0 = _time.time()
    risks: list[np.ndarray] = []
    medians: list[np.ndarray] = []
    t_cap = float(time_tr.max())
    for k, params in enumerate(hp_list):
        mdl = RSFModel({**params, "low_memory": False, "random_state": seed * 10 + k})
        mdl.fit(X_tr, time=time_tr, event=event_tr)
        risks.append(mdl.predict(X_te))
        sfs = mdl.predict_survival_curve(X_te)
        medians.append(
            np.array([median_survival_from_sf(sf, t_cap) for sf in sfs], dtype=float)
        )

    risk_e = np.mean(risks, axis=0)
    med_e = np.mean(medians, axis=0)

    c, *_ = concordance_index_censored(event_te, time_te, risk_e)
    c = float(c)

    # AUC at horizons via cumulative_dynamic_auc — must use train arrays as
    # the reference distribution and test arrays for the held-out signal.
    train_surv = Surv.from_arrays(event_tr, time_tr)
    test_surv = Surv.from_arrays(event_te, time_te)
    auc_at: dict[int, float] = {}
    for Nh in NS:
        try:
            auc_arr, _ = cumulative_dynamic_auc(
                train_surv, test_surv, risk_e, times=[Nh],
            )
            auc_at[Nh] = float(auc_arr[0])
        except Exception as e:
            log.warning(f"  [seed={seed}] AUC@{Nh} failed: {e}")
            auc_at[Nh] = float("nan")

    # MAE / RMSE on faded test cells (have ground-truth cycle life)
    faded_mask = event_te
    if faded_mask.any():
        err = med_e[faded_mask] - time_te[faded_mask].astype(float)
        mae_faded = float(np.mean(np.abs(err)))
        rmse_faded = float(np.sqrt(np.mean(err ** 2)))
    else:
        mae_faded = float("nan")
        rmse_faded = float("nan")

    log.info(
        f"  [seed={seed}] rsf: "
        f"n_train={len(idx_tr)}, n_test={len(idx_te)} "
        f"(faded={int(faded_mask.sum())}), "
        f"C-index={c:.4f}, AUC@200/300/400={auc_at[200]:.4f}/{auc_at[300]:.4f}/{auc_at[400]:.4f}, "
        f"MAE_faded={mae_faded:.1f}, RMSE_faded={rmse_faded:.1f} "
        f"(t={_time.time() - t0:.1f}s)"
    )

    preds_df = pd.DataFrame({
        "seed": seed,
        "task": "rsf",
        "cell_name": names_te,
        "event": event_te.astype(int),
        "time": time_te,
        "risk_ensemble": risk_e,
        "median_cycle_ensemble": med_e,
    })

    metrics = {
        "cindex": c,
        "auc_at_n200": auc_at[200],
        "auc_at_n300": auc_at[300],
        "auc_at_n400": auc_at[400],
        "mae_faded": mae_faded,
        "rmse_faded": rmse_faded,
        "n_test_rsf": len(idx_te),
        "n_test_rsf_faded": int(faded_mask.sum()),
        "n_train_rsf": len(idx_tr),
    }
    return metrics, preds_df


def run_seed(
    seed: int,
    hp: dict[str, Any],
    db_version: str,
    baseline_cycle: int,
    log: logging.Logger,
) -> tuple[dict[str, float], pd.DataFrame]:
    """All 4 tasks for one seed."""
    all_metrics: dict[str, float] = {"seed": seed}
    all_preds: list[pd.DataFrame] = []
    for N in NS:
        m, df = run_classifier_seed(
            N, seed, hp["ebm_classifier"][N], db_version, baseline_cycle, log,
        )
        all_metrics.update(m)
        all_preds.append(df)
    m, df = run_rsf_seed(
        seed, hp["rsf"], db_version, baseline_cycle, log,
    )
    all_metrics.update(m)
    all_preds.append(df)
    return all_metrics, pd.concat(all_preds, axis=0, ignore_index=True)


def aggregate(per_seed: list[dict[str, float]], hp: dict[str, Any]) -> dict[str, Any]:
    """Mean ± std across seeds, with production inner-CV reference."""
    keys = [k for k in per_seed[0] if k != "seed"]
    out: dict[str, Any] = {}
    for k in keys:
        vals = [s[k] for s in per_seed if not (isinstance(s[k], float) and np.isnan(s[k]))]
        if not vals:
            out[k] = {"mean": float("nan"), "std": float("nan"), "per_seed": [s[k] for s in per_seed]}
            continue
        out[k] = {
            "mean": float(np.mean(vals)),
            "std": float(np.std(vals, ddof=1)) if len(vals) > 1 else 0.0,
            "per_seed": [s[k] for s in per_seed],
        }
    # Inject production reference values for AUC / C-index comparison
    out["_production_reference"] = {
        "ebm_classifier_inner_cv_auc_mean": hp["ebm_classifier_inner_cv_mean"],
        "rsf_inner_cv_cindex_mean": hp["rsf_inner_cv_mean"],
    }
    return out


def write_summary_wide(per_seed: list[dict[str, float]], path: Path) -> None:
    rows = []
    keys = [k for k in per_seed[0] if k != "seed"]
    for k in keys:
        vals = [s[k] for s in per_seed if not (isinstance(s[k], float) and np.isnan(s[k]))]
        rows.append({
            "metric": k,
            "mean": float(np.mean(vals)) if vals else float("nan"),
            "std": float(np.std(vals, ddof=1)) if len(vals) > 1 else 0.0,
            "n_seeds": len(vals),
        })
    pd.DataFrame(rows).to_csv(path, index=False)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seeds", type=int, default=5)
    parser.add_argument("--db-version", default="A2.2")
    parser.add_argument("--baseline-cycle", type=int, default=1)
    parser.add_argument(
        "--prod-run", default=str(DEFAULT_PROD_RUN),
        help="Path to the production run dir holding best_params.json",
    )
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()

    if args.smoke:
        args.seeds = 1

    RUNS_DIR.mkdir(parents=True, exist_ok=True)
    LOG_DIR.mkdir(parents=True, exist_ok=True)

    log_path = LOG_DIR / f"run_{_time.strftime('%Y%m%d_%H%M%S')}.log"
    log = setup_logging(log_path)
    log.info(
        f"Exp K — production-ensemble validation. seeds={args.seeds}, "
        f"db_version={args.db_version}, baseline_cycle={args.baseline_cycle}, "
        f"prod_run={args.prod_run}"
    )

    hp = load_production_hyperparams(Path(args.prod_run))
    log.info(
        f"Loaded hyperparams: classifier N=200/300/400 inner-CV AUC mean = "
        f"{hp['ebm_classifier_inner_cv_mean'][200]:.4f} / "
        f"{hp['ebm_classifier_inner_cv_mean'][300]:.4f} / "
        f"{hp['ebm_classifier_inner_cv_mean'][400]:.4f}; "
        f"rsf inner-CV C-index mean = {hp['rsf_inner_cv_mean']:.4f}"
    )

    per_seed: list[dict[str, float]] = []
    for seed in range(args.seeds):
        log.info(f"=== seed {seed} ===")
        metrics, preds = run_seed(seed, hp, args.db_version, args.baseline_cycle, log)
        per_seed.append(metrics)
        preds.to_csv(RUNS_DIR / f"seed_{seed}.csv", index=False)

    summary = aggregate(per_seed, hp)
    (HERE / "summary.json").write_text(json.dumps(summary, indent=2, default=str))
    write_summary_wide(per_seed, HERE / "summary_wide.csv")

    log.info("=== Summary (mean ± std across seeds) ===")
    for N in NS:
        auc = summary[f"auc_n{N}"]
        ref = hp["ebm_classifier_inner_cv_mean"][N]
        log.info(
            f"  classifier N={N}: AUC = {auc['mean']:.4f} ± {auc['std']:.4f}  "
            f"(prod inner-CV mean = {ref:.4f}, delta = {auc['mean'] - ref:+.4f})"
        )
    c = summary["cindex"]
    log.info(
        f"  rsf: C-index = {c['mean']:.4f} ± {c['std']:.4f}  "
        f"(prod inner-CV mean = {hp['rsf_inner_cv_mean']:.4f}, "
        f"delta = {c['mean'] - hp['rsf_inner_cv_mean']:+.4f})"
    )
    for Nh in NS:
        a = summary[f"auc_at_n{Nh}"]
        log.info(f"  rsf AUC@{Nh}: {a['mean']:.4f} ± {a['std']:.4f}")
    mae = summary["mae_faded"]
    log.info(f"  rsf MAE on faded test cells: {mae['mean']:.1f} ± {mae['std']:.1f} cyc")
    log.info(f"Wrote summary.json + summary_wide.csv to {HERE}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

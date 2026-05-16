#!/usr/bin/env python
"""Ensemble blend of xgb_aft + rsf survival predictions.

For each (N, feature_subset) tuple, locate the latest run dirs for
xgb_aft and rsf, load their `predictions.csv` (per-cell test predictions
across all seeds), z-score normalize each model's risk scores, average,
and recompute C-index + AUC@N for the blend. Compares the blend against
the best single model.

Reads only — no new training runs.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from cell_lifetime.evaluation.survival_metrics import survival_metrics


HERE = Path(__file__).resolve().parent
PKG_DIR = HERE.parents[1]
RUNS_DIR = PKG_DIR / "out" / "runs" / "survival"


def _latest_run(model: str, N: int, features: str) -> Path | None:
    """Return the latest run dir matching slug pattern, or None if missing."""
    pattern = f"{model}__survival__N{N}__A2.2_b1__{features}__*"
    hits = sorted(RUNS_DIR.glob(pattern))
    return hits[-1] if hits else None


def _zscore(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=float)
    mu = float(np.mean(x))
    sd = float(np.std(x))
    if sd < 1e-12:
        return np.zeros_like(x)
    return (x - mu) / sd


def _blend_one_horizon(N: int, features: str) -> dict | None:
    aft_dir = _latest_run("xgb_aft", N, features)
    rsf_dir = _latest_run("rsf", N, features)
    if aft_dir is None or rsf_dir is None:
        print(f"[blend] N={N} {features}: missing aft={aft_dir}, rsf={rsf_dir}")
        return None
    aft = pd.read_csv(aft_dir / "predictions.csv")
    rsf = pd.read_csv(rsf_dir / "predictions.csv")
    # Predictions live as (seed, cell_name, time, event, raw_predict, risk_score).
    # The same train/test split was used per seed (StratifiedKFold seed=seed), so
    # cells appearing in each seed should match across aft and rsf.
    merged = aft.merge(
        rsf,
        on=["seed", "cell_name", "time", "event", "cohort"],
        suffixes=("_aft", "_rsf"),
    )
    if merged.empty:
        print(f"[blend] N={N} {features}: 0 overlap rows after merge")
        return None

    # Per-seed z-score normalization of each model's risk_score, then average.
    blend_per_seed = []
    for seed, sub in merged.groupby("seed"):
        z_aft = _zscore(sub["risk_score_aft"].values)
        z_rsf = _zscore(sub["risk_score_rsf"].values)
        sub = sub.assign(blend_risk=(z_aft + z_rsf) / 2.0)
        blend_per_seed.append(sub)
    blended = pd.concat(blend_per_seed, ignore_index=True)

    # Per-seed C-index + AUC for each model and the blend; then mean/std across seeds.
    rows = []
    for seed, sub in blended.groupby("seed"):
        for name, risk in (
            ("xgb_aft", sub["risk_score_aft"].values),
            ("rsf", sub["risk_score_rsf"].values),
            ("blend_zscore_avg", sub["blend_risk"].values),
        ):
            m = survival_metrics(sub["event"].values, sub["time"].values, risk)
            rows.append({"model": name, "seed": int(seed),
                         "c_index": m["c_index"],
                         f"auc_at_{N}": m.get(f"auc_at_{N}", float("nan"))})
    df = pd.DataFrame(rows)

    summary = {}
    for name in ("xgb_aft", "rsf", "blend_zscore_avg"):
        sub = df[df["model"] == name]
        summary[name] = {
            "c_index_mean": float(sub["c_index"].mean()),
            "c_index_std": float(sub["c_index"].std()),
            f"auc_at_{N}_mean": float(sub[f"auc_at_{N}"].mean()),
            f"auc_at_{N}_std": float(sub[f"auc_at_{N}"].std()),
            "n_seeds": int(sub["seed"].nunique()),
        }
    return {"N": N, "features": features, **summary, "per_seed": df.to_dict(orient="records")}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--features", default="fs_all", choices=["fs_cv", "fs_all"])
    ap.add_argument("--horizons", default="200,300,400")
    args = ap.parse_args()

    horizons = [int(x) for x in args.horizons.split(",")]
    out_records = []
    for N in horizons:
        result = _blend_one_horizon(N, args.features)
        if result is None:
            continue
        out_records.append(result)
        print()
        print(f"=== N={N} {args.features} ===")
        for name in ("xgb_aft", "rsf", "blend_zscore_avg"):
            r = result[name]
            print(f"  {name:20s}  C-index = {r['c_index_mean']:.4f} ± {r['c_index_std']:.4f}   "
                  f"AUC@{N} = {r[f'auc_at_{N}_mean']:.4f} ± {r[f'auc_at_{N}_std']:.4f}   "
                  f"(n_seeds={r['n_seeds']})")
        # Verdict
        best_single = max(result["xgb_aft"]["c_index_mean"], result["rsf"]["c_index_mean"])
        blend_c = result["blend_zscore_avg"]["c_index_mean"]
        blend_std = result["blend_zscore_avg"]["c_index_std"]
        if blend_c > best_single + blend_std:
            verdict = f"BLEND WINS by {blend_c - best_single:+.4f} (>1 std={blend_std:.4f})"
        elif blend_c > best_single:
            verdict = f"blend nudges up by {blend_c - best_single:+.4f} but inside 1 std"
        else:
            verdict = f"blend does NOT improve (Δ={blend_c - best_single:+.4f})"
        print(f"  → {verdict}")

    out_path = HERE / f"blend_summary_{args.features}.json"
    out_path.write_text(json.dumps(out_records, indent=2, default=str))
    print(f"\n[blend] wrote {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

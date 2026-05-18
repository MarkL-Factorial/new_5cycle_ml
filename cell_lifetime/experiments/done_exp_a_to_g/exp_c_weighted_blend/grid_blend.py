#!/usr/bin/env python
"""Experiment C: weighted blend grid search over predictions.csv.

For each N ∈ {200, 300, 400} and each seed, computes
    blend_risk(w) = w · z(rsf.risk) + (1 - w) · z(aft.risk)
for w ∈ {0.0, 0.1, ..., 1.0}, then recomputes C-index + AUC@N on the
test set. Reports the optimal w per N (max mean C-index across seeds).

Pure post-processing — no model training. Reads from
out/runs/survival/{model}__survival__N{N}__A2.2_b1__fs_cv__*/predictions.csv.

Exits 0 always; the interesting output is the printed table + the
saved blend_curves.csv.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from cell_lifetime.evaluation.survival_metrics import survival_metrics


HERE = Path(__file__).resolve().parent
PKG_DIR = HERE.parents[1]
RUNS_DIR = PKG_DIR / "out" / "runs" / "survival"

WEIGHTS = np.round(np.arange(0.0, 1.0001, 0.1), 2)


def _latest_run(model: str, N: int, features: str = "fs_cv") -> Path | None:
    hits = sorted(RUNS_DIR.glob(f"{model}__survival__N{N}__A2.2_b1__{features}__*"))
    return hits[-1] if hits else None


def _zscore(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=float)
    mu, sd = float(np.mean(x)), float(np.std(x))
    if sd < 1e-12:
        return np.zeros_like(x)
    return (x - mu) / sd


def _grid_one_horizon(N: int) -> tuple[pd.DataFrame, dict]:
    aft_dir = _latest_run("xgb_aft", N)
    rsf_dir = _latest_run("rsf", N)
    if aft_dir is None or rsf_dir is None:
        print(f"[exp_c] N={N}: missing aft={aft_dir!r}, rsf={rsf_dir!r}")
        return pd.DataFrame(), {}

    aft = pd.read_csv(aft_dir / "predictions.csv")
    rsf = pd.read_csv(rsf_dir / "predictions.csv")
    merged = aft.merge(rsf, on=["seed", "cell_name", "time", "event", "cohort"],
                       suffixes=("_aft", "_rsf"))

    rows = []
    for seed, sub in merged.groupby("seed"):
        z_aft = _zscore(sub["risk_score_aft"].values)
        z_rsf = _zscore(sub["risk_score_rsf"].values)
        # Per-model anchors at the grid endpoints — useful sanity check
        for w in WEIGHTS:
            blend = w * z_rsf + (1 - w) * z_aft
            m = survival_metrics(sub["event"].values, sub["time"].values, blend)
            rows.append({
                "seed": int(seed), "N": N, "w_rsf": float(w),
                "c_index": m["c_index"],
                f"auc_at_{N}": m.get(f"auc_at_{N}", float("nan")),
            })
    df = pd.DataFrame(rows)

    # Aggregate across seeds for each w
    agg = df.groupby("w_rsf").agg(
        c_mean=("c_index", "mean"),
        c_std=("c_index", "std"),
        auc_mean=(f"auc_at_{N}", "mean"),
        auc_std=(f"auc_at_{N}", "std"),
    ).reset_index()

    # Find optimal w
    best_idx = int(agg["c_mean"].idxmax())
    best_w = float(agg.loc[best_idx, "w_rsf"])
    best_c_mean = float(agg.loc[best_idx, "c_mean"])
    best_c_std = float(agg.loc[best_idx, "c_std"])

    # Compare against single-model endpoints
    aft_c = float(agg[agg["w_rsf"] == 0.0]["c_mean"].iloc[0])
    rsf_c = float(agg[agg["w_rsf"] == 1.0]["c_mean"].iloc[0])
    max_single = max(aft_c, rsf_c)
    delta = best_c_mean - max_single
    verdict = (
        "BLEND WINS" if (best_w not in (0.0, 1.0) and delta > best_c_std)
        else ("RSF Pareto-optimal" if best_w == 1.0 else
              "AFT Pareto-optimal" if best_w == 0.0 else
              "blend nudges (inside std)")
    )

    summary = {
        "N": N, "optimal_w_rsf": best_w,
        "blend_c_index_mean": best_c_mean, "blend_c_index_std": best_c_std,
        "aft_alone_c_index": aft_c, "rsf_alone_c_index": rsf_c,
        "max_single_c_index": max_single, "delta_vs_max_single": delta,
        "verdict": verdict,
    }

    # Long-form for the printed table
    return df, summary


def main() -> int:
    HERE.mkdir(parents=True, exist_ok=True)
    all_long: list[pd.DataFrame] = []
    summaries: list[dict] = []
    for N in (200, 300, 400):
        df, summ = _grid_one_horizon(N)
        if df.empty:
            continue
        all_long.append(df)
        summaries.append(summ)

    if not all_long:
        print("[exp_c] no horizons produced output; check that exp_b predictions exist.")
        return 1

    long_df = pd.concat(all_long, ignore_index=True)
    long_df.to_csv(HERE / "blend_curves_long.csv", index=False)

    # Wide pivot: rows=w_rsf, cols=(N, metric, mean/std)
    pivot_rows = []
    for w in WEIGHTS:
        row = {"w_rsf": float(w)}
        for N in (200, 300, 400):
            sub = long_df[(long_df["N"] == N) & (long_df["w_rsf"] == float(w))]
            if sub.empty:
                row[f"N{N}_c_mean"] = float("nan")
                row[f"N{N}_c_std"] = float("nan")
                continue
            row[f"N{N}_c_mean"] = float(sub["c_index"].mean())
            row[f"N{N}_c_std"] = float(sub["c_index"].std())
        pivot_rows.append(row)
    pivot_df = pd.DataFrame(pivot_rows)
    pivot_df.to_csv(HERE / "blend_curves.csv", index=False)

    # Print headline
    print("Weighted blend grid (mean ± std across 5 seeds, fs_cv):")
    print("=" * 78)
    for s in summaries:
        N = s["N"]
        print(f"\nN={N}")
        print(f"  AFT alone (w=0.0):  C-index = {s['aft_alone_c_index']:.4f}")
        print(f"  RSF alone (w=1.0):  C-index = {s['rsf_alone_c_index']:.4f}")
        print(f"  Optimal w_rsf:       {s['optimal_w_rsf']:.1f}")
        print(f"  Best blend C-index:  {s['blend_c_index_mean']:.4f} ± {s['blend_c_index_std']:.4f}")
        print(f"  Δ vs max(single):    {s['delta_vs_max_single']:+.4f}")
        print(f"  → {s['verdict']}")

    print()
    print("Full curve (C-index mean, by w_rsf):")
    print("=" * 78)
    cols = ["w_rsf", "N200_c_mean", "N300_c_mean", "N400_c_mean"]
    print(pivot_df[cols].to_string(index=False,
                                    formatters={c: "{:.4f}".format for c in cols[1:]}))

    (HERE / "summary.json").write_text(json.dumps(summaries, indent=2))
    print(f"\n[exp_c] wrote {HERE / 'blend_curves.csv'}, {HERE / 'blend_curves_long.csv'}, {HERE / 'summary.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

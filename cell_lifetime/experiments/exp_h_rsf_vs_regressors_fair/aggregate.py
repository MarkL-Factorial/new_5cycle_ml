#!/usr/bin/env python
"""Exp H aggregator: pretty tables from `metric_long.csv`.

Tables produced:
  1. Headline: mean ± std for MAE, MAPE, RMSE, R² across 3 models.
  2. Per-quartile MAE: how each model handles Q1 (shortest-lived) ...
     Q4 (longest-lived) cells.
  3. Runtime: how long each model took on average.

Reads `metric_long.csv` written by `run.py`.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


HERE = Path(__file__).resolve().parent
LONG_CSV = HERE / "metric_long.csv"


def fmt(mean: float, std: float, decimals: int = 3) -> str:
    if np.isnan(mean):
        return "—"
    return f"{mean:.{decimals}f} ± {std:.{decimals}f}"


def main() -> int:
    if not LONG_CSV.exists():
        print(f"No {LONG_CSV}; run run.py first.")
        return 1
    df = pd.read_csv(LONG_CSV)

    # Pivot: rows = (model, metric), columns = seed, values = value.
    # Then mean/std across seeds.
    summary = (
        df.groupby(["model", "metric"])["value"]
        .agg(["mean", "std", "count"])
        .reset_index()
    )

    models = ["rsf", "xgb_regressor", "ebm_regressor"]

    def get(metric: str, model: str) -> tuple[float, float]:
        row = summary[(summary["model"] == model) & (summary["metric"] == metric)]
        if row.empty:
            return float("nan"), float("nan")
        return float(row["mean"].iloc[0]), float(row["std"].iloc[0])

    # ----- 1. Headline -----
    print("=" * 90)
    print("Headline: cycle-life regression on held-out 20% faded cells (mean ± std, 5 seeds)")
    print("=" * 90)
    print(f"{'Model':<15} {'MAE (cyc)':>16} {'MAPE':>16} {'RMSE':>16} {'R²':>16}")
    for model in models:
        bits = [f"{model:<15}"]
        for m, dec in (("mae", 2), ("mape", 3), ("rmse", 2), ("r2", 3)):
            mn, sd = get(m, model)
            bits.append(f"{fmt(mn, sd, dec):>16}")
        print(" ".join(bits))

    # ----- 2. Per-quartile MAE -----
    print()
    print("=" * 90)
    print("Per-quartile MAE on test set (Q1: shortest-lived 25% ... Q4: longest-lived 25%)")
    print("=" * 90)
    print(f"{'Model':<15} {'Q1':>16} {'Q2':>16} {'Q3':>16} {'Q4':>16}")
    for model in models:
        bits = [f"{model:<15}"]
        for q in ("mae_q1", "mae_q2", "mae_q3", "mae_q4"):
            mn, sd = get(q, model)
            bits.append(f"{fmt(mn, sd, 1):>16}")
        print(" ".join(bits))

    # ----- 3. Runtime -----
    print()
    print("=" * 90)
    print("Runtime per seed (mean ± std, seconds)")
    print("=" * 90)
    print(f"{'Model':<15} {'Time (s)':>16}")
    for model in models:
        mn, sd = get("runtime_s", model)
        print(f"{model:<15} {fmt(mn, sd, 1):>16}")

    # ----- 4. Long → wide table for further analysis -----
    wide = summary.pivot_table(
        index="model", columns="metric", values="mean", aggfunc="first"
    )
    out_wide = HERE / "summary_wide.csv"
    wide.to_csv(out_wide)
    print(f"\n[exp_h] wrote {out_wide}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())

#!/usr/bin/env python
"""Exp I aggregator: side-by-side RSF-with vs RSF-no on the same Q4 cells."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


HERE = Path(__file__).resolve().parent
LONG_CSV = HERE / "metric_long.csv"


def fmt(mean: float, std: float, decimals: int = 2) -> str:
    if np.isnan(mean):
        return "—"
    return f"{mean:.{decimals}f} ± {std:.{decimals}f}"


def main() -> int:
    if not LONG_CSV.exists():
        print(f"No {LONG_CSV}; run run.py first.")
        return 1
    df = pd.read_csv(LONG_CSV)

    summary = (
        df.groupby(["variant", "metric"])["value"]
        .agg(["mean", "std", "count"])
        .reset_index()
    )

    variants = ["rsf_with_censored", "rsf_no_censored"]

    def get(metric: str, variant: str) -> tuple[float, float]:
        row = summary[(summary["variant"] == variant) & (summary["metric"] == metric)]
        if row.empty:
            return float("nan"), float("nan")
        return float(row["mean"].iloc[0]), float(row["std"].iloc[0])

    # ----- Headline -----
    print("=" * 92)
    print("Exp I — RSF censored-data ablation (5 seeds, fs_cv, same 20% faded test cells)")
    print("=" * 92)
    print(f"{'Variant':<22} {'MAE (cyc)':>16} {'MAPE':>16} {'RMSE':>16} {'R²':>16}")
    for v in variants:
        bits = [f"{v:<22}"]
        for m, dec in (("mae", 2), ("mape", 3), ("rmse", 2), ("r2", 3)):
            mn, sd = get(m, v)
            bits.append(f"{fmt(mn, sd, dec):>16}")
        print(" ".join(bits))

    # ----- Per-quartile -----
    print()
    print("=" * 92)
    print("Per-quartile MAE — does censored data specifically help Q4 (long-lived)?")
    print("=" * 92)
    print(f"{'Variant':<22} {'Q1':>16} {'Q2':>16} {'Q3':>16} {'Q4':>16}")
    for v in variants:
        bits = [f"{v:<22}"]
        for q in ("mae_q1", "mae_q2", "mae_q3", "mae_q4"):
            mn, sd = get(q, v)
            bits.append(f"{fmt(mn, sd, 1):>16}")
        print(" ".join(bits))

    # ----- Δ table — censored uplift = with − without -----
    print()
    print("=" * 92)
    print("Δ (RSF-no-censored MINUS RSF-with-censored). Positive = removing censoring hurts.")
    print("=" * 92)
    print(f"{'Metric':<22} {'Δ value':>16}")
    for m in ("mae", "rmse", "mape", "mae_q1", "mae_q2", "mae_q3", "mae_q4"):
        mn_w, _ = get(m, "rsf_with_censored")
        mn_n, _ = get(m, "rsf_no_censored")
        delta = mn_n - mn_w
        print(f"{m:<22} {delta:>+16.3f}")

    out_wide = HERE / "summary_wide.csv"
    summary.pivot_table(
        index="variant", columns="metric", values="mean", aggfunc="first"
    ).to_csv(out_wide)
    print(f"\n[exp_i] wrote {out_wide}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

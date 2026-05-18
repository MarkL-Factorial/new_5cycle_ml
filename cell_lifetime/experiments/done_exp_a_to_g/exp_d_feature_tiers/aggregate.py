#!/usr/bin/env python
"""Experiment D aggregator: compare per-tier feature subsets.

Walks out/runs/, picks the latest 5-seed run per (model, feature_subset)
at N=300, and prints a pivot table: rows=model, cols=feature_subset,
values=headline metric ± std.

Includes Exp A's fs_cv and fs_all results for direct comparison.
"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd


HERE = Path(__file__).resolve().parent
PKG_DIR = HERE.parents[1]
RUNS_DIR = PKG_DIR / "out" / "runs"

SUBSETS = ["fs_a_only", "fs_b_only", "fs_ab", "fs_cv", "fs_c_only", "fs_all"]
HEADLINE = {
    "classification": ("test_f1_mean", "test_f1_std", "lower_is_better=False"),
    "survival":       ("test_c_index_mean", "test_c_index_std", "lower_is_better=False"),
}


def _latest(task: str, model: str, features: str) -> dict | None:
    pattern = f"{model}__{task}__N300__A2.2_b1__{features}__*"
    hits = sorted((RUNS_DIR / task).glob(pattern))
    if not hits:
        return None
    sj = hits[-1] / "summary.json"
    if not sj.exists():
        return None
    d = json.loads(sj.read_text())
    if d.get("n_seeds", 0) < 5:
        return None
    return d


def main() -> int:
    print("Experiment D — feature-tier ablation (N=300, 5 seeds, fs_cv/fs_all in for reference)")
    print("=" * 95)
    rows = []
    for task in ("classification", "survival"):
        metric_key, std_key, _ = HEADLINE[task]
        for model in (["xgb_classifier"] if task == "classification" else ["rsf"]):
            print(f"\n[{task}] {model}")
            line = f"  {'feature_subset':<14s}{'n_cols':>8s}{'headline':>22s}"
            print(line)
            for fs in SUBSETS:
                d = _latest(task, model, fs)
                if d is None:
                    print(f"  {fs:<14s}{'?':>8s}{'(missing)':>22s}")
                    continue
                mean = d.get(metric_key, float("nan"))
                std = d.get(std_key, float("nan"))
                n_features = d.get("n_features", "?")
                head = f"{mean:.4f} ± {std:.4f}"
                print(f"  {fs:<14s}{str(n_features):>8s}{head:>22s}")
                rows.append({
                    "task": task, "model": model, "feature_subset": fs,
                    "n_features": n_features, "metric": metric_key,
                    "mean": mean, "std": std,
                })

    out = pd.DataFrame(rows)
    out.to_csv(HERE / "tier_comparison.csv", index=False)
    print(f"\n[exp_d] wrote {HERE / 'tier_comparison.csv'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

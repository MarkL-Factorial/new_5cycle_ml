#!/usr/bin/env python
"""Aggregate Experiment A run outputs into a single long-form CSV + comparison table.

Walks cell_lifetime/out/runs/{task}/, opens each {slug}__{timestamp}/summary.json,
filters to runs from Experiment A (model × feature_subset matching one of the
known A grid points), and writes:

  experiments/exp_a_feature_set/metric_long.csv  -- one row per (slug, metric)
  experiments/exp_a_feature_set/headline.csv      -- the comparison table
  stdout                                          -- printed comparison
"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd


HERE = Path(__file__).resolve().parent
PKG_DIR = HERE.parents[1]
RUNS_DIR = PKG_DIR / "out" / "runs"

# Per-task headline metric (one that we sort/compare on)
HEADLINE = {
    "classification": ("test_f1_mean", "test_f1_std", "higher_is_better"),
    "regression":     ("test_mae_mean", "test_mae_std", "lower_is_better"),
    "survival":       ("test_c_index_mean", "test_c_index_std", "higher_is_better"),
}


def _walk_summaries() -> list[dict]:
    rows = []
    for task_dir in sorted(RUNS_DIR.glob("*")):
        if not task_dir.is_dir():
            continue
        task = task_dir.name
        for run_dir in sorted(task_dir.iterdir()):
            sj = run_dir / "summary.json"
            if not sj.exists():
                continue
            try:
                summary = json.loads(sj.read_text())
            except json.JSONDecodeError:
                continue
            # Add path provenance
            summary["_run_dir"] = str(run_dir.relative_to(PKG_DIR))
            summary["_task"] = task
            rows.append(summary)
    return rows


def main() -> int:
    summaries = _walk_summaries()
    if not summaries:
        print(f"No run summaries found under {RUNS_DIR}; did you run experiments/exp_a_feature_set/run.sh?")
        return 1

    # Long-form: one row per (slug, metric)
    long_rows = []
    for s in summaries:
        for k, v in s.items():
            if isinstance(v, (int, float)) and (k.startswith("test_") or k == "n_seeds" or k == "n_rows"):
                long_rows.append({
                    "model": s.get("model"),
                    "task": s.get("task"),
                    "feature_subset": s.get("feature_subset"),
                    "N": s.get("N"),
                    "baseline_cycle": s.get("baseline_cycle"),
                    "n_seeds": s.get("n_seeds"),
                    "n_rows": s.get("n_rows"),
                    "run_dir": s.get("_run_dir"),
                    "metric": k,
                    "value": v,
                })
    long_df = pd.DataFrame(long_rows)
    out_long = HERE / "metric_long.csv"
    long_df.to_csv(out_long, index=False)
    print(f"[aggregate] wrote {out_long} ({len(long_df)} rows)")

    # Headline comparison: latest run per (model, feature_subset)
    # (in case run.sh was rerun, take the row whose run_dir sorts last)
    summaries_df = pd.DataFrame(summaries)
    summaries_df = summaries_df.sort_values("_run_dir").drop_duplicates(
        subset=["model", "feature_subset", "N", "baseline_cycle"], keep="last"
    )

    # Build the headline table
    headline_rows = []
    for _, row in summaries_df.iterrows():
        task = row.get("task")
        metric_key, std_key, direction = HEADLINE.get(task, (None, None, None))
        if metric_key is None or metric_key not in row:
            continue
        headline_rows.append({
            "model": row["model"],
            "task": task,
            "feature_subset": row["feature_subset"],
            "N": row.get("N"),
            "n_seeds": row.get("n_seeds"),
            "headline_metric": metric_key.replace("test_", "").replace("_mean", ""),
            "mean": row[metric_key],
            "std": row.get(std_key, float("nan")),
            "direction": direction,
            "run_dir": row.get("_run_dir"),
        })
    headline_df = pd.DataFrame(headline_rows)
    headline_df = headline_df.sort_values(["task", "model", "feature_subset"])
    out_headline = HERE / "headline.csv"
    headline_df.to_csv(out_headline, index=False)
    print(f"[aggregate] wrote {out_headline}")

    # Pretty-print the comparison: pivot model × feature_subset → mean ± std
    print()
    print("Headline comparison (mean ± std across seeds):")
    print("-" * 80)
    for task in ("classification", "regression", "survival"):
        sub = headline_df[headline_df["task"] == task]
        if sub.empty:
            continue
        metric_name = sub["headline_metric"].iloc[0]
        direction = sub["direction"].iloc[0]
        print(f"\n[{task}] metric={metric_name} ({direction.replace('_', ' ')})")
        pivot_rows = []
        for model in sub["model"].unique():
            row = {"model": model}
            for fs in ("fs_cv", "fs_all"):
                hit = sub[(sub["model"] == model) & (sub["feature_subset"] == fs)]
                if hit.empty:
                    row[fs] = "(missing)"
                else:
                    mean = float(hit["mean"].iloc[0])
                    std = float(hit["std"].iloc[0]) if pd.notna(hit["std"].iloc[0]) else 0.0
                    row[fs] = f"{mean:.4f} ± {std:.4f}"
            # Improvement
            try:
                cv_mean = float(sub[(sub["model"] == model) & (sub["feature_subset"] == "fs_cv")]["mean"].iloc[0])
                all_mean = float(sub[(sub["model"] == model) & (sub["feature_subset"] == "fs_all")]["mean"].iloc[0])
                if direction == "lower_is_better":
                    delta_pct = -100.0 * (all_mean - cv_mean) / cv_mean
                else:
                    delta_pct = 100.0 * (all_mean - cv_mean) / cv_mean
                row["delta_pct"] = f"{delta_pct:+.1f}%"
            except (IndexError, ZeroDivisionError):
                row["delta_pct"] = "n/a"
            pivot_rows.append(row)
        pivot_df = pd.DataFrame(pivot_rows)
        print(pivot_df.to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

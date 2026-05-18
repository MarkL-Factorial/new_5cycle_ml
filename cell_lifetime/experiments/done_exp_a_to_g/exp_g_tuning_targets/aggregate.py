#!/usr/bin/env python
"""Exp G aggregator: pivot AUC- vs F1-tuned runs across models × N × subsets.

Reads `summary.json` from each run dir in `experiments/exp_g_tuning_targets/runs/`,
groups by (model, feature_subset, N, optimize), and prints two pivots:

  1. Held-out F1 (classification) or F1@N (survival)  — does F1-tuning win?
  2. Held-out ROC-AUC (classification) or AUC@N (survival) — does AUC-tuning win?

A long-form CSV is written for downstream plotting.
"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd


HERE = Path(__file__).resolve().parent
RUNS_DIR = HERE / "runs" / "runs"

# Map model name → (task, primary_metric, primary_at_N_metric)
TASK_PRIMARIES = {
    "xgb_classifier": ("classification", "f1", "roc_auc"),
    "ebm_classifier": ("classification", "f1", "roc_auc"),
    "rsf":            ("survival",       "f1_at_{N}", "auc_at_{N}"),
    "xgb_aft":        ("survival",       "f1_at_{N}", "auc_at_{N}"),
}


def _walk() -> list[dict]:
    rows = []
    for task_dir in sorted(RUNS_DIR.glob("*")):
        if not task_dir.is_dir():
            continue
        for run_dir in sorted(task_dir.iterdir()):
            sj = run_dir / "summary.json"
            mj = run_dir / "manifest.json"
            if not sj.exists() or not mj.exists():
                continue
            try:
                summary = json.loads(sj.read_text())
                manifest = json.loads(mj.read_text())
            except json.JSONDecodeError:
                continue
            # tune_objective comes from manifest.json's optimize_metric;
            # summary.json doesn't carry it.
            summary["tune_objective"] = manifest.get("optimize_metric", "?")
            summary["_run_dir"] = str(run_dir.relative_to(HERE))
            rows.append(summary)
    return rows


def main() -> int:
    summaries = _walk()
    if not summaries:
        print(f"No summaries under {RUNS_DIR}; run experiments/exp_g_tuning_targets/run.sh first.")
        return 1
    print(f"Loaded {len(summaries)} run summaries from {RUNS_DIR}")

    # Long-form rows: extract per-row F1 + AUC for both tasks
    rows = []
    for s in summaries:
        N = int(s["N"])
        model = s["model"]
        if model not in TASK_PRIMARIES:
            continue
        task, f1_key, auc_key = TASK_PRIMARIES[model]
        if task != s["task"]:
            continue
        f1_key = f1_key.format(N=N)
        auc_key = auc_key.format(N=N)
        rows.append({
            "model": model,
            "task": task,
            "feature_subset": s["feature_subset"],
            "N": N,
            "n_seeds": s.get("n_seeds", 0),
            "tune_objective": s.get("tune_objective", "?"),
            "test_f1_mean": s.get(f"test_{f1_key}_mean", float("nan")),
            "test_f1_std": s.get(f"test_{f1_key}_std", float("nan")),
            "test_auc_mean": s.get(f"test_{auc_key}_mean", float("nan")),
            "test_auc_std": s.get(f"test_{auc_key}_std", float("nan")),
            "c_index_mean": s.get("test_c_index_mean", float("nan")),  # survival only
            "c_index_std": s.get("test_c_index_std", float("nan")),
            "run_dir": s.get("_run_dir", ""),
        })
    df = pd.DataFrame(rows)
    # Keep only n_seeds>=5 (real runs) and de-duplicate to the latest per slug
    df = df[df["n_seeds"] >= 5]
    df = df.sort_values("run_dir").drop_duplicates(
        subset=["model", "feature_subset", "N", "tune_objective"], keep="last"
    )

    out_long = HERE / "metric_long.csv"
    df.to_csv(out_long, index=False)
    print(f"[exp_g] wrote {out_long} ({len(df)} rows)")

    # Pivot tables: rows = (model, feature_subset, N), cols = (tune_objective × metric)
    print()
    print("=" * 100)
    print("Held-out F1 (classification) / F1@N (survival): mean across 5 seeds")
    print("=" * 100)
    pivot_f1 = df.pivot_table(
        index=["model", "feature_subset", "N"],
        columns="tune_objective",
        values="test_f1_mean",
        aggfunc="first",
    )
    print(pivot_f1.to_string(float_format="{:.4f}".format))

    print()
    print("=" * 100)
    print("Held-out ROC-AUC (classification) / AUC@N (survival): mean across 5 seeds")
    print("=" * 100)
    pivot_auc = df.pivot_table(
        index=["model", "feature_subset", "N"],
        columns="tune_objective",
        values="test_auc_mean",
        aggfunc="first",
    )
    print(pivot_auc.to_string(float_format="{:.4f}".format))

    # ----- Pretty mean ± std tables (the user wants these alongside Δ) -----
    df["tune_simple"] = df["tune_objective"].map(
        lambda obj: "f1" if "f1" in str(obj) else "auc"
    )
    df["f1_str"] = df.apply(
        lambda r: f"{r['test_f1_mean']:.4f} ± {r['test_f1_std']:.4f}", axis=1
    )
    df["auc_str"] = df.apply(
        lambda r: f"{r['test_auc_mean']:.4f} ± {r['test_auc_std']:.4f}", axis=1
    )
    df["c_str"] = df.apply(
        lambda r: (
            f"{r['c_index_mean']:.4f} ± {r['c_index_std']:.4f}"
            if pd.notna(r["c_index_mean"]) else "—"
        ),
        axis=1,
    )

    print()
    print("=" * 100)
    print("Held-out F1 / F1@N (mean ± std): values, AUC-tuned vs F1-tuned")
    print("=" * 100)
    vt_f1 = df.pivot_table(
        index=["model", "feature_subset", "N"], columns="tune_simple",
        values="f1_str", aggfunc="first",
    )
    if {"auc", "f1"}.issubset(vt_f1.columns):
        vt_f1 = vt_f1[["auc", "f1"]].rename(
            columns={"auc": "AUC-tuned", "f1": "F1-tuned"}
        )
    print(vt_f1.to_string())

    print()
    print("=" * 100)
    print("Held-out ROC-AUC / AUC@N (mean ± std): values, AUC-tuned vs F1-tuned")
    print("=" * 100)
    vt_auc = df.pivot_table(
        index=["model", "feature_subset", "N"], columns="tune_simple",
        values="auc_str", aggfunc="first",
    )
    if {"auc", "f1"}.issubset(vt_auc.columns):
        vt_auc = vt_auc[["auc", "f1"]].rename(
            columns={"auc": "AUC-tuned", "f1": "F1-tuned"}
        )
    print(vt_auc.to_string())

    print()
    print("=" * 100)
    print("Survival C-index (mean ± std): does tune target affect rank metric?")
    print("=" * 100)
    surv_df = df[df["task"] == "survival"]
    vt_c = surv_df.pivot_table(
        index=["model", "feature_subset", "N"], columns="tune_simple",
        values="c_str", aggfunc="first",
    )
    if {"auc", "f1"}.issubset(vt_c.columns):
        vt_c = vt_c[["auc", "f1"]].rename(
            columns={"auc": "AUC-tuned", "f1": "F1-tuned"}
        )
    print(vt_c.to_string())

    # Δ tables: F1-tuned MINUS AUC-tuned
    print()
    print("=" * 100)
    print("Δ F1 (F1-tuned − AUC-tuned). Positive = F1-tuning wins.")
    print("=" * 100)
    # tune_simple is already set above
    p_f1 = df.pivot_table(index=["model","feature_subset","N"], columns="tune_simple",
                           values="test_f1_mean", aggfunc="first")
    p_auc = df.pivot_table(index=["model","feature_subset","N"], columns="tune_simple",
                            values="test_auc_mean", aggfunc="first")
    if "f1" in p_f1.columns and "auc" in p_f1.columns:
        delta_f1 = p_f1["f1"] - p_f1["auc"]
        delta_auc = p_auc["f1"] - p_auc["auc"]
        combined = pd.DataFrame({"ΔF1": delta_f1, "ΔAUC": delta_auc})
        print(combined.to_string(float_format="{:+.4f}".format))

    return 0


if __name__ == "__main__":
    raise SystemExit(main())

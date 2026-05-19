#!/usr/bin/env python
"""Emit overfit diagnostics for any completed production run.

Reads `predictions.csv` and `best_params.json` from a run directory
(default: `cell_lifetime/results/run/latest/`) and prints:

  1. Per-seed inner-CV score spread per model (tight cluster ⇒ robust).
  2. Hyperparameter agreement across the K ensemble members.
  3. Per-cell prediction std distribution (high std ⇒ uncertain cells).

Usage:
    python scripts/diagnose_run.py
    python scripts/diagnose_run.py cell_lifetime/results/run/20260518_1903
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd


NS = (200, 300, 400)


def main() -> int:
    if len(sys.argv) > 1:
        run_dir = Path(sys.argv[1])
    else:
        # default: cell_lifetime/results/run/latest
        run_dir = (
            Path(__file__).resolve().parents[1] / "results" / "run" / "latest"
        )
    if not run_dir.exists():
        print(f"Run directory not found: {run_dir}")
        return 1
    print(f"Diagnosing run: {run_dir.resolve()}")

    # Predictions CSV is named predictions_<TIMESTAMP>.csv. The timestamp is
    # the run directory's basename (resolves the symlink first when needed).
    timestamp = run_dir.resolve().name
    csv_candidates = [
        run_dir / f"predictions_{timestamp}.csv",
        run_dir / "predictions.csv",  # backward-compat for older runs
    ]
    csv_path = next((p for p in csv_candidates if p.exists()), None)
    if csv_path is None:
        print(f"No predictions CSV found under {run_dir}")
        return 1
    df = pd.read_csv(csv_path)
    with (run_dir / "best_params.json").open() as f:
        bp = json.load(f)
    K = bp.get("ensemble_seeds", 1)
    print(f"Ensemble seeds (K): {K}")
    if K == 1:
        print("Single deterministic fit — no ensemble diagnostics.")
        return 0

    # 1. Per-seed inner-CV score spread
    print("\n--- Per-seed inner-CV score spread ---")
    for N in NS:
        per_seed = bp["models"][f"ebm_classifier_n{N}"]["inner_cv_auc_per_seed"]
        print(
            f"  AUC per seed N={N}: mean={np.mean(per_seed):.4f} ± "
            f"std={np.std(per_seed, ddof=1):.4f} "
            f"(range {min(per_seed):.4f} - {max(per_seed):.4f})"
        )
    per_seed_c = bp["models"]["rsf"]["inner_cv_cindex_per_seed"]
    print(
        f"  C-index per seed (rsf): mean={np.mean(per_seed_c):.4f} ± "
        f"std={np.std(per_seed_c, ddof=1):.4f} "
        f"(range {min(per_seed_c):.4f} - {max(per_seed_c):.4f})"
    )

    # 2. Hyperparameter agreement across seeds
    print("\n--- Hyperparameter spread across seeds ---")
    for N in NS:
        key = f"ebm_classifier_n{N}"
        sets = bp["models"][key]["best_params_per_seed"]
        for pname in ("max_bins", "max_interaction_bins", "interactions",
                      "learning_rate", "max_leaves"):
            vals = [s.get(pname) for s in sets if s.get(pname) is not None]
            if not vals:
                continue
            if isinstance(vals[0], float):
                summary = (
                    f"{min(vals):.4g}-{max(vals):.4g} "
                    f"(med {np.median(vals):.4g})"
                )
            else:
                summary = f"{min(vals)}-{max(vals)} (med {int(np.median(vals))})"
            print(f"  {key}[{pname}]: {summary}")
    sets = bp["models"]["rsf"]["best_params_per_seed"]
    for pname in ("n_estimators", "max_depth", "min_samples_split",
                  "min_samples_leaf"):
        vals = [s.get(pname) for s in sets if s.get(pname) is not None]
        if vals:
            print(f"  rsf[{pname}]: {min(vals)}-{max(vals)} (med {int(np.median(vals))})")
    print(f"  rsf[max_features]: {[s.get('max_features') for s in sets]}")

    # 3. Per-cell prediction std
    print("\n--- Per-cell prediction std (ensemble disagreement per cell) ---")
    for N in NS:
        col = f"prob_pass_n{N}_std"
        s = df[col].dropna()
        if len(s) == 0:
            continue
        n_high = int((s > 0.10).sum())
        print(
            f"  prob_pass_n{N}_std: mean={s.mean():.4f}, p50={s.median():.4f}, "
            f"p90={s.quantile(0.90):.4f}, p99={s.quantile(0.99):.4f}, "
            f"cells>0.10: {n_high}/{len(s)}"
        )
    s = df["rsf_median_cycle_std"]
    n_high = int((s > 50).sum())
    print(
        f"  rsf_median_cycle_std: mean={s.mean():.1f}, p50={s.median():.1f}, "
        f"p90={s.quantile(0.90):.1f}, p99={s.quantile(0.99):.1f}, "
        f"cells>50 cyc: {n_high}/{len(s)}"
    )
    print(
        "\nRead: tighter spread + lower per-cell std => "
        "ensemble is more robust to hyperparameter choice (lower overfit risk)."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

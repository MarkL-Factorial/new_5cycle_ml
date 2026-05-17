#!/usr/bin/env python
"""Experiment E: 4-way weighted blend of all four survival models.

Searches the simplex {w_rsf + w_aft + w_cox + w_weibull = 1, w_i ≥ 0}
on a coarse grid (step 0.1 → 286 points) for the weight vector that
maximizes mean C-index across seeds at N=300, fs_cv. Compares against
the Pareto-optimal RSF-alone baseline from Exp C.

Pure post-processing — reads predictions.csv from each model's latest
run.
"""

from __future__ import annotations

import json
from itertools import product
from pathlib import Path

import numpy as np
import pandas as pd

from cell_lifetime.evaluation.survival_metrics import survival_metrics


HERE = Path(__file__).resolve().parent
PKG_DIR = HERE.parents[1]
RUNS_DIR = PKG_DIR / "out" / "runs" / "survival"

MODELS = ["rsf", "xgb_aft", "lifelines_cox", "lifelines_weibull_aft"]
N_HORIZON = 300
FEATURES = "fs_cv"
GRID_STEP = 0.1  # 11 points along each axis; total simplex points = C(11+3, 3) = ~286


def _latest(model: str) -> Path | None:
    hits = sorted(RUNS_DIR.glob(f"{model}__survival__N{N_HORIZON}__A2.2_b1__{FEATURES}__*"))
    return hits[-1] if hits else None


def _zscore(x: np.ndarray) -> np.ndarray:
    mu, sd = float(np.mean(x)), float(np.std(x))
    if sd < 1e-12:
        return np.zeros_like(x)
    return (x - mu) / sd


def _simplex_grid(n: int, step: float) -> list[tuple[float, ...]]:
    """Enumerate non-negative weights summing to 1 with given step."""
    levels = int(round(1.0 / step))
    grid: list[tuple[float, ...]] = []
    for combo in product(range(levels + 1), repeat=n):
        if sum(combo) == levels:
            grid.append(tuple(c * step for c in combo))
    return grid


def main() -> int:
    # Locate predictions per model
    dirs = {m: _latest(m) for m in MODELS}
    missing = [m for m, p in dirs.items() if p is None]
    if missing:
        print(f"[exp_e/blend4way] missing predictions for: {missing}")
        return 1
    print("Using these run dirs:")
    for m, p in dirs.items():
        print(f"  {m:30s} {p.name}")

    # Load + merge (cells must match across seeds + models)
    preds = {m: pd.read_csv(p / "predictions.csv") for m, p in dirs.items()}
    merge_keys = ["seed", "cell_name", "time", "event", "cohort"]
    merged = preds[MODELS[0]][merge_keys + ["risk_score"]].rename(
        columns={"risk_score": f"risk_{MODELS[0]}"}
    )
    for m in MODELS[1:]:
        right = preds[m][merge_keys + ["risk_score"]].rename(
            columns={"risk_score": f"risk_{m}"}
        )
        merged = merged.merge(right, on=merge_keys)
    print(f"merged shape: {merged.shape} (= n_seeds × n_test_cells_per_seed)")

    # Per-seed z-score normalization of each model's risk_score
    blocks = []
    for seed, sub in merged.groupby("seed"):
        sub = sub.copy()
        for m in MODELS:
            sub[f"z_{m}"] = _zscore(sub[f"risk_{m}"].values)
        blocks.append(sub)
    blended = pd.concat(blocks, ignore_index=True)

    # Grid search over the simplex
    grid = _simplex_grid(len(MODELS), GRID_STEP)
    print(f"grid points: {len(grid)} (simplex with step={GRID_STEP})")

    results = []
    for w in grid:
        # Combine
        risk = sum(w[i] * blended[f"z_{MODELS[i]}"].values for i in range(len(MODELS)))
        # C-index per seed, then mean
        c_per_seed = []
        for seed, sub in blended.groupby("seed"):
            mask = blended["seed"] == seed
            r = risk[mask.values]
            out = survival_metrics(sub["event"].values, sub["time"].values, r)
            c_per_seed.append(out["c_index"])
        c_per_seed = np.array(c_per_seed)
        results.append({
            **{f"w_{m}": float(w[i]) for i, m in enumerate(MODELS)},
            "c_mean": float(np.mean(c_per_seed)),
            "c_std": float(np.std(c_per_seed)),
        })

    res_df = pd.DataFrame(results)
    res_df = res_df.sort_values("c_mean", ascending=False).reset_index(drop=True)
    res_df.to_csv(HERE / "blend4way_grid.csv", index=False)

    print()
    print("Top 10 weight vectors by mean C-index:")
    print(res_df.head(10).to_string(index=False,
                                     formatters={c: "{:.3f}".format
                                                 for c in res_df.columns}))

    # Single-model anchors for reference (corners of simplex)
    print()
    print("Single-model baselines (each model alone, w=1):")
    for m in MODELS:
        anchor = res_df[(res_df[f"w_{m}"] == 1.0)].iloc[0]
        print(f"  {m:30s} C-index = {anchor['c_mean']:.4f} ± {anchor['c_std']:.4f}")

    # Best blend vs best single
    best_single_c = max(
        res_df[(res_df[f"w_{m}"] == 1.0)].iloc[0]["c_mean"] for m in MODELS
    )
    best_blend = res_df.iloc[0]
    is_pure = any(best_blend[f"w_{m}"] == 1.0 for m in MODELS)
    delta = best_blend["c_mean"] - best_single_c
    print()
    print(f"Best blend C-index: {best_blend['c_mean']:.4f} ± {best_blend['c_std']:.4f}")
    weight_str = ", ".join(f"{m}={best_blend['w_' + m]:.1f}" for m in MODELS)
    print(f"  weights: {weight_str}")
    print(f"Best single-model C-index: {best_single_c:.4f}")
    print(f"Δ blend vs best single: {delta:+.4f}")
    if is_pure:
        verdict = "best 'blend' is actually a single model — no ensemble gain"
    elif delta > best_blend["c_std"]:
        verdict = "BLEND WINS by >1 std"
    elif delta > 0:
        verdict = "blend nudges up but inside std band"
    else:
        verdict = "blend does NOT improve"
    print(f"→ {verdict}")

    (HERE / "blend4way_summary.json").write_text(json.dumps({
        "best_blend": {**best_blend.to_dict(), "is_pure_single": bool(is_pure)},
        "best_single_c_index": float(best_single_c),
        "delta": float(delta),
        "verdict": verdict,
        "n_grid_points": len(grid),
        "models": MODELS,
    }, indent=2))
    print(f"\n[exp_e/blend4way] wrote {HERE / 'blend4way_grid.csv'} + summary.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

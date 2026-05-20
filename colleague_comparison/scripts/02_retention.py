"""Retention comparison.

A has one scalar `retention` per cell. B has multiple flavors:
  - B features: discharge_capacity_retention_final = cap_dis[c5]/cap_dis[c1]
  - B labels:  final_retention                     = cap_dis[c_last]/cap_dis[c1]

We construct several truth-derived candidates and ask which one A.retention
correlates with most strongly (correlation + mean-abs-diff).
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
from load_data import (  # noqa: E402
    ensure_out_dir,
    load_colleague,
    load_features,
    load_labels,
    load_registry_regular,
    overlap_cells,
)


def per_cell_retention_curves(reg: pd.DataFrame, a_max_by_cell: dict) -> pd.DataFrame:
    """For each cell, compute the family of candidate retention definitions.

    All anchored on regular_cycle == 1's discharge capacity (= the documented
    baseline in B's manifest, baseline_cycle=1).

    Includes a "snapshot" hypothesis: retention at A.max_regular_cycle, in
    case A's `retention` is retention-at-A's-then-latest-cycle (because A is
    a frozen earlier export of this same data).
    """
    rows = []
    for cell, grp in reg.groupby("cell_name"):
        grp = grp.sort_values("regular_cycle")
        rc = grp["regular_cycle"].to_numpy()
        cap = grp["capacity_discharge_ah"].to_numpy()
        if len(cap) == 0 or cap[0] == 0:
            continue
        c1 = cap[0]
        row = {"cell_name": cell, "n_regular_truth": int(rc.max()), "cap_dis_c1_truth": c1}
        mask5 = rc == 5
        row["ret_c5_truth"] = cap[mask5][0] / c1 if mask5.any() else np.nan
        row["ret_clast_truth"] = cap[-1] / c1
        row["ret_min_truth"] = cap.min() / c1
        for N in (200, 300, 400):
            mask = rc >= N
            row[f"ret_at_n{N}_truth"] = cap[mask][0] / c1 if mask.any() else np.nan

        # Snapshot hypothesis: retention at A.max_regular_cycle (if available)
        a_max = a_max_by_cell.get(cell)
        if a_max is not None and not pd.isna(a_max):
            mask_at_amax = rc <= int(a_max)
            row["ret_at_A_snapshot_truth"] = cap[mask_at_amax][-1] / c1 if mask_at_amax.any() else np.nan
        else:
            row["ret_at_A_snapshot_truth"] = np.nan
        rows.append(row)
    return pd.DataFrame(rows)


def main() -> None:
    out = ensure_out_dir()
    A = load_colleague()[["cell_name", "retention", "max_regular_cycle"]]
    Bf = load_features()[["cell_name", "discharge_capacity_retention_final"]]
    Bl = load_labels()[["cell_name", "final_retention", "n_regular", "status", "baseline_dis_ah"]]
    reg = load_registry_regular()

    overlap = overlap_cells()
    print(f"[02_retention] overlap cells: {len(overlap)}")

    a_max_by_cell = dict(zip(A["cell_name"], A["max_regular_cycle"]))
    truth = per_cell_retention_curves(reg, a_max_by_cell)
    df = pd.DataFrame({"cell_name": overlap})
    for t in (A, Bf, Bl, truth):
        df = df.merge(t, on="cell_name", how="left")

    # B's c5/c1 retention from registry truth (sanity check on B.features formula)
    df["B_ret_c5_check"] = df["discharge_capacity_retention_final"] - df["ret_c5_truth"]

    df.to_csv(out / "retention_per_cell.csv", index=False)
    print(f"[02_retention] wrote {out / 'retention_per_cell.csv'} ({len(df)} rows)")

    # Verify B's c5-retention formula
    d = df["B_ret_c5_check"].dropna()
    print(f"\n[02_retention] B.discharge_capacity_retention_final vs truth cap_dis[c5]/cap_dis[c1]:")
    print(f"  n={len(d)} mean={d.mean():+.5f} max|d|={d.abs().max():.5f}")

    # Search for which truth-flavor A.retention matches best
    candidates = [
        "ret_c5_truth", "ret_clast_truth", "ret_min_truth",
        "ret_at_n200_truth", "ret_at_n300_truth", "ret_at_n400_truth",
        "ret_at_A_snapshot_truth",  # snapshot hypothesis
        "final_retention",  # B-labels: cap[c_last]/cap[c1]
        "discharge_capacity_retention_final",  # B-features: cap[c5]/cap[c1]
    ]
    print("\n[02_retention] A.retention vs each candidate (Pearson r, mean|d|, n):")
    scores = []
    for c in candidates:
        sub = df[["retention", c]].dropna()
        if len(sub) < 5:
            print(f"  {c:40s}  n={len(sub):3d}  (skip)")
            continue
        r = sub["retention"].corr(sub[c])
        mad = (sub["retention"] - sub[c]).abs().mean()
        bias = (sub["retention"] - sub[c]).mean()
        scores.append((c, r, mad, bias, len(sub)))
        print(f"  {c:40s}  n={len(sub):3d}  r={r:+.4f}  mean|d|={mad:.5f}  bias={bias:+.5f}")

    best = max(scores, key=lambda s: s[1])
    print(f"\n[02_retention] BEST match for A.retention → {best[0]} (r={best[1]:.4f}, mean|d|={best[2]:.5f})")

    # Flag A.retention > 1 oddities
    odd = df[df["retention"] > 1.0]
    print(f"\n[02_retention] cells where A.retention > 1.0: {len(odd)}")
    if len(odd) > 0:
        same_in_truth = ((odd["retention"] > 1.0) & (odd["ret_clast_truth"] > 1.0)).sum()
        print(f"  of those, also retention>1 vs truth(c_last/c1): {same_in_truth} / {len(odd)}")

    # Plot 2x3 grid: A.retention vs each major candidate
    plot_candidates = [
        "ret_c5_truth", "ret_clast_truth", "ret_min_truth",
        "ret_at_n300_truth", "ret_at_A_snapshot_truth", "final_retention",
    ]
    fig, axes = plt.subplots(2, 3, figsize=(15, 9))
    for ax, c in zip(axes.flat, plot_candidates):
        sub = df[["retention", c]].dropna()
        ax.scatter(sub["retention"], sub[c], s=12, alpha=0.6)
        lo, hi = 0.2, 1.2
        ax.plot([lo, hi], [lo, hi], "k--", lw=0.8)
        ax.set_xlim(lo, hi); ax.set_ylim(lo, hi)
        ax.set_xlabel("A.retention")
        ax.set_ylabel(c)
        if len(sub) >= 5:
            r = sub["retention"].corr(sub[c])
            ax.set_title(f"{c}  (r={r:+.3f}, n={len(sub)})")
        else:
            ax.set_title(f"{c} (n={len(sub)})")
    fig.suptitle("A.retention vs candidate truth retentions")
    fig.tight_layout()
    fig.savefig(out / "retention_definitions.png", dpi=120)
    print(f"\n[02_retention] wrote {out / 'retention_definitions.png'}")


if __name__ == "__main__":
    main()

"""Fade-cycle / lifetime comparison.

Verifies:
  1. A.max_regular_cycle == registry max regular_cycle? (Is A.max_regular_cycle
     just lifetime, NOT a fade cycle?)
  2. B.n_regular == registry max regular_cycle?
  3. B.last_fade_cycle matches a locally recomputed sticky-crossing fade.

The sticky-crossing rule (mirrors ml_label_preprocess/labels.py::_last_crossing_into_bad):
  - A 'crossing into bad' at index i means ret[i] < 0.85 AND (i == 0 OR
    ret[i-1] >= 0.85). Only transitions count, not every bad cycle.
  - Among all crossings, last_fade_cycle is the LAST whose post-crossing
    window has fewer than RECOVERY_MIN (=3) cycles with ret > 0.85 (strict).
  - Earlier crossings with >= RECOVERY_MIN recovered cycles after are
    treated as transient and skipped.
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
    load_labels,
    load_registry_regular,
    overlap_cells,
)

FADE_THRESHOLD = 0.85
RECOVERY_MIN = 3


def fade_cycle_for(rc: np.ndarray, cap: np.ndarray) -> tuple[float, int]:
    """Return (last_fade_cycle_or_nan, n_recovered_crossings)."""
    if len(cap) < 1 or cap[0] <= 0:
        return np.nan, 0
    ret = cap / cap[0]
    n = len(ret)
    last_fade = np.nan
    n_recovered = 0
    for i, r in enumerate(ret):
        if r >= FADE_THRESHOLD:
            continue
        # only count crossings (transition from healthy or start)
        if i > 0 and ret[i - 1] < FADE_THRESHOLD:
            continue
        n_good_after = int(np.sum(ret[i + 1 :] > FADE_THRESHOLD))
        if n_good_after >= RECOVERY_MIN:
            n_recovered += 1
        else:
            last_fade = float(rc[i])  # keep walking; LAST sticky crossing wins
    return last_fade, n_recovered


def per_cell_truth(reg: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for cell, grp in reg.groupby("cell_name"):
        grp = grp.sort_values("regular_cycle")
        rc = grp["regular_cycle"].to_numpy()
        cap = grp["capacity_discharge_ah"].to_numpy()
        last_fade, n_recov = fade_cycle_for(rc, cap)
        rows.append({
            "cell_name": cell,
            "n_regular_truth": int(rc.max()) if len(rc) else 0,
            "fade_cycle_truth_0p85": last_fade,
            "n_recovered_crossings_truth": n_recov,
        })
    return pd.DataFrame(rows)


def main() -> None:
    out = ensure_out_dir()
    A = load_colleague()[["cell_name", "max_regular_cycle"]]
    Bl = load_labels()[["cell_name", "n_regular", "last_fade_cycle", "status",
                        "n_recovered_crossings"]]
    reg = load_registry_regular()

    overlap = overlap_cells()
    print(f"[03_fade] overlap cells: {len(overlap)}")

    truth = per_cell_truth(reg)
    df = pd.DataFrame({"cell_name": overlap})
    for t in (A, Bl, truth):
        df = df.merge(t, on="cell_name", how="left")

    df["A_lifetime_matches_truth"] = (df["max_regular_cycle"] == df["n_regular_truth"]).astype(int)
    df["B_lifetime_matches_truth"] = (df["n_regular"] == df["n_regular_truth"]).astype(int)

    # Compare last_fade_cycle (B) to locally recomputed truth fade
    both = df["last_fade_cycle"].notna() & df["fade_cycle_truth_0p85"].notna()
    df["B_fade_matches_truth"] = (
        df["last_fade_cycle"].where(both, np.nan) == df["fade_cycle_truth_0p85"].where(both, np.nan)
    ).astype(float)
    df.loc[~both, "B_fade_matches_truth"] = np.nan

    df.to_csv(out / "fade_per_cell.csv", index=False)
    print(f"[03_fade] wrote {out / 'fade_per_cell.csv'} ({len(df)} rows)")

    print(f"\n[03_fade] A.max_regular_cycle == truth.n_regular: "
          f"{int(df['A_lifetime_matches_truth'].sum())}/{len(df)}")
    print(f"[03_fade] B.n_regular         == truth.n_regular: "
          f"{int(df['B_lifetime_matches_truth'].sum())}/{len(df)}")

    n_both = int(both.sum())
    n_match = int(((df["last_fade_cycle"] == df["fade_cycle_truth_0p85"]) & both).sum())
    print(f"\n[03_fade] B.last_fade_cycle vs locally-recomputed truth fade cycle:")
    print(f"  cells where both are non-null: {n_both}")
    print(f"  exact match: {n_match}")
    diff_when_both = (df["last_fade_cycle"] - df["fade_cycle_truth_0p85"]).where(both).dropna()
    if len(diff_when_both):
        print(f"  median diff: {diff_when_both.median():+.1f}  mean|d|: {diff_when_both.abs().mean():.2f}  "
              f"max|d|: {diff_when_both.abs().max():.0f}")

    n_a_only = int((df["fade_cycle_truth_0p85"].notna() & df["last_fade_cycle"].isna()).sum())
    n_b_only = int((df["fade_cycle_truth_0p85"].isna() & df["last_fade_cycle"].notna()).sum())
    print(f"  truth fades but B says none: {n_a_only}")
    print(f"  B says fade but truth says none: {n_b_only}")

    # A doesn't have a fade-cycle column — but we can ask: among A.label=='BAD', do they have a truth fade?
    A_label = load_colleague()[["cell_name", "label"]]
    df = df.merge(A_label, on="cell_name", how="left")
    n_bad_with_fade = int(((df["label"] == "BAD") & df["fade_cycle_truth_0p85"].notna()).sum())
    n_bad_total = int((df["label"] == "BAD").sum())
    n_good_with_fade = int(((df["label"] == "GOOD") & df["fade_cycle_truth_0p85"].notna()).sum())
    n_good_total = int((df["label"] == "GOOD").sum())
    print(f"\n[03_fade] A.label=='BAD' cells:  {n_bad_with_fade}/{n_bad_total} have a truth fade cycle")
    print(f"[03_fade] A.label=='GOOD' cells: {n_good_with_fade}/{n_good_total} have a truth fade cycle")

    # Plot
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    axes[0].scatter(df["n_regular_truth"], df["max_regular_cycle"], s=12, alpha=0.6)
    m = df[["n_regular_truth", "max_regular_cycle"]].dropna().to_numpy()
    if len(m):
        hi = max(m.max(), 1100)
        axes[0].plot([0, hi], [0, hi], "k--", lw=0.8)
        axes[0].set_xlim(0, hi); axes[0].set_ylim(0, hi)
    axes[0].set_xlabel("truth max regular_cycle")
    axes[0].set_ylabel("A.max_regular_cycle")
    axes[0].set_title("A.max_regular_cycle is lifetime (not fade)")

    sub = df.dropna(subset=["fade_cycle_truth_0p85", "last_fade_cycle"])
    axes[1].scatter(sub["fade_cycle_truth_0p85"], sub["last_fade_cycle"], s=12, alpha=0.6, color="C1")
    if len(sub):
        hi = max(sub[["fade_cycle_truth_0p85", "last_fade_cycle"]].max().max(), 1100)
        axes[1].plot([0, hi], [0, hi], "k--", lw=0.8)
        axes[1].set_xlim(0, hi); axes[1].set_ylim(0, hi)
    axes[1].set_xlabel("locally-recomputed truth fade cycle (0.85, sticky)")
    axes[1].set_ylabel("B.last_fade_cycle")
    axes[1].set_title(f"B fade detector: {n_match}/{n_both} exact")

    fig.tight_layout()
    fig.savefig(out / "fade_scatter.png", dpi=120)
    print(f"\n[03_fade] wrote {out / 'fade_scatter.png'}")


if __name__ == "__main__":
    main()

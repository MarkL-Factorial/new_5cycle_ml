"""Coulombic-efficiency comparison.

Goal: identify which cycle each side's CE column actually refers to, by
matching against the per-cycle registry truth.

Candidates for what A.ce2 might mean:
  - regular_cycle == 1 or == 2 (CE)
  - cd_index == 1 or == 2 (1st / 2nd entry overall, often formation)
  - the second formation-only event (event_kind == 'formation', 2nd one)

B's coulombic_efficiency_final is documented as cycle-5 CE in percent.
We verify against regular_cycle == 5.
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
    load_registry_all,
    load_registry_regular,
    overlap_cells,
    truth_at_regular,
)


def main() -> None:
    out = ensure_out_dir()
    A = load_colleague()[["cell_name", "ce2"]]
    Bf = load_features()[["cell_name", "coulombic_efficiency_final"]]
    reg = load_registry_regular()
    reg_all = load_registry_all()

    overlap = overlap_cells()
    print(f"[01_ce] overlap cells: {len(overlap)}")

    # truth at several regular-cycle indices
    truth_c1 = truth_at_regular(reg, 1)[["cell_name", "ce_c1"]]
    truth_c2 = truth_at_regular(reg, 2)[["cell_name", "ce_c2"]]
    truth_c5 = truth_at_regular(reg, 5)[["cell_name", "ce_c5"]]

    # truth at cd_index == 1 (2nd cd_event overall — often the 2nd formation)
    cdidx1 = reg_all[reg_all["cd_index"] == 1][["cell_name", "coulombic_efficiency"]].rename(
        columns={"coulombic_efficiency": "ce_cdidx1"}
    )
    # truth at the 2nd formation event
    form = reg_all[reg_all["event_kind"] == "formation"].copy()
    form["form_order"] = form.groupby("cell_name").cumcount()
    form2 = form[form["form_order"] == 1][["cell_name", "coulombic_efficiency"]].rename(
        columns={"coulombic_efficiency": "ce_form2"}
    )

    df = pd.DataFrame({"cell_name": overlap})
    for t in (A, Bf, truth_c1, truth_c2, truth_c5, cdidx1, form2):
        df = df.merge(t, on="cell_name", how="left")

    df["ce_mine_c5_as_frac"] = df["coulombic_efficiency_final"] / 100.0
    df["diff_mine_c5"] = df["ce_mine_c5_as_frac"] - df["ce_c5"]

    # Test A.ce2 against each candidate
    for candidate in ("ce_c1", "ce_c2", "ce_c5", "ce_cdidx1", "ce_form2"):
        df[f"diff_A_vs_{candidate}"] = df["ce2"] - df[candidate]

    df.to_csv(out / "ce_per_cell.csv", index=False)
    print(f"[01_ce] wrote {out / 'ce_per_cell.csv'} ({len(df)} rows)")

    # Print fit summaries
    print("\n[01_ce] mine (B.coulombic_efficiency_final / 100) vs truth ce_c5:")
    d = df["diff_mine_c5"].dropna()
    print(f"  n={len(d)} mean={d.mean():+.5f} std={d.std():.5f} max|diff|={d.abs().max():.5f}"
          f"  n(|d|>0.01)={(d.abs() > 0.01).sum()}")

    print("\n[01_ce] A.ce2 vs each candidate truth (mean / std / max-abs / n(|d|>0.01)):")
    for candidate in ("ce_c1", "ce_c2", "ce_c5", "ce_cdidx1", "ce_form2"):
        d = df[f"diff_A_vs_{candidate}"].dropna()
        print(f"  {candidate:11s}  n={len(d):3d}  mean={d.mean():+.5f}  "
              f"std={d.std():.5f}  max|d|={d.abs().max():.5f}  "
              f"n(|d|>0.01)={(d.abs() > 0.01).sum()}")

    # Identify best candidate (lowest mean-abs-diff)
    best = min(
        ("ce_c1", "ce_c2", "ce_c5", "ce_cdidx1", "ce_form2"),
        key=lambda c: df[f"diff_A_vs_{c}"].abs().mean(),
    )
    print(f"\n[01_ce] BEST match for A.ce2 → {best} (lowest mean|diff|)")

    # Plot
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    axes[0].scatter(df[best], df["ce2"], s=12, alpha=0.6)
    lo, hi = 0.0, 1.1
    axes[0].plot([lo, hi], [lo, hi], "k--", lw=0.8)
    axes[0].set_xlim(lo, hi); axes[0].set_ylim(lo, hi)
    axes[0].set_xlabel(f"truth {best} (fraction)")
    axes[0].set_ylabel("A.ce2 (fraction)")
    axes[0].set_title(f"A.ce2 vs truth {best}")

    axes[1].scatter(df["ce_c5"], df["ce_mine_c5_as_frac"], s=12, alpha=0.6, color="C1")
    axes[1].plot([0.8, 1.2], [0.8, 1.2], "k--", lw=0.8)
    axes[1].set_xlabel("truth ce_c5 (fraction)")
    axes[1].set_ylabel("B.coulombic_efficiency_final/100")
    axes[1].set_title("B.CE_final vs truth ce_c5")

    axes[2].hist(df[f"diff_A_vs_{best}"].dropna(), bins=30, alpha=0.6, label=f"A.ce2 - {best}")
    axes[2].hist(df["diff_mine_c5"].dropna(), bins=30, alpha=0.6, label="B.CE_final/100 - ce_c5")
    axes[2].axvline(0, color="k", lw=0.8)
    axes[2].set_xlabel("diff (fraction)")
    axes[2].set_ylabel("count")
    axes[2].set_title("error histograms")
    axes[2].legend()

    fig.tight_layout()
    fig.savefig(out / "ce_distributions.png", dpi=120)
    print(f"[01_ce] wrote {out / 'ce_distributions.png'}")


if __name__ == "__main__":
    main()

"""Categorical-label agreement.

A.label ∈ {GOOD, BAD} is implicitly a binary survival call, but the cycle
threshold is unstated. For each N ∈ {200, 300, 400}, we build a confusion
matrix against B.label_n{N} ∈ {pass, bad, censor, excluded}.

Then, for the off-diagonal cells, we attach the truth retention at cycle N
(from the registry) as a third-party witness so each disagreement can be
adjudicated.
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

N_THRESHOLDS = (200, 300, 400)


def truth_at_or_after(reg: pd.DataFrame, n: int) -> pd.DataFrame:
    """For each cell, retention at the first regular_cycle >= n (NaN if not reached)."""
    rows = []
    for cell, grp in reg.groupby("cell_name"):
        grp = grp.sort_values("regular_cycle")
        rc = grp["regular_cycle"].to_numpy()
        cap = grp["capacity_discharge_ah"].to_numpy()
        if len(cap) == 0 or cap[0] == 0:
            rows.append({"cell_name": cell, f"truth_ret_at_n{n}": np.nan,
                         f"truth_cycle_at_n{n}": np.nan,
                         f"truth_min_ret_through_n{n}": np.nan})
            continue
        mask = rc >= n
        if mask.any():
            i = int(np.argmax(mask))  # first True
            ret_n = float(cap[i] / cap[0])
            cyc_n = int(rc[i])
            min_through = float((cap[: i + 1] / cap[0]).min())
        else:
            ret_n = np.nan
            cyc_n = np.nan
            min_through = float((cap / cap[0]).min())
        rows.append({"cell_name": cell, f"truth_ret_at_n{n}": ret_n,
                     f"truth_cycle_at_n{n}": cyc_n,
                     f"truth_min_ret_through_n{n}": min_through})
    return pd.DataFrame(rows)


def main() -> None:
    out = ensure_out_dir()
    A = load_colleague()[["cell_name", "label"]].rename(columns={"label": "A_label"})
    Bl = load_labels()[["cell_name", "status", "n_regular", "last_fade_cycle",
                        "label_n200", "label_n300", "label_n400"]]
    reg = load_registry_regular()

    overlap = overlap_cells()
    print(f"[04_label] overlap cells: {len(overlap)}")

    df = pd.DataFrame({"cell_name": overlap})
    df = df.merge(A, on="cell_name", how="left").merge(Bl, on="cell_name", how="left")
    for N in N_THRESHOLDS:
        df = df.merge(truth_at_or_after(reg, N), on="cell_name", how="left")

    print(f"\n[04_label] A.label distribution: {df['A_label'].value_counts(dropna=False).to_dict()}")
    for N in N_THRESHOLDS:
        print(f"[04_label] B.label_n{N} distribution: "
              f"{df[f'label_n{N}'].value_counts(dropna=False).to_dict()}")

    # Build confusion matrices, one per N
    rows = []
    print()
    for N in N_THRESHOLDS:
        ct = pd.crosstab(df["A_label"], df[f"label_n{N}"], dropna=False, margins=False)
        print(f"[04_label] confusion A.label × B.label_n{N}:")
        print(ct.to_string())
        print()
        for a_val in ct.index:
            for b_val in ct.columns:
                rows.append({"N": N, "A_label": a_val, "B_label": b_val,
                             "count": int(ct.loc[a_val, b_val])})

    pd.DataFrame(rows).to_csv(out / "label_confusion.csv", index=False)
    print(f"[04_label] wrote {out / 'label_confusion.csv'}")

    # Off-diagonal cells (mapping GOOD↔pass, BAD↔bad), per N
    A_to_B = {"GOOD": {"pass"}, "BAD": {"bad"}}
    disagree_rows = []
    for N in N_THRESHOLDS:
        for _, row in df.iterrows():
            a_lab = row["A_label"]
            b_lab = row[f"label_n{N}"]
            if pd.isna(a_lab) or pd.isna(b_lab):
                continue
            expected = A_to_B.get(a_lab, set())
            if b_lab in expected:
                continue
            disagree_rows.append({
                "N": N,
                "cell_name": row["cell_name"],
                "A_label": a_lab,
                f"B_label_n{N}": b_lab,
                "B_status": row["status"],
                "B_n_regular": row["n_regular"],
                "B_last_fade_cycle": row["last_fade_cycle"],
                f"truth_ret_at_n{N}": row[f"truth_ret_at_n{N}"],
                f"truth_cycle_at_n{N}": row[f"truth_cycle_at_n{N}"],
                f"truth_min_ret_through_n{N}": row[f"truth_min_ret_through_n{N}"],
            })
    if disagree_rows:
        dd = pd.DataFrame(disagree_rows)
        dd.to_csv(out / "label_disagreements.csv", index=False)
        print(f"\n[04_label] wrote {out / 'label_disagreements.csv'} ({len(dd)} disagreements across all N)")
        print(f"[04_label] disagreement counts by N: {dd.groupby('N').size().to_dict()}")
    else:
        print("\n[04_label] no disagreements (unexpected — sanity check the mapping)")

    # 3-panel heatmap
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))
    for ax, N in zip(axes, N_THRESHOLDS):
        ct = pd.crosstab(df["A_label"], df[f"label_n{N}"], dropna=False)
        im = ax.imshow(ct.to_numpy(), cmap="Blues", aspect="auto")
        ax.set_xticks(range(len(ct.columns)))
        ax.set_xticklabels(ct.columns, rotation=30, ha="right")
        ax.set_yticks(range(len(ct.index)))
        ax.set_yticklabels(ct.index)
        ax.set_xlabel(f"B.label_n{N}")
        ax.set_ylabel("A.label")
        ax.set_title(f"N = {N}")
        for i in range(ct.shape[0]):
            for j in range(ct.shape[1]):
                v = int(ct.iat[i, j])
                ax.text(j, i, str(v), ha="center", va="center",
                        color="white" if v > ct.to_numpy().max() / 2 else "black",
                        fontsize=11)
    fig.tight_layout()
    fig.savefig(out / "label_confusion.png", dpi=120)
    print(f"[04_label] wrote {out / 'label_confusion.png'}")


if __name__ == "__main__":
    main()

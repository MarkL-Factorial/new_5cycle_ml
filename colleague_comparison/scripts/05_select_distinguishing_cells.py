"""Select a small set of cells that distinguish A's vs B's labeling.

For each disagreement category, pick the clearest example(s). Output:
  - out/cells_to_review.csv         (one row per selected cell, full context)
  - out/cells_to_review_curves.png  (retention curve per cell, with key cycles annotated)

Categories selected:

  cat1_GOOD_vs_bad_at_n300   — A:GOOD, B:bad at N=300. Both labels apply at the
                                same cycle; the cleanest case for "who is right".

  cat2_BAD_vs_pass_at_n200   — A:BAD, B:pass at N=200. Reveals A's BAD threshold:
                                these cells survive N=200 but A still calls bad.

  cat3_stale_GOOD_to_bad_n400 — A:GOOD, B:bad at N=400. Snapshot staleness
                                blowing up — cell faded after A was exported.

  cat4_negative_gap          — A.max_regular_cycle > truth.n_regular. Investigates
                                cells where A claims MORE cycles than the registry.

  cat5_retention_anomaly     — biggest |A.retention - any-truth-flavor| disagreement.
                                These are where A.retention's mystery formula
                                deviates most starkly from a registry-derived one.
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


PER_CATEGORY = 2  # how many cells per category to pick


def _truth_retention_at_amax(rc: np.ndarray, cap: np.ndarray, a_max: float) -> tuple[float, int]:
    if pd.isna(a_max):
        return np.nan, np.nan
    mask = rc <= int(a_max)
    if not mask.any() or cap[0] == 0:
        return np.nan, np.nan
    return float(cap[mask][-1] / cap[0]), int(rc[mask][-1])


def build_master_table() -> pd.DataFrame:
    """One row per overlap cell with every value the manual reviewer might want."""
    A = load_colleague()[["cell_name", "label", "retention", "max_regular_cycle", "ce2"]]
    Bf = load_features()[["cell_name", "coulombic_efficiency_final",
                          "discharge_capacity_retention_final"]]
    Bl = load_labels()[["cell_name", "status", "n_regular", "last_fade_cycle",
                        "final_retention", "baseline_dis_ah", "cohort",
                        "label_n200", "label_n300", "label_n400"]]
    reg = load_registry_regular()
    overlap = overlap_cells()

    # Per-cell truth aggregates
    rows = []
    for cell, grp in reg.groupby("cell_name"):
        grp = grp.sort_values("regular_cycle")
        rc = grp["regular_cycle"].to_numpy()
        cap = grp["capacity_discharge_ah"].to_numpy()
        if len(cap) == 0 or cap[0] == 0:
            continue
        c1 = cap[0]
        mask5 = rc == 5
        row = {
            "cell_name": cell,
            "n_regular_truth": int(rc.max()),
            "ret_c5_truth": float(cap[mask5][0] / c1) if mask5.any() else np.nan,
            "ret_clast_truth": float(cap[-1] / c1),
            "ret_min_truth": float(cap.min() / c1),
        }
        for N in (200, 300, 400):
            mask = rc >= N
            row[f"ret_at_n{N}_truth"] = float(cap[mask][0] / c1) if mask.any() else np.nan
        rows.append(row)
    truth = pd.DataFrame(rows)

    df = pd.DataFrame({"cell_name": overlap})
    for t in (A, Bf, Bl, truth):
        df = df.merge(t, on="cell_name", how="left")

    # Snapshot gap + retention at A's snapshot cycle
    a_map = dict(zip(A["cell_name"], A["max_regular_cycle"]))
    snap_ret = []
    snap_actual_cycle = []
    for cell in df["cell_name"]:
        sub = reg[reg["cell_name"] == cell].sort_values("regular_cycle")
        ret, used_cyc = _truth_retention_at_amax(
            sub["regular_cycle"].to_numpy(),
            sub["capacity_discharge_ah"].to_numpy(),
            a_map.get(cell, np.nan),
        )
        snap_ret.append(ret)
        snap_actual_cycle.append(used_cyc)
    df["ret_at_A_snapshot_truth"] = snap_ret
    df["truth_cycle_used_for_snapshot"] = snap_actual_cycle
    df["snapshot_gap"] = df["n_regular_truth"] - df["max_regular_cycle"]

    return df


def pick_cells(df: pd.DataFrame) -> pd.DataFrame:
    picked = []

    # cat1: A:GOOD ∩ B:bad @ N=300 — there are only 3 of these; take all
    cat1 = df[(df["label"] == "GOOD") & (df["label_n300"] == "bad")].copy()
    cat1["why"] = "A:GOOD but B:bad @ N=300 — cleanest 'who's right' case at primary N"
    cat1["category"] = "cat1_GOOD_vs_bad_at_n300"
    picked.append(cat1.head(PER_CATEGORY + 1))  # take all if few

    # cat2: A:BAD ∩ B:pass @ N=200 — sort by HIGH ret_at_n200_truth (clearly fine at 200)
    cat2 = df[(df["label"] == "BAD") & (df["label_n200"] == "pass")].copy()
    cat2 = cat2.sort_values("ret_at_n200_truth", ascending=False)
    cat2["why"] = "A:BAD but B:pass @ N=200 — A is stricter than B at N=200 (likely A targets later N)"
    cat2["category"] = "cat2_BAD_vs_pass_at_n200"
    picked.append(cat2.head(PER_CATEGORY))

    # cat3: A:GOOD ∩ B:bad @ N=400 — sort by LARGEST snapshot_gap (most stale)
    cat3 = df[(df["label"] == "GOOD") & (df["label_n400"] == "bad")].copy()
    cat3 = cat3.sort_values("snapshot_gap", ascending=False)
    cat3["why"] = "A:GOOD but B:bad @ N=400 — A's snapshot pre-dates the fade (snapshot staleness)"
    cat3["category"] = "cat3_stale_GOOD_to_bad_n400"
    picked.append(cat3.head(PER_CATEGORY))

    # cat4: negative gap — A.max_regular_cycle > truth.n_regular
    cat4 = df[df["snapshot_gap"] < 0].copy()
    cat4 = cat4.sort_values("snapshot_gap", ascending=True)  # most negative first
    cat4["why"] = "A.max_regular_cycle > truth.n_regular — A counts cycles B/registry don't"
    cat4["category"] = "cat4_negative_gap"
    picked.append(cat4.head(PER_CATEGORY))

    # cat5: retention anomaly — biggest |A.retention - ret_at_A_snapshot_truth|
    cat5 = df.copy()
    cat5["retention_dev"] = (cat5["retention"] - cat5["ret_at_A_snapshot_truth"]).abs()
    cat5 = cat5.sort_values("retention_dev", ascending=False)
    cat5["why"] = "Largest |A.retention - registry retention at A's snapshot cycle| — formula mystery"
    cat5["category"] = "cat5_retention_anomaly"
    picked.append(cat5.head(PER_CATEGORY))

    out = pd.concat(picked, ignore_index=True)
    # If a cell appears in multiple categories, keep the first occurrence
    out = out.drop_duplicates(subset="cell_name", keep="first").reset_index(drop=True)
    return out


def plot_curves(selected: pd.DataFrame, reg: pd.DataFrame, out_path: Path) -> None:
    n = len(selected)
    ncols = 3
    nrows = (n + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(5 * ncols, 3.5 * nrows), squeeze=False)
    A = load_colleague()[["cell_name", "max_regular_cycle", "label", "retention"]]
    a_map = A.set_index("cell_name").to_dict("index")

    for i, (_, row) in enumerate(selected.iterrows()):
        ax = axes[i // ncols][i % ncols]
        cell = row["cell_name"]
        sub = reg[reg["cell_name"] == cell].sort_values("regular_cycle")
        rc = sub["regular_cycle"].to_numpy()
        cap = sub["capacity_discharge_ah"].to_numpy()
        if len(cap) == 0 or cap[0] == 0:
            ax.set_title(f"{cell} (no data)")
            continue
        ret = cap / cap[0]
        ax.plot(rc, ret, lw=1.0, color="C0")
        ax.axhline(0.85, color="red", lw=0.6, ls=":")
        for N in (200, 300, 400):
            ax.axvline(N, color="grey", lw=0.5, ls=":")
        a_max = a_map.get(cell, {}).get("max_regular_cycle", np.nan)
        if not pd.isna(a_max):
            ax.axvline(a_max, color="orange", lw=1.0, ls="--", label=f"A.max={int(a_max)}")
        last_fade = row.get("last_fade_cycle", np.nan)
        if not pd.isna(last_fade):
            ax.axvline(last_fade, color="purple", lw=1.0, ls="--", label=f"B.fade={int(last_fade)}")
        ax.set_ylim(0.4, 1.1)
        ax.set_xlabel("regular_cycle")
        ax.set_ylabel("cap_dis / cap_dis[c1]")
        title = (f"{cell}\n"
                 f"A={a_map.get(cell, {}).get('label', '?')}({a_map.get(cell, {}).get('retention', np.nan):.2f})  "
                 f"B@300={row['label_n300']}  B@400={row['label_n400']}\n"
                 f"cat={row['category'].replace('cat', '')[:25]}")
        ax.set_title(title, fontsize=8)
        ax.legend(loc="lower left", fontsize=7)

    # Hide unused
    for j in range(n, nrows * ncols):
        axes[j // ncols][j % ncols].axis("off")

    fig.tight_layout()
    fig.savefig(out_path, dpi=120)


def main() -> None:
    out = ensure_out_dir()
    df = build_master_table()
    print(f"[05] master table: {len(df)} overlap cells")

    selected = pick_cells(df)
    print(f"[05] selected {len(selected)} cells across {selected['category'].nunique()} categories")
    print(selected.groupby("category").size())

    # Reorder columns: identity + category first, then A side, B side, truth, snapshot
    front = ["cell_name", "category", "why",
             "label", "retention", "max_regular_cycle", "ce2",
             "status", "label_n200", "label_n300", "label_n400",
             "n_regular", "last_fade_cycle", "final_retention",
             "discharge_capacity_retention_final", "coulombic_efficiency_final",
             "n_regular_truth", "ret_c5_truth", "ret_clast_truth", "ret_min_truth",
             "ret_at_n200_truth", "ret_at_n300_truth", "ret_at_n400_truth",
             "ret_at_A_snapshot_truth", "truth_cycle_used_for_snapshot",
             "snapshot_gap", "cohort", "baseline_dis_ah"]
    cols = front + [c for c in selected.columns if c not in front]
    selected[cols].to_csv(out / "cells_to_review.csv", index=False)
    print(f"[05] wrote {out / 'cells_to_review.csv'}")

    # Plot
    reg = load_registry_regular()
    plot_curves(selected, reg, out / "cells_to_review_curves.png")
    print(f"[05] wrote {out / 'cells_to_review_curves.png'}")

    # Print a compact summary table to stdout
    print("\n=== Selected cells ===")
    short = selected[["cell_name", "category", "label", "retention",
                       "max_regular_cycle", "label_n300", "label_n400",
                       "n_regular_truth", "last_fade_cycle",
                       "ret_at_n300_truth", "ret_at_n400_truth"]].copy()
    print(short.to_string(index=False))


if __name__ == "__main__":
    main()

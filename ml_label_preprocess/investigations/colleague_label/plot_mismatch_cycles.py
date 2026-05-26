"""Plot per-regular-cycle charge / discharge / CE / discharge-retention curves
for every cell where our pipeline and the colleague don't agree on a pass/bad call:
the `disagree_*` rows plus `ours_censor` / `ours_excluded` rows from
`out/comparison_table.parquet`.

One 2x2 PNG per cell; saves to `out/mismatch_cycles/`.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import matplotlib.pyplot as plt
import polars as pl

HERE = Path(__file__).resolve().parent
OUT_DIR = HERE / "out"
PLOT_DIR = OUT_DIR / "mismatch_cycles"
COMPARISON_PARQUET = OUT_DIR / "comparison_table.parquet"
NON_AGREE_CATEGORIES = [
    "disagree_we_pass_they_bad",
    "disagree_we_bad_they_pass",
    "ours_censor",
    "ours_excluded",
]
ANNOT_DIR = Path(os.environ.get(
    "BAT_ANNOT_DIR",
    "/mnt/data/mliao/battery-ml-workbench/data/A2.2/annotations",
))


def load_regular_cycles(cell_name: str) -> pl.DataFrame:
    annot = json.loads((ANNOT_DIR / f"{cell_name}.annotations.json").read_text())
    rows = [
        {
            "regular_cycle": e["regular_cycle"],
            "capacity_charge_ah": e["capacity_charge_ah"],
            "capacity_discharge_ah": e["capacity_discharge_ah"],
            "coulombic_efficiency": e["coulombic_efficiency"],
        }
        for e in annot["cd_events"]
        if e.get("regular_cycle") is not None
        and e.get("event_kind") == "regular_cd"
    ]
    return pl.DataFrame(rows).sort("regular_cycle")


def plot_cell(row: dict, cycles: pl.DataFrame, out_path: Path) -> None:
    cell = row["cell_name"]
    ours_label = row["label_n300"]
    theirs_label = row["colleague_label"]
    last_fade = row.get("last_fade_cycle")
    coll_max = row.get("colleague_max_cycle")
    coll_ret = row.get("colleague_retention")

    rc = cycles["regular_cycle"].to_numpy()
    qc = cycles["capacity_charge_ah"].to_numpy()
    qd = cycles["capacity_discharge_ah"].to_numpy()
    ce = cycles["coulombic_efficiency"].to_numpy()

    baseline = float(qd[0]) if len(qd) else float("nan")
    qd_ret = qd / baseline if baseline else qd

    excl_reason = row.get("exclusion_reason") if ours_label == "excluded" else None
    ours_descr = (
        f"label_n300={ours_label} (reason={excl_reason})"
        if excl_reason
        else f"label_n300={ours_label}"
    )

    fig, axes = plt.subplots(2, 2, figsize=(13, 8))
    fig.suptitle(
        f"{cell}  —  ours: {ours_descr}   "
        f"colleague: {theirs_label} (retention={coll_ret:.3f}, "
        f"max_cycle={int(coll_max) if coll_max is not None else 'n/a'})  "
        f"category={row['category']}",
        fontsize=11,
    )

    def overlay_refs(ax, *, with_ret: bool = False):
        ax.axvline(300, color="black", linestyle="--", linewidth=1, alpha=0.5)
        ax.text(305, ax.get_ylim()[0], "N=300", fontsize=8, alpha=0.6)
        if last_fade is not None and not (isinstance(last_fade, float) and last_fade != last_fade):
            ax.axvline(float(last_fade), color="red", linestyle=":", linewidth=1, alpha=0.7,
                       label=f"ours last_fade_cycle={int(last_fade)}")
        if coll_max is not None:
            ax.axvline(float(coll_max), color="green", linestyle=":", linewidth=1, alpha=0.5,
                       label=f"colleague max_cycle={int(coll_max)}")
        if with_ret:
            ax.axhline(0.85, color="black", linestyle="--", linewidth=1, alpha=0.5)
            ax.text(ax.get_xlim()[1] * 0.98, 0.855, "ret=0.85",
                    ha="right", fontsize=8, alpha=0.6)
        ax.legend(loc="best", fontsize=8)

    ax = axes[0, 0]
    ax.plot(rc, qc, marker=".", ms=3, lw=0.8, color="#1f77b4")
    ax.set_xlabel("regular_cycle")
    ax.set_ylabel("charge capacity (Ah)")
    ax.set_title("Charge capacity vs. cycle")
    overlay_refs(ax)

    ax = axes[0, 1]
    ax.plot(rc, qd, marker=".", ms=3, lw=0.8, color="#2ca02c")
    ax.set_xlabel("regular_cycle")
    ax.set_ylabel("discharge capacity (Ah)")
    ax.set_title("Discharge capacity vs. cycle")
    overlay_refs(ax)

    ax = axes[1, 0]
    ax.plot(rc, ce, marker=".", ms=3, lw=0.8, color="#9467bd")
    ax.set_xlabel("regular_cycle")
    ax.set_ylabel("coulombic efficiency")
    ax.set_title("Coulombic efficiency vs. cycle")
    ax.set_ylim(0.95, 1.02)
    overlay_refs(ax)

    ax = axes[1, 1]
    ax.plot(rc, qd_ret, marker=".", ms=3, lw=0.8, color="#d62728")
    ax.set_xlabel("regular_cycle")
    ax.set_ylabel("discharge capacity / cycle-1 baseline")
    ax.set_title(f"Discharge retention vs. cycle (baseline = {baseline:.4f} Ah)")
    overlay_refs(ax, with_ret=True)

    fig.tight_layout(rect=(0, 0, 1, 0.96))
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def main() -> None:
    PLOT_DIR.mkdir(parents=True, exist_ok=True)
    cells = (
        pl.read_parquet(COMPARISON_PARQUET)
        .filter(pl.col("category").is_in(NON_AGREE_CATEGORIES))
        .sort(["category", "cell_name"])
    )
    if cells.is_empty():
        print("no non-agreement cells to plot")
        return

    print(f"plotting {cells.height} non-agreement cells -> {PLOT_DIR}")
    for row in cells.iter_rows(named=True):
        cell = row["cell_name"]
        cycles = load_regular_cycles(cell)
        out_path = PLOT_DIR / f"{cell}.png"
        plot_cell(row, cycles, out_path)
        print(f"  wrote {out_path.name}  "
              f"(category={row['category']}, n_regular_cycles={cycles.height})")


if __name__ == "__main__":
    main()

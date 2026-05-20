"""Per-cell deep-dive figures for the 11 review cells.

For each cell in out/cells_to_review.csv, loads the renumbered step-level
parquet via the toolkit's load_raw_tagged, recomputes per-cycle capacity
and CE from scratch (trapezoid I·dt), and produces a 4-panel figure:

  (1) capacity (charge + discharge) vs regular_cycle  — recomputed + registry
  (2) CE vs regular_cycle — with A.ce2 + B.coulombic_efficiency_final/100
  (3) retention (cap_dis / cap_dis[c1]) vs regular_cycle — with 0.85 line,
      A.retention horiz, B.discharge_capacity_retention_final at c5,
      B.final_retention at c_last
  (4) voltage profile at {c1, c5, c=A.max, c=B.last_fade}

Annotations: N=200/300/400 vertical refs (grey dotted), A.max_regular_cycle
(orange dashed), B.last_fade_cycle (purple dashed).

Output: out/deep_dive/{cell_name}.png (one per cell).
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
    OUT_DIR,
    attach_regular_cycle,
    ensure_out_dir,
    load_colleague,
    load_features,
    load_labels,
    load_raw_tagged_for,
    load_registry_regular,
    recompute_per_cycle_from_raw,
)

N_REFS = (200, 300, 400)


def _add_vrefs(ax, a_max, b_fade):
    for N in N_REFS:
        ax.axvline(N, color="grey", lw=0.4, ls=":")
    if not pd.isna(a_max):
        ax.axvline(float(a_max), color="orange", lw=1.0, ls="--",
                   label=f"A.max={int(a_max)}")
    if not pd.isna(b_fade):
        ax.axvline(float(b_fade), color="purple", lw=1.0, ls="--",
                   label=f"B.fade={int(b_fade)}")


def plot_one_cell(cell_name: str, A_row, B_feat_row, B_lab_row,
                  out_dir: Path) -> dict:
    raw = load_raw_tagged_for(cell_name)
    per = recompute_per_cycle_from_raw(raw)
    per = attach_regular_cycle(per, cell_name).dropna(subset=["regular_cycle"]).copy()
    per["regular_cycle"] = per["regular_cycle"].astype(int)
    per = per.sort_values("regular_cycle").reset_index(drop=True)

    # Registry (for overlay markers and sanity checks)
    reg = load_registry_regular()
    reg_one = reg.loc[reg["cell_name"] == cell_name].sort_values("regular_cycle")

    # Retention from recomputed (anchor on cycle-1 discharge cap)
    if len(per) and per["cap_dis_ah"].iloc[0] > 0:
        cap1 = float(per["cap_dis_ah"].iloc[0])
        per["retention"] = per["cap_dis_ah"] / cap1
    else:
        per["retention"] = np.nan

    a_max = A_row.get("max_regular_cycle", np.nan)
    a_ret = A_row.get("retention", np.nan)
    a_ce2 = A_row.get("ce2", np.nan)
    a_label = A_row.get("label", "?")
    b_status = B_lab_row.get("status", "?") if B_lab_row is not None else "?"
    b_fade = B_lab_row.get("last_fade_cycle", np.nan) if B_lab_row is not None else np.nan
    b_n_regular = B_lab_row.get("n_regular", np.nan) if B_lab_row is not None else np.nan
    b_label_n300 = B_lab_row.get("label_n300", "?") if B_lab_row is not None else "?"
    b_label_n400 = B_lab_row.get("label_n400", "?") if B_lab_row is not None else "?"
    b_final_ret = B_lab_row.get("final_retention", np.nan) if B_lab_row is not None else np.nan
    b_disc_ret_c5 = B_feat_row.get("discharge_capacity_retention_final", np.nan) if B_feat_row is not None else np.nan
    b_ce_c5_pct = B_feat_row.get("coulombic_efficiency_final", np.nan) if B_feat_row is not None else np.nan

    fig, axes = plt.subplots(2, 2, figsize=(15, 9))
    suptitle = (
        f"{cell_name}    "
        f"A.label={a_label}   A.retention={a_ret:.3f}   A.max={int(a_max) if not pd.isna(a_max) else '?'}    "
        f"B.status={b_status}   B@N300={b_label_n300}   B@N400={b_label_n400}   "
        f"B.n_regular={int(b_n_regular) if not pd.isna(b_n_regular) else '?'}"
    )
    fig.suptitle(suptitle, fontsize=10)

    # ---- Panel 1: capacity vs cycle
    ax = axes[0, 0]
    ax.plot(per["regular_cycle"], per["cap_chg_ah"], color="C0", lw=1.0,
            label="recomputed Q_charge")
    ax.plot(per["regular_cycle"], per["cap_dis_ah"], color="C3", lw=1.0,
            label="recomputed Q_discharge")
    if len(reg_one):
        ax.scatter(reg_one["regular_cycle"], reg_one["capacity_charge_ah"],
                   s=8, color="C0", alpha=0.35, marker=".", label="registry Q_chg")
        ax.scatter(reg_one["regular_cycle"], reg_one["capacity_discharge_ah"],
                   s=8, color="C3", alpha=0.35, marker=".", label="registry Q_dis")
    ax.set_xlabel("regular_cycle")
    ax.set_ylabel("capacity (Ah)")
    ax.set_title("(1) Capacity per cycle")
    _add_vrefs(ax, a_max, b_fade)
    ax.legend(fontsize=7, loc="best")

    # ---- Panel 2: CE vs cycle
    ax = axes[0, 1]
    ax.plot(per["regular_cycle"], per["ce"], color="C2", lw=1.0,
            label="recomputed CE")
    if len(reg_one):
        ax.scatter(reg_one["regular_cycle"], reg_one["coulombic_efficiency"],
                   s=8, color="grey", alpha=0.35, marker=".", label="registry CE")
    if not pd.isna(a_ce2):
        ax.axhline(a_ce2, color="orange", lw=0.8, ls=":",
                   label=f"A.ce2={a_ce2:.3f}")
    if not pd.isna(b_ce_c5_pct):
        ax.scatter([5], [b_ce_c5_pct / 100.0], color="purple", marker="X",
                   s=70, zorder=5, label=f"B.CE_final/100={b_ce_c5_pct/100:.3f}@c5")
    ax.set_xlabel("regular_cycle")
    ax.set_ylabel("Coulombic efficiency (fraction)")
    ax.set_title("(2) CE per cycle")
    ax.set_ylim(0.90, 1.02)
    _add_vrefs(ax, a_max, b_fade)
    ax.legend(fontsize=7, loc="best")

    # ---- Panel 3: retention vs cycle
    ax = axes[1, 0]
    ax.plot(per["regular_cycle"], per["retention"], color="C0", lw=1.0,
            label="recomputed retention (cap_dis/cap_dis[c1])")
    ax.axhline(0.85, color="red", lw=0.7, ls=":", label="0.85 threshold")
    if not pd.isna(a_ret):
        ax.axhline(a_ret, color="orange", lw=0.8, ls=":",
                   label=f"A.retention={a_ret:.3f}")
    if not pd.isna(b_disc_ret_c5):
        ax.scatter([5], [b_disc_ret_c5], color="purple", marker="X", s=70,
                   zorder=5, label=f"B.dis_ret_final={b_disc_ret_c5:.3f}@c5")
    if not pd.isna(b_final_ret) and len(per):
        ax.scatter([per["regular_cycle"].iloc[-1]], [b_final_ret], color="green",
                   marker="D", s=60, zorder=5,
                   label=f"B.final_retention={b_final_ret:.3f}@c_last")
    ax.set_xlabel("regular_cycle")
    ax.set_ylabel("retention (cap_dis / cap_dis[c1])")
    ax.set_title("(3) Retention per cycle")
    ax.set_ylim(0.40, 1.15)
    _add_vrefs(ax, a_max, b_fade)
    ax.legend(fontsize=7, loc="best")

    # ---- Panel 4: voltage profile at selected cycles
    ax = axes[1, 1]
    candidates = []
    candidates.append((1, "cycle 1 (baseline)", "C0"))
    if (per["regular_cycle"] == 5).any():
        candidates.append((5, "cycle 5 (B's feature cycle)", "C2"))
    if not pd.isna(a_max):
        amax_int = int(a_max)
        if (per["regular_cycle"] == amax_int).any():
            candidates.append((amax_int, f"A.max={amax_int}", "orange"))
    if not pd.isna(b_fade):
        bf_int = int(b_fade)
        if (per["regular_cycle"] == bf_int).any():
            candidates.append((bf_int, f"B.fade={bf_int}", "purple"))

    # Convert raw polars → pandas just for the cycles we need
    raw_pd = raw.filter(raw["event_kind"] == "regular_cd").to_pandas()
    # Build a cd_index → regular_cycle map from the registry, then back-resolve
    cd_to_rc = reg_one.set_index("cd_index")["regular_cycle"].to_dict()
    raw_pd["regular_cycle"] = raw_pd["cd_index"].map(cd_to_rc)

    for rc, label_txt, color in candidates:
        sub = raw_pd[raw_pd["regular_cycle"] == rc]
        if len(sub) == 0:
            continue
        # within a cycle, plot V vs cumulative time (re-zeroed for readability)
        t0 = sub["elapsed_time"].min()
        ax.plot(sub["elapsed_time"] - t0, sub["voltage"], lw=0.8,
                color=color, label=label_txt)
    ax.set_xlabel("time within cycle (s)")
    ax.set_ylabel("voltage (V)")
    ax.set_title("(4) Voltage profile at key cycles")
    ax.legend(fontsize=7, loc="best")

    fig.tight_layout(rect=(0, 0, 1, 0.97))
    out_path = out_dir / f"{cell_name}.png"
    fig.savefig(out_path, dpi=110)
    plt.close(fig)

    # Sanity-check vs registry
    merged = per.merge(
        reg_one[["regular_cycle", "capacity_charge_ah", "capacity_discharge_ah"]],
        on="regular_cycle", how="inner",
    )
    max_dchg = float((merged["cap_chg_ah"] - merged["capacity_charge_ah"]).abs().max()) if len(merged) else np.nan
    max_ddis = float((merged["cap_dis_ah"] - merged["capacity_discharge_ah"]).abs().max()) if len(merged) else np.nan
    return {
        "cell_name": cell_name,
        "n_regular_recomputed": int(per["regular_cycle"].max()) if len(per) else 0,
        "max_abs_diff_chg_ah": max_dchg,
        "max_abs_diff_dis_ah": max_ddis,
        "out": str(out_path),
    }


def main() -> None:
    out_root = ensure_out_dir()
    deep_dir = out_root / "deep_dive"
    deep_dir.mkdir(parents=True, exist_ok=True)

    cells = pd.read_csv(out_root / "cells_to_review.csv")["cell_name"].tolist()
    print(f"[06] plotting {len(cells)} cells")

    A = load_colleague().set_index("cell_name")
    Bf = load_features().set_index("cell_name")
    Bl = load_labels().set_index("cell_name")

    rows = []
    for cell in cells:
        try:
            r = plot_one_cell(
                cell,
                A.loc[cell] if cell in A.index else pd.Series(dtype=object),
                Bf.loc[cell] if cell in Bf.index else None,
                Bl.loc[cell] if cell in Bl.index else None,
                deep_dir,
            )
            print(f"  {cell}: n_regular={r['n_regular_recomputed']}  "
                  f"|Δcap_chg|_max={r['max_abs_diff_chg_ah']:.5f}Ah  "
                  f"|Δcap_dis|_max={r['max_abs_diff_dis_ah']:.5f}Ah  → {r['out']}")
            rows.append(r)
        except Exception as e:
            print(f"  {cell}: ERROR {type(e).__name__}: {e}")
            rows.append({"cell_name": cell, "error": f"{type(e).__name__}: {e}"})

    summary = pd.DataFrame(rows)
    summary.to_csv(out_root / "deep_dive_summary.csv", index=False)
    print(f"\n[06] wrote {out_root / 'deep_dive_summary.csv'}")
    if "max_abs_diff_chg_ah" in summary.columns:
        bad = summary[(summary["max_abs_diff_chg_ah"] > 0.01) |
                      (summary["max_abs_diff_dis_ah"] > 0.01)]
        if len(bad):
            print(f"[06] ⚠ {len(bad)} cells have |Δ| > 0.01 Ah vs registry:")
            print(bad[["cell_name", "max_abs_diff_chg_ah", "max_abs_diff_dis_ah"]].to_string(index=False))
        else:
            print("[06] all cells reproduce registry to < 0.01 Ah ✓")


if __name__ == "__main__":
    main()

"""Shared loaders for the colleague-vs-mine annotation audit.

All paths are hard-coded to this workspace — this is a one-off investigation,
not a redistributable package.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

# Toolkit path so we can reuse load_raw_tagged (the renumbered+annotated loader).
_TOOLKIT_DASHBOARD = "/mnt/data/mliao/battery-ml-workbench/battery-annotation-toolkit/dashboard"
if _TOOLKIT_DASHBOARD not in sys.path:
    sys.path.insert(0, _TOOLKIT_DASHBOARD)

ROOT = Path("/mnt/data/mliao/battery-ml-workbench")
COLLEAGUE_PARQUET = ROOT / "new_5cycle_ml/colleague_annoation/all_features.parquet"
B_FEATURES = ROOT / "new_5cycle_ml/ml_label_preprocess/datasets/A2.2_b1/cell_features.parquet"
B_LABELS = ROOT / "new_5cycle_ml/ml_label_preprocess/datasets/A2.2_b1/cell_labels.parquet"
REGISTRY = ROOT / "data/A2.2/annotations/_annotations.parquet"
OUT_DIR = ROOT / "new_5cycle_ml/colleague_comparison/out"


def load_colleague() -> pd.DataFrame:
    return pd.read_parquet(COLLEAGUE_PARQUET)


def load_features() -> pd.DataFrame:
    return pd.read_parquet(B_FEATURES)


def load_labels() -> pd.DataFrame:
    return pd.read_parquet(B_LABELS)


def load_registry_regular() -> pd.DataFrame:
    """Per-cycle truth, filtered to regular cycling events.

    Columns kept: cell_name, regular_cycle (int), capacity_charge_ah,
    capacity_discharge_ah, coulombic_efficiency, cd_index, tester_cycle.
    """
    r = pd.read_parquet(REGISTRY)
    r = r[r["event_kind"] == "regular_cd"].copy()
    r["regular_cycle"] = r["regular_cycle"].astype(int)
    keep = [
        "cell_name", "regular_cycle", "cd_index", "tester_cycle",
        "capacity_charge_ah", "capacity_discharge_ah", "coulombic_efficiency",
    ]
    return r[keep].sort_values(["cell_name", "regular_cycle"]).reset_index(drop=True)


def load_registry_all() -> pd.DataFrame:
    """Full registry incl. formation / rate_test / pulse events. Kept lean."""
    r = pd.read_parquet(REGISTRY)
    keep = [
        "cell_name", "event_kind", "cd_index", "tester_cycle", "regular_cycle",
        "capacity_charge_ah", "capacity_discharge_ah", "coulombic_efficiency",
    ]
    return r[keep].sort_values(["cell_name", "cd_index"]).reset_index(drop=True)


def overlap_cells() -> list[str]:
    """Cells present in both colleague + mine (intersected on cell_name)."""
    a = set(load_colleague()["cell_name"])
    b = set(load_features()["cell_name"]) | set(load_labels()["cell_name"])
    return sorted(a & b)


def truth_at_regular(reg: pd.DataFrame, n: int) -> pd.DataFrame:
    """One row per cell with capacities + CE at regular_cycle == n."""
    sub = reg[reg["regular_cycle"] == n][[
        "cell_name", "capacity_charge_ah", "capacity_discharge_ah", "coulombic_efficiency",
    ]].copy()
    sub = sub.rename(columns={
        "capacity_charge_ah": f"cap_chg_c{n}",
        "capacity_discharge_ah": f"cap_dis_c{n}",
        "coulombic_efficiency": f"ce_c{n}",
    })
    return sub


def ensure_out_dir() -> Path:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    return OUT_DIR


def load_raw_tagged_for(cell_name: str):
    """Polars DataFrame: renumbered raw + (cd_index, cd_phase, event_kind).

    Reuses the toolkit's loader at battery-annotation-toolkit/dashboard/verify.py.
    Columns: cycle, step, elapsed_time, step_time, current, voltage, state,
             cd_index, cd_phase, event_kind.
    """
    import verify  # toolkit dashboard module (path added at module load)
    return verify.load_raw_tagged(cell_name)


def recompute_per_cycle_from_raw(raw_pl) -> pd.DataFrame:
    """Trapezoid-integrate current·dt per regular-cycle charge/discharge phase.

    Input: Polars DataFrame from `load_raw_tagged_for`.
    Returns pandas DataFrame: cd_index, regular_cycle, cap_chg_ah, cap_dis_ah, ce.

    Filters to event_kind == 'regular_cd' and cd_phase in {'charge', 'discharge'},
    so formation / rate_test / pulse / rebalance events are excluded.
    """
    import polars as pl

    # restrict to regular CD events with a labeled phase
    df = raw_pl.filter(
        (pl.col("event_kind") == "regular_cd")
        & (pl.col("cd_phase").is_in(["charge", "discharge"]))
    ).sort(["cd_index", "cd_phase", "elapsed_time"])

    if df.height == 0:
        return pd.DataFrame(columns=["cd_index", "regular_cycle",
                                      "cap_chg_ah", "cap_dis_ah", "ce"])

    # Trapezoid: contribution_i = (I_i + I_{i+1})/2 * (t_{i+1} - t_i)
    # We compute the per-row contribution between consecutive samples within
    # each (cd_index, cd_phase) group, then sum to get Coulombs (A·s); /3600 → Ah.
    df = df.with_columns(
        dt=(pl.col("elapsed_time").diff().over(["cd_index", "cd_phase"])),
        i_prev=(pl.col("current").shift(1).over(["cd_index", "cd_phase"])),
    )
    df = df.with_columns(
        seg=((pl.col("current") + pl.col("i_prev")) / 2.0 * pl.col("dt")),
    )

    per_phase = (
        df.group_by(["cd_index", "cd_phase"])
        .agg(pl.col("seg").sum().alias("A_s"))
        .with_columns((pl.col("A_s").abs() / 3600.0).alias("cap_ah"))
        .pivot(index="cd_index", on="cd_phase", values="cap_ah")
        .rename({"charge": "cap_chg_ah", "discharge": "cap_dis_ah"})
        .sort("cd_index")
    )

    out = per_phase.to_pandas()
    # CE = discharge / charge, NaN if either side missing or zero
    out["ce"] = out["cap_dis_ah"] / out["cap_chg_ah"].where(out["cap_chg_ah"] > 0)

    # Recover regular_cycle by joining with the registry (regular_cd only)
    reg = load_registry_regular()
    # we don't have the cell name here, but the cd_index space is per-cell and the
    # caller's raw_pl was for one cell — the caller can attach cell_name + regular_cycle:
    # we just return cd_index-keyed values; helper below joins.
    return out


def attach_regular_cycle(per_cycle: pd.DataFrame, cell_name: str) -> pd.DataFrame:
    """Join recomputed per-cycle table with registry to add `regular_cycle`."""
    reg = load_registry_regular()
    reg_one = reg.loc[reg["cell_name"] == cell_name, ["cd_index", "regular_cycle"]]
    return per_cycle.merge(reg_one, on="cd_index", how="left").sort_values("regular_cycle")

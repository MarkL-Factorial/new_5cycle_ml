"""Diff two DOP-feature snapshots produced by `run_investigation.py`
(or the equivalent candidate runner).

Usage:
    python compare_snapshots.py <new_snapshot_dir> <ref_snapshot_dir>

Both args are paths to a timestamped snapshot dir containing:
    cell_dop_features.parquet
    cell_dop_features_status.csv
    manifest.json

Outputs a per-feature numeric diff summary, status-flag diff counts, and
a manifest-delta block to stdout, plus a per-cell diff CSV written to
``<new_snapshot_dir>/comparison_vs_<ref_dir_name>.csv`` listing every
row where any feature differs by more than TOL.
"""
from __future__ import annotations

import json
import math
import sys
from pathlib import Path

import polars as pl

from dop_features import FEATURE_COLUMNS

TOL = 1e-9
STATUS_FLAGS = (
    "has_c1_chg", "has_c5_chg", "has_c1_dis", "has_c5_dis",
    "dop_ok_c1_chg", "dop_ok_c5_chg", "dop_ok_c1_dis", "dop_ok_c5_dis",
    "n_features_success",
)


def _abs_diff(a: float | None, b: float | None) -> float | None:
    if a is None or b is None:
        return None
    if (isinstance(a, float) and math.isnan(a)) or (isinstance(b, float) and math.isnan(b)):
        return None
    return abs(float(a) - float(b))


def _both_nan(a, b) -> bool:
    return (
        a is None or (isinstance(a, float) and math.isnan(a))
    ) and (
        b is None or (isinstance(b, float) and math.isnan(b))
    )


def _is_nan(v) -> bool:
    return v is None or (isinstance(v, float) and math.isnan(v))


def main() -> int:
    if len(sys.argv) != 3:
        print(__doc__, file=sys.stderr)
        return 2
    new_dir = Path(sys.argv[1]).resolve()
    ref_dir = Path(sys.argv[2]).resolve()
    for d in (new_dir, ref_dir):
        if not (d / "cell_dop_features.parquet").exists():
            print(f"missing cell_dop_features.parquet in {d}", file=sys.stderr)
            return 1

    new_feat = pl.read_parquet(new_dir / "cell_dop_features.parquet")
    ref_feat = pl.read_parquet(ref_dir / "cell_dop_features.parquet")
    new_stat = pl.read_csv(new_dir / "cell_dop_features_status.csv")
    ref_stat = pl.read_csv(ref_dir / "cell_dop_features_status.csv")
    new_manifest = json.loads((new_dir / "manifest.json").read_text())
    ref_manifest = json.loads((ref_dir / "manifest.json").read_text())

    # Cohort parity -------------------------------------------------------
    new_cells = set(new_feat["cell_name"].to_list())
    ref_cells = set(ref_feat["cell_name"].to_list())
    only_new = sorted(new_cells - ref_cells)
    only_ref = sorted(ref_cells - new_cells)
    both = sorted(new_cells & ref_cells)
    print(f"NEW    snapshot : {new_dir.name}   n_cells={len(new_cells)}")
    print(f"REF    snapshot : {ref_dir.name}   n_cells={len(ref_cells)}")
    print(f"in both         : {len(both)}")
    print(f"only in NEW     : {len(only_new)}  {only_new[:10]}{'...' if len(only_new) > 10 else ''}")
    print(f"only in REF     : {len(only_ref)}  {only_ref[:10]}{'...' if len(only_ref) > 10 else ''}")
    print()

    # Per-feature numeric diffs ------------------------------------------
    # Outer join so we keep all cells.
    join = new_feat.select(["cell_name", *FEATURE_COLUMNS]).rename(
        {c: f"{c}__new" for c in FEATURE_COLUMNS}
    ).join(
        ref_feat.select(["cell_name", *FEATURE_COLUMNS]).rename(
            {c: f"{c}__ref" for c in FEATURE_COLUMNS}
        ),
        on="cell_name", how="full", coalesce=True,
    )

    print("per-feature numeric diff (over cells present in both snapshots):")
    print(f"  TOL = {TOL:g}")
    print(f"  {'column':<36s}  {'n_identical':>11s}  {'n_diff':>6s}  {'max|Δ|':>11s}  {'mean|Δ|':>11s}  {'n_nan_flip':>10s}")
    per_col_summary = []
    for col in FEATURE_COLUMNS:
        new_vals = join[f"{col}__new"].to_list()
        ref_vals = join[f"{col}__ref"].to_list()
        cells_col = join["cell_name"].to_list()
        n_identical = 0
        n_diff = 0
        max_abs = 0.0
        sum_abs = 0.0
        n_nan_flip = 0
        for cell, a, b in zip(cells_col, new_vals, ref_vals):
            if cell not in new_cells or cell not in ref_cells:
                continue  # outer-join padding row; skip
            if _both_nan(a, b):
                n_identical += 1
                continue
            a_nan = _is_nan(a); b_nan = _is_nan(b)
            if a_nan != b_nan:
                n_nan_flip += 1
                continue
            d = abs(float(a) - float(b))
            if d <= TOL:
                n_identical += 1
            else:
                n_diff += 1
                sum_abs += d
                if d > max_abs:
                    max_abs = d
        mean_abs = (sum_abs / n_diff) if n_diff else 0.0
        print(f"  {col:<36s}  {n_identical:>11d}  {n_diff:>6d}  {max_abs:>11.3e}  {mean_abs:>11.3e}  {n_nan_flip:>10d}")
        per_col_summary.append({
            "column": col, "n_identical": n_identical, "n_diff": n_diff,
            "max_abs_diff": max_abs, "mean_abs_diff": mean_abs,
            "n_nan_flip": n_nan_flip,
        })

    # Status flag diffs --------------------------------------------------
    print()
    print("status-flag diff counts (over cells in both):")
    status_join = new_stat.select(["cell_name", *STATUS_FLAGS]).rename(
        {c: f"{c}__new" for c in STATUS_FLAGS}
    ).join(
        ref_stat.select(["cell_name", *STATUS_FLAGS]).rename(
            {c: f"{c}__ref" for c in STATUS_FLAGS}
        ),
        on="cell_name", how="inner",
    )
    for flag in STATUS_FLAGS:
        n_changed = int(status_join.filter(
            pl.col(f"{flag}__new") != pl.col(f"{flag}__ref")
        ).height)
        print(f"  {flag:<24s}  n_changed = {n_changed}")

    # Manifest delta -----------------------------------------------------
    print()
    print("manifest delta:")
    for k in ("generated_at", "mode", "n_cells_attempted", "n_cells_full"):
        print(f"  {k:<20s}  NEW={new_manifest.get(k)!r:<30s}  REF={ref_manifest.get(k)!r}")
    new_fit = new_manifest.get("params", {}).get("fit_config", {})
    ref_fit = ref_manifest.get("params", {}).get("fit_config", {})
    fit_changed = {k for k in set(new_fit) | set(ref_fit) if new_fit.get(k) != ref_fit.get(k)}
    if fit_changed:
        print(f"  fit_config diff keys: {sorted(fit_changed)}")
        for k in sorted(fit_changed):
            print(f"    {k}: NEW={new_fit.get(k)} REF={ref_fit.get(k)}")
    else:
        print("  fit_config: identical")

    # Per-cell diff CSV --------------------------------------------------
    out_path = new_dir / f"comparison_vs_{ref_dir.name}.csv"
    diff_rows = []
    new_lookup = {row["cell_name"]: row for row in new_feat.iter_rows(named=True)}
    ref_lookup = {row["cell_name"]: row for row in ref_feat.iter_rows(named=True)}
    for cell in sorted(new_cells & ref_cells):
        any_diff = False
        rec: dict[str, object] = {"cell_name": cell}
        for col in FEATURE_COLUMNS:
            a = new_lookup[cell][col]; b = ref_lookup[cell][col]
            rec[f"{col}__new"] = a
            rec[f"{col}__ref"] = b
            if _both_nan(a, b):
                rec[f"{col}__abs_diff"] = 0.0
                continue
            if _is_nan(a) or _is_nan(b):
                rec[f"{col}__abs_diff"] = None
                any_diff = True
                continue
            d = abs(float(a) - float(b))
            rec[f"{col}__abs_diff"] = d
            if d > TOL:
                any_diff = True
        if any_diff:
            diff_rows.append(rec)
    pl.DataFrame(diff_rows).write_csv(out_path) if diff_rows else out_path.write_text(
        "cell_name\n"  # empty header-only file when nothing differs
    )
    print()
    print(f"per-cell diff CSV: {out_path}  (n_rows={len(diff_rows)})")
    return 0


if __name__ == "__main__":
    sys.exit(main())

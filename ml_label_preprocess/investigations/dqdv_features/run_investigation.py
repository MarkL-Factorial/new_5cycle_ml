"""CLI: iterate annotation JSONs, extract 8 dQ/dV features per cell.

Writes a timestamped sub-folder under ``out/`` with one parquet per
feature-set version (v1 + v2), a unified status CSV, and a manifest:

    cell_dqdv_features_v1.parquet / .csv      (4 v1 cols + cell_name + cohort)
    cell_dqdv_features_v2.parquet / .csv      (4 v2 cols + cell_name + cohort)
    cell_dqdv_features_status.csv             (per-cell QC for both sets)
    manifest.json                             (params + counts)
    plots/                                    (only when --plots or --pilot)
        overlay_<cell>.png                    (dQ/dV c1 + c3 + c5)
        deltaQ_<cell>.png                     (Q_c5(V) − Q_c1(V))

Run modes:
    default                full A2.2 sweep, no plots
    --pilot                5 cells per cohort + plots (eyeball test)
    --cells X Y ...        subset (plots if --plots also given)
    --plots                generate diagnostic plots
    --no-plots             explicit no-plot, even with --cells (default)

The orchestration here is deliberately thin — the per-cell logic lives
in dqdv_features.featurize_cell. This file's job is iteration, output
plumbing, and plotting.

Author: Mark Liao (Sheng-Lun Liao)
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from datetime import datetime, timezone
from pathlib import Path

import polars as pl

# Parent dirs added to sys.path so we can import _common from ../../
_THIS_DIR = Path(__file__).resolve().parent
_PIPELINE_DIR = _THIS_DIR.parent.parent
sys.path.insert(0, str(_PIPELINE_DIR))

from _common import (  # noqa: E402
    ANNOT_DIR,
    _cohort,
    db_version_from_path,
    iter_annotations,
)
from dqdv_features import (  # noqa: E402
    COMMON_GRID_STEP_V,
    FEATURE_COLUMNS,
    FEATURE_COLUMNS_V1,
    FEATURE_COLUMNS_V2,
    PEAK_MIN_DISTANCE_SAMPLES,
    PEAK_PROMINENCE_FRAC,
    _CYCLES_FOR_PLOTS,
    _load_dqdv,
    _regular_cycle_cd_index,
    featurize_cell,
)
from battery_workbench.core.data.annotations import load_raw_tagged  # noqa: E402
from battery_workbench.core.analysis.dqdv import (  # noqa: E402
    DIRECTION_CHARGE,
    DIRECTION_DISCHARGE,
)


PILOT_PER_COHORT = 5
OUT_DIR = _THIS_DIR / "out"


def _parse_args(argv: list[str]) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        prog="run_investigation.py",
        description="Extract 8 dQ/dV features per cell from the first 5 cycles.",
    )
    p.add_argument("--pilot", action="store_true",
                   help=f"run on {PILOT_PER_COHORT} cells per cohort with plots")
    p.add_argument("--cells", nargs="+", default=None,
                   help="restrict to these cell names (debugging)")
    plot_group = p.add_mutually_exclusive_group()
    plot_group.add_argument("--plots", action="store_true",
                            help="generate diagnostic plots (forced on with --pilot)")
    plot_group.add_argument("--no-plots", action="store_true",
                            help="suppress plots (default for full runs)")
    return p.parse_args(argv)


# ---------------------------------------------------------------------------
# Plotting helpers (matplotlib loaded lazily — only when requested).
# ---------------------------------------------------------------------------

def _import_matplotlib():
    import matplotlib
    matplotlib.use("Agg")  # headless
    import matplotlib.pyplot as plt
    return plt


def _plot_overlay(cell_name: str, annot: dict, raw_tagged: pl.DataFrame,
                  plots_dir: Path) -> None:
    """Overlay dQ/dV curves for cycles 1, 3, 5 (charge + discharge) and
    mark the dominant peaks."""
    plt = _import_matplotlib()
    fig, (ax_c, ax_d) = plt.subplots(2, 1, figsize=(7, 7), sharex=True)
    colors = {1: "C0", 3: "C1", 5: "C2"}

    for cyc in _CYCLES_FOR_PLOTS:
        cd = _regular_cycle_cd_index(annot, cyc)
        if cd is None:
            continue
        for direction, ax in [(DIRECTION_CHARGE, ax_c), (DIRECTION_DISCHARGE, ax_d)]:
            _, _, V, d = _load_dqdv(raw_tagged, cd, direction)
            if V.size == 0:
                continue
            ax.plot(V, d, color=colors[cyc], label=f"c{cyc}", lw=1)
            from dqdv_features import find_dominant_peak  # local import for plotting
            peak = find_dominant_peak(V, d, direction=direction)
            if peak is not None:
                ax.plot(peak[0], peak[1], marker="o", color=colors[cyc], ms=6, mfc="none")

    ax_c.set_title(f"{cell_name} — dQ/dV charge (CC only)")
    ax_d.set_title(f"{cell_name} — dQ/dV discharge")
    ax_d.set_xlabel("Voltage (V)")
    for ax in (ax_c, ax_d):
        ax.axhline(0, color="k", lw=0.5, alpha=0.3)
        ax.set_ylabel("dQ/dV (Ah/V)")
        ax.legend(loc="best", fontsize=8)
    fig.tight_layout()
    fig.savefig(plots_dir / f"overlay_{cell_name}.png", dpi=120)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Orchestration.
# ---------------------------------------------------------------------------

def _select_pilot_cells(annot_iter) -> set[str]:
    """First PILOT_PER_COHORT cells per cohort, by name-sorted order."""
    by_cohort: dict[str, list[str]] = {"0MC": [], "AR": []}
    for path, _ in annot_iter:
        name = path.stem.replace(".annotations", "")
        cohort = _cohort(name)
        if cohort in by_cohort and len(by_cohort[cohort]) < PILOT_PER_COHORT:
            by_cohort[cohort].append(name)
        if all(len(v) >= PILOT_PER_COHORT for v in by_cohort.values()):
            break
    return set(by_cohort["0MC"] + by_cohort["AR"])


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv if argv is not None else sys.argv[1:])

    # Plot policy.
    if args.pilot:
        do_plots = True
    elif args.plots:
        do_plots = True
    else:
        do_plots = False

    # Snapshot dir.
    ts = datetime.now().strftime("%Y%m%d_%H%M")
    snapshot_dir = OUT_DIR / ts
    snapshot_dir.mkdir(parents=True, exist_ok=True)
    plots_dir = snapshot_dir / "plots"
    if do_plots:
        plots_dir.mkdir(exist_ok=True)

    # Cell selection.
    pilot_set = None
    if args.pilot:
        # Need to iterate twice; iter_annotations is cheap, so do it again.
        pilot_set = _select_pilot_cells(iter_annotations())
        print(f"pilot set ({len(pilot_set)} cells): {sorted(pilot_set)}")

    user_subset = set(args.cells) if args.cells else None

    feature_rows: list[dict] = []
    status_rows: list[dict] = []
    n_attempted = 0

    for path, annot in iter_annotations():
        cell_name = path.stem.replace(".annotations", "")
        if pilot_set is not None and cell_name not in pilot_set:
            continue
        if user_subset is not None and cell_name not in user_subset:
            continue

        n_attempted += 1
        if n_attempted % 25 == 0:
            print(f"  [{n_attempted}] processed; latest cell={cell_name}")

        # One raw load per cell; reused by featurize_cell and any plots.
        try:
            raw_tagged = load_raw_tagged(cell_name)
        except Exception as exc:
            raw_tagged = None
            print(f"  load WARN {cell_name}: {type(exc).__name__}: {exc}", file=sys.stderr)

        feat, stat = featurize_cell(cell_name, annot, raw_tagged=raw_tagged)
        feat["cohort"] = _cohort(cell_name)
        stat["cohort"] = _cohort(cell_name)
        feature_rows.append(feat)
        status_rows.append(stat)

        if do_plots and raw_tagged is not None:
            try:
                _plot_overlay(cell_name, annot, raw_tagged, plots_dir)
            except Exception as exc:
                print(f"  plot WARN {cell_name}: {type(exc).__name__}: {exc}", file=sys.stderr)

    if not feature_rows:
        print("no cells selected — nothing to write", file=sys.stderr)
        return 1

    # Build polars frames. One row per cell carries all 8 features;
    # we split into two parquets (v1 / v2) for downstream consumers.
    feat_all = pl.DataFrame(feature_rows)
    feat_v1 = feat_all.select(["cell_name", "cohort", *FEATURE_COLUMNS_V1])
    feat_v2 = feat_all.select(["cell_name", "cohort", *FEATURE_COLUMNS_V2])
    stat_df = pl.DataFrame(status_rows).select([
        "cell_name", "cohort",
        "has_c1_dis", "has_c5_dis", "has_c5_chg",
        "c1_discharge_cap_ah",
        "dqdv_v1_n_success", "dqdv_v2_n_success",
        "error_msg",
    ])

    feat_v1.write_parquet(snapshot_dir / "cell_dqdv_features_v1.parquet")
    feat_v1.write_csv(snapshot_dir / "cell_dqdv_features_v1.csv")
    feat_v2.write_parquet(snapshot_dir / "cell_dqdv_features_v2.parquet")
    feat_v2.write_csv(snapshot_dir / "cell_dqdv_features_v2.csv")
    stat_df.write_csv(snapshot_dir / "cell_dqdv_features_status.csv")

    # Manifest.
    n_v1_full = int(stat_df.filter(
        pl.col("dqdv_v1_n_success") == len(FEATURE_COLUMNS_V1)
    ).height)
    n_v2_full = int(stat_df.filter(
        pl.col("dqdv_v2_n_success") == len(FEATURE_COLUMNS_V2)
    ).height)
    manifest = {
        "schema_version": 2,
        "db_version": db_version_from_path(ANNOT_DIR),
        "annot_dir": str(ANNOT_DIR),
        "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "mode": "pilot" if args.pilot else ("subset" if args.cells else "full"),
        "n_cells_attempted": n_attempted,
        "n_cells_v1_full": n_v1_full,
        "n_cells_v2_full": n_v2_full,
        "feature_columns_v1": list(FEATURE_COLUMNS_V1),
        "feature_columns_v2": list(FEATURE_COLUMNS_V2),
        "params": {
            "smoothing": {"method": "savgol", "window_length": 51, "polyorder": 3},
            "common_grid_step_v": COMMON_GRID_STEP_V,
            "peak_detection": {
                "prominence_frac": PEAK_PROMINENCE_FRAC,
                "min_distance_samples": PEAK_MIN_DISTANCE_SAMPLES,
            },
        },
    }
    (snapshot_dir / "manifest.json").write_text(json.dumps(manifest, indent=2))

    # Quick stdout report.
    feat_df = feat_all.select(["cell_name", "cohort", *FEATURE_COLUMNS])
    print()
    print(f"snapshot     = {snapshot_dir}")
    print(f"attempted    = {n_attempted}")
    print(f"v1 full rows = {n_v1_full} ({n_v1_full / max(n_attempted, 1):.1%})")
    print(f"v2 full rows = {n_v2_full} ({n_v2_full / max(n_attempted, 1):.1%})")
    print()
    print("per-feature null counts:")
    for col in FEATURE_COLUMNS:
        n_null = int(feat_df.filter(pl.col(col).is_null() | pl.col(col).is_nan()).height)
        print(f"  {col:<40s}  null = {n_null}/{n_attempted}")
    print()
    print("status dqdv_v1_n_success distribution:")
    print(stat_df.group_by("dqdv_v1_n_success").len().sort("dqdv_v1_n_success"))
    print("status dqdv_v2_n_success distribution:")
    print(stat_df.group_by("dqdv_v2_n_success").len().sort("dqdv_v2_n_success"))

    # Per-cohort distribution of each feature — sanity check that the
    # c1-capacity normalization actually equalises the cohorts.
    print()
    print("per-cohort feature summary (mean ± std, n):")
    for col in FEATURE_COLUMNS:
        sub = feat_df.filter(pl.col(col).is_not_nan())
        if sub.is_empty():
            print(f"  {col:<32s}  (all NaN)")
            continue
        grp = sub.group_by("cohort").agg([
            pl.col(col).mean().alias("mean"),
            pl.col(col).std().alias("std"),
            pl.col(col).count().alias("n"),
        ]).sort("cohort")
        print(f"  {col}:")
        for row in grp.iter_rows(named=True):
            print(f"    {row['cohort']:5s}  {row['mean']:+.4f} ± {row['std']:.4f}   n={row['n']}")

    return 0


if __name__ == "__main__":
    sys.exit(main())

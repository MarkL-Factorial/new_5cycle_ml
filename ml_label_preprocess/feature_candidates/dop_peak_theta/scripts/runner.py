"""CLI: iterate annotation JSONs, extract 6 DOP-peak features per cell.

Mirrors investigations/dqdv_features/run_investigation.py in shape.
Writes a timestamped sub-folder under ``out/`` with one parquet, a
status CSV, and a manifest:

    cell_dop_features.parquet / .csv      (6 feature cols + cell_name + cohort)
    cell_dop_features_status.csv          (per-segment QC + timing)
    manifest.json                         (params + counts + db_version)
    plots/                                (only when --plots or --pilot)
        overlay_<cell>.png                (4 panels: c1/c5 × chg/dis ρ(θ))

Run modes:
    default                full A2.2 sweep, no plots
    --pilot                5 cells per cohort + plots (eyeball + timing test)
    --cells X Y ...        subset (plots if --plots also given)
    --plots                generate diagnostic plots
    --no-plots             explicit no-plot, even with --cells (default)

The orchestration here is deliberately thin — the per-cell logic lives
in dop_features.featurize_cell. This file owns iteration, output
plumbing, and plotting.

Author: Mark Liao (Sheng-Lun Liao)
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from dataclasses import asdict
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
from dop_features import (  # noqa: E402
    CHARGE_WINDOW_MIN,
    DISCHARGE_WINDOW_MIN,
    FEATURE_COLUMNS,
    _SEGMENTS,
    featurize_cell,
)
from battery_workbench.core.analysis.drt_wrapper import (  # noqa: E402
    DRTAnalyzer,
    FitConfig,
)


PILOT_PER_COHORT = 5
OUT_DIR = _THIS_DIR / "out"


def _parse_args(argv: list[str]) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        prog="run_investigation.py",
        description="Extract 6 DOP-peak features per cell from cycles 1+5.",
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


def _plot_overlay(
    cell_name: str,
    segment_results: dict,
    feat_row: dict,
    plots_dir: Path,
) -> None:
    """Four panels: c1/c5 × chg/dis. Each shows ρ(θ) with the dominant
    peak marked. If a fit failed, panel shows an error annotation."""
    plt = _import_matplotlib()
    fig, axes = plt.subplots(2, 2, figsize=(10, 7))
    # _SEGMENTS order: c1_chg, c5_chg, c1_dis, c5_dis
    panel_axes = [axes[0, 0], axes[0, 1], axes[1, 0], axes[1, 1]]

    for (seg_key, cycle, direction, minutes), ax in zip(_SEGMENTS, panel_axes):
        direction_abbrev = "chg" if direction == "charge" else "dis"
        feat_key = f"dop_peak_theta_c{cycle}_{direction_abbrev}"
        title = f"c{cycle} {direction} ({minutes:.0f} min window)"
        ax.set_title(title)
        ax.set_xlabel("θ (degrees)")
        ax.set_ylabel("ρ (normalised)")

        result = segment_results.get(seg_key)
        if result is None or result.theta is None or result.rho is None:
            ax.text(0.5, 0.5, "fit failed / no DOP",
                    transform=ax.transAxes, ha="center", va="center",
                    fontsize=11, color="C3")
            continue

        # ρ(θ) curve. Both arrays come out of the wrapper sorted to
        # match θ as returned (descending or ascending — accept both).
        ax.plot(result.theta, result.rho, color="C0", lw=1.2)

        # Mark the dominant peak from result.dop_peaks (post-Ohmic-filter
        # already applied inside drt_wrapper). Theta of the chosen peak
        # is exactly feat_row[feat_key].
        theta_pick = feat_row.get(feat_key, math.nan)
        if isinstance(theta_pick, float) and math.isfinite(theta_pick):
            # ρ at θ_pick — look up the dominant peak object directly.
            dominant = max(result.dop_peaks, key=lambda p: p.rho_max)
            ax.plot(dominant.theta_center, dominant.rho_max,
                    marker="o", color="C3", ms=8, mfc="none")
            ax.axvline(dominant.theta_center, color="C3", lw=0.8, alpha=0.4)

        ax.axhline(0, color="k", lw=0.5, alpha=0.3)

    fig.suptitle(f"{cell_name} — DOP ρ(θ)")
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
    do_plots = args.pilot or args.plots

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
        pilot_set = _select_pilot_cells(iter_annotations())
        print(f"pilot set ({len(pilot_set)} cells): {sorted(pilot_set)}")

    user_subset = set(args.cells) if args.cells else None

    # One analyzer per process; each fit constructs a fresh internal model.
    analyzer = DRTAnalyzer(fit_dop=True)

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
        if n_attempted % 10 == 0:
            print(f"  [{n_attempted}] processed; latest cell={cell_name}", flush=True)

        feat, stat, seg_results = featurize_cell(cell_name, annot, analyzer=analyzer)
        feat["cohort"] = _cohort(cell_name)
        stat["cohort"] = _cohort(cell_name)
        feature_rows.append(feat)
        status_rows.append(stat)

        if do_plots:
            try:
                _plot_overlay(cell_name, seg_results, feat, plots_dir)
            except Exception as exc:
                print(f"  plot WARN {cell_name}: {type(exc).__name__}: {exc}",
                      file=sys.stderr)

    if not feature_rows:
        print("no cells selected — nothing to write", file=sys.stderr)
        return 1

    feat_df = pl.DataFrame(feature_rows).select(
        ["cell_name", "cohort", *FEATURE_COLUMNS]
    )
    stat_df = pl.DataFrame(status_rows).select([
        "cell_name", "cohort",
        "has_c1_chg", "has_c5_chg", "has_c1_dis", "has_c5_dis",
        "dop_ok_c1_chg", "dop_ok_c5_chg", "dop_ok_c1_dis", "dop_ok_c5_dis",
        "fit_time_s_total", "n_features_success", "error_msg",
    ])

    feat_df.write_parquet(snapshot_dir / "cell_dop_features.parquet")
    feat_df.write_csv(snapshot_dir / "cell_dop_features.csv")
    stat_df.write_csv(snapshot_dir / "cell_dop_features_status.csv")

    # Manifest.
    n_full = int(stat_df.filter(
        pl.col("n_features_success") == len(FEATURE_COLUMNS)
    ).height)
    manifest = {
        "schema_version": 1,
        "db_version": db_version_from_path(ANNOT_DIR),
        "annot_dir": str(ANNOT_DIR),
        "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "mode": "pilot" if args.pilot else ("subset" if args.cells else "full"),
        "n_cells_attempted": n_attempted,
        "n_cells_full": n_full,
        "feature_columns": list(FEATURE_COLUMNS),
        "params": {
            "charge_window_min": CHARGE_WINDOW_MIN,
            "discharge_window_min": DISCHARGE_WINDOW_MIN,
            "fit_config": asdict(FitConfig()),
        },
    }
    (snapshot_dir / "manifest.json").write_text(json.dumps(manifest, indent=2))

    # Stdout report.
    print()
    print(f"snapshot     = {snapshot_dir}")
    print(f"attempted    = {n_attempted}")
    print(f"full rows    = {n_full} ({n_full / max(n_attempted, 1):.1%})")
    print()
    print("per-feature null counts:")
    for col in FEATURE_COLUMNS:
        n_null = int(feat_df.filter(
            pl.col(col).is_null() | pl.col(col).is_nan()
        ).height)
        print(f"  {col:<36s}  null = {n_null}/{n_attempted}")
    print()
    print("DOP convergence per segment (≥1 peak):")
    for seg_key, cycle, direction, _ in _SEGMENTS:
        direction_abbrev = "chg" if direction == "charge" else "dis"
        col = f"dop_ok_c{cycle}_{direction_abbrev}"
        n_ok = int(stat_df.filter(pl.col(col)).height)
        print(f"  {col:<24s}  {n_ok}/{n_attempted} ({n_ok / max(n_attempted, 1):.0%})")
    print()
    print("n_features_success distribution:")
    print(stat_df.group_by("n_features_success").len()
          .sort("n_features_success"))
    print()
    print("fit_time_s_total per cell (4 fits each):")
    timing_stats = stat_df.select([
        pl.col("fit_time_s_total").min().alias("min"),
        pl.col("fit_time_s_total").median().alias("median"),
        pl.col("fit_time_s_total").mean().alias("mean"),
        pl.col("fit_time_s_total").max().alias("max"),
        pl.col("fit_time_s_total").sum().alias("sum"),
    ])
    print(timing_stats)

    # Per-cohort feature summary — sanity check that the DOP peak θ
    # actually varies across cohorts.
    print()
    print("per-cohort feature summary (mean ± std, n):")
    for col in FEATURE_COLUMNS:
        sub = feat_df.filter(pl.col(col).is_not_nan() & pl.col(col).is_not_null())
        if sub.is_empty():
            print(f"  {col:<36s}  (all NaN)")
            continue
        grp = sub.group_by("cohort").agg([
            pl.col(col).mean().alias("mean"),
            pl.col(col).std().alias("std"),
            pl.col(col).count().alias("n"),
        ]).sort("cohort")
        print(f"  {col}:")
        for row in grp.iter_rows(named=True):
            mean_str = f"{row['mean']:+.3f}" if row['mean'] is not None else "  NaN"
            std_str = f"{row['std']:.3f}" if row['std'] is not None else " NaN"
            print(f"    {row['cohort']:5s}  {mean_str} ± {std_str}   n={row['n']}")

    return 0


if __name__ == "__main__":
    sys.exit(main())

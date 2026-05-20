"""Outlier detector — full-cohort dry-run (Pattern A).

Iterates every annotation JSON, runs ``detect_outliers`` on its
retention curve, and emits:

  outlier_sidecar.json                  per-cell {n_outliers, outliers: [...]}
                                        — GIT-TRACKED; consumed by labels.py
  reports/outlier_report.csv            long-format: one row per (cell, outlier-cycle)
  reports/outlier_summary.txt           aggregate stats + per-cell rollup
  plots/outliers/with_outliers/*.png    every cell with ≥1 flagged outlier
  plots/outliers/known_glitches_audit/  the Pattern A truth set
                                        (0MC20-251126-R001, AR-3422, AR3941, AR4084)
  plots/outliers/no_outliers_audit/     random sample of clean cells (FN check)

No labels are altered. Purely diagnostic.

Author: Mark Liao (Sheng-Lun Liao)
"""
from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import polars as pl

# Make the parent project's _common importable (we live one level deeper now).
HERE = Path(__file__).resolve().parent
ROOT = HERE.parent  # .../ml_label_preprocess
sys.path.insert(0, str(ROOT))
from _common import (  # noqa: E402
    ANNOT_DIR,
    _cohort,
    iter_annotations,
    iter_regulars,
)

from curation.jump_detection import compute_retentions  # noqa: E402
from curation.outlier_detector import (  # noqa: E402
    OutlierParams,
    OutlierReport,
    detect_outliers,
)


REPORTS_DIR = HERE / "reports"
PLOTS_DIR = HERE / "plots" / "outliers"
SIDECAR_PATH = HERE / "outlier_sidecar.json"

# Cells the user identified in AUDIT_FINDINGS.md as Pattern A truth set.
# Forced into known_glitches_audit/ regardless of detector output, so the
# audit can compare the flagged cycles to the user's notes.
KNOWN_GLITCH_CELLS = (
    "0MC20-251126-R001",
    "AR-3422",
    "AR3941",
    "AR4084",
)

FN_AUDIT_SAMPLE_SIZE = 20
FN_AUDIT_MIN_REGULARS = 100
FN_AUDIT_SEED = 20260520


# ---------------- helpers duplicated from jump_detection ----------------
# These are lightweight and the explore agent confirmed duplication is
# preferred over cross-import when the two investigations may evolve
# independently.

def _regime_boundaries(annot: dict) -> list[int]:
    """Cumulative-sum cycle boundaries from regular_rate_regimes.

    For an annotation with regimes of length [267, 205], returns [267].
    Final regime's terminus is NOT included (it's just end-of-life).
    """
    regimes = annot.get("regular_rate_regimes") or []
    if len(regimes) < 2:
        return []
    out: list[int] = []
    acc = 0
    for r in regimes[:-1]:
        n = r.get("n_regular_cd")
        if n is None:
            continue
        acc += int(n)
        out.append(acc)
    return out


def _max_rate_delta_pct(annot: dict) -> float | None:
    regimes = annot.get("regular_rate_regimes") or []
    bs = [r.get("baseline_i_a") for r in regimes
          if r.get("baseline_i_a") is not None]
    if len(bs) < 2:
        return None
    lo, hi = min(bs), max(bs)
    if lo <= 0:
        return None
    return (hi - lo) / lo * 100.0


def _is_near_boundary(cycle: int, boundaries: list[int], tol: int = 2) -> bool:
    return any(abs(cycle - b) <= tol for b in boundaries)


# ---------------- per-cell processing ----------------

def _row_dict_from_outlier(
    cell: str, cohort: str, consistency: str, protocol: str,
    n_regulars: int, n_regimes: int, max_rate_delta: float | None,
    boundaries: list[int], n_outliers: int,
    rpt: OutlierReport | None,
) -> dict:
    """Build one CSV row. ``rpt=None`` means 'cell has zero outliers'."""
    row = {
        "cell_name": cell,
        "cohort": cohort,
        "cycling_consistency": consistency,
        "protocol_pattern": protocol,
        "n_regulars": n_regulars,
        "n_regimes": n_regimes,
        "max_regime_rate_delta_pct": max_rate_delta,
        "regime_boundary_cycles": ",".join(str(b) for b in boundaries) or None,
        "n_outliers": n_outliers,
        "outlier_cycle": None,
        "retention": None,
        "predicted": None,
        "residual": None,
        "z_score": None,
        "pre_post_disagreement": None,
        "pre_n_points": None,
        "post_n_points": None,
        "near_regime_boundary": None,
    }
    if rpt is None:
        return row
    row.update({
        "outlier_cycle": rpt.cycle,
        "retention": rpt.retention,
        "predicted": rpt.predicted,
        "residual": rpt.residual,
        "z_score": rpt.z_score,
        "pre_post_disagreement": rpt.pre_post_disagreement,
        "pre_n_points": rpt.pre_n_points,
        "post_n_points": rpt.post_n_points,
        "near_regime_boundary": _is_near_boundary(rpt.cycle, boundaries),
    })
    return row


# ---------------- plotting ----------------

def _plot_cell(
    cell: str, cycles: list[int], retentions: list[float],
    outliers: list[OutlierReport], boundaries: list[int],
    consistency: str, n_regulars: int, out_path: Path,
) -> None:
    """One PNG per cell, mirrors jump_detection's visual style.

    Blue scatter + 0.85 line + dotted regime boundaries; outlier cycles
    overlaid as red X markers + cycle-number labels.
    """
    fig, ax = plt.subplots(figsize=(9, 4.5))
    ax.scatter(cycles, retentions, s=8, color="tab:blue", alpha=0.6,
               label="retention")
    ax.axhline(0.85, color="grey", linestyle="--", linewidth=0.8,
               label="0.85 fade threshold")

    for b in boundaries:
        ax.axvline(b, color="grey", linestyle=":", linewidth=0.8, alpha=0.6)

    if outliers:
        xs = [r.cycle for r in outliers]
        ys = [r.retention for r in outliers]
        ax.scatter(xs, ys, s=80, marker="x", color="red", linewidths=1.6,
                   label="flagged outlier", zorder=5)
        # Label each flagged cycle so the audit can identify it without
        # consulting the CSV.
        for r in outliers:
            ax.annotate(
                f"{r.cycle}",
                xy=(r.cycle, r.retention),
                xytext=(4, 6), textcoords="offset points",
                fontsize=7, color="red",
            )

    title = (f"{cell} | consistency={consistency} | n_reg={n_regulars} | "
             f"n_outliers={len(outliers)}")
    ax.set_xlabel("regular_cycle")
    ax.set_ylabel("discharge retention")
    ax.set_title(title, fontsize=10)
    ax.set_ylim(min(0.3, min(retentions) - 0.05),
                max(1.05, max(retentions) + 0.05))
    ax.legend(loc="lower left", fontsize=8)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=120)
    plt.close(fig)


# ---------------- main ----------------

def main(params: OutlierParams) -> int:
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    for sub in ("with_outliers", "known_glitches_audit", "no_outliers_audit"):
        (PLOTS_DIR / sub).mkdir(parents=True, exist_ok=True)

    rows: list[dict] = []
    sidecar: dict[str, dict] = {}
    plot_data: dict[str, tuple[list[int], list[float], list[OutlierReport],
                                list[int], str, int]] = {}

    n_total = 0
    for path, d in iter_annotations():
        n_total += 1
        cell = d.get("cell_name", path.stem.replace(".annotations", ""))
        cohort = _cohort(cell)
        consistency = d.get("cycling_consistency", "no_regular")
        protocol = d.get("protocol_pattern", "other")
        regimes = d.get("regular_rate_regimes") or []
        n_regimes = len(regimes)
        max_rate_delta = _max_rate_delta_pct(d)
        boundaries = _regime_boundaries(d)

        regulars = iter_regulars(d)
        n_regulars = int(regulars[-1]["regular_cycle"]) if regulars else 0

        cycles, retentions = compute_retentions(regulars)
        outliers: list[OutlierReport]
        if not cycles:
            outliers = []
        else:
            outliers = detect_outliers(cycles, retentions, boundaries, params)

        # CSV: one row per outlier, or one sentinel row per clean cell.
        if outliers:
            for rpt in outliers:
                rows.append(_row_dict_from_outlier(
                    cell, cohort, consistency, protocol, n_regulars,
                    n_regimes, max_rate_delta, boundaries,
                    len(outliers), rpt,
                ))
        else:
            rows.append(_row_dict_from_outlier(
                cell, cohort, consistency, protocol, n_regulars,
                n_regimes, max_rate_delta, boundaries, 0, None,
            ))

        # Sidecar JSON
        sidecar[cell] = {
            "n_outliers": len(outliers),
            "outliers": [
                {
                    "cycle": r.cycle,
                    "list_index": r.list_index,
                    "retention": r.retention,
                    "predicted": r.predicted,
                    "residual": r.residual,
                    "z_score": r.z_score,
                    "pre_post_disagreement": r.pre_post_disagreement,
                    "pre_n_points": r.pre_n_points,
                    "post_n_points": r.post_n_points,
                }
                for r in outliers
            ],
        }

        plot_data[cell] = (cycles, retentions, outliers, boundaries,
                           consistency, n_regulars)

    if not rows:
        print("ERROR: no cells processed", file=sys.stderr)
        return 1

    df = pl.DataFrame(rows).sort(
        ["cohort", "cell_name", "outlier_cycle"], nulls_last=True)
    csv_path = REPORTS_DIR / "outlier_report.csv"
    df.write_csv(csv_path)
    print(f"wrote {csv_path}  ({df.height} rows)")

    json_path = SIDECAR_PATH
    json_path.write_text(json.dumps(sidecar, indent=2))
    print(f"wrote {json_path}  ({len(sidecar)} cells)")

    # ---------------- summary ----------------
    summary_lines: list[str] = []
    summary_lines.append("Outlier-detection dry-run summary")
    summary_lines.append(f"  annot_dir         = {ANNOT_DIR}")
    summary_lines.append(f"  cells scanned     = {n_total}")
    summary_lines.append(f"  window_half       = {params.window_half}")
    summary_lines.append(f"  n_trim            = {params.n_trim}")
    summary_lines.append(f"  boundary_skip     = {params.boundary_skip}")
    summary_lines.append(f"  skip_last_n       = {params.skip_last_n}")
    summary_lines.append(f"  discontinuity_max = {params.discontinuity_max}")
    summary_lines.append(f"  mad_multiplier    = {params.mad_multiplier}")
    summary_lines.append(f"  sigma_floor       = {params.sigma_floor}")
    summary_lines.append("")

    n_with = sum(1 for c in sidecar.values() if c["n_outliers"] > 0)
    n_total_flags = sum(c["n_outliers"] for c in sidecar.values())
    summary_lines.append(f"Cells with ≥1 flagged outlier: {n_with}/{n_total}")
    summary_lines.append(f"Total flagged cycles: {n_total_flags}")
    summary_lines.append("")

    n_outliers_hist: dict[int, int] = {}
    for c in sidecar.values():
        n_outliers_hist[c["n_outliers"]] = (
            n_outliers_hist.get(c["n_outliers"], 0) + 1)
    summary_lines.append("Per-cell n_outliers histogram:")
    for k in sorted(n_outliers_hist):
        summary_lines.append(f"  n={k:>3d}  cells={n_outliers_hist[k]:>4d}")
    summary_lines.append("")

    flagged_cells = sorted(c for c, v in sidecar.items() if v["n_outliers"] > 0)
    if flagged_cells:
        summary_lines.append(f"Flagged cells ({len(flagged_cells)}):")
        for cell in flagged_cells:
            v = sidecar[cell]
            cyc_str = ",".join(str(o["cycle"]) for o in v["outliers"])
            summary_lines.append(
                f"  {cell:<22s}  n={v['n_outliers']:>2d}  cycles=[{cyc_str}]"
            )
    summary_lines.append("")

    # Truth-set status
    summary_lines.append("Known Pattern A cells (truth set):")
    for cell in KNOWN_GLITCH_CELLS:
        v = sidecar.get(cell)
        if v is None:
            summary_lines.append(f"  {cell:<22s}  NOT FOUND IN COHORT")
        else:
            cyc_str = ",".join(str(o["cycle"]) for o in v["outliers"])
            summary_lines.append(
                f"  {cell:<22s}  n={v['n_outliers']:>2d}  cycles=[{cyc_str}]"
            )

    summary_text = "\n".join(summary_lines) + "\n"
    summary_path = REPORTS_DIR / "outlier_summary.txt"
    summary_path.write_text(summary_text)
    print(summary_text)
    print(f"wrote {summary_path}")

    # ---------------- plots ----------------
    print()
    print("Generating plots...")

    # with_outliers/
    for cell in flagged_cells:
        cycles, retentions, outliers, boundaries, consistency, n_regulars = (
            plot_data[cell])
        _plot_cell(cell, cycles, retentions, outliers, boundaries,
                   consistency, n_regulars,
                   PLOTS_DIR / "with_outliers" / f"{cell}.png")
    print(f"  with_outliers:        {len(flagged_cells)} plots")

    # known_glitches_audit/ — forced inclusion regardless of n_outliers
    n_known = 0
    for cell in KNOWN_GLITCH_CELLS:
        if cell not in plot_data:
            continue
        cycles, retentions, outliers, boundaries, consistency, n_regulars = (
            plot_data[cell])
        _plot_cell(cell, cycles, retentions, outliers, boundaries,
                   consistency, n_regulars,
                   PLOTS_DIR / "known_glitches_audit" / f"{cell}.png")
        n_known += 1
    print(f"  known_glitches_audit: {n_known} plots "
          f"(truth set; flagged-or-not)")

    # no_outliers_audit/ — random sample of clean cells with enough cycles
    clean_candidates = [c for c, v in sidecar.items()
                        if v["n_outliers"] == 0
                        and plot_data[c][5] >= FN_AUDIT_MIN_REGULARS]
    rng = random.Random(FN_AUDIT_SEED)
    fn_sample = rng.sample(clean_candidates,
                            min(FN_AUDIT_SAMPLE_SIZE, len(clean_candidates)))
    for cell in fn_sample:
        cycles, retentions, outliers, boundaries, consistency, n_regulars = (
            plot_data[cell])
        _plot_cell(cell, cycles, retentions, outliers, boundaries,
                   consistency, n_regulars,
                   PLOTS_DIR / "no_outliers_audit" / f"{cell}.png")
    print(f"  no_outliers_audit:    {len(fn_sample)} plots "
          f"(seeded {FN_AUDIT_SEED}, min n_regulars={FN_AUDIT_MIN_REGULARS})")

    return 0


def _build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__)
    d = OutlierParams()
    p.add_argument("--window-half", type=int, default=d.window_half)
    p.add_argument("--min-pre-len", type=int, default=d.min_pre_len)
    p.add_argument("--min-post-len", type=int, default=d.min_post_len)
    p.add_argument("--boundary-skip", type=int, default=d.boundary_skip)
    p.add_argument("--skip-last-n", type=int, default=d.skip_last_n)
    p.add_argument("--discontinuity-max", type=float, default=d.discontinuity_max)
    p.add_argument("--mad-multiplier", type=float, default=d.mad_multiplier)
    p.add_argument("--n-trim", type=int, default=d.n_trim)
    p.add_argument("--sigma-floor", type=float, default=d.sigma_floor)
    return p


if __name__ == "__main__":
    args = _build_argparser().parse_args()
    params = OutlierParams(
        window_half=args.window_half,
        min_pre_len=args.min_pre_len,
        min_post_len=args.min_post_len,
        boundary_skip=args.boundary_skip,
        skip_last_n=args.skip_last_n,
        discontinuity_max=args.discontinuity_max,
        mad_multiplier=args.mad_multiplier,
        n_trim=args.n_trim,
        sigma_floor=args.sigma_floor,
    )
    sys.exit(main(params))

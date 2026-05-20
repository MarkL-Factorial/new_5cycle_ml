"""Sustained-step detection — strict params, outlier-masked, rate_changed-excluded.

Pipeline (per cell):

  1. Skip cells with ``cycling_consistency == "rate_changed"``. These are
     Pattern B (rate-change step) per AUDIT_FINDINGS and are dropped
     wholesale from the ML cohort; no sustained-step analysis needed.
  2. Compute retentions from regular cycles (same baseline as labels.py).
  3. Mask outlier cycles flagged by ``curation/outlier_sidecar.json``
     (these are Pattern A measurement glitches; we don't want them
     contaminating the OLS pre-fit).
  4. Run ``curation.jump_detection.detect_jumps`` on the masked series
     with **stricter** defaults than the original dry-run
     (``bump_min = persist_min = 0.05`` vs the original 0.03).
  5. For every cell that has at least one sustained candidate after
     masking, emit an annotated plot for manual review.

Output:

  reports/sustained_step_report.csv  one row per (cell, sustained candidate)
  reports/sustained_step_summary.txt histograms + per-cell rollup
  plots/sustained_step/*.png         annotated plots for manual eyeballing

Diagnostic / review aid — feeds curation/validation.py.

Author: Mark Liao (Sheng-Lun Liao)
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import polars as pl

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent  # .../ml_label_preprocess
sys.path.insert(0, str(ROOT))
from _common import (  # noqa: E402
    ANNOT_DIR,
    _cohort,
    iter_annotations,
    iter_regulars,
)
from curation.jump_detection import (  # noqa: E402
    DetectorParams,
    JumpReport,
    compute_retentions,
    detect_jumps,
)


REPORTS_DIR = HERE / "reports"
PLOTS_DIR = HERE / "plots" / "sustained_step"
SIDECAR_PATH = HERE / "outlier_sidecar.json"

EXCLUDED_CONSISTENCIES = {"rate_changed"}

# Stricter defaults than jump_detection (which used 0.03 / 0.03).
DEFAULT_BUMP_MIN = 0.05
DEFAULT_PERSIST_MIN = 0.05


# ---------------- helpers duplicated from jump_detection ----------------

def _regime_boundaries(annot: dict) -> list[int]:
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


# ---------------- plotting ----------------

def _plot_cell(
    cell: str,
    cycles_orig: list[int], rets_orig: list[float],
    outlier_cycles: set[int],
    reports: list[JumpReport],
    boundaries: list[int],
    consistency: str,
    n_regulars: int,
    max_rate_delta: float | None,
    params: DetectorParams,
    out_path: Path,
) -> None:
    """Per-cell PNG for manual review.

    Visual conventions:
      blue scatter         — retention curve (all cycles)
      red ✕                — outlier cycle (masked from detection)
      grey dotted vertical — regime boundary
      red vertical         — sustained candidate cycle (with text annotation)
      orange vertical      — transient candidate cycle
      grey vertical        — edge_skip candidate cycle
      solid coloured line  — pre-window OLS fit
      dashed coloured line — pre-trend extrapolated into post window
    """
    fig, ax = plt.subplots(figsize=(10, 5))

    # Separate outliers from kept cycles for color coding
    keep_x: list[int] = []
    keep_y: list[float] = []
    out_x: list[int] = []
    out_y: list[float] = []
    for c, r in zip(cycles_orig, rets_orig):
        if c in outlier_cycles:
            out_x.append(c)
            out_y.append(r)
        else:
            keep_x.append(c)
            keep_y.append(r)

    ax.scatter(keep_x, keep_y, s=8, color="tab:blue", alpha=0.6,
               label="retention")
    if out_x:
        ax.scatter(out_x, out_y, s=70, marker="x", color="red",
                   linewidths=1.4, label="masked outlier (Pattern A)",
                   zorder=4)

    ax.axhline(0.85, color="grey", linestyle="--", linewidth=0.8,
               label="0.85 fade threshold")

    for b in boundaries:
        ax.axvline(b, color="grey", linestyle=":", linewidth=0.8, alpha=0.6)

    min_keep_cycle = min(keep_x) if keep_x else 0
    for r in reports:
        color = {
            "sustained": "tab:red",
            "transient": "tab:orange",
            "edge_skip":  "tab:gray",
        }.get(r.classification, "black")
        ax.axvline(r.jump_cycle_ordinal, color=color, linewidth=1.2, alpha=0.85)
        if r.pre_n_points > 0 and r.classification != "edge_skip":
            xs_pre = list(range(
                max(min_keep_cycle, r.jump_cycle_ordinal - r.pre_n_points),
                r.jump_cycle_ordinal,
            ))
            ys_pre = [r.pre_slope * x + r.pre_intercept for x in xs_pre]
            ax.plot(xs_pre, ys_pre, color=color, linewidth=1.0)
            xs_ext = list(range(
                r.jump_cycle_ordinal,
                r.jump_cycle_ordinal + r.post_n_points,
            ))
            ys_ext = [r.pre_slope * x + r.pre_intercept for x in xs_ext]
            ax.plot(xs_ext, ys_ext, color=color, linewidth=1.0, linestyle="--")

        # Annotations for SUSTAINED candidates only (the ones the
        # reviewer needs to judge). Anchor near the pre-fit endpoint.
        if r.classification == "sustained":
            anchor_y = r.pre_slope * r.jump_cycle_ordinal + r.pre_intercept
            ax.annotate(
                f"cycle={r.jump_cycle_ordinal}\n"
                f"Δ={r.jump_magnitude:+.3f}\n"
                f"persist={r.persistence_score:+.3f}",
                xy=(r.jump_cycle_ordinal, anchor_y),
                xytext=(8, 14), textcoords="offset points",
                fontsize=8, color="tab:red",
                bbox=dict(boxstyle="round,pad=0.3", facecolor="white",
                          edgecolor="tab:red", alpha=0.85),
                arrowprops=dict(arrowstyle="-", color="tab:red",
                                linewidth=0.6, alpha=0.7),
            )

    n_sus = sum(1 for r in reports if r.classification == "sustained")
    n_trn = sum(1 for r in reports if r.classification == "transient")
    rate_str = (f"max_rate_delta={max_rate_delta:.1f}%"
                if max_rate_delta is not None else "single regime")
    title = (f"{cell} | consistency={consistency} | n_reg={n_regulars} | "
             f"sustained={n_sus} transient={n_trn} | {rate_str}\n"
             f"params: bump_min={params.bump_min}  "
             f"persist_min={params.persist_min}  "
             f"pre/post window={params.pre_window}/{params.post_window}")
    ax.set_xlabel("regular_cycle")
    ax.set_ylabel("discharge retention")
    ax.set_title(title, fontsize=9)
    all_y = rets_orig
    ax.set_ylim(min(0.3, min(all_y) - 0.05),
                max(1.05, max(all_y) + 0.05))
    ax.legend(loc="lower left", fontsize=8)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=120)
    plt.close(fig)


# ---------------- main ----------------

def main(params: DetectorParams) -> int:
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    PLOTS_DIR.mkdir(parents=True, exist_ok=True)

    if not SIDECAR_PATH.exists():
        print(
            f"ERROR: outlier sidecar not found at {SIDECAR_PATH}.\n"
            f"Run: python -m curation.outlier_detection",
            file=sys.stderr,
        )
        return 1
    sidecar = json.loads(SIDECAR_PATH.read_text())

    rows: list[dict] = []
    n_total = 0
    n_excluded_rate_changed = 0
    n_excluded_no_data = 0
    n_in_scope = 0
    n_with_sustained = 0

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

        if consistency in EXCLUDED_CONSISTENCIES:
            n_excluded_rate_changed += 1
            continue

        regulars = iter_regulars(d)
        n_regulars = int(regulars[-1]["regular_cycle"]) if regulars else 0

        cycles_orig, rets_orig = compute_retentions(regulars)
        if not cycles_orig:
            n_excluded_no_data += 1
            continue
        n_in_scope += 1

        outlier_cycles = {
            int(o["cycle"]) for o in sidecar.get(cell, {}).get("outliers", [])
        }
        # Mask: drop outlier cycles before running jump detection
        masked = [(c, r) for c, r in zip(cycles_orig, rets_orig)
                  if c not in outlier_cycles]
        mcycles = [c for c, _ in masked]
        mrets = [r for _, r in masked]

        reports = detect_jumps(mcycles, mrets, params)
        sustained_reports = [r for r in reports if r.classification == "sustained"]

        if not sustained_reports:
            continue
        n_with_sustained += 1

        for r in sustained_reports:
            rows.append({
                "cell_name": cell,
                "cohort": cohort,
                "cycling_consistency": consistency,
                "protocol_pattern": protocol,
                "n_regulars": n_regulars,
                "n_regimes": n_regimes,
                "max_regime_rate_delta_pct": max_rate_delta,
                "regime_boundary_cycles": (
                    ",".join(str(b) for b in boundaries) or None),
                "n_outliers_masked": len(outlier_cycles),
                "sustained_cycle": r.jump_cycle_ordinal,
                "jump_magnitude": r.jump_magnitude,
                "jump_direction": r.jump_direction,
                "pre_slope": r.pre_slope,
                "pre_n_points": r.pre_n_points,
                "post_n_points": r.post_n_points,
                "persistence_score": r.persistence_score,
                "near_regime_boundary": _is_near_boundary(
                    r.jump_cycle_ordinal, boundaries),
            })

        out_path = PLOTS_DIR / f"{cell}.png"
        _plot_cell(
            cell, cycles_orig, rets_orig, outlier_cycles, reports,
            boundaries, consistency, n_regulars, max_rate_delta,
            params, out_path,
        )

    if not rows:
        print("No cells flagged as sustained after masking.")

    df = pl.DataFrame(rows).sort(
        ["cohort", "cell_name", "sustained_cycle"], nulls_last=True
    ) if rows else None
    csv_path = REPORTS_DIR / "sustained_step_report.csv"
    if df is not None:
        df.write_csv(csv_path)
        print(f"wrote {csv_path}  ({df.height} rows)")

    # ---------------- summary ----------------
    summary_lines: list[str] = []
    summary_lines.append("Sustained-step detection (strict, masked) summary")
    summary_lines.append(f"  annot_dir              = {ANNOT_DIR}")
    summary_lines.append(f"  cells scanned          = {n_total}")
    summary_lines.append(f"  excluded rate_changed  = {n_excluded_rate_changed}")
    summary_lines.append(f"  excluded no_data       = {n_excluded_no_data}")
    summary_lines.append(f"  in scope               = {n_in_scope}")
    summary_lines.append(f"  with sustained flag    = {n_with_sustained}")
    summary_lines.append("")
    summary_lines.append("Detector params (stricter than jump_detection defaults):")
    summary_lines.append(f"  bump_min      = {params.bump_min}   (jump_detection default 0.03)")
    summary_lines.append(f"  persist_min   = {params.persist_min}   (jump_detection default 0.03)")
    summary_lines.append(f"  pre_window    = {params.pre_window}")
    summary_lines.append(f"  post_window   = {params.post_window}")
    summary_lines.append(f"  min_pre_len   = {params.min_pre_len}")
    summary_lines.append(f"  min_post_len  = {params.min_post_len}")
    summary_lines.append("")

    if rows:
        # Per-cell rollup, sorted by earliest sustained cycle
        per_cell: dict[str, list[dict]] = {}
        for row in rows:
            per_cell.setdefault(row["cell_name"], []).append(row)
        ordered_cells = sorted(
            per_cell.keys(),
            key=lambda c: min(r["sustained_cycle"] for r in per_cell[c]),
        )
        summary_lines.append(
            f"Cells with sustained flag ({len(ordered_cells)}), "
            f"sorted by earliest sustained cycle:"
        )
        for cell in ordered_cells:
            cell_rows = per_cell[cell]
            primary = min(cell_rows, key=lambda r: r["sustained_cycle"])
            n_extra = len(cell_rows) - 1
            extra_str = f"  (+{n_extra} more)" if n_extra else ""
            near = primary["near_regime_boundary"]
            n_out = primary["n_outliers_masked"]
            summary_lines.append(
                f"  {cell:<22s}  cycle={primary['sustained_cycle']:>4d}  "
                f"dir={primary['jump_direction']:<4s}  "
                f"Δ={primary['jump_magnitude']:+.4f}  "
                f"persist={primary['persistence_score']:+.4f}  "
                f"near_boundary={near}  outliers_masked={n_out}{extra_str}"
            )

    text = "\n".join(summary_lines) + "\n"
    summary_path = REPORTS_DIR / "sustained_step_summary.txt"
    summary_path.write_text(text)
    print(text)
    print(f"wrote {summary_path}")
    print(f"wrote {n_with_sustained} plots to {PLOTS_DIR}")
    return 0


def _build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__)
    d = DetectorParams()
    p.add_argument("--bump-min", type=float, default=DEFAULT_BUMP_MIN)
    p.add_argument("--persist-min", type=float, default=DEFAULT_PERSIST_MIN)
    p.add_argument("--pre-window", type=int, default=d.pre_window)
    p.add_argument("--post-window", type=int, default=d.post_window)
    p.add_argument("--min-pre-len", type=int, default=d.min_pre_len)
    p.add_argument("--min-post-len", type=int, default=d.min_post_len)
    return p


if __name__ == "__main__":
    args = _build_argparser().parse_args()
    params = DetectorParams(
        bump_min=args.bump_min,
        persist_min=args.persist_min,
        pre_window=args.pre_window,
        post_window=args.post_window,
        min_pre_len=args.min_pre_len,
        min_post_len=args.min_post_len,
    )
    sys.exit(main(params))

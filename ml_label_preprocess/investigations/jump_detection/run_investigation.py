"""Capacity-jump detector — full-cohort dry-run.

Iterates every annotation JSON, runs ``detect_jumps`` on its retention
curve, and emits:

  out/jump_detection_report.csv     long-format census (one row per cell+candidate)
  out/jump_detection_summary.txt    aggregate stats
  out/plots/sustained/*.png         all 'sustained' cells
  out/plots/transient/*.png         all 'transient' cells
  out/plots/multi_regime_audit/*.png  the multi-regime single_rate cohort
  out/plots/false_negative_audit/*.png  random sample of 'none' cells

No production labels are altered. This is purely diagnostic.

Author: Mark Liao (Sheng-Lun Liao)
"""
from __future__ import annotations

import argparse
import random
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import polars as pl

# Make the parent project importable without modifying it.
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent.parent))  # .../ml_label_preprocess
from _common import (  # noqa: E402
    ANNOT_DIR,
    _cohort,
    iter_annotations,
    iter_regulars,
)

# Algorithm core lives in curation/ (graduated from this folder); the
# runner here is just the diagnostic harness around it.
from curation.jump_detection import (  # noqa: E402
    DetectorParams,
    JumpReport,
    compute_retentions,
    detect_jumps,
)


OUT_DIR = HERE / "out"
PLOTS_DIR = OUT_DIR / "plots"
FN_AUDIT_SAMPLE_SIZE = 20
FN_AUDIT_MIN_REGULARS = 100
FN_AUDIT_SEED = 20260519


# ---------------- per-cell processing ----------------

def _regime_boundaries(annot: dict) -> list[int]:
    """Cumulative-sum cycle boundaries from regular_rate_regimes.

    For an annotation with regimes of length [267, 205], this returns
    [267] (the index where regime 0 ends / regime 1 begins). The final
    regime's terminus is NOT included since it's just the end-of-life.
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


def _row_dict_from_report(
    cell: str, cohort: str, consistency: str, protocol: str,
    n_regulars: int, n_regimes: int, max_rate_delta: float | None,
    boundaries: list[int], rpt: JumpReport | None,
) -> dict:
    """Build one CSV row. ``rpt=None`` means 'cell has zero candidates'."""
    row = {
        "cell_name": cell,
        "cohort": cohort,
        "cycling_consistency": consistency,
        "protocol_pattern": protocol,
        "n_regulars": n_regulars,
        "n_regimes": n_regimes,
        "max_regime_rate_delta_pct": max_rate_delta,
        "regime_boundary_cycles": ",".join(str(b) for b in boundaries) or None,
        "jump_cycle_ordinal": None,
        "jump_magnitude": None,
        "jump_direction": None,
        "pre_slope": None,
        "pre_n_points": None,
        "post_n_points": None,
        "persistence_score": None,
        "classification": "none",
        "jump_near_regime_boundary": None,
    }
    if rpt is None:
        return row
    row.update({
        "jump_cycle_ordinal": rpt.jump_cycle_ordinal,
        "jump_magnitude": rpt.jump_magnitude,
        "jump_direction": rpt.jump_direction,
        "pre_slope": rpt.pre_slope,
        "pre_n_points": rpt.pre_n_points,
        "post_n_points": rpt.post_n_points,
        "persistence_score": rpt.persistence_score,
        "classification": rpt.classification,
        "jump_near_regime_boundary": _is_near_boundary(
            rpt.jump_cycle_ordinal, boundaries),
    })
    return row


# ---------------- plotting ----------------

def _plot_cell(
    cell: str, cycles: list[int], retentions: list[float],
    reports: list[JumpReport], boundaries: list[int],
    consistency: str, n_regulars: int, out_path: Path,
) -> None:
    """One PNG per cell. Mirrors the colleague_comparison plotting style."""
    fig, ax = plt.subplots(figsize=(9, 4.5))
    ax.scatter(cycles, retentions, s=8, color="tab:blue", alpha=0.6,
               label="retention")
    ax.axhline(0.85, color="grey", linestyle="--", linewidth=0.8,
               label="0.85 fade threshold")

    # Regime boundaries from the annotation
    for b in boundaries:
        ax.axvline(b, color="grey", linestyle=":", linewidth=0.8, alpha=0.6)

    # Each candidate jump
    for r in reports:
        color = {
            "sustained": "tab:red",
            "transient": "tab:orange",
            "edge_skip":  "tab:gray",
        }.get(r.classification, "black")
        ax.axvline(r.jump_cycle_ordinal, color=color, linewidth=1.2, alpha=0.85)
        if r.pre_n_points > 0 and r.classification != "edge_skip":
            # Pre-window fit line (solid) over its actual range
            xs_pre = list(range(
                max(min(cycles), r.jump_cycle_ordinal - r.pre_n_points),
                r.jump_cycle_ordinal,
            ))
            ys_pre = [r.pre_slope * x + r.pre_intercept for x in xs_pre]
            ax.plot(xs_pre, ys_pre, color=color, linewidth=1.0)
            # Extrapolation (dashed) over the post window
            xs_ext = list(range(
                r.jump_cycle_ordinal,
                r.jump_cycle_ordinal + r.post_n_points,
            ))
            ys_ext = [r.pre_slope * x + r.pre_intercept for x in xs_ext]
            ax.plot(xs_ext, ys_ext, color=color, linewidth=1.0, linestyle="--")

    primary = next((r for r in reports if r.classification == "sustained"), None)
    if primary is None:
        primary = next((r for r in reports if r.classification == "transient"), None)
    cls = primary.classification if primary else "none"
    direction = primary.jump_direction if primary else "-"
    title = (f"{cell} | consistency={consistency} | n_reg={n_regulars} | "
             f"class={cls} (dir={direction}) | candidates={len(reports)}")
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

def main(params: DetectorParams) -> int:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    for sub in ("sustained", "transient", "multi_regime_audit",
                "false_negative_audit"):
        (PLOTS_DIR / sub).mkdir(parents=True, exist_ok=True)

    rows: list[dict] = []
    # cell-level state: stash per-cell info needed for plotting
    plot_data: dict[str, tuple[list[int], list[float], list[JumpReport],
                                list[int], str, int]] = {}
    multi_regime_single_rate_cells: list[str] = []

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

        if consistency == "single_rate" and n_regimes > 1:
            multi_regime_single_rate_cells.append(cell)

        regulars = iter_regulars(d)
        n_regulars = int(regulars[-1]["regular_cycle"]) if regulars else 0

        cycles, retentions = compute_retentions(regulars)
        reports: list[JumpReport]
        if not cycles:
            reports = []
        else:
            reports = detect_jumps(cycles, retentions, params)

        if reports:
            for rpt in reports:
                rows.append(_row_dict_from_report(
                    cell, cohort, consistency, protocol, n_regulars,
                    n_regimes, max_rate_delta, boundaries, rpt,
                ))
        else:
            rows.append(_row_dict_from_report(
                cell, cohort, consistency, protocol, n_regulars,
                n_regimes, max_rate_delta, boundaries, None,
            ))

        plot_data[cell] = (cycles, retentions, reports, boundaries,
                           consistency, n_regulars)

    if not rows:
        print("ERROR: no cells processed", file=sys.stderr)
        return 1

    df = pl.DataFrame(rows).sort(["cohort", "cell_name", "jump_cycle_ordinal"],
                                  nulls_last=True)
    csv_path = OUT_DIR / "jump_detection_report.csv"
    df.write_csv(csv_path)
    print(f"wrote {csv_path}  ({df.height} rows)")

    # ---------------- per-cell classification (rollup) ----------------
    # A cell's classification = strongest among its candidates.
    # Order: sustained > transient > edge_skip > none.
    rank = {"sustained": 3, "transient": 2, "edge_skip": 1, "none": 0}
    cell_class: dict[str, str] = {}
    for cell, (_, _, reports, _, _, _) in plot_data.items():
        if not reports:
            cell_class[cell] = "none"
            continue
        cell_class[cell] = max((r.classification for r in reports),
                                key=lambda c: rank.get(c, 0))

    # ---------------- summary ----------------
    summary_lines: list[str] = []
    summary_lines.append(f"Jump-detection dry-run summary")
    summary_lines.append(f"  annot_dir       = {ANNOT_DIR}")
    summary_lines.append(f"  cells scanned   = {n_total}")
    summary_lines.append(f"  bump_min        = {params.bump_min}")
    summary_lines.append(f"  persist_min     = {params.persist_min}")
    summary_lines.append(f"  pre_window      = {params.pre_window}")
    summary_lines.append(f"  post_window     = {params.post_window}")
    summary_lines.append("")
    hist: dict[str, int] = {"none": 0, "transient": 0, "edge_skip": 0,
                             "sustained": 0}
    for c in cell_class.values():
        hist[c] = hist.get(c, 0) + 1
    summary_lines.append("Per-cell classification histogram:")
    for k in ("sustained", "transient", "edge_skip", "none"):
        summary_lines.append(f"  {k:>10s}  {hist[k]:>4d}")
    summary_lines.append("")

    sustained_cells = [c for c, k in cell_class.items() if k == "sustained"]
    summary_lines.append(f"Sustained cells ({len(sustained_cells)}):")
    for c in sustained_cells:
        # primary = earliest sustained candidate
        reports = plot_data[c][2]
        boundaries = plot_data[c][3]
        s_reports = [r for r in reports if r.classification == "sustained"]
        primary = min(s_reports, key=lambda r: r.jump_cycle_ordinal)
        near_boundary = _is_near_boundary(primary.jump_cycle_ordinal,
                                           boundaries)
        summary_lines.append(
            f"  {c:<20s}  cycle={primary.jump_cycle_ordinal:>4d}  "
            f"dir={primary.jump_direction:<4s}  Δ={primary.jump_magnitude:+.4f}  "
            f"persist={primary.persistence_score:+.4f}  "
            f"near_regime_boundary={near_boundary}"
        )
    summary_lines.append("")

    # Multi-regime single_rate cohort cross-tab
    summary_lines.append(
        f"Multi-regime single_rate cohort ({len(multi_regime_single_rate_cells)} cells):"
    )
    mr_hist = {"sustained": 0, "transient": 0, "edge_skip": 0, "none": 0}
    for c in multi_regime_single_rate_cells:
        k = cell_class.get(c, "none")
        mr_hist[k] = mr_hist.get(k, 0) + 1
    for k in ("sustained", "transient", "edge_skip", "none"):
        summary_lines.append(f"  {k:>10s}  {mr_hist[k]:>4d}")
    summary_lines.append("")

    # Boundary coincidence rate
    if sustained_cells:
        coincide = 0
        for c in sustained_cells:
            reports = plot_data[c][2]
            boundaries = plot_data[c][3]
            s_reports = [r for r in reports if r.classification == "sustained"]
            if s_reports and _is_near_boundary(
                min(s_reports, key=lambda r: r.jump_cycle_ordinal).jump_cycle_ordinal,
                boundaries
            ):
                coincide += 1
        summary_lines.append(
            f"Sustained-cell regime-boundary coincidence: "
            f"{coincide}/{len(sustained_cells)} "
            f"({100*coincide/len(sustained_cells):.0f}%)"
        )

    summary_text = "\n".join(summary_lines) + "\n"
    summary_path = OUT_DIR / "jump_detection_summary.txt"
    summary_path.write_text(summary_text)
    print(summary_text)
    print(f"wrote {summary_path}")

    # ---------------- plots ----------------
    print()
    print("Generating plots...")

    # Sustained
    for c in sustained_cells:
        cycles, retentions, reports, boundaries, consistency, n_regulars = plot_data[c]
        _plot_cell(c, cycles, retentions, reports, boundaries,
                   consistency, n_regulars,
                   PLOTS_DIR / "sustained" / f"{c}.png")
    print(f"  sustained:           {len(sustained_cells)} plots")

    # Transient
    transient_cells = [c for c, k in cell_class.items() if k == "transient"]
    for c in transient_cells:
        cycles, retentions, reports, boundaries, consistency, n_regulars = plot_data[c]
        _plot_cell(c, cycles, retentions, reports, boundaries,
                   consistency, n_regulars,
                   PLOTS_DIR / "transient" / f"{c}.png")
    print(f"  transient:           {len(transient_cells)} plots")

    # Multi-regime single_rate audit (regardless of classification)
    for c in multi_regime_single_rate_cells:
        cycles, retentions, reports, boundaries, consistency, n_regulars = plot_data[c]
        if not cycles:
            continue
        _plot_cell(c, cycles, retentions, reports, boundaries,
                   consistency, n_regulars,
                   PLOTS_DIR / "multi_regime_audit" / f"{c}.png")
    print(f"  multi_regime_audit:  {len(multi_regime_single_rate_cells)} plots")

    # False-negative audit: random sample of cells classified 'none' with
    # enough regulars to be interesting
    none_candidates = [c for c, k in cell_class.items()
                       if k == "none" and plot_data[c][5] >= FN_AUDIT_MIN_REGULARS]
    rng = random.Random(FN_AUDIT_SEED)
    fn_sample = rng.sample(none_candidates,
                            min(FN_AUDIT_SAMPLE_SIZE, len(none_candidates)))
    for c in fn_sample:
        cycles, retentions, reports, boundaries, consistency, n_regulars = plot_data[c]
        _plot_cell(c, cycles, retentions, reports, boundaries,
                   consistency, n_regulars,
                   PLOTS_DIR / "false_negative_audit" / f"{c}.png")
    print(f"  false_negative_audit: {len(fn_sample)} plots "
          f"(seeded {FN_AUDIT_SEED}, min n_regulars={FN_AUDIT_MIN_REGULARS})")

    return 0


def _build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--bump-min", type=float, default=DetectorParams().bump_min)
    p.add_argument("--persist-min", type=float,
                   default=DetectorParams().persist_min)
    p.add_argument("--pre-window", type=int,
                   default=DetectorParams().pre_window)
    p.add_argument("--post-window", type=int,
                   default=DetectorParams().post_window)
    p.add_argument("--min-pre-len", type=int,
                   default=DetectorParams().min_pre_len)
    p.add_argument("--min-post-len", type=int,
                   default=DetectorParams().min_post_len)
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

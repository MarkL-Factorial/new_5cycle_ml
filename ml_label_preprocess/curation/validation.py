"""Manual cell-validation pipeline.

Integrates the two upstream curation modules (outlier_detection +
sustained_step) and surfaces, per run, the cells that still need a
human judgement. Once decided, cells drop out of the pending queue;
their review plots are archived to ``plots/validated/`` for audit.

Subcommands:

  refresh           Re-run both upstream pipelines (outlier_detection,
                    sustained_step) on the full cohort. Use after
                    annotation data changes or detector params change.

  sync              Surface cells matching any of three criteria;
                    schema-validate every entry; promote snapshot
                    plots. Default if no subcommand given. Criteria:
                      1. cell is in sustained_step report
                      2. outlier mask covers a cycle in the last
                         ``--tail-window`` (default 5) regular cycles
                      3. decision's ``n_regular_at_review`` no longer
                         matches the cohort (stale decision)

  migrate-snapshot  One-shot: fill ``n_regular_at_review`` for legacy
                    decisions that lack it (use current n_regular).
                    Idempotent; run once per workspace.

Output structure (under curation/):

  decisions.json          (committed — the artifact)
                          master JSON: one entry per validated cell
  plots/validated/<cell>.png  (committed) snapshot of plot used at validation

  pending/                (gitignored — regenerated each sync)
    cell_list.txt         pending cells + per-cell key stats
    plots/<cell>.png      copied from sustained_step (preferred) or
                          outliers/with_outliers (fallback for tail-only
                          candidates)
    template.json         paste-and-fill stub for new decisions

Schema for one decisions.json entry (all six fields required):

  {
    "exclude_from_ml":      bool,         If true, cell dropped from cohort.
    "last_available_cycle": int | null,   See labels.py interpretation table.
                                            Cannot be null when
                                            event_type='event'.
    "event_type":           "censor" | "event" | null,
                                          Drives status. Cannot be null
                                            when exclude_from_ml=false.
    "reason":               str,          Free-text audit note.
    "validated_at":         "YYYY-MM-DD"  ISO date.
    "n_regular_at_review":  int,          Cell's n_regular at review time;
                                            stale-decision check anchor.
  }

Author: Mark Liao (Sheng-Lun Liao)
"""
from __future__ import annotations

import argparse
import json
import shutil
import sys
from datetime import date
from pathlib import Path
from typing import Any

import polars as pl


HERE = Path(__file__).resolve().parent
ROOT = HERE.parent  # .../ml_label_preprocess

OUTLIER_SIDECAR = HERE / "outlier_sidecar.json"
OUTLIER_PLOTS = HERE / "plots" / "outliers" / "with_outliers"
SUSTAINED_CSV = HERE / "reports" / "sustained_step_report.csv"
SUSTAINED_PLOTS = HERE / "plots" / "sustained_step"

PENDING_DIR = HERE / "pending"
PENDING_PLOTS = PENDING_DIR / "plots"
PENDING_LIST = PENDING_DIR / "cell_list.txt"
PENDING_TEMPLATE = PENDING_DIR / "template.json"

VALIDATED_PLOTS = HERE / "plots" / "validated"
DECISIONS_PATH = HERE / "decisions.json"

# Cells whose outlier mask covers any cycle in the last TAIL_WINDOW
# regular cycles are auto-surfaced for manual review. Motivation:
# outlier mask near end of life can hide real fade events (AR4195 was
# the canonical case — cycles 32-35 masked of a 35-cycle cell that
# actually faded at cycle 35).
TAIL_WINDOW_DEFAULT = 5


# Allow ``from _common import ...`` from the parent project.
sys.path.insert(0, str(ROOT))
from _common import iter_annotations, iter_regulars  # noqa: E402


# ---------------- schema ----------------

REQUIRED_FIELDS: set[str] = {
    "exclude_from_ml", "last_available_cycle", "event_type",
    "reason", "validated_at", "n_regular_at_review",
}
ALLOWED_EVENT_TYPES: set[Any] = {"censor", "event", None}


def _validate_entry(cell: str, entry: dict) -> list[str]:
    """Return a list of human-readable errors for one decisions.json entry.

    Empty list ⇒ entry is valid.
    """
    errs: list[str] = []

    if not isinstance(entry, dict):
        return [f"entry must be a JSON object, got {type(entry).__name__}"]

    keys = set(entry.keys())
    missing = REQUIRED_FIELDS - keys
    if missing:
        errs.append(f"missing required field(s): {sorted(missing)}")
    extra = keys - REQUIRED_FIELDS
    if extra:
        errs.append(f"unknown field(s): {sorted(extra)}")

    exclude = entry.get("exclude_from_ml")
    if not isinstance(exclude, bool):
        errs.append(
            f"exclude_from_ml must be bool, got {type(exclude).__name__}"
        )

    lac = entry.get("last_available_cycle", "MISSING")
    if lac != "MISSING" and lac is not None and not isinstance(lac, int):
        errs.append(
            f"last_available_cycle must be int or null, "
            f"got {type(lac).__name__}"
        )

    et = entry.get("event_type", "MISSING")
    if et != "MISSING" and et not in ALLOWED_EVENT_TYPES:
        errs.append(
            f"event_type must be 'censor', 'event', or null; "
            f"got {et!r}"
        )

    reason = entry.get("reason")
    if reason is not None and not isinstance(reason, str):
        errs.append(f"reason must be str, got {type(reason).__name__}")

    va = entry.get("validated_at")
    if va is not None and not isinstance(va, str):
        errs.append(
            f"validated_at must be str (ISO date), "
            f"got {type(va).__name__}"
        )

    nar = entry.get("n_regular_at_review", "MISSING")
    if nar != "MISSING" and nar is not None and not isinstance(nar, int):
        errs.append(
            f"n_regular_at_review must be int or null, "
            f"got {type(nar).__name__}"
        )

    # Cross-field consistency rules. labels.py treats decisions.json
    # as authoritative — these rules guarantee every kept-cell entry
    # has the fields labels.py needs to produce a valid row.
    if isinstance(exclude, bool) and not exclude:
        et = entry.get("event_type")
        if et is None:
            errs.append(
                "event_type cannot be null when exclude_from_ml=false"
            )
        # event_type="event" → last_available_cycle becomes last_fade_cycle.
        # Cannot be null (no fade cycle to assert).
        if et == "event" and entry.get("last_available_cycle") is None:
            errs.append(
                "last_available_cycle cannot be null when event_type='event' "
                "(it is the asserted fade cycle)"
            )

    return errs


def _validate_decisions(decisions: dict) -> dict[str, list[str]]:
    """Validate the full decisions.json mapping. Returns {cell: [errs]}.

    Cells without errors do not appear in the result.
    """
    return {
        cell: errs
        for cell, errs in (
            (c, _validate_entry(c, e)) for c, e in decisions.items()
        )
        if errs
    }


def _load_decisions() -> dict:
    if not DECISIONS_PATH.exists():
        return {}
    return json.loads(DECISIONS_PATH.read_text())


# ---------------- refresh ----------------

def cmd_refresh() -> int:
    """Re-run outlier_detection + sustained_step on the full cohort."""
    print("=" * 72)
    print("STEP 1/2: outlier_detection")
    print("=" * 72)
    from curation.outlier_detector import OutlierParams  # noqa: E402
    from curation import outlier_detection as od  # noqa: E402
    rc = od.main(OutlierParams())
    if rc != 0:
        print(f"\noutlier_detection exited with code {rc}", file=sys.stderr)
        return rc

    print()
    print("=" * 72)
    print("STEP 2/2: sustained_step")
    print("=" * 72)
    from curation.jump_detection import DetectorParams  # noqa: E402
    from curation import sustained_step as ss  # noqa: E402
    rc = ss.main(DetectorParams(
        bump_min=ss.DEFAULT_BUMP_MIN,
        persist_min=ss.DEFAULT_PERSIST_MIN,
    ))
    if rc != 0:
        print(f"\nsustained_step exited with code {rc}", file=sys.stderr)
        return rc

    print()
    print("refresh complete. Now run: python run_validation.py sync")
    return 0


# ---------------- sync ----------------

def _load_sustained_cells() -> tuple[set[str], pl.DataFrame]:
    """Return (flagged_cell_set, full_csv_dataframe).

    If the CSV doesn't exist, exit with an actionable error.
    """
    if not SUSTAINED_CSV.exists():
        print(
            f"ERROR: sustained_step CSV not found at {SUSTAINED_CSV}\n"
            f"Run: python run_validation.py refresh",
            file=sys.stderr,
        )
        sys.exit(1)
    df = pl.read_csv(SUSTAINED_CSV)
    cells = set(df["cell_name"].unique().to_list())
    return cells, df


def _cell_summary_row(df: pl.DataFrame, cell: str) -> dict:
    """Pull the earliest-sustained-cycle row for ``cell`` from the CSV.

    sustained_step emits one row per (cell, sustained candidate). For
    summary listing we want the primary candidate per cell.
    """
    cell_df = df.filter(pl.col("cell_name") == cell).sort("sustained_cycle")
    return cell_df.row(0, named=True)


def _n_regular_by_cell() -> dict[str, int]:
    """Return {cell_name: last_regular_cycle} from the annotation cohort.

    Same source labels.py uses; one pass over all annotations.
    """
    out: dict[str, int] = {}
    for _, d in iter_annotations():
        regulars = iter_regulars(d)
        if not regulars:
            continue
        out[d.get("cell_name", "?")] = int(regulars[-1]["regular_cycle"])
    return out


def _stale_decisions(
    decisions: dict, n_regular_by_cell: dict[str, int],
) -> dict[str, str]:
    """Return ``{cell_name: stale_reason_text}`` for entries whose
    recorded ``n_regular_at_review`` no longer matches the cohort.

    Two failure modes are surfaced:
      - cell extended / shortened since the review
      - cell missing from current annotations (orphan)

    A returned non-empty string is the human-readable reason used in
    the pending queue's ``review_reason`` column.
    """
    stale: dict[str, str] = {}
    for cell, entry in decisions.items():
        n_at = entry.get("n_regular_at_review")
        n_now = n_regular_by_cell.get(cell)
        if n_at is None:
            stale[cell] = "stale_decision(was:legacy,now:" + (
                str(n_now) if n_now is not None else "missing") + ")"
        elif n_now is None:
            stale[cell] = f"stale_decision(was:{n_at},now:missing)"
        elif int(n_now) != int(n_at):
            stale[cell] = f"stale_decision(was:{n_at},now:{n_now})"
    return stale


def _tail_outlier_candidates(
    outlier_sidecar: dict,
    tail_window: int,
    n_regular_by_cell: dict[str, int],
) -> dict[str, tuple[int, list[int]]]:
    """Find cells with outliers in their last ``tail_window`` cycles.

    Returns ``{cell: (n_regular, sorted_tail_outlier_cycles)}``. A cell
    is included iff at least one of its sidecar-flagged outlier cycles
    falls inside ``[n_regular - tail_window + 1, n_regular]``.

    Cells the sidecar doesn't know about, or cells without an
    annotation entry, are silently skipped.
    """
    result: dict[str, tuple[int, list[int]]] = {}
    for cell, entry in outlier_sidecar.items():
        n_reg = n_regular_by_cell.get(cell)
        if n_reg is None:
            continue
        tail_lo = n_reg - tail_window + 1
        outliers = entry.get("outliers") or []
        tail_cycles = sorted(
            int(o["cycle"]) for o in outliers
            if tail_lo <= int(o["cycle"]) <= n_reg
        )
        if tail_cycles:
            result[cell] = (n_reg, tail_cycles)
    return result


def cmd_sync(tail_window: int = TAIL_WINDOW_DEFAULT) -> int:
    if not DECISIONS_PATH.exists():
        print(
            f"ERROR: decisions.json not found at {DECISIONS_PATH}\n"
            f"Initialize it with: echo '{{}}' > {DECISIONS_PATH}",
            file=sys.stderr,
        )
        return 1
    decisions = _load_decisions()

    # Schema-validate first; refuse to do anything else until clean.
    val_errors = _validate_decisions(decisions)
    if val_errors:
        print("decisions.json failed schema validation:", file=sys.stderr)
        for cell, errs in sorted(val_errors.items()):
            for e in errs:
                print(f"  {cell}: {e}", file=sys.stderr)
        print(
            "\nFix the entries above and re-run sync.\n"
            f"Schema reference: docstring at top of {__file__}",
            file=sys.stderr,
        )
        return 1

    flagged_cells, csv_df = _load_sustained_cells()

    # Load outlier sidecar + compute tail-outlier candidates.
    sidecar: dict = (
        json.loads(OUTLIER_SIDECAR.read_text())
        if OUTLIER_SIDECAR.exists() else {}
    )
    n_regular_by_cell = _n_regular_by_cell()
    tail_outliers = _tail_outlier_candidates(
        sidecar, tail_window, n_regular_by_cell,
    )
    tail_outlier_cells: set[str] = set(tail_outliers.keys())

    validated_cells = set(decisions.keys())

    # Stale-decision detection. A validated cell whose n_regular_at_review
    # no longer matches the cohort's current n_regular re-enters the
    # pending queue (the prior decision may have been based on stale data).
    stale = _stale_decisions(decisions, n_regular_by_cell)
    stale_cells = set(stale.keys())

    all_candidates = flagged_cells | tail_outlier_cells | stale_cells
    # Stale cells stay in pending even though they're in decisions.json;
    # other validated cells drop out of pending as usual.
    pending_cells = (
        (flagged_cells | tail_outlier_cells) - validated_cells
    ) | stale_cells

    # Cells that are decided AND no longer flagged AND not stale: clean.
    # Cells that are decided AND no longer flagged BUT also tail-outlier
    # (matches today's "stale" message) — keep the legacy info line.
    informational_stale_cells = (
        validated_cells - (flagged_cells | tail_outlier_cells) - stale_cells
    )

    # Refresh pending/ from scratch.
    if PENDING_DIR.exists():
        shutil.rmtree(PENDING_DIR)
    PENDING_PLOTS.mkdir(parents=True, exist_ok=True)

    list_lines: list[str] = [
        "# cell\treview_reason\tsustained_cycle\tdelta\tpersist\t"
        "tail_outlier_cycles\tn_outliers_masked\tn_regulars"
    ]
    template: dict[str, dict] = {}
    today = date.today().isoformat()
    for cell in sorted(pending_cells):
        in_sustained = cell in flagged_cells
        in_tail = cell in tail_outlier_cells
        in_stale = cell in stale_cells
        reasons: list[str] = []
        if in_stale:
            # Detail embedded in the reason string itself, e.g.
            # ``stale_decision(was:113,now:163)``.
            reasons.append(stale[cell])
        if in_sustained:
            reasons.append("sustained_step")
        if in_tail:
            reasons.append("tail_outlier")
        review_reason = ",".join(reasons)

        # Sustained-side stats (if applicable)
        sus_cycle = "-"
        sus_delta = "-"
        sus_persist = "-"
        n_masked = 0
        if in_sustained:
            row = _cell_summary_row(csv_df, cell)
            sus_cycle = str(row["sustained_cycle"])
            sus_delta = f"{row['jump_magnitude']:+.4f}"
            sus_persist = f"{row['persistence_score']:+.4f}"
            n_masked = int(row["n_outliers_masked"])
            n_regulars = int(row["n_regulars"])
        else:
            n_regulars = n_regular_by_cell.get(cell, 0)

        # Tail-outlier-side stats (if applicable)
        if in_tail:
            tail_cycles = tail_outliers[cell][1]
            tail_cycles_str = ",".join(str(c) for c in tail_cycles)
            # If the sustained side didn't report a mask count, use the
            # total outliers for this cell from the sidecar.
            if not in_sustained:
                n_masked = len(sidecar.get(cell, {}).get("outliers", []))
        else:
            tail_cycles_str = "-"

        list_lines.append(
            f"{cell}\t{review_reason}\t{sus_cycle}\t{sus_delta}\t"
            f"{sus_persist}\t{tail_cycles_str}\t{n_masked}\t{n_regulars}"
        )
        # Template stub: only for cells NOT already in decisions.json.
        # Stale cells already have entries; the reviewer edits in place.
        if not in_stale:
            template[cell] = {
                "exclude_from_ml": False,
                # Pre-filled with the cell's last regular cycle — i.e., "use
                # the whole cell". Reviewer either keeps this (no truncation)
                # or lowers it to truncate before a sustained step.
                "last_available_cycle": int(n_regulars),
                "event_type": None,
                "reason": "",
                "validated_at": today,
                "n_regular_at_review": int(n_regulars),
            }
        # Plot copy: prefer the sustained_step annotated plot; fall back
        # to the outlier-detection plot (with red X markers on flagged
        # cycles) for tail-only candidates.
        sus_src = SUSTAINED_PLOTS / f"{cell}.png"
        out_src = OUTLIER_PLOTS / f"{cell}.png"
        if sus_src.exists():
            shutil.copy2(sus_src, PENDING_PLOTS / f"{cell}.png")
        elif out_src.exists():
            shutil.copy2(out_src, PENDING_PLOTS / f"{cell}.png")

    PENDING_LIST.write_text("\n".join(list_lines) + "\n")
    PENDING_TEMPLATE.write_text(json.dumps(template, indent=2) + "\n")

    # Promote plots for any validated cells that don't yet have a snapshot.
    VALIDATED_PLOTS.mkdir(parents=True, exist_ok=True)
    n_promoted = 0
    n_promoted_missing_src = 0
    for cell in sorted(validated_cells):
        dst = VALIDATED_PLOTS / f"{cell}.png"
        if dst.exists():
            continue
        src = SUSTAINED_PLOTS / f"{cell}.png"
        if not src.exists():
            n_promoted_missing_src += 1
            continue
        shutil.copy2(src, dst)
        n_promoted += 1

    # Summary
    n_pending_sustained = sum(
        1 for c in pending_cells if c in flagged_cells
    )
    n_pending_tail = sum(
        1 for c in pending_cells if c in tail_outlier_cells
    )
    n_pending_stale = sum(
        1 for c in pending_cells if c in stale_cells
    )
    print(f"Sustained flagged cells:       {len(flagged_cells)}")
    print(f"Tail-outlier cells (last {tail_window}): {len(tail_outlier_cells)}")
    print(f"Total validated:               {len(validated_cells)}")
    print(f"Stale decisions:               {len(stale_cells)}")
    print(f"Pending review:                {len(pending_cells)} "
          f"({n_pending_sustained} sustained, {n_pending_tail} tail-outlier, "
          f"{n_pending_stale} stale)")
    print(f"Newly promoted plot snapshots: {n_promoted}")
    if n_promoted_missing_src:
        print(
            f"Validated cells with no source plot to snapshot: "
            f"{n_promoted_missing_src} "
            f"(detector params likely changed; existing snapshots retained)"
        )
    if informational_stale_cells:
        print(
            f"\nDecided but no longer flagged by any candidate criterion "
            f"(decisions stay; cells just aren't current candidates):"
        )
        for c in sorted(informational_stale_cells):
            print(f"  {c}")
    if pending_cells:
        print()
        print(f"Plots to review:    {PENDING_PLOTS}")
        print(f"Per-cell stats:     {PENDING_LIST}")
        print(f"Decision template:  {PENDING_TEMPLATE}")
        print(f"Add entries to:     {DECISIONS_PATH}")
    else:
        print("\n✓ All flagged cells are validated.")
    return 0


# ---------------- migrate-snapshot ----------------

def cmd_migrate_snapshot() -> int:
    """One-shot migration: fill ``n_regular_at_review`` for legacy entries.

    For each entry in ``decisions.json`` that lacks
    ``n_regular_at_review`` (or has it as null), set it to the cell's
    current ``n_regular`` (the cell's last regular_cycle in the
    current cohort). Effectively records "I just confirmed this
    against the current data" as of run time.

    Idempotent: re-running has no effect on entries that already have
    the field populated. Orphan cells (no longer in annotations) are
    left untouched and will be flagged stale by the next ``sync``.
    """
    if not DECISIONS_PATH.exists():
        print(
            f"ERROR: decisions.json not found at {DECISIONS_PATH}",
            file=sys.stderr,
        )
        return 1
    decisions = _load_decisions()
    n_regular_by_cell = _n_regular_by_cell()
    n_added = 0
    n_skipped_orphan = 0
    n_already = 0
    for cell, entry in decisions.items():
        existing = entry.get("n_regular_at_review")
        if existing is not None:
            n_already += 1
            continue
        n_now = n_regular_by_cell.get(cell)
        if n_now is None:
            print(
                f"  skipping {cell}: cell not present in current annotations",
                file=sys.stderr,
            )
            n_skipped_orphan += 1
            continue
        entry["n_regular_at_review"] = int(n_now)
        n_added += 1
        print(f"  {cell}: n_regular_at_review = {n_now}")

    DECISIONS_PATH.write_text(json.dumps(decisions, indent=2) + "\n")
    print(f"\nMigration complete:")
    print(f"  filled in:                 {n_added}")
    print(f"  already populated:         {n_already}")
    print(f"  skipped (orphan / missing): {n_skipped_orphan}")
    print(f"  total entries:             {len(decisions)}")
    return 0


# ---------------- main ----------------

def _build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    sub = p.add_subparsers(dest="cmd", metavar="COMMAND")
    sub.add_parser(
        "refresh",
        help="Re-run outlier_detection + sustained_step on the full cohort.",
    )
    p_sync = sub.add_parser(
        "sync",
        help="Identify pending cells, validate decisions.json, promote plots. "
             "Default when no subcommand is given.",
    )
    p_sync.add_argument(
        "--tail-window", type=int, default=TAIL_WINDOW_DEFAULT,
        help=f"Surface cells whose outlier mask covers any cycle in the "
             f"last N regular cycles (default {TAIL_WINDOW_DEFAULT}).",
    )
    sub.add_parser(
        "migrate-snapshot",
        help="One-shot: fill n_regular_at_review (using current n_regular) "
             "for legacy decisions.json entries that lack it.",
    )
    # Allow --tail-window on the top-level too so `python -m curation.validation
    # --tail-window 10` works (sync is the default subcommand).
    p.add_argument(
        "--tail-window", type=int, default=TAIL_WINDOW_DEFAULT,
        help=argparse.SUPPRESS,
    )
    return p


def main(argv: list[str]) -> int:
    args = _build_argparser().parse_args(argv)
    cmd = args.cmd or "sync"
    if cmd == "refresh":
        return cmd_refresh()
    if cmd == "migrate-snapshot":
        return cmd_migrate_snapshot()
    return cmd_sync(tail_window=args.tail_window)


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))

# Capacity-jump detector — dry-run investigation

## Why

Cell **AR4313** is currently labeled `label_n300 = pass` in
`datasets/A2.2_b1/cell_labels.parquet` even though its discharge retention
crossed 0.85 around cycle ~218. The cause is a +15% step in discharge
capacity at cycle 268 (0.1039 Ah → 0.1198 Ah), unexplained by the modest
2.1% rate change the annotation toolkit recorded for the second regime.
The recovery-aware fade detector in `labels.py::_last_crossing_into_bad`
interprets those zombie post-jump cycles as legitimate recovery and pushes
`last_fade_cycle` from ~218 → 400, flipping the cell's true `bad` →
incorrect `pass` at N=300.

This folder is a **standalone dry-run** that finds cells like AR4313 in
both directions (up *and* down) and discriminates *pathological regime
shifts* from *normal RPT-style transient recoveries*. **No production
labels are modified by anything here.** Once the parameters are validated
against eyeballed plots, a follow-up plan will wire truncation into
`labels.py`.

## Algorithm

For each cell, walk regular cycles 1..N:

1. **Trigger**: list-index `i` is a candidate when `|ret[i] - ret[i-1]| >= bump_min`.
2. **Pre-trend fit**: ordinary least squares on cycles `[i-pre_window, i-1]`.
3. **Extrapolate**: project the pre-trend forward over `[i, i+post_window)`.
4. **Persistence score**: signed median of `actual - extrapolated` over the post window.
5. **Classify**:
   - `|persist| >= persist_min` → **sustained** (post-jump curve permanently offset from pre-trend — looks like a regime shift).
   - otherwise → **transient** (curve returns to pre-trend within K cycles — looks like RPT recovery).
   - Either window too short → **edge_skip** (no judgement).

By using absolute values for both the trigger and the persistence test, the
detector is symmetric: upward and downward steps are treated identically.
The signed `jump_magnitude` and `jump_direction` columns preserve the
direction so the report can be filtered.

## Parameters (initial defaults)

| Param | Default | Tunable via |
|---|---:|---|
| `bump_min` | 0.03 | `--bump-min` |
| `persist_min` | 0.03 | `--persist-min` |
| `pre_window` | 20 cycles | `--pre-window` |
| `post_window` | 10 cycles | `--post-window` |
| `min_pre_len` | 10 points | `--min-pre-len` |
| `min_post_len` | 5 points | `--min-post-len` |

These defaults flag AR4313 at cycle 268 with `persistence ≈ +0.14`,
roughly 4.7× the threshold. They are intentionally sensitive on the
trigger side (`bump_min`) so the persistence test does the discrimination
work — that way, RPT bumps still surface in the CSV (as `transient`) for
audit, rather than being silently dropped.

## How to run

```bash
source /home/mliao/miniconda3/etc/profile.d/conda.sh && conda activate eis

# Synthetic selftest (must pass before running on real data)
python detector.py --selftest

# Full investigation
python run_investigation.py

# Tune parameters
python run_investigation.py --bump-min 0.05 --persist-min 0.05
```

## Outputs

```
out/
├── jump_detection_report.csv     # one row per (cell, candidate jump); cells with no candidates get one sentinel row
├── jump_detection_summary.txt    # histograms + named sustained-cell list
└── plots/
    ├── sustained/         # every cell with at least one 'sustained' candidate
    ├── transient/         # every cell whose strongest candidate is 'transient'
    ├── multi_regime_audit/  # the 69 multi-regime single_rate cells regardless of classification
    └── false_negative_audit/  # random sample of cells classified 'none' (seeded for reproducibility)
```

Each plot shows: retention scatter, 0.85 fade line, regime boundaries
(dotted grey), and for every candidate jump: a vertical line at the jump
cycle plus the pre-window fit (solid) and post-window extrapolation
(dashed) so the residual is visually obvious. Color coding: red =
sustained, orange = transient, grey = edge_skip.

### CSV columns

| Column | Meaning |
|---|---|
| `cell_name`, `cohort`, `cycling_consistency`, `protocol_pattern` | identifiers / cohort metadata |
| `n_regulars` | lifetime regular_cycle count |
| `n_regimes` | number of `regular_rate_regimes` in the annotation |
| `max_regime_rate_delta_pct` | (max - min) / min baseline_i_a across regimes, % |
| `regime_boundary_cycles` | cumulative cycle counts where regimes change (CSV string) |
| `jump_cycle_ordinal` | regular_cycle of the candidate (NULL if cell has no candidate) |
| `jump_magnitude` | signed Δret at the candidate |
| `jump_direction` | `up` (Δ > 0) or `down` (Δ < 0) |
| `pre_slope` | slope of pre-window OLS fit |
| `pre_n_points`, `post_n_points` | actual window lengths used |
| `persistence_score` | signed median residual over the post window |
| `classification` | `sustained` / `transient` / `edge_skip` / `none` |
| `jump_near_regime_boundary` | True if `jump_cycle_ordinal` within ±2 of any boundary |

## Audit procedure (recommended)

1. **Check AR4313** — should be `sustained, dir=up, cycle=268,
   persist≈+0.14, near_regime_boundary=True`.
2. **Skim `out/plots/sustained/`** — every plot should show a visually
   unambiguous step (either up or down) with the post-jump curve sitting
   permanently offset from the extrapolated pre-trend.
3. **Skim `out/plots/transient/`** — every plot should show a single-cycle
   spike that returns to the pre-trend within ~5 cycles. If any of these
   look like real regime shifts, lower `persist_min`.
4. **Multi-regime audit (`out/plots/multi_regime_audit/`)** — most of the
   69 multi-regime single_rate cells have <1% rate delta and should
   show no visible step. Confirm that the few with visible steps overlap
   with the `sustained` set.
5. **False-negative spot check (`out/plots/false_negative_audit/`)** —
   20 random cells classified `none`. They should look smooth (clean
   monotone fade or healthy plateau). If any show obvious steps,
   investigate why the detector missed them (likely `bump_min` too high
   or near a window edge).

## Known limitations

- **Baseline = cycle 1 always**. The retention curve uses the first regular
  cycle as denominator; this matches `labels.py`'s default
  `baseline_cycle=1`. If you want to investigate `baseline_cycle=3`,
  modify `compute_retentions` in `detector.py`.
- **Multiple candidates per cell**: RPT bumps produce two adjacent
  candidates (an up-trigger followed by a down-trigger as the curve
  returns). Both will be classified `transient` and that is the correct
  behavior — the long-format CSV preserves them; the per-cell rollup in
  `summary.txt` reports the strongest classification.
- **No truncation suggested in the report yet**. The plan is: the
  earliest `sustained` candidate per cell becomes the truncation cycle
  in the follow-up plan. That logic is intentionally not wired here.
- **`bump_min` is sensitive on purpose**. Lowering it surfaces more RPT
  bumps as `transient` candidates (no harm; they are filtered by
  classification). Raising it can hide small but persistent regime shifts.

## Files

```
detector.py            pure-function core: compute_retentions, detect_jumps, selftest
run_investigation.py   CLI: iterate cells → CSV + summary + plots
README.md              this file
.gitignore             out/ is reproducible — only sources are committed
out/                   generated artifacts (gitignored)
```

Shared with the production pipeline (read-only):
- `iter_annotations`, `iter_regulars`, `_cohort`, `ANNOT_DIR` from
  `../../_common.py`.

Nothing in `../../labels.py`, `../../features.py`, `../../preprocess.py`,
or `../../datasets/` is modified.

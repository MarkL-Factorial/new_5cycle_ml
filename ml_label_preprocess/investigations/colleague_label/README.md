# Colleague-label cross-check

Cross-checks our pipeline's `label_n300` against a colleague's
independent GOOD/BAD labels on the 208 cells shared by both sets.

## Inputs

- `all_features.parquet` — colleague's deliverable. 208 cells. Relevant
  columns: `cell_name`, `label` ∈ {GOOD, BAD}, `retention`,
  `max_regular_cycle`. (The 40 engineered feature columns are ignored.)
- `../../datasets/A2.2_b1/A2.2_b1_latest/cell_labels.parquet` — our
  latest snapshot (470 cells). Relevant columns: `cell_name`,
  `label_n300` ∈ {pass, bad, censor, excluded}, `status`,
  `last_fade_cycle`, `final_retention`, `n_regular`,
  `truncation_cycle`.

## Mapping rule

The colleague's labels were empirically derived as: **BAD iff retention
< 0.85**, and every GOOD cell has `max_regular_cycle ≥ 310`. So:

| colleague | ↔ our `label_n300` |
|---|---|
| `good` | `pass` (cell passed N=300 with retention ≥ 0.85) |
| `bad`  | `bad`  (cell faded by N=300) |

Our `censor` (still in testing, `n_regular < 300`) and `excluded`
(unusable cell) are shown separately and **not** counted in the
primary agreement %.

## How to run

```bash
source /home/mliao/miniconda3/etc/profile.d/conda.sh && conda activate eis
cd /mnt/data/mliao/battery-ml-workbench/new_5cycle_ml/ml_label_preprocess/investigations/colleague_label
python compare_labels.py
```

Stdout prints category counts and the agreement %. All artifacts go to
`out/`.

## Outputs (`out/`)

| File | Contents |
|---|---|
| `comparison_table.parquet` / `.csv` | 208-row per-cell join with category & cohort columns. |
| `mismatches.csv` | Subset where ours can't confirm theirs: `disagree_we_*` plus `ours_censor` / `ours_excluded` (where we suspend comparison). |
| `colleague_only_cells.csv` | Colleague cells with no row in our latest labels (annotation gap on our side). |
| `summary_stats.json` | Total/primary counts, agreement %, confusion matrix, per-cohort breakdown. |
| `confusion_matrix.png` | colleague label × our `label_n300` (annotated heatmap). |
| `retention_vs_cycle_scatter.png` | `max_regular_cycle` vs `final_retention`, colored by agreement category; reference lines at N=300 and retention=0.85. |
| `cohort_agreement_bar.png` | Agreement breakdown by `AR` / `0MC2` / `other` cohorts. |

## Agreement categories

- `agree_pass` — both label this cell GOOD/pass.
- `agree_bad` — both label this cell BAD/bad.
- `disagree_we_pass_they_bad` — we say pass at N=300, colleague says BAD.
- `disagree_we_bad_they_pass` — we say bad at N=300, colleague says GOOD.
- `ours_censor` — our pipeline censors (cell hasn't reached N=300); shown but excluded from agreement %.
- `ours_excluded` — our pipeline excludes the cell entirely; shown but excluded from agreement %.

## Notes

- **6 colleague cells missing from our cohort** (5 `AR*` + 1 `R1348`)
  appear in `colleague_only_cells.csv`. These cells don't have an
  annotation JSON in our current `BAT_ANNOT_DIR` snapshot — worth
  flagging upstream.

- **`0MC2-251022-004`** — labelled GOOD by the colleague despite a
  final retention of 0.32. This is the lone exception to the
  BAD-iff-retention<0.85 rule, and turns out NOT to be a disagreement:
  our pipeline also labels it `pass` at N=300 (the cell faded long
  after cycle 300, so it correctly passed at N=300 even though its
  *final* retention later dropped below 0.85). The script prints this
  cell's category to stdout for visibility.

# colleague_comparison

One-off audit: why does the colleague's `all_features.parquet` disagree with
`ml_label_preprocess/datasets/A2.2_b1/` on cells they both cover?

Spoiler: not bugs — the two sides use different definitions for "the same"
column (different cycle index, different formula, fraction vs percent).
This folder quantifies which gap is which, using the annotation registry
at `/mnt/data/mliao/battery-ml-workbench/data/A2.2/annotations/_annotations.parquet`
as ground truth.

## Inputs (read-only)

- `colleague_annoation/all_features.parquet` — colleague, 208 cells × 44 cols
- `ml_label_preprocess/datasets/A2.2_b1/cell_features.parquet` — mine (B), 459 × 41
- `ml_label_preprocess/datasets/A2.2_b1/cell_labels.parquet` — mine (B), 470 × 18
- `data/A2.2/annotations/_annotations.parquet` — per-cycle truth, 143k rows

## How to run

```bash
conda activate eis
cd /mnt/data/mliao/battery-ml-workbench/new_5cycle_ml/colleague_comparison
python scripts/01_ce.py
python scripts/02_retention.py
python scripts/03_fade_cycle.py
python scripts/04_label_agreement.py
python scripts/05_select_distinguishing_cells.py
python scripts/06_deep_dive_plots.py
```

Outputs (CSVs + PNGs) land in `out/`. Read `report.md` for the verdict.

## Layout

| File | Purpose |
|---|---|
| `scripts/load_data.py` | Shared loaders for A / B-features / B-labels / registry |
| `scripts/01_ce.py` | Compare A.ce2 vs B.coulombic_efficiency_final vs registry truth at several cycle candidates |
| `scripts/02_retention.py` | Compare A.retention vs each plausible truth-derived retention formula |
| `scripts/03_fade_cycle.py` | A.max_regular_cycle, B.n_regular, B.last_fade_cycle vs recomputed truth |
| `scripts/04_label_agreement.py` | A.label (GOOD/BAD) vs B.label_n{200,300,400}, with per-cell truth witness |
| `scripts/05_select_distinguishing_cells.py` | Picks 2–3 cells per disagreement category + plots retention curves for manual review |
| `scripts/06_deep_dive_plots.py` | Per-cell 4-panel figure: capacity / CE / retention / voltage profile, recomputed from raw step-level parquet |

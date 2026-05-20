# exp_l — Does adding `discharge_capacity_c5` improve classification?

The production classifier uses `fs_a_only` (3 retention features). This
experiment tests whether adding the **absolute** discharge capacity at
cycle 5 (`baseline_dis_ah × discharge_capacity_retention_final`, in Ah)
improves AUC/F1 on the held-out test set.

**Cohort confound**: `baseline_dis_ah` separates AR (~0.1 Ah) from 0MC
(~6 Ah) almost deterministically. Per-cohort metrics (AR / 0MC) tell us
whether any improvement is real signal or a cohort shortcut.

## Design

- **Compare**: `fs_a_only` (3 feat) vs `fs_a_plus_cap_c5` (4 feat)
- **Tuning**: paired — both feature sets re-tuned per seed with the
  same Optuna budget (30 trials × 5-fold inner CV)
- **Splits**: stratified 80/20 × 5 seeds — same `random_state` for the
  two feature sets per seed (paired comparison)
- **Universe**: `n_reg ≥ 6 & status != 'excluded' & trainable_n{N}=True`
  → 295 / 255 / 241 cells for N=200/300/400

## Run

```bash
# Smoke (1 seed × 5 trials × 3 CV, ~3 min)
python experiments/exp_l_classifier_add_cap_c5/run.py --smoke

# Full (5 seeds × 30 trials × 5 CV, ~50-90 min)
python experiments/exp_l_classifier_add_cap_c5/run.py --seeds 5 --trials 30 --inner-cv 5
```

## Outputs

- `runs/seed_{s}.csv` — per-cell test predictions (both feature sets)
- `run_logs/run_<TS>.log` — execution log
- `summary.json` — mean ± std per metric, both feature sets, plus deltas
- `summary_wide.csv` — tidy table
- `comparison_by_cohort.csv` — per-cohort breakdown (the rubric below)

## Shortcut-detection rubric

| Pattern | Interpretation |
|---|---|
| AUC↑ overall AND in BOTH AR and 0MC | Real signal — absolute capacity adds info |
| AUC↑ overall AND in 0MC only (flat AR) | Real signal helping the minority cohort |
| AUC↑ overall but FLAT in both cohorts | Cohort shortcut — model learned "AR vs 0MC" |
| AUC flat or ↓ | New feature doesn't help |

# exp_k — production-ensemble held-out validation

Validates the [`results/run/20260519_1145/`](../../results/run/20260519_1145/)
production fit by holding out 20% of the trainable cells per seed and
measuring how the K=5 production ensemble generalizes.

**Design (chosen in chat — see plan):**

- **Split**: 80/20 stratified for classifiers, 80/20 random for RSF.
  5 seeds via `train_test_split(random_state=seed)`.
- **Hyperparameters**: frozen — loaded verbatim from production's
  `best_params.json` (5 sets per model, one per production seed).
- **Per split**: refit all 5 production hyperparameter sets on the 80%,
  ensemble-average predictions on the 20%. Mirrors production fidelity
  one-to-one.
- **Cell universe** (the "meaningful" cohort):
  - Both: `n_regular ≥ 6 & status != 'excluded'` → 417 cells in A2.2_b1
  - Classifier additionally: `trainable_n{N} == True` → 295/255/241 for N=200/300/400
  - Drops the 27 n_reg=5 cells (predict-only in production) and the
    15 rate_changed cells (predict-only in production)

**Run:**

```bash
# Smoke (1 seed, ~2 min)
python experiments/exp_k_production_validation/run.py --smoke

# Full (5 seeds, ~5-10 min)
python experiments/exp_k_production_validation/run.py --seeds 5
```

**Outputs**:

- `runs/seed_{s}.csv` — per-cell test predictions for seed s
- `run_logs/seed_{s}.log` — per-seed log
- `summary.json` — mean ± std per metric across seeds, plus the
  production inner-CV value for comparison
- `summary_wide.csv` — tidy table for reporting

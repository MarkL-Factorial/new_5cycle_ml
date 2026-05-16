# cell_lifetime experiments — REPORT

**Branch:** `feature/cell_lifetime` &nbsp;·&nbsp;
**Compute:** 5 seeds × 30 Optuna trials × 5 inner CV per run, N=300 unless noted, A2.2_b1, 10-core cap

This report consolidates two experiments aimed at improving the
Phase 1+2+3 in-session baselines.

---

## TL;DR

| Decision | Outcome |
|---|---|
| **Use all 40 features (`fs_all`) instead of `fs_cv` (12)?** | **YES for classification & regression; NO for survival (RSF overfits at 40)** |
| **Use a z-score blend of RSF + XGB-AFT?** | **NO. Blend is more stable (lower std) but consistently 1 std below RSF alone.** |
| **Headline cycle-life model?** | **RSF + fs_cv** — C-index 0.801 ± 0.021, AUC@300 0.879 ± 0.048. |

The five models now have honest multi-seed estimates instead of the
1-seed smokes from Phases 1+2+3. Phase 1's optimistic single-seed
numbers (e.g. RSF C-index = 0.807) collapse toward the 5-seed mean
(0.801 ± 0.021) — small but consistent.

---

## Experiment A — feature-set expansion (fs_cv → fs_all)

**Question**: does using the full 40 feature-role columns beat the
12-column `fs_cv` subset?

### Results (mean ± std across 5 seeds, N=300)

| Task | Model | fs_cv | fs_all | Δ relative | Verdict |
|---|---|---|---|---|---|
| classification | xgb_classifier | F1 = 0.838 ± 0.029 | **F1 = 0.866 ± 0.037** | **+3.3%** | **fs_all wins** |
| regression | xgb_regressor | MAE = 137.7 ± **16.9** | MAE = 136.5 ± **9.6** | +0.9% mean, **std ↓ 43%** | fs_all wins on stability |
| regression | ebm_regressor | MAE = 142.1 ± 14.6 | **MAE = 136.2 ± 12.3** | **+4.2%** | **fs_all wins** |
| survival | xgb_aft | C-index = 0.770 ± 0.026 | C-index = 0.778 ± 0.024 | +1.0% | tie (inside 1 std) |
| survival | rsf | **C-index = 0.801 ± 0.021** | C-index = 0.787 ± 0.020 | **−1.8%** | **fs_cv wins (rsf overfits)** |

### Interpretation

- **Classification and regression benefit from more features** — F1 and
  MAE both improve; for xgb_regressor the std halves, meaning models
  trained on fs_all are far less seed-sensitive.
- **RSF actually does worse with 40 features.** With only ~330 cells
  in each 80% training fold (415 × 0.8), 40 features × deep trees
  produces noisier splits. Forest models need enough samples per
  feature; survival forests are picky.
- **XGB-AFT is approximately neutral** (within 1 std band either way).

### Why regression hits a ceiling

Regression is **fundamentally biased** because it trains on faded-only
cells (187/415; the 228 right-censored cells are dropped). Per-quartile
MAE for xgb_regressor × fs_cv:

| True cycle-life quartile | n | MAE (cycles) |
|---|---:|---:|
| Q1 (short, 6–~93) | 49 | 92 |
| Q2 (~93–~310) | 46 | 98 |
| Q3 (~310–~524) | 48 | 77 |
| **Q4 (long, ~524+)** | **47** | **286** ← 3× worse |

The model has never seen a 900-cycle cell during training (those were
all still in testing and got censored). Survival models avoid this by
using the censored cells.

---

## Experiment B — survival horizon profile + ensemble blend

**Question**: how do RSF and XGB-AFT perform across N ∈ {200, 300, 400},
and does a z-score blend beat either alone?

### Results (mean ± std across 5 seeds, fs_cv)

| Model | C-index | AUC@200 | AUC@300 | AUC@400 |
|---|---|---|---|---|
| xgb_aft | 0.770 ± 0.026 | 0.910 ± 0.044 | 0.836 ± 0.057 | 0.803 ± 0.074 |
| **rsf** | **0.801 ± 0.021** | **0.924 ± 0.030** | **0.879 ± 0.048** | **0.862 ± 0.042** |
| blend (z-score avg) | 0.792 ± 0.015 | 0.925 ± 0.034 | 0.865 ± 0.050 | 0.836 ± 0.052 |

### Findings

1. **C-index is constant across N** (it's a rank metric on the full
   data; doesn't depend on the threshold). The headline survival
   metric is unchanged: RSF 0.801, AFT 0.770.
2. **AUC@N drops with N**: 0.924 → 0.879 → 0.862 for RSF.
   Long-horizon prediction is harder, as expected (more censored
   cells haven't reached the threshold yet).
3. **RSF beats AFT at every horizon by a comfortable margin** (Δ ≈ 0.03
   C-index, > 1 std).
4. **The blend never beats RSF alone.** Plan acceptance criterion was
   "blend > max(rsf, aft) by ≥ 1 std" → fails at every horizon:
   - N=200: blend 0.793 vs RSF 0.803, Δ = −0.010
   - N=300: blend 0.790 vs RSF 0.801, Δ = −0.011
   - N=400: blend 0.792 vs RSF 0.802, Δ = −0.010
5. **The blend is more stable** (std ~0.015 vs RSF's ~0.021) — useful
   if you specifically need lower seed variance, but you pay 1
   point of accuracy for it.

### Why the blend fails

Naive z-score averaging gives both models equal weight. RSF is
meaningfully better than AFT (0.801 vs 0.770), so equal weight dilutes
the better model's signal. A weighted blend (e.g. 70% RSF + 30% AFT,
based on out-of-fold C-index) would likely recover, but the plan
specified the simple z-score average so we report the null result and
recommend the simpler RSF-only approach.

---

## Headline recommendation

**For binary classification (pass/bad at N=300):**
- `xgb_classifier` + `fs_all` (40 features). **F1 = 0.866 ± 0.037**.

**For regression on faded cells (cycle life):**
- `ebm_regressor` + `fs_all` (interpretable) or `xgb_regressor` +
  `fs_all` (slightly tighter). **MAE ≈ 136 cycles, R² ≈ 0.28**.
- ⚠ **Note the regression ceiling is selection-bias-driven.** Long-life
  cells (Q4: > 524 cycles) have MAE 286 because they're underrepresented
  in training. If you want better long-horizon prediction, use survival.

**For survival (censoring-aware, the full cohort):**
- `rsf` + `fs_cv` (12 features). **C-index = 0.801 ± 0.021,
  AUC@300 = 0.879 ± 0.048**.
- This uses all 415 cells (187 observed + 228 censored).

---

## Methodology

### Protocol
- Multi-seed: 5 seeds (1, 2, 3, 4, 5).
- Optuna: 30 trials per seed (TPE sampler).
- Inner CV for tuning: 5 folds.
- Outer split: 80/20, stratified by class label (classification) or
  event (survival). Stratification disabled for regression.
- 10-core cap (`OMP_NUM_THREADS=10` + `n_jobs=10` in fixed_params).

### Data
- Bundle: `ml_label_preprocess/datasets/A2.2_b1/` (461 cells in
  labels, 439 with features, `n_regular ≥ 6` filter → 415 trainable).
- Faded subset: 187 cells (for regression).
- Survival subset: 415 cells (187 observed events + 228 right-censored).
- Classification subset at N=300: 250 trainable (pass + bad, dropping
  censored at N).

### Three small fixes landed in this round

1. **Box-Cox inverse clip** — `(λy + 1)^(1/λ)` was producing NaN when
   the transformed-y went below `-1/λ ≈ −1.87` for λ=0.534. Clipped
   the inside-power to a small positive ε, so the inverse becomes a
   tiny-but-finite cycle life rather than NaN. Caught during EBM fs_cv
   run with `interactions` tunable, which Optuna sometimes pushed
   into unstable regions.
2. **EBM `interactions` upper bound 0..3** (was 0..10). Higher
   interactions exploded wall-clock without accuracy gain; the
   half-hour-per-fit blowup of EBM × fs_all dropped under control.
3. **`predictions.csv` artifact per run.** New file per run dir
   listing per-cell test predictions across all seeds. Enables the
   blend experiment without re-running models, and supports future
   per-cell error analysis.

Plus a per-trial `n_skipped_folds` Optuna user_attr for diagnosability
when survival inner-CV folds emit NaN C-index.

### Where regression breaks (worth knowing for future work)

The regression models' per-seed Pearson r was 0.48–0.64; Pearson²
≈ 0.27–0.41, which matches the reported R² = 0.28. The model knows
the relative ordering OK; it just can't get the magnitudes right at
the long-life tail. This is the canonical "regression with sample
selection on the target" trap, well-documented in the survival
literature (Heckman 1979). For cycle-life prediction with right-
censoring, survival is the correct framework — not because it
predicts magnitudes better, but because it uses all the data.

---

## Files

| Path | Contents |
|---|---|
| `experiments/exp_a_feature_set/run.sh` | Driver for all 5 models × 2 feature subsets |
| `experiments/exp_a_feature_set/aggregate.py` | Walks `out/runs/`, builds metric_long.csv + headline.csv |
| `experiments/exp_a_feature_set/metric_long.csv` | 228 rows: every (model, fs, metric) tuple |
| `experiments/exp_a_feature_set/headline.csv` | Comparison table (10 rows) |
| `experiments/exp_a_feature_set/logs/*.log` | Per-run stdout/stderr |
| `experiments/exp_b_horizon_and_blend/run.sh` | Driver for rsf + xgb_aft × N |
| `experiments/exp_b_horizon_and_blend/blend.py` | z-score blend + survival_metrics recompute |
| `experiments/exp_b_horizon_and_blend/blend_summary_fs_cv.json` | Per-N blend results |
| `experiments/exp_b_horizon_and_blend/logs/*.log` | Per-run stdout/stderr |

## Code changes (additive — no existing files outside cell_lifetime/ touched)

| File | What changed |
|---|---|
| `src/cell_lifetime/data/loader.py` | New `fs_all` subset support (reads `role: feature` from manifest) |
| `src/cell_lifetime/preprocessing/target_transform.py` | Box-Cox inverse clip to prevent NaN |
| `src/cell_lifetime/models/xgb_aft.py` | `predict_cycle_life` clip [-50, 50] before exp |
| `src/cell_lifetime/models/ebm_regressor.py` | `interactions` upper bound 0..3, removed fixed `interactions=0` |
| `src/cell_lifetime/pipelines/validation.py` | `predictions.csv` artifact, `n_skipped_folds` user_attr |
| `tests/test_loader_fs_all.py` | New (3 tests) |

Test suite: **50/50 passing.**

---

## Open items for future work

| Item | Why deferred | What to try next |
|---|---|---|
| Weighted ensemble | Plan specified simple z-score blend; null result is the report's finding | Weight by out-of-fold C-index (likely 70/30 RSF/AFT) |
| Multi-baseline (b=2, b=3, b=4) | Only b=1 bundle exists today | Run `ml_label_preprocess/preprocess.py --baseline-cycle N` for N ∈ {2,3,4} |
| Per-tier feature analysis | Did fs_cv beat fs_all for RSF because of Tier-A (retention) or Tier-C (KWW) features? | Need additional sweeps: `fs_tier_a`, `fs_tier_b`, `fs_tier_c` |
| Calibration | Probabilities from xgb_classifier are uncalibrated | `CalibratedClassifierCV` wrapper, Brier score |
| Survival SHAP | sksurv RSF has no SHAP support; AFT SHAP is interpretation-fragile | Wait for upstream; permutation importance is sufficient for now |
| Multi-seed cohort-stratified scoring | per-cohort C-index has wide CIs due to 0MC scarcity (~20 faded cells) | Need either more 0MC data or a Bayesian cohort prior |

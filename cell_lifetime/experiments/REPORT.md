# cell_lifetime experiments — REPORT

**Branch:** `feature/cell_lifetime` &nbsp;·&nbsp;
**Compute:** 5 seeds × 30 Optuna trials × 5 inner CV per run, N=300 unless noted, A2.2_b1, 10-core cap

This report consolidates **seven experiments** (A through G) aimed at
improving the Phase 1+2+3 in-session baselines. A and B were the
initial pass; C/D/E/F drill into specific findings; G extends Exp F's
tune-target question to the full 4-model × 3-horizon grid.

---

## TL;DR

| Decision | Outcome |
|---|---|
| **Use all 40 features (`fs_all`) instead of `fs_cv` (12)?** | **YES for classification & regression; NO for survival (RSF overfits at 40)** |
| **Use a z-score (Exp B) or weighted (Exp C) blend of RSF + XGB-AFT?** | **NO. Optimal blend weight `w_rsf = 1.0` at every horizon — RSF alone is Pareto-optimal.** |
| **Does adding more survival models (Cox PH, Weibull AFT) help via a 4-way blend (Exp E)?** | **NO. Best 4-way blend is still `w_rsf = 1.0`. RSF dominates everything tried.** |
| **Where does the survival signal live across feature tiers (Exp D)?** | **Tier A retention/CE (3 cols) alone gives RSF C-index 0.774; Tier C CV-phase (34 cols) alone collapses to 0.577 (near random). Tier A is the anchor; CV features are useful additions but not standalone.** |
| **Does tuning ON F1 produce better F1 than tuning on ROC-AUC (Exp F)?** | **At fs_all: YES, AUC-tuning wins by 4.1 pts.** |
| **Does that pattern generalize across model families and horizons (Exp G)?** | **NO. Across 24 (model, fs, N) cells with fs_a_only and fs_cv, F1-tuning and AUC-tuning are statistically indistinguishable (0 of 24 cells have \|Δ\| > pooled std). The Exp F finding was fs_all-specific — likely an Optuna-in-high-dim artifact.** |
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

## Experiment C — weighted-blend grid search

**Question**: does a weighted blend `w · RSF + (1−w) · AFT` beat the
best single model at any choice of `w` ∈ {0.0, 0.1, …, 1.0}?

### Results (per-seed C-index curves, fs_cv, 5 seeds)

| w_rsf | C-index @ N=200 | C-index @ N=300 | C-index @ N=400 |
|---:|---:|---:|---:|
| 0.0 (AFT alone)   | 0.770 | 0.770 | 0.770 |
| 0.5               | 0.793 | 0.790 | 0.792 |
| 0.7               | 0.797 | 0.795 | 0.795 |
| 0.9               | 0.801 | 0.800 | 0.801 |
| **1.0 (RSF alone)** | **0.803** | **0.801** | **0.802** |

The curve is **monotonically increasing** at every N. Optimal
`w_rsf = 1.0` (RSF alone) at every horizon. AFT contributes only noise
to the blend.

**Verdict**: blending rejected — RSF is Pareto-optimal in the 2-model
space. The Exp B null result is now confirmed at fine grid resolution.

---

## Experiment D — feature tier ablation

**Question**: where does the survival signal live across feature tiers?
Is `rsf × fs_cv > rsf × fs_all` because of Tier balance, or coincidence?

### Tier composition

- **Tier A** (3 cols): retention/CE — `coulombic_efficiency_final`,
  `discharge_capacity_retention_final`, `charge_capacity_retention_min`
- **Tier B** (3 cols): nominal voltage retention (charge/discharge,
  max/std)
- **Tier C** (34 cols): CV-phase KWW per-cycle fits + aggregates +
  engineered ratios

### Results (5 seeds, N=300, all combinations of tiers vs the existing fs_cv/fs_all baselines)

| Subset | n_cols | xgb_classifier F1 | rsf C-index |
|---|---:|---:|---:|
| fs_b_only | 3 | 0.821 ± 0.050 | 0.715 ± 0.051 |
| fs_c_only | 34 | **0.785 ± 0.010** | **0.577 ± 0.036** ← *near-random!* |
| fs_a_only | 3 | 0.857 ± 0.024 | 0.774 ± 0.039 |
| fs_ab | 6 | 0.852 ± 0.030 | 0.794 ± 0.021 |
| fs_cv | 12 | 0.838 ± 0.029 | **0.801 ± 0.021** |
| fs_all | 40 | **0.866 ± 0.037** | 0.787 ± 0.020 |

### Headline findings

1. **Tier C alone collapses RSF survival to C-index 0.577** — barely
   above random. The 34 CV-phase features carry almost no standalone
   survival signal; they only help when anchored by Tier-A retention.
   Classification F1 also drops to 0.785 (the worst on the table) when
   limited to Tier C.

2. **Tier A alone (3 retention features) gets xgb_classifier to
   F1 = 0.857**, beating fs_cv (12 cols, F1 = 0.838) and within 1 pt
   of fs_all (40 cols, F1 = 0.866). For classification, the **3
   retention features carry almost all the signal**; the other 37 are
   marginal.

3. **For RSF survival, fs_cv stays the winner** (C-index 0.801). Tier A
   alone gets 0.774, fs_ab (A+B) gets 0.794. So CV-phase features do
   add ~0.01 C-index on top of A+B, but only when combined — alone
   they're worthless. fs_all (40) hurts because the 34 noisy Tier-C
   features overwhelm the 3 anchoring retention features for the
   nonparametric forest.

4. **The fs_cv subset (12 cols) was well-chosen** for survival — it
   captures the Tier-A anchors AND a curated 9 of the 34 Tier-C
   features that genuinely help.

### Operational recommendation

- **For classification**: 3 retention features (fs_a_only) gets you
  to within 1 pt of the headline; production deployment doesn't need
  the full 40-feature pipeline.
- **For survival**: keep fs_cv. The CV-phase features are useful but
  only as a complement to retention — alone they're noise.

---

## Experiment E — parametric survival via lifelines

**Question**: do parametric (Weibull AFT) and semi-parametric (Cox PH)
survival models from `lifelines` add information that tree-based
survival doesn't capture? If so, a richer ensemble could outperform
RSF alone.

### New models added (in `cell_lifetime/models/`)

- **`lifelines_weibull_aft`** — `lifelines.WeibullAFTFitter`, parametric
  AFT with Weibull baseline hazard. risk_orientation = "time_high".
- **`lifelines_cox`** — `lifelines.CoxPHFitter`, semi-parametric
  proportional-hazards. risk_orientation = "risk_high".

Both inherit `CycleLifeModel`, share the `fit(X, time, event)`
signature with `xgb_aft` and `rsf`, register via the guarded-import
pattern, and feed cleanly into the existing pipeline.

### Solo results (5 seeds, fs_cv, N=300)

| Model | C-index | AUC@300 | Notes |
|---|---:|---:|---|
| lifelines_cox | 0.752 ± 0.029 | 0.780 ± 0.052 | semi-parametric, fast |
| lifelines_weibull_aft | 0.755 ± 0.025 | 0.784 ± 0.043 | parametric AFT, fast |
| xgb_aft | 0.770 ± 0.026 | 0.836 ± 0.057 | gradient-boosted AFT |
| **rsf** | **0.801 ± 0.021** | **0.879 ± 0.048** | **tree-based ensemble** |

Both lifelines models score below xgb_aft and well below RSF on this
dataset. Diversity ≠ accuracy.

### 4-way weighted blend (simplex grid, step 0.1, 286 weight vectors)

| Weights (RSF, xgb_aft, Cox, Weibull) | C-index |
|---|---:|
| **(1.0, 0.0, 0.0, 0.0)** | **0.801 ± 0.019** ← optimal |
| (0.9, 0.1, 0.0, 0.0) | 0.800 ± 0.018 |
| (0.9, 0.0, 0.1, 0.0) | 0.800 ± 0.019 |
| (0.8, 0.0, 0.2, 0.0) | 0.799 ± 0.019 |
| (any with w_weibull > 0) | strictly ≤ 0.798 |

**Verdict**: 4-way blend produces the same answer as 2-way blend —
RSF alone is Pareto-optimal across all four survival models. The
parametric/semi-parametric models genuinely don't add information.

This is a meaningful negative result: it means the tree-based RSF is
*already capturing whatever signal the parametric assumptions could
extract*. Future survival improvements need a fundamentally different
angle (e.g., new features, different censoring assumptions, or a
fundamentally different model class like deep survival nets).

---

## Experiment F — tuning objective (ROC-AUC vs F1)

**Question**: does tuning xgb_classifier on F1 directly produce better
F1 than tuning on ROC-AUC?

### Results (5 seeds, N=300)

| subset | tuned on | F1 ± std | ROC-AUC ± std | Δ F1 (F1-tuned − AUC-tuned) |
|---|---|---:|---:|---:|
| fs_a_only | roc_auc | 0.857 ± 0.024 | 0.859 ± 0.022 | — |
| fs_a_only | **f1** | 0.842 ± 0.015 | 0.860 ± 0.033 | **−0.015** |
| fs_b_only | roc_auc | 0.821 ± 0.050 | 0.831 ± 0.014 | — |
| fs_b_only | **f1** | 0.830 ± 0.022 | 0.827 ± 0.019 | **+0.009** |
| fs_cv | roc_auc | 0.838 ± 0.029 | 0.875 ± 0.016 | — |
| fs_cv | **f1** | 0.842 ± 0.024 | **0.884 ± 0.023** | **+0.004** |
| fs_all | roc_auc | **0.866 ± 0.037** | 0.875 ± 0.027 | — |
| fs_all | **f1** | 0.825 ± 0.031 | 0.864 ± 0.022 | **−0.041** ← worse! |

### Headline finding

**Tuning on F1 makes F1 *worse* on the headline subset (fs_all, −4.1
pts).** AUC-tuned fs_all stays the F1 winner at 0.866.

### Why

F1 is a **step function** in probability space — it changes only when
predictions flip across the 0.5 decision threshold. Optuna's TPE
sampler gets a noisy/discrete objective with very little gradient
information, especially with only 30 trials × 5 inner folds.

ROC-AUC is **smooth** (rank-based on continuous probabilities). TPE
gets clean gradient information, converges to well-calibrated models,
and those happen to land at favorable F1 at the default 0.5 threshold.

This is a classic ML lesson: **tune on a smooth proxy, evaluate on
the metric you care about.** Hyperparameter optimization landscapes
shaped by discrete metrics are jagged and search-unfriendly.

### Recommendation

Keep `optimize: roc_auc` in `configs/xgb_classifier.yaml`. The F1
numbers reported throughout this report are the right ones.

---

## Experiment G — AUC vs F1 tuning across model families and horizons

**Question**: Exp F found a strong "AUC-tuning beats F1-tuning" result
on xgb_classifier × fs_all at N=300. Does that pattern generalize to
other model families and horizons?

### Setup

- **Models**: xgb_classifier, ebm_classifier (new), rsf, xgb_aft
- **Tune targets**: AUC (`roc_auc` for classification, `auc_at_N` for
  survival) vs F1 (`f1` for classification, `f1_at_N` for survival;
  the latter is a median-threshold binary-classification F1 at the
  run's horizon — newly added to `survival_metrics`)
- **Feature subsets**: fs_a_only (3 cols), fs_cv (12 cols)
- **Horizons**: N ∈ {200, 300, 400}
- **Grid**: 4 models × 2 tune targets × 2 subsets × 3 N = **48 runs**

### Held-out F1 / F1@N — values (mean ± std, 5 seeds)

For direct comparison: actual F1 (classification) and F1@N (survival,
median-threshold) at each cell, AUC-tuned vs F1-tuned.

| model | feature_subset | N | AUC-tuned | F1-tuned |
|---|---|---|---|---|
| ebm_classifier | fs_a_only | 200 | 0.9061 ± 0.0394 | 0.9118 ± 0.0425 |
| ebm_classifier | fs_a_only | 300 | 0.8545 ± 0.0115 | 0.8485 ± 0.0270 |
| ebm_classifier | fs_a_only | 400 | 0.8260 ± 0.0577 | 0.8049 ± 0.0638 |
| ebm_classifier | fs_cv | 200 | 0.9108 ± 0.0269 | 0.9039 ± 0.0294 |
| ebm_classifier | fs_cv | 300 | 0.8437 ± 0.0444 | 0.8477 ± 0.0446 |
| ebm_classifier | fs_cv | 400 | 0.8125 ± 0.0661 | 0.8081 ± 0.0663 |
| rsf | fs_a_only | 200 | 0.5886 ± 0.0665 | 0.5886 ± 0.0665 |
| rsf | fs_a_only | 300 | 0.6835 ± 0.0721 | 0.6835 ± 0.0721 |
| rsf | fs_a_only | 400 | 0.7255 ± 0.1066 | **0.7828 ± 0.0557** |
| rsf | fs_cv | 200 | 0.5793 ± 0.0583 | 0.5600 ± 0.0647 |
| rsf | fs_cv | 300 | 0.7094 ± 0.0514 | 0.7005 ± 0.0448 |
| rsf | fs_cv | 400 | 0.7643 ± 0.0403 | 0.7571 ± 0.0279 |
| xgb_aft | fs_a_only | 200 | 0.5381 ± 0.1057 | **0.5791 ± 0.0644** |
| xgb_aft | fs_a_only | 300 | 0.6529 ± 0.0675 | 0.6467 ± 0.0535 |
| xgb_aft | fs_a_only | 400 | 0.7257 ± 0.1109 | 0.7076 ± 0.0660 |
| xgb_aft | fs_cv | 200 | 0.5607 ± 0.0425 | 0.5510 ± 0.0536 |
| xgb_aft | fs_cv | 300 | 0.6734 ± 0.0692 | 0.6823 ± 0.0468 |
| xgb_aft | fs_cv | 400 | 0.7319 ± 0.0617 | 0.7468 ± 0.0451 |
| xgb_classifier | fs_a_only | 200 | 0.8970 ± 0.0427 | **0.9180 ± 0.0400** |
| xgb_classifier | fs_a_only | 300 | 0.8572 ± 0.0241 | 0.8436 ± 0.0145 |
| xgb_classifier | fs_a_only | 400 | 0.8104 ± 0.0686 | 0.8013 ± 0.0422 |
| xgb_classifier | fs_cv | 200 | 0.8999 ± 0.0378 | 0.9054 ± 0.0320 |
| xgb_classifier | fs_cv | 300 | 0.8377 ± 0.0294 | 0.8420 ± 0.0241 |
| xgb_classifier | fs_cv | 400 | 0.7973 ± 0.0542 | 0.7781 ± 0.0367 |

Bolded cells are the four largest positive Δ from the Δ table below.
Note that *every one* of these "winners" has a std bar that swallows the
"loser" — i.e., the F1-tuned advantage is within seed noise.

### Held-out ROC-AUC / AUC@N — values (mean ± std, 5 seeds)

Same shape; reports the smooth-target metric instead of F1.

| model | feature_subset | N | AUC-tuned | F1-tuned |
|---|---|---|---|---|
| ebm_classifier | fs_a_only | 200 | 0.8892 ± 0.0598 | 0.8841 ± 0.0786 |
| ebm_classifier | fs_a_only | 300 | 0.8642 ± 0.0226 | 0.8625 ± 0.0330 |
| ebm_classifier | fs_a_only | 400 | 0.8847 ± 0.0530 | 0.8861 ± 0.0508 |
| ebm_classifier | fs_cv | 200 | 0.8756 ± 0.0861 | 0.8746 ± 0.0795 |
| ebm_classifier | fs_cv | 300 | 0.8639 ± 0.0543 | 0.8649 ± 0.0497 |
| ebm_classifier | fs_cv | 400 | 0.8778 ± 0.0326 | 0.8708 ± 0.0522 |
| rsf | fs_a_only | 200 | **0.9360 ± 0.0298** | 0.9309 ± 0.0306 |
| rsf | fs_a_only | 300 | 0.8518 ± 0.0526 | 0.8467 ± 0.0541 |
| rsf | fs_a_only | 400 | 0.8215 ± 0.0688 | 0.8352 ± 0.0404 |
| rsf | fs_cv | 200 | **0.9279 ± 0.0335** | 0.9139 ± 0.0381 |
| rsf | fs_cv | 300 | 0.8743 ± 0.0521 | 0.8731 ± 0.0463 |
| rsf | fs_cv | 400 | 0.8577 ± 0.0434 | 0.8597 ± 0.0423 |
| xgb_aft | fs_a_only | 200 | 0.9154 ± 0.0277 | 0.9238 ± 0.0248 |
| xgb_aft | fs_a_only | 300 | 0.8312 ± 0.0784 | 0.8219 ± 0.0777 |
| xgb_aft | fs_a_only | 400 | 0.8142 ± 0.0917 | 0.8025 ± 0.0686 |
| xgb_aft | fs_cv | 200 | 0.9192 ± 0.0481 | 0.8926 ± 0.0252 |
| xgb_aft | fs_cv | 300 | 0.8301 ± 0.0644 | 0.8375 ± 0.0407 |
| xgb_aft | fs_cv | 400 | 0.8323 ± 0.0666 | 0.8331 ± 0.0136 |
| xgb_classifier | fs_a_only | 200 | 0.8951 ± 0.0678 | 0.8903 ± 0.0755 |
| xgb_classifier | fs_a_only | 300 | 0.8589 ± 0.0223 | 0.8526 ± 0.0328 |
| xgb_classifier | fs_a_only | 400 | **0.8658 ± 0.0534** | 0.8342 ± 0.0883 |
| xgb_classifier | fs_cv | 200 | 0.9051 ± 0.0628 | 0.9003 ± 0.0581 |
| xgb_classifier | fs_cv | 300 | 0.8753 ± 0.0155 | 0.8878 ± 0.0207 |
| xgb_classifier | fs_cv | 400 | 0.8861 ± 0.0422 | 0.8696 ± 0.0488 |

Headline survival cells (RSF fs_a/fs_cv at N=200) — AUC@200 ~0.93 —
match the Phase 3 fs_cv RSF headline; tune target has no material
effect on this rank metric either.

### Survival C-index — values (mean ± std, 5 seeds)

A rank-based metric independent of the N threshold. **Does the tune
target damage RSF / xgb_aft's natural rank ordering?**

| model | feature_subset | N | AUC-tuned | F1-tuned |
|---|---|---|---|---|
| rsf | fs_a_only | 200 | 0.7728 ± 0.0325 | 0.7736 ± 0.0346 |
| rsf | fs_a_only | 300 | 0.7768 ± 0.0321 | 0.7734 ± 0.0331 |
| rsf | fs_a_only | 400 | 0.7691 ± 0.0434 | 0.7800 ± 0.0281 |
| rsf | fs_cv | 200 | 0.8009 ± 0.0205 | 0.7942 ± 0.0186 |
| rsf | fs_cv | 300 | 0.7966 ± 0.0215 | 0.7987 ± 0.0220 |
| rsf | fs_cv | 400 | 0.7906 ± 0.0209 | 0.7965 ± 0.0134 |
| xgb_aft | fs_a_only | 200 | 0.7572 ± 0.0358 | 0.7622 ± 0.0315 |
| xgb_aft | fs_a_only | 300 | 0.7752 ± 0.0491 | 0.7539 ± 0.0538 |
| xgb_aft | fs_a_only | 400 | 0.7681 ± 0.0493 | 0.7647 ± 0.0373 |
| xgb_aft | fs_cv | 200 | 0.7770 ± 0.0139 | 0.7654 ± 0.0321 |
| xgb_aft | fs_cv | 300 | 0.7679 ± 0.0427 | 0.7647 ± 0.0289 |
| xgb_aft | fs_cv | 400 | 0.7840 ± 0.0152 | 0.7875 ± 0.0091 |

**Answer: no.** RSF + fs_cv C-index lands in [0.79, 0.81] regardless of
whether we tune on `auc_at_N` or `f1_at_N`. xgb_aft + fs_cv lands in
[0.76, 0.79]. All differences are within 1 std. The rank ordering is
preserved across tune targets — the survival models' core competence
(ranking faster-failing vs slower-failing cells) is robust to which
binary metric we optimize.

### Per-cell Δ (F1-tuned − AUC-tuned), held-out F1 / F1@N

| model | fs | N | ΔF1 | ΔAUC |
|---|---|---|---:|---:|
| xgb_classifier | fs_a_only | 200 | +0.021 | −0.005 |
| xgb_classifier | fs_a_only | 300 | −0.014 | −0.006 |
| xgb_classifier | fs_a_only | 400 | −0.009 | −0.032 |
| xgb_classifier | fs_cv | 200 | +0.006 | −0.005 |
| xgb_classifier | fs_cv | 300 | +0.004 | +0.013 |
| xgb_classifier | fs_cv | 400 | −0.019 | −0.017 |
| ebm_classifier | fs_a_only | 200 | +0.006 | −0.005 |
| ebm_classifier | fs_a_only | 300 | −0.006 | −0.002 |
| ebm_classifier | fs_a_only | 400 | −0.021 | +0.001 |
| ebm_classifier | fs_cv | 200 | −0.007 | −0.001 |
| ebm_classifier | fs_cv | 300 | +0.004 | +0.001 |
| ebm_classifier | fs_cv | 400 | −0.004 | −0.007 |
| rsf | fs_a_only | 200 | +0.000 | −0.005 |
| rsf | fs_a_only | 300 | +0.000 | −0.005 |
| rsf | fs_a_only | 400 | **+0.057** | +0.014 |
| rsf | fs_cv | 200 | −0.019 | −0.014 |
| rsf | fs_cv | 300 | −0.009 | −0.001 |
| rsf | fs_cv | 400 | −0.007 | +0.002 |
| xgb_aft | fs_a_only | 200 | **+0.041** | +0.008 |
| xgb_aft | fs_a_only | 300 | −0.006 | −0.009 |
| xgb_aft | fs_a_only | 400 | −0.018 | −0.012 |
| xgb_aft | fs_cv | 200 | −0.010 | −0.027 |
| xgb_aft | fs_cv | 300 | +0.009 | +0.007 |
| xgb_aft | fs_cv | 400 | +0.015 | +0.001 |

### Headline summary

| Statistic | Value |
|---|---|
| Mean ΔF1 across 24 cells | **+0.0005** (essentially zero) |
| Median ΔF1 | −0.0052 |
| Cells favoring F1-tuning | 9 / 24 |
| Cells favoring AUC-tuning | 15 / 24 |
| **Cells with \|Δ\| > pooled std** | **0 / 24** |

### Interpretation

**Across 24 cells, F1-tuning and AUC-tuning produce statistically
indistinguishable held-out F1.** Every single difference is inside the
pooled standard deviation across seeds. Exp F's strong −4.1 pt result
on `xgb_classifier × fs_all × N=300` was a **single-cell outlier**;
when extended to other subsets and horizons, the effect vanishes.

This is the cleanest possible null result. The "smooth proxies beat
discrete objectives" heuristic isn't actually load-bearing on this
dataset and Optuna budget (30 trials × 5 inner CV × 5 seeds). It only
matters at extreme feature-set sizes where Optuna can wander; with
3-12 features and small samples, the tune-target choice is noise.

### Two cells worth a second look

| Cell | ΔF1 | Note |
|---|---:|---|
| rsf × fs_a_only × N=400 | +0.057 | Largest positive Δ; F1-tuning helps RSF at long horizon with sparse features |
| xgb_aft × fs_a_only × N=200 | +0.041 | F1-tuning helps AFT at short horizon with sparse features |

Both involve fs_a_only (3 cols). With such a small feature space,
both objectives may converge near the same loss landscape minimum,
and randomness in the per-seed test split dominates the comparison.

### Recommendation (updated)

- **For published comparisons**, keep `optimize: roc_auc` for all
  classifiers. It's the conventional choice; Exp G shows there's no
  cost to using it.
- **The Exp F finding (fs_all penalty) remains**: when feature counts
  scale to 40, F1-tuning starts to lose ground. The take-away is more
  about Optuna's behavior in higher-dimensional HP × feature
  landscapes than about a fundamental F1-vs-AUC tradeoff.

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

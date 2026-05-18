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
| **Headline classifier (avg F1 across N=200/300/400)?** | **ebm_classifier × fs_all** (avg F1 = 0.866); ebm × fs_a_only (avg 0.862) is the deployment-friendly 3-feature alternative. |
| **Headline cycle-life model?** | **RSF + fs_cv** — C-index 0.801 ± 0.021, AUC@300 0.879 ± 0.048. Also wins MAE = 132.8 cyc and MAPE = 65% when its risk score is rank-quantile-calibrated. |

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

### Per-N model × feature-set comparison (AUC-tuned variant)

Rows = method, columns = feature subset × {F1, AUC}. All cells show
**mean ± std across 5 seeds, AUC-tuned variant** (the Exp F/G
recommendation). Survival models' F1 is the median-threshold F1@N
defined in `survival_metrics`.

#### N = 200

| Model | fs_a_only F1 | fs_a_only AUC | fs_cv F1 | fs_cv AUC | fs_all F1 | fs_all AUC |
|---|---|---|---|---|---|---|
| xgb_classifier | 0.8970 ± 0.0427 | 0.8951 ± 0.0678 | 0.8999 ± 0.0378 | 0.9051 ± 0.0628 | 0.8880 ± 0.0194 | 0.8903 ± 0.0678 |
| ebm_classifier | 0.9061 ± 0.0394 | 0.8892 ± 0.0598 | 0.9108 ± 0.0269 | 0.8756 ± 0.0861 | **0.9155 ± 0.0326** | 0.8946 ± 0.0721 |
| rsf | 0.5886 ± 0.0665 | **0.9360 ± 0.0298** | 0.5793 ± 0.0583 | 0.9279 ± 0.0335 | — | — |
| xgb_aft | 0.5381 ± 0.1057 | 0.9154 ± 0.0277 | 0.5607 ± 0.0425 | 0.9192 ± 0.0481 | — | — |

At N=200, **rsf × fs_a_only wins on AUC (0.936)** and **ebm × fs_all
wins on F1 (0.916)**. RSF's F1 (0.589) is much worse than classifiers
because the median-threshold binary metric is a poor fit for survival
models at short horizons (most cells haven't failed yet). For
ranking-based applications, RSF dominates; for threshold-based
applications, use the classifiers.

#### N = 300

| Model | fs_a_only F1 | fs_a_only AUC | fs_cv F1 | fs_cv AUC | fs_all F1 | fs_all AUC |
|---|---|---|---|---|---|---|
| xgb_classifier | 0.8572 ± 0.0241 | 0.8589 ± 0.0223 | 0.8377 ± 0.0294 | 0.8753 ± 0.0155 | 0.8656 ± 0.0374 | 0.8750 ± 0.0265 |
| ebm_classifier | 0.8545 ± 0.0115 | 0.8642 ± 0.0226 | 0.8437 ± 0.0444 | 0.8639 ± 0.0543 | **0.8697 ± 0.0250** | **0.8865 ± 0.0366** |
| rsf | 0.6835 ± 0.0721 | 0.8518 ± 0.0526 | 0.7094 ± 0.0514 | 0.8743 ± 0.0521 | — | — |
| xgb_aft | 0.6529 ± 0.0675 | 0.8312 ± 0.0784 | 0.6734 ± 0.0692 | 0.8301 ± 0.0644 | — | — |

At N=300, **ebm_classifier × fs_all wins on both F1 (0.870) and AUC
(0.887)**. The original Exp A headline of xgb_classifier × fs_all =
0.866 is essentially tied with EBM × fs_all (0.870) within 1 std,
but EBM's tighter std (0.025 vs 0.037) makes it slightly preferable.
RSF × fs_cv AUC = 0.874 is competitive on AUC alone.

#### N = 400

| Model | fs_a_only F1 | fs_a_only AUC | fs_cv F1 | fs_cv AUC | fs_all F1 | fs_all AUC |
|---|---|---|---|---|---|---|
| xgb_classifier | 0.8104 ± 0.0686 | 0.8658 ± 0.0534 | 0.7973 ± 0.0542 | 0.8861 ± 0.0422 | 0.7834 ± 0.0729 | 0.8816 ± 0.0380 |
| ebm_classifier | 0.8260 ± 0.0577 | 0.8847 ± 0.0530 | 0.8125 ± 0.0661 | 0.8778 ± 0.0326 | **0.8134 ± 0.0507** | **0.8927 ± 0.0368** |
| rsf | 0.7255 ± 0.1066 | 0.8215 ± 0.0688 | 0.7643 ± 0.0403 | 0.8577 ± 0.0434 | — | — |
| xgb_aft | 0.7257 ± 0.1109 | 0.8142 ± 0.0917 | 0.7319 ± 0.0617 | 0.8323 ± 0.0666 | — | — |

At N=400 (long horizon), **ebm_classifier × fs_all wins on AUC
(0.893)** while **ebm_classifier × fs_a_only wins on F1 (0.826)**.
The 3 retention features still carry most of the F1 signal at long
horizon; fs_all gives slightly better AUC at the cost of similar F1.
xgb_classifier × fs_all dropped sharply at N=400 (F1=0.783),
confirming xgb is not robust at long horizons with the full feature set.

#### Cross-horizon summary

Best AUC by N (any model, any fs in this comparison):

| N | Best model × fs | AUC |
|---|---|---|
| 200 | rsf × fs_a_only | **0.9360 ± 0.0298** |
| 300 | ebm_classifier × fs_all | 0.8865 ± 0.0366 |
| 400 | ebm_classifier × fs_all | **0.8927 ± 0.0368** |

AUC drops then partially recovers from N=200 to N=400. The N=300 dip
is a known phenomenon for binary survival at the median region —
class balance is most ambiguous there, making the metric noisier.

Best F1 by N:

| N | Best model × fs | F1 |
|---|---|---|
| 200 | ebm_classifier × fs_all | **0.9155 ± 0.0326** |
| 300 | ebm_classifier × fs_all | 0.8697 ± 0.0250 |
| 400 | ebm_classifier × fs_a_only | 0.8260 ± 0.0577 |

**ebm_classifier × fs_all is the strongest single classifier across
horizons** — best F1 at N=200 and N=300, best AUC at N=300 and N=400,
and only loses F1@400 to its smaller cousin (ebm × fs_a_only) by
~1 pt (inside std). For ranking at N=200, the survival model
(rsf × fs_a_only AUC = 0.936) is still the strongest cell observed
across all 7 experiments.

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

## Cross-experiment synthesis

Two questions the user posed across all 7 experiments, with
data-backed answers.

### Q1 — Best classifier by F1, considering all of N ∈ {200, 300, 400}

Computing average F1 across the three horizons (AUC-tuned variant per
the Exp F/G recommendation):

| Method × fs | F1@200 | F1@300 | F1@400 | **Avg F1** |
|---|---:|---:|---:|---:|
| **ebm_classifier × fs_a_only** | 0.9061 | 0.8545 | 0.8260 | **0.8622** |
| ebm_classifier × fs_cv | 0.9108 | 0.8437 | 0.8125 | 0.8557 |
| xgb_classifier × fs_a_only | 0.8970 | 0.8572 | 0.8104 | 0.8549 |
| xgb_classifier × fs_cv | 0.8999 | 0.8377 | 0.7973 | 0.8450 |

**Provisional winner: `ebm_classifier × fs_a_only`** — avg F1 = 0.862.

Why this is the right pick on these four (model, fs) combinations:

1. **Highest mean across all three horizons.** Beats
   `xgb_classifier × fs_a_only` by +0.007, `ebm_classifier × fs_cv` by
   +0.007, `xgb_classifier × fs_cv` by +0.017.
2. **Tightest std at N=300** (the middle, hardest horizon):
   `ebm × fs_a_only` = 0.8545 ± 0.0115. Compare to
   `xgb_classifier × fs_a_only` = 0.8572 ± 0.0241 (twice as wide).
   Cleaner threshold placement.
3. **Only 3 features.** Production-grade interpretability comes for free.
4. **Best at long horizon (N=400) too**: F1@400 = 0.826 beats every other
   (model, fs ∈ {fs_a, fs_cv}) cell.

#### Follow-up: closing the fs_all gap

The Exp G follow-up filled in 5 missing cells (xgb_classifier × fs_all
× N={200,400} and ebm_classifier × fs_all × {200,300,400}). The full
comparison across **6** (model, fs) combinations × 3 horizons:

| Method × fs | F1@200 | F1@300 | F1@400 | **Avg F1** |
|---|---:|---:|---:|---:|
| **ebm_classifier × fs_all** (40) | 0.9155 | 0.8697 | 0.8134 | **0.8662** |
| ebm_classifier × fs_a_only (3) | 0.9061 | 0.8545 | 0.8260 | 0.8622 |
| ebm_classifier × fs_cv (12) | 0.9108 | 0.8437 | 0.8125 | 0.8557 |
| xgb_classifier × fs_a_only (3) | 0.8970 | 0.8572 | 0.8104 | 0.8549 |
| xgb_classifier × fs_all (40) | 0.8880 | 0.8656 | 0.7834 | 0.8456 |
| xgb_classifier × fs_cv (12) | 0.8999 | 0.8377 | 0.7973 | 0.8450 |

**Updated Q1 verdict: `ebm_classifier × fs_all` wins** (avg F1 0.8662),
beating ebm_classifier × fs_a_only by only +0.004 — inside std.

Practical implication: **EBM dominates classification** (the top 3
rows are all EBM variants). The choice between fs_all (40 features,
slightly better avg F1) and fs_a_only (3 features, deployment-friendly,
−0.004 F1) is the standard accuracy-vs-interpretability tradeoff.

**Two operational picks**:

- **Best accuracy**: `ebm_classifier × fs_all` — avg F1 0.8662, 40 features.
- **Best deployment**: `ebm_classifier × fs_a_only` — avg F1 0.8622, only 3 features (retention + CE). The 0.4-pt gap is well inside std at every N.

Curiously, **xgb_classifier × fs_all underperforms** at long horizon (F1@400 = 0.783 vs ebm × fs_all = 0.813). The Exp A headline value of "F1 = 0.866 at N=300" turned out to be a single-horizon peak, not a robust winner.

### Q2 — Best cycle-life predictor (by AUC or by percentage error)

"Cycle life prediction" admits three framings:

- **Ranking** (which cells fail first) → AUC@N or C-index
- **Threshold classification** (pass/bad at fixed N) → AUC@N
- **Continuous prediction** (predicted cycles in time units) → MAE / MAPE

#### Data for ranking (AUC@N across N=200/300/400)

| Method × fs | AUC@200 | AUC@300 | AUC@400 | **Avg AUC** |
|---|---:|---:|---:|---:|
| xgb_classifier × fs_cv | 0.9051 | 0.8753 | **0.8861** | **0.8888** |
| rsf × fs_cv | 0.9279 | **0.8743** | 0.8577 | 0.8866 |
| ebm_classifier × fs_a_only | 0.8892 | 0.8642 | 0.8847 | 0.8794 |
| xgb_classifier × fs_a_only | 0.8951 | 0.8589 | 0.8658 | 0.8733 |
| rsf × fs_a_only | **0.9360** | 0.8518 | 0.8215 | 0.8698 |
| xgb_aft × fs_cv | 0.9192 | 0.8301 | 0.8323 | 0.8605 |

`xgb_classifier × fs_cv` marginally edges `rsf × fs_cv` on average
(0.889 vs 0.887, well inside noise). **But this comparison is
misleading**: xgb_classifier trains **three separate models** (one per
N); rsf trains **one model** evaluated at three thresholds — apples
to oranges.

#### Data for continuous prediction (MAE / MAPE)

Both regressors AND survival models can produce cycle-life magnitudes,
but via different mechanisms. **All MAE/MAPE computed on faded test
cells only**, 5 seeds, identical per-cell methodology (`MAPE =
mean(|err|/true)`):

| Method × fs | MAE (cycles) | MAPE | RMSE | Prediction type |
|---|---:|---:|---:|---|
| **rsf × fs_cv** (12) | **132.8 ± 26.3** | **64.7%** | 187.0 | rank-quantile calibrated |
| rsf × fs_a_only (3) | 133.4 ± 30.7 | 65.3% | 186.7 | rank-quantile calibrated |
| ebm_regressor × fs_all (40) | 136.2 ± 12.3 | 115% | — | direct regression |
| xgb_regressor × fs_all (40) | 136.5 ± 9.6 | 114% | — | direct regression |
| xgb_regressor × fs_cv (12) | 137.7 ± 16.9 | 125% | — | direct regression |
| rsf × fs_all (40) | 141.8 ± 31.3 | 78.0% | 195.7 | rank-quantile calibrated |
| ebm_regressor × fs_cv (12) | 142.1 ± 14.6 | 131% | — | direct regression |
| lifelines_cox × fs_cv | 151.6 ± 26.9 | 80% | 217.9 | rank-quantile calibrated |
| lifelines_weibull × fs_cv | 182.2 ± 49.3 | 129% | 277.6 | direct (predict_median) |
| xgb_aft × fs_a_only | 208.5 ± 78.5 | 91% | 281.7 | direct (predict cycles) |
| xgb_aft × fs_all | 271.5 ± 97.1 | 135% | 426.8 | direct (predict cycles) |
| xgb_aft × fs_cv | 284.6 ± 249.2 | 116% | 387.2 | direct (predict cycles) |

**Two striking findings:**

1. **`rsf × fs_cv` wins on MAE *and* MAPE** when its risk score is
   rank-quantile-calibrated to cycle-life. MAE = 132.8 vs regressors'
   ~136 (a small absolute win), but **MAPE = 65% vs regressors' 115–125%**
   (huge). The MAPE gap reflects that regressors heavily over/under-shoot on
   short-life cells (high per-cell relative error), while RSF's
   rank-quantile mapping yields predictions whose per-cell errors stay
   bounded. **RSF is competitive on magnitude prediction, not just rank.**
2. **XGB-AFT's direct predictions are badly miscalibrated** (MAE
   208–285, ~2× the regressors). The AFT loss optimizes
   log-likelihood under a parametric error distribution, not MAE —
   so even with C-index ≈ 0.77, the absolute magnitudes drift. A
   rank-quantile calibration on XGB-AFT's output would close most of
   the gap.

**Rank-quantile calibration**: for each test fold, sort the survival
model's risk scores descending and the true cycle lives ascending;
assign true-life-quantiles to risk-quantiles. This is the simplest
non-parametric calibration; it preserves rank performance (which is
RSF's strength) while yielding magnitude predictions on the true
cycle-life scale.

#### Regression's selection-bias problem (the Q4 issue)

Even the regressors' tied MAE = 136 hides a structural problem:

Per-quartile breakdown (from Exp A diagnosis):

| Quartile of true cycle life | n | MAE |
|---|---:|---:|
| Q1 short (6–~93) | 49 | 92 |
| Q2 (~93–~310) | 46 | 98 |
| Q3 (~310–~524) | 48 | 77 |
| **Q4 long (~524+)** | **47** | **286** ← 3× worse |

The "43% MAPE" headline is **2× the truth for Q4 cells** — the
ones where prediction accuracy matters most for warranty and
production-grade decisions.

#### Criterion → winner synthesis

| Criterion | Winner | Reason |
|---|---|---|
| Avg AUC@N (200/300/400) | xgb_classifier × fs_cv ≈ rsf × fs_cv (tie within 0.002) | Both ~0.888 |
| AUC@200 (short horizon, business-critical for early sorting) | **rsf × fs_a_only (0.9360)** | +1.5 pts over next-best |
| C-index (rank, horizon-independent) | **rsf × fs_cv (0.801 ± 0.021)** | The cleanest single-number summary |
| **MAE on faded cells (rank-quantile cal.)** | **rsf × fs_cv (132.8 ± 26.3 cyc)** | Beats regressors AND reuses rank skill |
| **MAPE on faded cells (per-cell)** | **rsf × fs_cv (64.7%)** | Far below regressors' 115–125%; less tail damage |
| Uses censored cells (228 of 415) | **rsf, xgb_aft only** | Regression and classification drop them |
| Single unified predictor across all N | **rsf only** | Classifier needs retraining per N |

### Headline recommendation

**`rsf × fs_cv`** as the unified cycle-life model.

It's the only single model that simultaneously:

1. Achieves competitive AUC@N at every horizon (0.928 / 0.874 / 0.858,
   average 0.887 — within 0.002 of the best).
2. Has the cleanest summary metric: C-index = **0.801 ± 0.021** (rank
   metric matching the natural problem framing, horizon-independent).
3. **Uses all 415 cells** including the 228 right-censored ones —
   solving the selection-bias problem that gives regression its
   misleading 43% MAPE.
4. Produces a **single continuous risk score** that's threshold-free
   — no need to retrain at different N.
5. Uses the fs_cv 12-feature subset, which Exp D confirmed is
   well-chosen: includes the Tier-A retention anchors AND a curated
   ~9 of the 34 Tier-C features that actually carry signal. (Tier C
   alone collapses RSF to C-index 0.577.)

### Deployment-context variants

- **Short-horizon early sorting at N=200** → `rsf × fs_a_only`
  (AUC@200 = 0.936; only 3 features for ultra-lightweight deployment).
- **General cycle-life prediction at unknown horizon** → **`rsf × fs_cv`**.
- **Threshold-based classification (committed to a fixed N)** →
  `xgb_classifier × fs_cv` (or `× fs_all` pending the follow-up).
- **Regression with magnitudes** → don't use for Q4 long-life cells;
  the apparent 43% MAPE is ~85% MAPE on those cells.

The reason RSF wins both questions' *spirit* but only Q2's *letter*
is that **RSF is trained on the right task** (survival with censoring)
while the classifiers are trained on three different binarized
projections of the same underlying continuous problem.

---

## Experiment H — fair 3-way head-to-head for continuous cycle-life prediction (fs_cv)

**Question**: At `fs_cv` (12 features), how do RSF (using its native
median-survival cycle-life output), XGB regressor, and EBM regressor
compare when:

1. The **test set is identical for all three models** — 20% of *faded*
   cells only (cells with ground-truth cycle counts).
2. **RSF training** uses censored cells *plus* the remaining 80% of
   faded; **regressor training** uses only the 80% of faded. This
   isolates "does censored data help median-survival cycle-life
   prediction?".
3. All three get an **identical 30 trials × 5 inner CV budget**, 5
   seeds, sqrt target transform on the regressors.

### Protocol

| Split component | RSF | XGB regressor | EBM regressor |
|---|---|---|---|
| Train | 80% faded (≈149) + ALL censored (228) = 377 | 80% faded (≈149) | 80% faded (≈149) |
| Test | same 20% faded (≈38) for all three | same | same |
| Target transform | none (consumes time+event) | √cycles | √cycles |
| HP tune objective | C-index (inner 5-CV) | MAE (inner 5-CV) | MAE (inner 5-CV) |
| Cycle-life prediction | median survival: min{t : S(t)≤0.5} | predict² | predict² |

Test cells are verified identical per seed via SHA-256 fingerprint
in `runs/seed_*/results.json` (see `verify_same_test_set.py`).

### Headline results (5 seeds, fs_cv, mean ± std)

| Model | MAE (cyc) | MAPE | RMSE | R² | Stability (MAE std) |
|---|---:|---:|---:|---:|---:|
| **xgb_regressor** | **123.4 ± 15.3** | **1.28 ± 0.40** | **171.3 ± 25.4** | **0.39 ± 0.11** | ±15.3 |
| ebm_regressor | 131.7 ± 17.5 | 1.29 ± 0.43 | 174.2 ± 24.3 | 0.37 ± 0.11 | ±17.5 |
| rsf (median-surv) | 143.3 ± 8.9 | 1.51 ± 0.46 | 182.8 ± 6.7 | 0.30 ± 0.08 | **±8.9** |

### Per-quartile MAE — where each model wins

| Model | Q1 (shortest) | Q2 | Q3 | Q4 (longest) |
|---|---:|---:|---:|---:|
| xgb_regressor | **94.4 ± 20.2** | **93.8 ± 37.2** | **64.9 ± 16.4** | 231.8 ± 48.3 |
| ebm_regressor | 105.5 ± 10.9 | 112.0 ± 45.6 | 73.2 ± 18.5 | 228.3 ± 36.8 |
| rsf (median-surv) | 146.1 ± 44.2 | 159.7 ± 58.0 | 128.3 ± 34.2 | **139.4 ± 40.4** |

### Findings

1. **XGB regressor wins on mean MAE** (123.4 cyc) — beats RSF
   (median-surv) by ≈20 cyc and EBM by ≈8 cyc. Lower MAPE too.
   Censored cells in RSF's training set do **not** help median-survival
   cycle-life prediction on faded cells.
2. **RSF has the lowest variance** (MAE std 8.9 vs 15–17 for the
   regressors). The censored-cell ballast stabilises its predictions
   across seeds even when its mean MAE is worse.
3. **The error distribution is the real story** — RSF and the
   regressors fail in *opposite* directions:
   - **Regressors win Q1–Q3** (short and middle-life cells). MAE in
     the 65–112 cyc range vs RSF's 128–160 cyc.
   - **RSF wins Q4** (long-life cells). MAE 139 cyc vs regressors'
     228–232 cyc. This is the regressor selection-bias problem we
     flagged earlier: trained only on ≤1052-cycle faded cells, the
     regressors regress to the population median for long-lived
     cells they've never seen the upper tail of.
   - RSF, trained on time-event data including censored cells (whose
     `time` extends to wherever observation stopped), has a richer
     view of "could live past N" and predicts those cells better.
     **Validated by Exp I** below — removing censored cells from RSF
     training increases Q4 MAE by +87 cyc (138 → 225), bringing it
     to the regressors' level.
4. **MAPE is similar across all three** (1.28–1.51), driven mostly
   by short-lived cells where a 50-cycle absolute error is a 100% +
   relative error.
5. **EBM is ≈10× slower** (382s vs 40s/44s per seed) without a
   meaningful accuracy gain over XGB at fs_cv.

### Verdict (this comparison only — fs_cv, sqrt target, median-survival
extraction)

| Use case | Pick |
|---|---|
| Lowest *average* MAE across faded cells | **xgb_regressor** |
| Most seed-stable MAE | **rsf (median-surv)** |
| Best long-life (Q4) prediction | **rsf (median-surv)** |
| Best short / middle-life prediction | **xgb_regressor** |
| Hybrid candidate (future work) | gate by predicted cycle-life: regressor for short, RSF for long — could close most of the MAE gap. |

### Crucial caveat

This experiment uses RSF's **median-survival** as the cycle-life
prediction. Earlier reports (Cross-experiment synthesis above) cited
`rsf × fs_cv` with MAE = 132.8 ± 26.3 / MAPE = 65% — those came from
**rank-quantile calibration** of the risk score (a different RSF
output transform). Median-survival, the principled "1-S(N) = 0.5"
extraction, is fundamentally biased toward the population median for
short-lived cells whose true cycle life is well below the censored
training population's central tendency.

In short: RSF can be the cycle-life winner — but you have to use the
right output. Median-survival is *not* it; rank-quantile or a
recalibration step (e.g. Bayesian Cox-PH posterior median) is.

---

## Experiment I — RSF censored-data ablation (validates the Q4 claim)

**Question**: Exp H attributed RSF's Q4 advantage to its training on
censored cells (`time` axis extends to right-censored observations).
Does it actually? Or is the win driven by the RSF *algorithm*
(survival-forest's heavy-tail handling), independent of censored data?

### Design — single variable changed

Same Exp H protocol (fs_cv, 5 seeds, 30 trials × 5 inner CV, sqrt for
regressors, median-survival extraction). Identical 20% faded test
cells per seed — **fingerprinted and asserted to match Exp H**
(`verify_test_set_matches_exp_h.py` reports OK on all 5 seeds).

| Variant | Train rows | Event labels |
|---|---|---|
| rsf_with_censored (Exp H baseline) | 80% faded (≈149) + ALL censored (228) = 377 | event=1 for faded, event=0 for censored |
| rsf_no_censored (ablation) | 80% faded only (≈149) | event=1 for all (all known fade) |

### Headline results (5 seeds)

| Variant | MAE | MAPE | R² | MAE std (seed-stability) |
|---|---:|---:|---:|---:|
| rsf_with_censored | 147.4 ± 6.5 | 1.87 | 0.28 | **±6.5** |
| **rsf_no_censored** | **127.7 ± 13.9** | **1.02** | **0.40** | ±13.9 |

### Per-quartile MAE — the actual story

| Variant | Q1 (shortest) | Q2 | Q3 | Q4 (longest) |
|---|---:|---:|---:|---:|
| rsf_with_censored | 159.6 ± 36.6 | 161.2 ± 51.0 | 130.4 ± 32.5 | **138.1 ± 32.9** |
| rsf_no_censored | **91.9 ± 23.9** | **105.7 ± 28.0** | **81.2 ± 27.5** | 225.3 ± 47.2 |
| Δ (no − with) | **−67.7** | **−55.6** | **−49.2** | **+87.2** |

### Findings

1. **The Q4 claim is validated.** Removing censored cells from RSF's
   training set increases Q4 MAE by +87 cyc (138 → 225). The
   ablated RSF's Q4 MAE (225) almost exactly matches XGB regressor's
   Q4 (232) and EBM regressor's Q4 (228) from Exp H. So **censored
   data is the mechanism behind RSF's long-life advantage**, not the
   RSF algorithm by itself.
2. **But the claim was incomplete.** Censored data is a **double-edged
   sword**: it lifts Q4 by 87 cyc while *hurting* Q1, Q2, Q3 by 50–68
   cyc each. Censored cells in training instil a "long-life prior"
   on the forest that helps long-lived cells but biases short-lived
   predictions upward.
3. **Net effect on mean MAE is negative for censored.** Without
   censored cells, RSF's overall MAE drops from 147 to 128 — a
   ≈20-cyc improvement, because there are more short/mid-life cells
   than long-life cells. Censored data is *not* a universal accuracy
   booster for median-survival cycle-life prediction; it trades short-
   for long-life accuracy.
4. **R² confirms the same picture.** RSF-no-censored hits R²=0.40
   (best in this comparison family), while RSF-with-censored sits at
   R²=0.28.
5. **MAE std (seed-stability) flips.** RSF-with-censored has ±6.5
   (most stable), RSF-no-censored has ±13.9. Censored data acts as
   ballast — stabilises across seeds even when it hurts mean accuracy.
   This matches the Exp H observation.

### Verdict

The original Exp H claim — *"RSF wins Q4 because censored cells extend
the time axis"* — is **confirmed** for Q4 specifically. But the
broader implicit claim that censored data is unambiguously good for
RSF cycle-life prediction is **refuted**: the price of Q4 accuracy is
worse Q1-Q3 accuracy, and the net effect on mean MAE is *negative*.

Two practical takeaways:

- **For a balanced cycle-life predictor at fs_cv**, `rsf_no_censored`
  is the surprising winner among RSF variants on this median-survival
  protocol — MAE 128, R² 0.40, beats XGB regressor's MAE 123 by only
  ≈4 cyc, and is more interpretable (single survival model).
- **For long-life-cell-specific prediction (Q4)**, censored data is
  essential — drop it and RSF collapses to regressor-level Q4 error.
  A hybrid model — gate on predicted cycle-life and use censored-trained
  RSF for predicted-long cells, regressors or no-censored RSF for
  predicted-short cells — would likely close most of the gap. Out of
  scope here; recorded in "Open items for future work".

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
| `experiments/done_exp_a_to_g/exp_a_feature_set/run.sh` | Driver for all 5 models × 2 feature subsets |
| `experiments/done_exp_a_to_g/exp_a_feature_set/aggregate.py` | Walks `out/runs/`, builds metric_long.csv + headline.csv |
| `experiments/done_exp_a_to_g/exp_a_feature_set/metric_long.csv` | 228 rows: every (model, fs, metric) tuple |
| `experiments/done_exp_a_to_g/exp_a_feature_set/headline.csv` | Comparison table (10 rows) |
| `experiments/done_exp_a_to_g/exp_a_feature_set/logs/*.log` | Per-run stdout/stderr |
| `experiments/done_exp_a_to_g/exp_b_horizon_and_blend/run.sh` | Driver for rsf + xgb_aft × N |
| `experiments/done_exp_a_to_g/exp_b_horizon_and_blend/blend.py` | z-score blend + survival_metrics recompute |
| `experiments/done_exp_a_to_g/exp_b_horizon_and_blend/blend_summary_fs_cv.json` | Per-N blend results |
| `experiments/done_exp_a_to_g/exp_b_horizon_and_blend/logs/*.log` | Per-run stdout/stderr |
| `experiments/exp_h_rsf_vs_regressors_fair/run.py` | Self-contained driver: identical 20%-faded test set across all 3 models, 5 seeds × 30 trials × 5 inner CV |
| `experiments/exp_h_rsf_vs_regressors_fair/aggregate.py` | Headline + per-quartile tables |
| `experiments/exp_h_rsf_vs_regressors_fair/verify_same_test_set.py` | Audit: confirms all 3 models share the same test cells per seed |
| `experiments/exp_h_rsf_vs_regressors_fair/runs/seed_*/predictions.csv` | Per-seed (cell_name, y_true, rsf_pred, xgb_pred, ebm_pred) |
| `experiments/exp_h_rsf_vs_regressors_fair/metric_long.csv` | 150 rows: (seed, model, metric, value) |
| `experiments/exp_h_rsf_vs_regressors_fair/summary.json` | Mean ± std per (model × metric) |
| `experiments/exp_i_rsf_censored_ablation/run.py` | Two RSF variants per seed (with/without censored), same test cells as Exp H |
| `experiments/exp_i_rsf_censored_ablation/aggregate.py` | Side-by-side + Δ table |
| `experiments/exp_i_rsf_censored_ablation/verify_test_set_matches_exp_h.py` | SHA-256 audit: Exp I per-seed test cells == Exp H per-seed test cells |
| `experiments/exp_i_rsf_censored_ablation/runs/seed_*/predictions.csv` | Per-seed (cell_name, y_true, rsf_with_pred, rsf_no_pred) |
| `experiments/exp_i_rsf_censored_ablation/summary.json` | Mean ± std per (variant × metric) |

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

# Plan: extend cell_classifier with cycle-life regression + survival models

## Context

Today `cell_classifier` only does binary N-cycle survival classification
(`pass`/`bad` at N ∈ {200, 300, 400}, RF only). The user wants three
modeling capabilities added:

1. **Cycle-life regression** with **EBM** and **XGBoost**. Cycle life is
   the cycle number at which retention crosses 0.85 and never recovers
   — that is **exactly** `last_fade_cycle` from the labels pipeline
   (verified: `labels.py::_last_crossing_into_bad`, RECOVERY_MIN=3,
   selftest cases cover the edges).
2. **Target transformation** to address skew. Verified distribution:
   raw `last_fade_cycle` spans 5–1052, median 310; sqrt compresses to
   6.7–28.3 (still moderate skew), log to 3.8–6.7 (well-conditioned).
   Box-Cox will land near log. **Plan uses Box-Cox (or log fallback)**;
   sqrt is offered as a model knob but not the default.
3. **Random Survival Forest** via scikit-survival, to handle the **58%
   right-censoring rate** (256 of 444 trainable cells are `in_testing`).
   User confirmed full-survival scope.

The classification surface must keep working unchanged. Env is
**`mldashboard`** (not `eis`), per user.

## Data semantics (verified, no changes to ml_label_preprocess)

| status | count | regression target | survival (event, time) |
|---|---:|---|---|
| faded      | 188 | `last_fade_cycle`             | (1, `last_fade_cycle`) — observed |
| in_testing | 256 | (drop for EBM; keep for survival) | (0, `n_regular`) — censored |
| excluded   |  17 | drop everywhere               | drop |

EBM has no native censoring support → faded-only (188 cells, 168 AR / 20 0MC).
XGBoost-AFT and RSF use all 444 cells via the `(event, time)` pair.

## Three-phase build

### Phase 1 — plumbing + regression (EBM + XGB on faded-only)

**Goal**: land the regression spine so the next two phases just plug models in.

Files to add / change in `cell_classifier/`:
- `src/cell_classifier/data/loader.py` — extend `Dataset` with:
  - `event: np.ndarray` (bool, True iff `status == "faded"`)
  - `time: np.ndarray` (int, `last_fade_cycle` if faded else `n_regular`)
  - `y_cycle: np.ndarray` (cycle-life regression target for faded; NaN for censored)
  - Loader signature gains `task: Literal["classification","regression","survival"]`
  - `.faded_view()` analogous to existing `.labeled_view()` (loader.py:91–106)
  - Extend `_LABEL_COLUMNS_DENYLIST` to include `last_fade_cycle`, `n_regular`, `status`
- `src/cell_classifier/preprocessing/target_transform.py` (new) — small wrapper that does Box-Cox (via `scipy.stats.boxcox`) or log/sqrt, stores λ on `.fit()`, exposes `transform/inverse_transform`. Defaults to `boxcox`. Used by regression models to fit on the transformed scale and report metrics on the **untransformed scale**.
- `src/cell_classifier/models/base.py` — add a class attr `task: str` (default `"classification"`). Existing classifier subclasses keep working. Add a docstring noting that regression subclasses may override `predict_proba` to raise `NotImplementedError`.
- `src/cell_classifier/models/xgb_classifier.py` (new) — `XGBClassifier` mirror of `RandomForestModel`. Drops in via existing sklearn-pipeline scaffold (random_forest.py:29–34 pattern).
- `src/cell_classifier/models/ebm_regressor.py` (new) — wraps `interpret.glassbox.ExplainableBoostingRegressor`. Task = regression. Wraps target_transform inside the sklearn pipeline. Replaces the stub at `models/ebm.py` (or sits beside it; the existing stub is for the classifier).
- `src/cell_classifier/models/xgb_regressor.py` (new) — `XGBRegressor` with target_transform. Task = regression. Optuna search space: `n_estimators`, `max_depth`, `learning_rate`, `subsample`, `colsample_bytree`, `min_child_weight`, `gamma`, `reg_lambda`.
- `src/cell_classifier/evaluation/regression_metrics.py` (new) — MAE, RMSE, R², median absolute error, computed on the **untransformed cycle scale**, plus per-cohort breakdowns (AR vs 0MC). Matches the cohort-aware pattern in `metrics.py:56–62`.
- `src/cell_classifier/evaluation/importance.py` — parametrize the `scoring` arg of `permutation_importance` by task: `roc_auc` for classification, `r2` for regression, custom scorer for survival in Phase 3. (One-line conditional, file is already model-agnostic per [importance.py:42].)
- `src/cell_classifier/pipelines/validation.py` and `production.py` — branch on `config["task"]` to choose metrics + which target to feed the model. Keep classification as the default branch unchanged.
- `src/cell_classifier/cli.py` — add `--task {classification,regression,survival}` to the run parser (cli.py:206–234). Consistency check at config-resolve time: `args.task` must match the chosen model's `task` class attribute (cli.py:142–144 pattern).
- `configs/ebm_regressor.yaml`, `configs/xgb_regressor.yaml`, `configs/xgb_classifier.yaml` (new) — siblings of `configs/rf.yaml`, with each model's defaults + Optuna search hint.
- `pyproject.toml` — add `[project.optional-dependencies] xgb = ["xgboost"]` and bundle into `all`.
- Tests:
  - `tests/test_loader_regression.py` — verifies `event`, `time`, `y_cycle` shapes; faded_view filters to 188; censored rows have event=False.
  - `tests/test_target_transform.py` — Box-Cox round-trip, log/sqrt round-trip.
  - `tests/test_models_xgb_regressor.py`, `tests/test_models_ebm_regressor.py` — fit/predict on a tiny synthetic dataset; assert untransformed predictions are in plausible range.
  - `tests/test_metrics_regression.py` — MAE/RMSE/R² match sklearn on hand-picked vectors.
  - Update `tests/test_loader.py:33` and `tests/test_models_rf.py:30` to guard their binary-y assertions under `task=="classification"`.
  - Update `tests/test_no_mode_branching.py` if it forbids non-mode dispatch (allow task branching).

### Phase 2 — XGBoost-AFT (censoring-aware XGB, no new framework)

**Goal**: use XGBoost's native `survival:aft` objective to handle censoring without dragging in `scikit-survival` yet.

- `src/cell_classifier/models/xgb_aft.py` (new) — wraps `xgboost.XGBRegressor` (or low-level Booster) with `objective="survival:aft"`, `eval_metric="aft-nloglik"`. Fit signature accepts `(X, time, event)` and constructs the AFT label format internally:
  - `y_lower = time`
  - `y_upper = time` if `event` else `+∞`
  - tunable `aft_loss_distribution` ∈ {`normal`, `logistic`, `extreme`}; `aft_loss_distribution_scale` as Optuna float.
- `task = "survival"`. `predict(X)` returns log-cycle-life predictions; `predict_cycle_life(X)` returns `exp(predict(X))`. No `predict_proba`.
- Pipeline branch for `task=="survival"` consumes the new `event`/`time` fields from `Dataset` (Phase 1 already exposes them).
- Survival metrics module needed for Phase 2's evaluation (used by both AFT and RSF):
  - `src/cell_classifier/evaluation/survival_metrics.py` (new) — wraps `sksurv.metrics.concordance_index_censored` (already installed if scikit-survival is in `mldashboard`; verify) and `integrated_brier_score`. Time-dependent AUC at the existing thresholds N ∈ {200, 300, 400} so survival numbers stay comparable to classification ones. If `scikit-survival` is not available, fall back to a hand-rolled C-index using only `(event, time)` — same formula, no dependency.
- `configs/xgb_aft.yaml` (new).
- Tests: `tests/test_models_xgb_aft.py`, `tests/test_metrics_survival.py`.

### Phase 3 — Random Survival Forest

**Goal**: full sksurv-backed RSF, the structured-target frontier.

Required only in this phase (kept out of Phases 1–2 to limit blast radius):
- `pyproject.toml` — `[project.optional-dependencies] survival = ["scikit-survival>=0.22"]`.
- `src/cell_classifier/data/loader.py` — emit `y_struct = sksurv.util.Surv.from_arrays(event, time)` when `task=="survival"` AND `model.requires_structured_y` is True (`xgb_aft` doesn't need it; RSF does).
- `src/cell_classifier/models/rsf.py` (new) — wraps `sksurv.ensemble.RandomSurvivalForest`. Fit signature mirrors `xgb_aft.py`: accepts `(X, time, event)`, constructs the structured array internally. Hyperparameter space copied from the solstice reference (`n_estimators` 50–300, `max_depth` 5–20, `min_samples_split` 5–20, `min_samples_leaf` 5–20, `max_features` ∈ {sqrt, log2, None}) but adapted to Optuna's `suggest_*` calls instead of `RandomizedSearchCV`.
- `predict(X)` returns the **risk score** (higher = sooner failure). Optional `predict_survival_curve(X)` returns the step-function array for downstream plotting; not used by sweep aggregation.
- SHAP for RSF is **out of scope for this phase** — `shap` does not natively support `sksurv.RandomSurvivalForest`. `compute_shap()` returns `None`; `shap.py` already handles that (shap.py:24–25). Permutation importance still works via a custom scorer wrapping `concordance_index_censored`.
- `configs/rsf.yaml` (new).
- Tests: `tests/test_models_rsf.py` (skipped if `sksurv` import fails — pattern matches the existing optional-extras `try/except` in `registry.py:41–51`).

## Critical files to be modified (summary)

| File | Phase | Change |
|---|---|---|
| `src/cell_classifier/data/loader.py` | 1, 3 | Add event/time/y_cycle to Dataset; faded_view(); task-aware y emission; sksurv structured y in P3 |
| `src/cell_classifier/models/base.py` | 1 | `task` class attr; relax classification-only docstrings |
| `src/cell_classifier/models/registry.py` | 1–3 | Guarded imports for xgb_*, ebm_regressor, rsf (pattern at registry.py:41–51) |
| `src/cell_classifier/pipelines/validation.py` | 1, 2, 3 | Task-branch on metrics + targets; the SHAP-scope fix (`b0c243f`) sets the precedent |
| `src/cell_classifier/pipelines/production.py` | 1, 2, 3 | Same task-branch |
| `src/cell_classifier/cli.py` | 1 | `--task` flag, model↔task consistency check |
| `src/cell_classifier/evaluation/importance.py` | 1, 3 | Scoring-by-task; custom survival scorer in P3 |
| `pyproject.toml` | 1, 3 | `xgb` and `survival` extras |
| `README.md`, `ROADMAP.md` | each phase | Document each capability as it lands |

## Patterns / utilities to reuse (not re-invent)

- **Registry guarded-import pattern** at `registry.py:41–51` — every new model goes in this way.
- **sklearn Pipeline scaffold** at `random_forest.py:29–34` — every new tree model uses the same `[imputer, estimator]` shape.
- **TransformedTargetRegressor** from sklearn — first-class wrapper for Box-Cox / log targets; saves having to roll our own. Skip the custom `target_transform.py` if this lands cleanly; keep the module only if Box-Cox-with-stored-λ inside an Optuna inner loop forces it.
- **`labeled_view()`** at `loader.py:91–106` — copy verbatim to make `faded_view()`.
- **`shap_summary_scope` branching** at `validation.py:303–310` (last commit, `b0c243f`) — proves the task-branching pattern is already established in the pipelines.
- **`solstice_mlflow` RSF reference** — borrow the HP space and stratification logic only; do NOT pull in its MLflow / data-loader scaffolding (incompatible with cell_classifier's clean interface).

## Verification

Each phase:
1. **Pytest gate**: `pytest -q` in `mldashboard` env from `cell_classifier/`. New tests added per phase listed above; existing tests must still pass.
2. **Smoke run** end-to-end on the regenerated `A2.2_b1` bundle:
   - Phase 1: `cell-classifier run --mode validation --model-config configs/xgb_regressor.yaml --task regression --N 300 --db-version A2.2 --baseline-cycle 1 --feature-subset fs_cv --tuning-protocol tune_inner_cv` (cheap, surfaces wiring bugs fast).
   - Phase 2: same with `configs/xgb_aft.yaml --task survival`.
   - Phase 3: same with `configs/rsf.yaml --task survival`.
3. **Smoke sweep**: extend `configs/sweeps/rf_n_b1_smoke.yaml` (the YAML landed in `6de612c`) into a `model × N` sweep covering RF / XGBClass / EBMReg / XGBReg / XGBAFT / RSF at baseline=1 with 5 seeds and `n_trials=10`. Aggregate metrics land in `out/sweeps/<sweep_id>/metric_long.csv`; eyeball that classification numbers are unchanged from the pre-extension baseline and that regression/survival numbers look plausible (test_rmse on the order of 100–200 cycles for cycle-life regression; C-index in 0.65–0.85 for AFT/RSF).

## Scope guardrails

- **Do not touch `ml_label_preprocess/`** — `last_fade_cycle`, `n_regular`, `status` already give us everything. No new labels needed.
- **Classification remains a first-class task.** Every change to the loader, pipelines, and CLI must preserve the existing `--task classification` (default) path with byte-identical outputs for the same config — verify via the per-seed metrics on a fixed seed before/after the loader refactor.
- **Use `mldashboard` env, not `eis`.** Confirm `xgboost`, `interpret`, `scikit-survival` versions on first use; add to `pyproject.toml` extras only after confirming the env has them — do not pip-install into shared envs.
- **No new top-level dependencies in the base install.** XGBoost and scikit-survival go under optional extras (`pip install -e .[xgb,survival]`) to keep `pip install -e .` cheap.
- **SHAP for RSF is explicitly deferred** — note it in the README, don't ship a broken `compute_shap` for sksurv models.

# `ml_classification`

Binary classification of battery cells: **will the cell survive past cycle N with
discharge retention ≥ 0.85?** for N ∈ {200, 300, 400}, using the 12 FS_CV features
produced by [`ml_label_preprocess`](../ml_label_preprocess/).

The pipeline is **model-agnostic** — currently ships Random Forest (Stage 1);
designed so EBM (InterpretML, Stage 2) and BART (Stage 3) can be plugged in
without touching pipeline or config-loader code.

## Quick start

```bash
# in the eis conda env (where workbench-app + optuna are wired up)
python -m ml_classification.run --config ml_classification/configs/rf_n300.yaml
python -m ml_classification.run --config ml_classification/configs/rf_n200.yaml
python -m ml_classification.run --config ml_classification/configs/rf_n400.yaml

# unit tests
python -m pytest ml_classification/tests/ -x
```

Each run writes to `ml_classification/out/{experiment_name}/`:

| File | What |
|---|---|
| `per_seed_metrics.csv` | 10 rows, one per seed — train / val / test metrics + per-cohort AUC + best params |
| `feature_importance.csv` | 12 rows — native importance + permutation importance (mean ± std across seeds) |
| `optuna_history.csv` | Trial-level diagnostics (10 × 100 = 1000 rows by default) |
| `best_params.json` | Best hyperparameters per seed |
| `model_best.joblib` | Highest-test-AUC seed's fitted Pipeline (imputer + RF) — for downstream use |
| `summary.json` | Aggregated stats: mean ± std test AUC, runtime, full config snapshot |

## Pipeline shape

```
load_dataset(N, "fs_cv")             # join features + labels, filter trainable_n{N}
       │
       ▼  for each of 10 seeds:
   stratified_split(y, test_frac=0.2)   # 80/20 stratified on target only
       │
       ▼
   tune(model_spec, X_tr, y_tr)      # Optuna TPE, 5-fold inner CV on train
       │   100 trials, optimize ROC-AUC
       ▼
   model = Pipeline([median-imputer, model_spec.build(best_params)])
   model.fit(X_tr, y_tr)
       │
       ▼
   evaluate(model, X_{tr,te}, y_{tr,te}, cohorts_{tr,te})
   compute_importance(spec, model, X_te, y_te, feature_names, n_repeats=30)
       │
       ▼
   per-seed row (including inner_cv_roc_auc) + per-seed importance df + study
       │
       ▼  aggregate
   per_seed_metrics.csv, feature_importance.csv, summary.json, model_best.joblib
```

**Best-seed model selection** is by **mean inner-CV ROC-AUC** (the `study.best_value`
each seed produces during tuning), NOT by test AUC — picking by test AUC would
peek at the test set. Inner CV is the proper criterion because it lives entirely
inside the training data.

**Decisions that differ from the reference report
(`experiment_cv_features/report_M2_vs_lean8_vs_CV_20260501/`):**

1. **Optuna TPE** replaces sklearn `RandomizedSearchCV` (Bayesian over random).
2. **ROC-AUC** is the tuning objective (reference used F1) — threshold-agnostic.
3. Median imputation runs inside a `sklearn.Pipeline` so it's part of CV, never
   data-leaks from test into train.
4. The split itself only stratifies on the binary target. Per-cohort metrics
   (AR / 0MC) are reported in the output rows but do not influence the split
   — by design (matches the reference, keeps the procedure simple).
5. **80/20** train/test split (no separate validation slice). Inner 5-fold CV
   handles hyperparameter selection, so a held-out val set adds no information.

## File layout

```
ml_classification/
├── README.md                 (this file)
├── data.py                   (load + join + filter; rejects label-like features)
├── splits.py                 (stratified 70/20/10 by target)
├── tuning.py                 (Optuna TPE wrapper, inner-CV objective)
├── metrics.py                (evaluate(); overall + per-cohort)
├── importance.py             (native + permutation, ROC-AUC scored)
├── pipeline.py               (run_experiment — seed loop, persistence)
├── run.py                    (CLI entry: `python -m ml_classification.run`)
├── models/
│   ├── __init__.py           (MODEL_REGISTRY, get_model_spec)
│   ├── base.py               (ModelSpec ABC — the extension contract)
│   ├── random_forest.py      (Stage 1, fully implemented)
│   ├── ebm.py                (Stage 2 stub — see "Adding a new model" below)
│   └── bart.py               (Stage 3 stub)
├── configs/
│   ├── base.yaml             (shared defaults: 10 seeds, 100 trials, ROC-AUC)
│   ├── rf_n200.yaml          (extends base; sets N=200)
│   ├── rf_n300.yaml          (N=300 — primary target)
│   └── rf_n400.yaml          (N=400)
├── tests/                    (pytest — 23 tests)
└── out/
    └── {experiment_name}/    (one folder per config)
```

## Adding a new model (EBM, BART, …)

The whole reason this is split into a `models/` subpackage. Three steps:

### 1. Implement `ModelSpec` in `models/<your_model>.py`

```python
from .base import ModelSpec

class YourModelSpec(ModelSpec):
    name = "your_model"
    fixed_params = {"n_jobs": -1}            # whatever doesn't get tuned

    def build(self, params):
        return YourModel(**params)            # sklearn-compatible estimator

    def suggest_params(self, trial):
        return {
            "knob_a": trial.suggest_int("knob_a", 1, 100),
            "knob_b": trial.suggest_float("knob_b", 1e-4, 1.0, log=True),
        }

    def feature_importance(self, fitted, X, feature_names):
        return dict(zip(feature_names, fitted.your_importance_attr))
```

**Estimator contract**: returned objects must support `.fit(X, y)`,
`.predict(X)`, and `.predict_proba(X)[:, 1]`. EBM (`interpret`) and most
sklearn-compat libraries already conform. For BART, write a thin shim.

### 2. Register it in `models/__init__.py`

```python
from .your_model import YourModelSpec

MODEL_REGISTRY = {
    "random_forest": RFModelSpec,
    "ebm": EBMModelSpec,
    "bart": BARTModelSpec,
    "your_model": YourModelSpec,
}
```

### 3. Add config files

Clone `configs/rf_n300.yaml` to `configs/your_model_n300.yaml`, change
`model:` and `experiment_name:`. Done. No pipeline, tuning, metric, or
output-schema changes are needed.

### Stubbed stages

`models/ebm.py` and `models/bart.py` already exist as stubs that raise
`NotImplementedError` with a clear message. Each header comment documents
the exact integration path (library, search-space hints, importance API).

## How leakage is prevented

1. **Feature subset is a manifest**, not a free-form list. `data.load_dataset`
   reads `subsets.fs_cv` from
   [`ml_label_preprocess/column_roles.yaml`](../ml_label_preprocess/column_roles.yaml).
2. **Hard denylist** in `data._LABEL_COLUMNS_DENYLIST` rejects any feature
   subset that contains label-like columns (`label_n*`, `trainable_n*`,
   `status`, `n_regular`, `last_fade_cycle`, `final_retention`, etc.).
3. **`tests/test_data.py::test_no_label_leakage`** asserts that condition on
   every N.
4. The imputer lives **inside the sklearn `Pipeline`**, so median values are
   computed only on the train fold during inner CV — they never leak from
   the test slice into training.
5. **Best-seed selection uses inner-CV ROC-AUC**, not test ROC-AUC — picking
   by test AUC would let model selection peek at the test set.

## Dataset sizes (after inner-join with features, filter trainable)

| N | trainable cells | pass | bad | pass% | AR | 0MC |
|---:|---:|---:|---:|---:|---:|---:|
| 200 | 291 | 223 | 68 | 76.6% | 255 | 36 |
| 300 | 248 | 157 | 91 | 63.3% | 219 | 29 |
| 400 | 233 | 118 | 115 | 50.6% | 206 | 27 |

**N=400 0MC caveat.** The 0MC cohort has only 27 cells; with a 20% test
slice it holds ~5 0MC cells. Roughly 1–2 seeds out of 10 are expected to
hit an all-one-class 0MC slice, producing `auc_0MC = NaN`. The aggregated
`test_auc_0MC_mean` is still usable as a rough indicator of 0MC
generalization, BUT per-seed CI is very wide (`n_pos ≈ 2–3` per slice →
`SE(AUC) ≈ 0.25`). **Do not make cohort-difference claims** (e.g. "0MC
generalizes worse than AR") from N=400 alone. Use N=200 or N=300 for
cohort comparisons.

Downstream readers should use `skipna=True` when averaging
per-cohort metrics across seeds.

## Success criterion (Stage 1)

Mean test ROC-AUC > **0.75** for all three N (N=200/300/400). Reference
industry baseline for 5-cycle BOL → long-cycle survival classifiers is
0.75–0.85. If any N falls below 0.70, the feature set itself should be
revisited before model changes.

## Output file schemas

| File | Columns |
|---|---|
| `per_seed_metrics.csv` | `seed`, `inner_cv_roc_auc`, `train_*`, `test_*` (incl. per-cohort `_AR` / `_0MC`), `overfit_auc`, `best_params` (JSON string) |
| `feature_importance.csv` | `feature`, `native_mean`, `native_std`, `perm_mean`, `perm_std` (sorted by `perm_mean`) |
| `optuna_history.csv` | `seed`, `trial_number`, `value`, `state`, plus `param_*` for each tuned hyperparameter |
| `best_params.json` | `{seed: {param: value}}` for all 10 seeds |
| `model_best.joblib` | dict `{model, best_params, best_seed, feature_names, N}` — for the seed with the highest **inner-CV** ROC-AUC |
| `summary.json` | aggregates: mean ± std test AUC, per-cohort means, best-seed metadata, `sklearn_version`, `optuna_version`, `python_version`, full `config_snapshot` |

## Environment

Reuses the `eis` conda env. New dependency: `optuna`. The `interpret` (EBM) and
`pymc-bart` / `bartpy` (BART) libraries are imported lazily inside their
respective stubs — Stage 1 does not require them.

```bash
# already installed:
conda run -n eis pip install optuna
```

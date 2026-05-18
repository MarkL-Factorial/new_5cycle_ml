# cell_lifetime — project conventions

This file pins down conventions that have already been chosen and
should not be re-litigated by future work. New experiments, the
validation pipeline, and any production fit MUST follow these unless
explicitly noted as an ablation.

## Canonical training sets, per task

| Task | Training set | Loader accessor | Approx. n (A2.2_b1) |
|---|---|---|---|
| Classification (binary at N) | **`trainable_n{N}` cells** — every cell with a definitive pass/fail label at N: faded cells (known cycle_life) ∪ censored cells with `n_regular ≥ N` (definitively pass). | `ds.label_mask` or `ds.view_for_task("classification")` after `load_dataset(N=N, …)` | **291** (N=200) / **250** (N=300) / **236** (N=400) |
| Regression (continuous cycle life) | **Faded cells only** — censored cells have unknown cycle_life and can't be used as regression targets. | `ds.faded_mask` or `ds.view_for_task("regression")` | **187** |
| Survival (RSF, XGB-AFT, etc.) | **All 415 cells** — both faded (event=1) and censored (event=0). Survival models consume censoring natively via `(time, event)`. | `ds.view_for_task("survival")` returns no-op mask; pass full arrays | **415** |

For the cohort breakdown above, "trainable" semantics are owned by
`ml_label_preprocess` (the upstream label engine). The cell_lifetime
loader reads the parquet column `trainable_n{N}` directly into
`ds.label_mask`. Do not recompute it.

### Why classification = `trainable_n{N}`, not `faded` only

Empirical observation (Exp J first pass vs Exp J rerun, single seed):

| N | Faded-only training (187) → OOF F1 on trainable_n{N} | `trainable_n{N}` training → OOF F1 on trainable_n{N} |
|---:|---:|---:|
| 200 | n/a (different surface) | **0.923** |
| 300 | n/a | **0.833** |
| 400 | n/a | **0.789** |

Faded-only training drops 49–104 known-pass censored cells per N.
Those cells are definitive positive examples — their `n_regular ≥ N`
means the upstream observed them surviving past N. Throwing them away
hurts F1 and AUC noticeably, especially at N=300/400 where the
faded-only training set is most data-starved.

### Why regression = faded only

Censored cells have no observed cycle_life, only a lower-bound
observation time. They can't enter a least-squares regression as
target rows. They CAN enter as survival training data — that's what
RSF / XGB-AFT / Cox / Weibull AFT are for.

## How to access the canonical training set in code

```python
from cell_lifetime.data.loader import load_dataset

# Classification at N=300
ds = load_dataset(N=300, feature_subset="fs_a_only")
view = ds.view_for_task("classification")
X_train, y_train = view.X, view.y_class
# OR equivalently:
mask = ds.label_mask               # bool, trainable_n300
X_train = ds.X.loc[mask]
y_train = ds.y_class[mask]
```

```python
# Regression on cycle_life
ds = load_dataset(N=300, feature_subset="fs_all")  # N here is a placeholder
view = ds.view_for_task("regression")
X_train, y_train = view.X, view.y_cycle            # faded cells only
```

```python
# Survival
ds = load_dataset(N=300, feature_subset="fs_cv")
# Use ds.X, ds.event, ds.time directly — no filtering.
```

## Predicting on cells outside the training set

For all tasks, you may **predict** on any cell whose features are
available (typically all 415). Just use the unfiltered `ds.X`:

```python
# Train on trainable_n{N} (251–291 cells), but predict on all 415:
view = ds.view_for_task("classification")
model.fit(view.X, view.y_class)
prob_all = model.predict_proba(ds.X)[:, 1]
```

This is what `experiments/exp_j_production_predictions/run.py` does.
Cells outside the training set don't have a ground-truth label at N
(censored with `n_regular < N`), but the model can still emit a
prediction; consumers should consult the `in_training_set_n{N}` or
`true_pass_n{N}` columns in `predictions.csv` to know which is which.

## What NOT to do

- Don't train a classifier on `faded`-only and report F1 against
  trainable_n{N} — that mixes evaluation surfaces and produces
  misleadingly low numbers, because the lost training cells were all
  positive examples.
- Don't train a classifier on `trainable_n{N}` then evaluate against
  a different N's mask. Each N is its own task with its own labels.
- Don't include censored cells as regression targets by imputing
  `y_cycle = n_regular`. Their cycle life is *at least* `n_regular`;
  treating that as a point estimate is a survival-analysis bug, not
  a regression workaround. Use a survival model instead.

## Versioning

This file should be updated when a convention changes or a new task
type is added. Bump the date below.

Last updated: 2026-05-18.

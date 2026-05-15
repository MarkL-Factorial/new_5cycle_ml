# cell_classifier

Battery cell N-cycle survival classifier. Predicts, from features extracted in
the first 5 regular cycles, whether a cell will retain ≥ 0.85 discharge
capacity past cycle N (N ∈ {200, 300, 400}).

Two pipeline modes:

- **Validation** — held-out evaluation. Stratified 80/20 split (or nested CV).
  Reports F1, accuracy, precision, recall, ROC-AUC + per-cohort AUC.
- **Production / inference** — train on all labeled cells; predict for every
  cell with features (label status irrelevant at inference time). No metrics
  computed.

## Install

```bash
pip install -e .            # core (RF only)
pip install -e ".[dev]"     # core + pytest + ruff
pip install -e ".[ebm]"     # add EBM (interpret)
pip install -e ".[bart]"    # add BART (pymc-bart)
pip install -e ".[all]"     # everything
```

## Usage

```bash
# Validation run — 5-fold nested CV
cell-classifier run \
    --mode validation \
    --model-config configs/rf.yaml \
    --N 300 --db-version A2.2 --baseline-cycle 1 --feature-subset fs_cv \
    --tuning-protocol nested_cv --outer-k 5

# Production / inference — reuse hyperparameters from the matching validation run
cell-classifier run \
    --mode production \
    --model-config configs/rf.yaml \
    --N 300 --db-version A2.2 --baseline-cycle 1 --feature-subset fs_cv
```

Each invocation writes to
`results/runs/{mode}/{model}__N{N}__{db}_b{baseline}__{feature_subset}__{YYYYMMDD_HHMMSS}/`.
A `{slug}` symlink (without the timestamp) is repointed at the latest
finished run, so re-running with the same config preserves prior runs
on disk while keeping `from_validation_run` / sweep lookups stable.
Each run folder contains:

- `manifest.json` — resolved config provenance, versions, SHA-256 idempotency hash
- `resolved_config.yaml` — the exact config dict that produced this run
- `inputs/` — a copy of the upstream `cell_features.parquet`,
  `cell_labels.parquet`, the preprocess `manifest.json`, and
  `column_roles.yaml` (so the folder is independently reproducible)
- per-seed metrics, plots, feature importance, SHAP, Optuna history, etc.

## Notebooks

Once installed via `pip install -e .`, notebooks can import directly:

```python
from cell_classifier.data.loader import load_dataset
from cell_classifier.utils.discover import find_runs
```

**Do not** use `sys.path.insert(...)` or relative imports — the editable
install handles path resolution. A starter notebook lives at
`notebooks/quickstart.ipynb`.

## Data source

The classifier reads preprocess bundles from
`/mnt/data/mliao/battery-ml-workbench/new_5cycle_ml/ml_label_preprocess/datasets/{db_version}_b{baseline_cycle}/`.

Override the preprocess root via the `BCC_PREPROCESS_ROOT` environment
variable or `data.preprocess_root` in the config YAML.

# cell_lifetime

Cycle-life regression and survival models for battery cells. Sibling
package to `cell_classifier/` — imports from it as a read-only consumer
and **never** modifies it.

Predicts, from the 12 `fs_cv` features of a cell's first 5 regular
cycles:

- **Classification** (XGBoost) — pass/bad at N ∈ {200, 300, 400} (mirrors `cell_classifier`'s RF baseline)
- **Regression** (XGBoost, EBM) — cycle-life = `last_fade_cycle`, with Box-Cox / log / sqrt target transform
- **Survival** (XGBoost-AFT, Random Survival Forest) — censoring-aware cycle-life modeling for the 58% of cells that haven't yet faded

## Install

```bash
source /home/mliao/miniconda3/etc/profile.d/conda.sh && conda activate mldashboard
pip install -e .[xgb,ebm,survival]    # full
pip install -e .[xgb,ebm]              # P1 (regression spine)
```

## Usage

```bash
# Classification (smoke; expect AUC within ~5pt of cell_classifier's RF)
cell-lifetime run --task classification --model-config configs/xgb_classifier.yaml \
    --N 300 --db-version A2.2 --baseline-cycle 1 --feature-subset fs_cv \
    --tuning-protocol tune_inner_cv

# Regression (log target)
cell-lifetime run --task regression --model-config configs/xgb_regressor.yaml \
    --N 300 --db-version A2.2 --baseline-cycle 1 --feature-subset fs_cv \
    --tuning-protocol tune_inner_cv --target-transform log

# Regression (Box-Cox target, EBM)
cell-lifetime run --task regression --model-config configs/ebm_regressor.yaml \
    --N 300 --db-version A2.2 --baseline-cycle 1 --feature-subset fs_cv \
    --tuning-protocol tune_inner_cv --target-transform boxcox
```

Each run writes to `out/runs/{task}/{slug}/` with a `manifest.json` and
per-seed metrics.

## Data source

Reads `ml_label_preprocess/datasets/{db_version}_b{baseline_cycle}/`.
Three columns drive the new targets:
- `label_n{N}` — binary classification (pass/bad)
- `last_fade_cycle` — regression target (cycle life)
- `(status, last_fade_cycle, n_regular)` — survival (event, time) pair

## Layout

See [INDEX.md](INDEX.md) for the live file/phase inventory, and
[../cell_classifier_regression_survival_roadmap.md](../cell_classifier_regression_survival_roadmap.md)
for the technical roadmap.

## Operating rules

This package follows five inviolable rules:

1. **No existing scripts touched** — `cell_classifier/`, `ml_label_preprocess/`, and root files are read-only.
2. **All new work in `cell_lifetime/`** — no leakage outside this directory.
3. **Automated execution** — `scripts/run_routine.sh` and `scripts/monday_smoke_real_data.sh` are fully scripted.
4. **Comprehensive logging** — every phase writes `.log`, `.test.xml`, `.summary.md` under `run_logs/`.
5. **Index maintained** — `INDEX.md` is appended by every phase.

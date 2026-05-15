# cell_lifetime INDEX

Branch: `feature/cell_lifetime`
Last updated: 2026-05-15 (initialized by Phase 1 in-session)

## Phase log (append-only)

| Phase | Surface | Started (UTC) | Ended (UTC) | Status | Commit | Files added | Tests run | Tests pass | Log/Summary |
|-------|---------|---------------|-------------|--------|--------|-------------|-----------|------------|-------------|
| phase1_regression_spine | in-session | 2026-05-15 | (pending) | (pending) | (pending) | ~30 | (pending) | (pending) | run_logs/phase1.summary.md |

## Files (alphabetical by path)

| Path | Purpose | Added by | Lines | Imports from cell_classifier? |
|------|---------|----------|-------|-------------------------------|
| `INDEX.md` | This file | phase1 | — | no |
| `README.md` | Quickstart | phase1 | — | no |
| `ROUTINES.md` | Cloud routine state machine | phase1 | — | no |
| `pyproject.toml` | Package metadata | phase1 | — | no |
| `.gitignore` | Local ignores | phase1 | — | no |
| `src/cell_lifetime/__init__.py` | Package init | phase1 | — | no |
| `src/cell_lifetime/cli.py` | CLI entry point | phase1 | — | yes (registry, paths) |
| `src/cell_lifetime/data/loader.py` | CycleLifeDataset + load_dataset | phase1 | — | yes (column_roles_path) |
| `src/cell_lifetime/data/synthetic.py` | Synthetic Dataset for cloud tests | phase1 | — | no |
| `src/cell_lifetime/preprocessing/target_transform.py` | Box-Cox / log / sqrt target wrapper | phase1 | — | no |
| `src/cell_lifetime/models/base.py` | CycleLifeModel(BaseModel) | phase1 | — | yes (BaseModel) |
| `src/cell_lifetime/models/registry.py` | Guarded-import registry | phase1 | — | yes (pattern) |
| `src/cell_lifetime/models/xgb_classifier.py` | XGBoost binary classifier | phase1 | — | yes (BaseModel) |
| `src/cell_lifetime/models/xgb_regressor.py` | XGBoost regressor + transform | phase1 | — | yes (BaseModel) |
| `src/cell_lifetime/models/ebm_regressor.py` | EBM regressor + transform | phase1 | — | yes (BaseModel) |
| `src/cell_lifetime/evaluation/regression_metrics.py` | MAE/RMSE/R²/MedAE | phase1 | — | no |
| `src/cell_lifetime/pipelines/validation.py` | Task-branched orchestrator (P1: class+reg) | phase1 | — | yes (paths, manifest helpers) |
| `configs/xgb_classifier.yaml` | XGB classifier template | phase1 | — | no |
| `configs/xgb_regressor.yaml` | XGB regressor template | phase1 | — | no |
| `configs/ebm_regressor.yaml` | EBM regressor template | phase1 | — | no |
| `tests/*.py` | Phase 1 unit + integration tests | phase1 | — | — |
| `scripts/install_env.sh` | Idempotent env install | phase1 | — | no |
| `scripts/run_routine.sh` | Cloud routine prologue wrapper | phase1 | — | no |
| `scripts/monday_smoke_real_data.sh` | Real-data smoke (all phases) | phase1 | — | no |

## External dependencies introduced

| Package | Version | Phase | Why |
|---------|---------|-------|-----|
| xgboost | 3.2.0 | phase1 | XGB classifier + regressor + AFT (P2) |
| interpret | 0.7.8 | phase1 | EBM regressor |
| scikit-survival | 0.25.0 | phase1 | RSF (P3); installed early so the env is one-shot |

## Open issues / blockers

(none yet)

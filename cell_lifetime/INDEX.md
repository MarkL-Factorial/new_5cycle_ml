# cell_lifetime INDEX

Branch: `feature/cell_lifetime`
Last updated: 2026-05-15T22:00:00Z

## Phase log (append-only)

| Phase | Surface | Started (UTC) | Ended (UTC) | Status | Commit | Files added | Tests run | Tests pass | Log/Summary |
|-------|---------|---------------|-------------|--------|--------|-------------|-----------|------------|-------------|
| phase1_regression_spine | in-session | 2026-05-15T21:14Z | 2026-05-15T21:52Z | OK | cfb5ce4 | 38 | 28 | 28 | run_logs/20260515T215136Z_smoke_phase1.log |
| phase2_xgb_aft | in-session | 2026-05-16T18:30Z | 2026-05-16T18:52Z | OK | (pending) | 5 | 40 | 40 | run_logs/20260516T185120Z_phase2_xgb_aft_smoke.log |

## Phase 1 real-data smoke (A2.2_b1, N=300, 1 seed)

| Model | Task | Transform | Test metric | Runtime |
|-------|------|-----------|-------------|---------|
| xgb_classifier | classification | — | F1 = 0.853, AUC = (1 seed) | 9.5s |
| xgb_regressor  | regression     | sqrt | MAE = 129.7 cycles | 7.6s |
| ebm_regressor  | regression     | boxcox (λ≈0.534) | MAE = 138.3 cycles | 64.8s |

Sanity bounds: faded-cell cycle life median = 317, range 6–1052; MAE ~130 cycles is ~40% of median (12 features, no tuning depth). XGB classifier F1=0.85 matches the cell_classifier RF baseline magnitude at N=300.

## Phase 2 real-data smoke (A2.2_b1, N=300, 1 seed, all 415 cells incl. censored)

| Model | Task | Metric | Value |
|-------|------|--------|-------|
| xgb_aft | survival | test C-index | **0.778** |
| xgb_aft | survival | test AUC@200 | 0.914 |
| xgb_aft | survival | test AUC@300 | 0.842 |
| xgb_aft | survival | test AUC@400 | 0.831 |

XGB-AFT uses censoring-aware loss (objective='survival:aft') and trains on all 415 cells (187 observed + 228 right-censored). The C-index 0.778 beats random (0.5) cleanly; AUC@N tracks the classification threshold the existing pipeline already uses, so survival numbers are directly comparable to xgb_classifier's F1.

## Cloud routines (scheduled but will no-op due to in-session completion)

| Phase | Routine ID | Fires (UTC) | Expected outcome |
|-------|------------|-------------|------------------|
| phase2_xgb_aft | trig_016VCZuFhZmi3piDF1xG6ZhD | 2026-05-16T14:00Z | no-op (INDEX shows OK; idempotency check exits cleanly) |
| phase3_rsf_and_summary | trig_01TUVoTc8oA6Utag2wEcbWJS | 2026-05-17T02:00Z | no-op once Phase 3 lands in-session |

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

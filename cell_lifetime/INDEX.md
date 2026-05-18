# cell_lifetime INDEX

Branch: `feature/cell_lifetime`
Last updated: 2026-05-18T18:30:00Z

## Project conventions

See [`CONVENTIONS.md`](CONVENTIONS.md) for canonical training sets per
task. **Classification trains on `trainable_n{N}` (faded ∪ censored
with `n_regular ≥ N`), not faded-only.** This applies to both the
`cell-lifetime run` validation pipeline and any production fit.

## Phase log (append-only)

| Phase | Surface | Started (UTC) | Ended (UTC) | Status | Commit | Files added | Tests run | Tests pass | Log/Summary |
|-------|---------|---------------|-------------|--------|--------|-------------|-----------|------------|-------------|
| phase1_regression_spine | in-session | 2026-05-15T21:14Z | 2026-05-15T21:52Z | OK | cfb5ce4 | 38 | 28 | 28 | run_logs/20260515T215136Z_smoke_phase1.log |
| phase2_xgb_aft | in-session | 2026-05-16T18:30Z | 2026-05-16T18:52Z | OK | fa9ee48 | 5 | 40 | 40 | run_logs/20260516T185120Z_phase2_xgb_aft_smoke.log |
| phase3_rsf_and_summary | in-session | 2026-05-16T18:54Z | 2026-05-16T18:56Z | OK | 4b4b394 | 3 | 47 | 47 | run_logs/20260516T185357Z_phase3_rsf_smoke.log |
| phase_experiments_a_b | in-session | 2026-05-16T19:10Z | 2026-05-16T23:35Z | OK | b41f0ce | 5 src + 14 exp | 50 | 50 | experiments/REPORT.html |
| phase_experiments_c_d_e_f | in-session | 2026-05-17T00:00Z | 2026-05-17T00:25Z | OK | 29c6857 | 4 src + 16 exp | 66 | 66 | experiments/REPORT.html |
| phase_experiments_g | in-session | 2026-05-17T00:42Z | 2026-05-17T03:23Z | OK | a9573d2 | 3 src + 10 exp | 75 | 75 | experiments/REPORT.html |
| phase_experiments_g_followup_fsall | in-session | 2026-05-17T14:20Z | 2026-05-17T15:14Z | OK | aed9924 | 0 src + 7 exp | 75 | 75 | experiments/REPORT.html |
| phase_experiments_h_rsf_vs_regressors | in-session | 2026-05-17T23:34Z | 2026-05-18T00:13Z | OK | dbedbc6 | 0 src + 4 exp | 75 | 75 | experiments/exp_h_rsf_vs_regressors_fair/summary.json |
| phase_experiments_i_rsf_censored_ablation | in-session | 2026-05-18T01:00Z | 2026-05-18T01:21Z | OK | 87ba00d | 0 src + 3 exp | 75 | 75 | experiments/exp_i_rsf_censored_ablation/summary.json |
| phase_experiments_j_production_predictions | in-session | 2026-05-18T17:01Z | 2026-05-18T18:13Z | OK | b5fd949 | 0 src + 1 exp | 75 | 75 | experiments/exp_j_production_predictions/predictions.csv |
| phase_experiments_j_rerun_trainable_n | in-session | 2026-05-18T18:11Z | 2026-05-18T18:23Z | OK | (pending) | 0 src + 1 exp (rerun) | 75 | 75 | experiments/exp_j_production_predictions/predictions.csv |

## Phase 1 real-data smoke (A2.2_b1, N=300, 1 seed)

| Model | Task | Transform | Test metric | Runtime |
|-------|------|-----------|-------------|---------|
| xgb_classifier | classification | — | F1 = 0.853, AUC = (1 seed) | 9.5s |
| xgb_regressor  | regression     | sqrt | MAE = 129.7 cycles | 7.6s |
| ebm_regressor  | regression     | boxcox (λ≈0.534) | MAE = 138.3 cycles | 64.8s |

Sanity bounds: faded-cell cycle life median = 317, range 6–1052; MAE ~130 cycles is ~40% of median (12 features, no tuning depth). XGB classifier F1=0.85 matches the cell_classifier RF baseline magnitude at N=300.

## Phase 2 + 3 real-data smoke (A2.2_b1, N=300, 1 seed, all 415 cells incl. censored)

| Model | Task | C-index | AUC@200 | AUC@300 | AUC@400 | Runtime |
|-------|------|---------|---------|---------|---------|---------|
| xgb_aft | survival | 0.778 | 0.914 | 0.842 | 0.831 | 6.8s |
| rsf     | survival | **0.807** | 0.911 | **0.880** | **0.896** | 11.8s |

Both models train on all 415 cells (187 observed + 228 right-censored). XGB-AFT uses censoring-aware loss (`objective='survival:aft'`) and predicts log-cycle-life; RSF predicts a risk score directly. The validation pipeline normalises both into the same sign via the `risk_orientation` class attribute before computing C-index.

RSF outperforms AFT at every horizon in this 1-seed smoke. The cross-model comparison is the headline cycle_lifetime deliverable: same data, same split logic, same metrics — RSF's nonparametric forest is a better fit than AFT's parametric (normal/logistic/extreme) AFT assumption on this data.

## Cross-task comparison at N=300 (all from 1-seed smoke runs)

| Model | Task | Headline | Note |
|-------|------|----------|------|
| xgb_classifier | classification | F1=0.853 | binary pass/bad, 250 cells |
| xgb_regressor (sqrt) | regression | MAE=129.7 cyc | 187 faded cells only |
| ebm_regressor (boxcox) | regression | MAE=138.3 cyc | 187 faded cells only |
| xgb_aft | survival | C-index=0.778, AUC@300=0.842 | 415 cells incl. censored |
| **rsf** | survival | **C-index=0.807, AUC@300=0.880** | 415 cells incl. censored |

## Experiment A + B headline (5 seeds × 30 trials × 5 inner CV)

Multi-seed estimates. See [`experiments/REPORT.md`](experiments/REPORT.md)
or `REPORT.html` for full breakdown.

| Task | Best model | Best feature set | Multi-seed headline |
|------|-----------|------------------|---------------------|
| classification | xgb_classifier | **fs_all** (40) | F1 = **0.866 ± 0.037** (+3.3% vs fs_cv) |
| regression | ebm_regressor | **fs_all** (40) | MAE = **136.2 ± 12.3** cyc (+4.2% vs fs_cv) |
| survival | **rsf** | **fs_cv** (12) | C-index = **0.801 ± 0.021**, AUC@300 = **0.879 ± 0.048** |

Findings:
- More features help classification + regression; RSF overfits at 40.
- Phase 1-3 single-seed numbers were optimistic vs 5-seed means.
- Z-score blend of RSF + AFT does NOT improve (≈ 1 std below RSF alone).
- Regression has a Q4 (long-life) ceiling driven by selection bias —
  survival is the right framework for those cells.

## Experiments C+D+E+F headline (5 seeds, 5 inner CV, fs_cv unless noted)

| Exp | Question | Headline finding |
|-----|----------|------------------|
| C | Weighted RSF+AFT blend? | **No** — optimal w_rsf=1.0 at every N; AFT contributes only noise |
| D | Where does the signal live across feature tiers? | **Tier A retention/CE alone (3 cols)** gets xgb_classifier F1=0.857; **Tier C alone collapses RSF C-index to 0.577** (near-random) |
| E | Adds Cox + Weibull AFT via lifelines; 4-way blend? | Cox=0.752, Weibull=0.755; **4-way blend also picks w_rsf=1.0** — RSF dominates |
| F | Tune classifier on F1 instead of ROC-AUC? | **No** — F1-tuned fs_all gets F1=0.825 (worse than AUC-tuned 0.866); F1 is a noisy/discrete objective for TPE |

The headline survival model is unchanged: **rsf + fs_cv** (C-index = 0.801 ± 0.021).
The headline classifier is unchanged: **xgb_classifier + fs_all** (F1 = 0.866 ± 0.037).
For lighter deployment: **xgb_classifier + fs_a_only** (3 features, F1 = 0.857 ± 0.024).

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

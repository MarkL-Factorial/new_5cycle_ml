# cell_lifetime experiments

Systematic experiments to improve cycle-life predictions on top of the
Phase 1+2+3 in-session baseline.

## Completed experiments (Exp A–G)

All seven prior experiments are archived under [`done_exp_a_to_g/`](done_exp_a_to_g/).
See [`REPORT.md`](REPORT.md) / [`REPORT.html`](REPORT.html) for the
consolidated narrative.

| Experiment | Question | Status |
|---|---|---|
| [exp_a_feature_set](done_exp_a_to_g/exp_a_feature_set/) | Does using all 40 feature-role columns beat the 12-column `fs_cv` subset? | done |
| [exp_b_horizon_and_blend](done_exp_a_to_g/exp_b_horizon_and_blend/) | How do top survival models perform across N, and does ensembling help? | done |
| [exp_c_weighted_blend](done_exp_a_to_g/exp_c_weighted_blend/) | Weighted RSF+AFT blend across N? | done |
| [exp_d_feature_tiers](done_exp_a_to_g/exp_d_feature_tiers/) | Where does the signal live across feature tiers? | done |
| [exp_e_parametric_survival](done_exp_a_to_g/exp_e_parametric_survival/) | Cox + Weibull AFT via lifelines; 4-way blend? | done |
| [exp_f_optimize_target](done_exp_a_to_g/exp_f_optimize_target/) | Tune classifier on F1 instead of ROC-AUC? | done |
| [exp_g_tuning_targets](done_exp_a_to_g/exp_g_tuning_targets/) | AUC vs F1 tuning across 4 models × 2 fs × 3 N. | done |
| [exp_g_followup_fsall](done_exp_a_to_g/exp_g_followup_fsall/) | Close the fs_all classifier gap at N=200/400. | done |

Each experiment dir contains:
- `run.sh` — orchestrator that issues `cell-lifetime run` invocations
- `aggregate.py` — walks `out/runs/` and parses `summary.json` per run
- `logs/` — tee'd stdout/stderr for each model × config invocation
- `metric_long.csv` — final aggregated comparison table

Final consolidated outputs:
- `REPORT.md` — markdown narrative
- `REPORT.html` — pandoc-rendered standalone HTML

Run from `cell_lifetime/`:

```bash
source /home/mliao/miniconda3/etc/profile.d/conda.sh && conda activate mldashboard
export OMP_NUM_THREADS=10 OPENBLAS_NUM_THREADS=10 MKL_NUM_THREADS=10 NUMEXPR_NUM_THREADS=10
bash experiments/done_exp_a_to_g/exp_a_feature_set/run.sh
python experiments/done_exp_a_to_g/exp_a_feature_set/aggregate.py
```

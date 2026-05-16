# cell_lifetime experiments

Systematic experiments to improve cycle-life predictions on top of the
Phase 1+2+3 in-session baseline.

| Experiment | Question | Status |
|---|---|---|
| [exp_a_feature_set](exp_a_feature_set/) | Does using all 40 feature-role columns beat the 12-column `fs_cv` subset? | running |
| [exp_b_horizon_and_blend](exp_b_horizon_and_blend/) | How do top survival models perform across N, and does ensembling help? | pending Exp A |

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
bash experiments/exp_a_feature_set/run.sh
python experiments/exp_a_feature_set/aggregate.py
```

#!/bin/bash
#SBATCH --job-name=cell_classifier
#SBATCH --time=96:00:00
#SBATCH --partition=compute
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=24
#SBATCH --mem=64GB
#SBATCH --output=cell_classifier_%j.log.out
#SBATCH --error=cell_classifier_%j.err.out

# Generic Slurm template for cell_classifier jobs whose wall time exceeds ~1 hr.
# Use this for any of:
#   - validation runs (tune_inner_cv or nested_cv)
#   - production runs (from_validation_run or retune)
#   - multi-axis sweeps
#
# Usage:
#   cd /mnt/data/mliao/battery-ml-workbench/new_5cycle_ml/cell_classifier
#   # 1. Edit the "JOB COMMAND" block at the bottom of this file.
#   # 2. Submit. The active example reads N from --export; examples that
#   #    don't use N just ignore the unused variable:
#   sbatch --export=ALL,N=300 scripts/slurm_train.sh
#
# Resources: 24 cores, 64 GB, 96 hr. RF and Optuna both use n_jobs=-1, so all
# 24 cores are exercised. Bring the time down if your job is short.
# Logs: cwd/cell_classifier_<jobid>.{log,err}.out

set -e
set -o pipefail

# ----- diagnostics -----
echo "=== SLURM job ==="
echo "Job ID:        $SLURM_JOB_ID"
echo "Job Name:      $SLURM_JOB_NAME"
echo "CPUs per task: $SLURM_CPUS_PER_TASK"
echo "Memory:        $SLURM_MEM_PER_NODE MB"
echo "Node:          $SLURM_NODELIST"
echo "Submit time:   $(date)"
echo "Working dir:   $(pwd)"
echo "================="
echo

# ----- activate env -----
source ~/miniconda3/etc/profile.d/conda.sh
conda activate /home/mliao/miniconda3/envs/mldashboard
conda --version
conda env list

# Sanity check: cell-classifier must be installed in this env (pip install -e .)
command -v cell-classifier >/dev/null || {
    echo "ERROR: cell-classifier console entry not found on PATH." >&2
    echo "Run 'pip install -e .' inside the active env." >&2
    exit 1
}

# ----- common defaults (override below if needed) -----
OUT_ROOT="$(pwd)/results"
export BCC_PREPROCESS_ROOT=/mnt/data/mliao/battery-ml-workbench/new_5cycle_ml/ml_label_preprocess

# =====================================================================
# JOB COMMAND — edit this block. Uncomment ONE of the examples below
# (or write your own). Everything else above is boilerplate.
# =====================================================================

# ----- Example 1: validation, nested_cv (the slow, defensible protocol) -----
# Submit one job per N:
#   sbatch -J cc_rf_N200_nested --export=ALL,N=200 scripts/slurm_train.sh
#   sbatch -J cc_rf_N300_nested --export=ALL,N=300 scripts/slurm_train.sh
#   sbatch -J cc_rf_N400_nested --export=ALL,N=400 scripts/slurm_train.sh
: "${N:?usage: sbatch --export=ALL,N=<200|300|400> scripts/slurm_train.sh}"
echo "Running N=${N}"
cell-classifier run \
    --mode validation \
    --model-config configs/rf.yaml \
    --N "$N" --db-version A2.2 --baseline-cycle 1 --feature-subset fs_cv \
    --tuning-protocol nested_cv --outer-k 5 \
    --tune.n-trials 100 --tune.inner-cv 5 \
    --seeds-preset fresh \
    --out-root "$OUT_ROOT" \
    --force

# ----- Example 2: validation, tune_inner_cv (cheaper, includes overfit_*) -----
# cell-classifier run \
#     --mode validation \
#     --model-config configs/rf.yaml \
#     --N 300 --db-version A2.2 --baseline-cycle 1 --feature-subset fs_cv \
#     --tuning-protocol tune_inner_cv --test-frac 0.2 \
#     --tune.n-trials 100 --tune.inner-cv 5 \
#     --seeds-preset fresh \
#     --out-root "$OUT_ROOT"

# ----- Example 3: production, reuse the matching validation run's HPs -----
# cell-classifier run \
#     --mode production \
#     --model-config configs/rf.yaml \
#     --N 300 --db-version A2.2 --baseline-cycle 1 --feature-subset fs_cv \
#     --production-params-source from_validation_run \
#     --seeds-preset fresh \
#     --out-root "$OUT_ROOT"

# ----- Example 4: production, fresh Optuna on the full labeled set -----
# cell-classifier run \
#     --mode production \
#     --model-config configs/rf.yaml \
#     --N 300 --db-version A2.2 --baseline-cycle 1 --feature-subset fs_cv \
#     --production-params-source retune \
#     --tune.n-trials 100 --tune.inner-cv 5 \
#     --seeds-preset fresh \
#     --out-root "$OUT_ROOT"

# ----- Example 5: sweep (Cartesian product over data axes) -----
# cell-classifier sweep \
#     --sweep configs/sweeps/rf_n_x_baseline.yaml

echo
echo "=== Job finished at $(date) ==="

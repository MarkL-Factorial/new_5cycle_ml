#!/bin/bash
#SBATCH --job-name=cc_sweep
#SBATCH --time=24:00:00
#SBATCH --partition=compute
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=24
#SBATCH --mem=64GB
#SBATCH --output=cc_sweep_%j.log.out
#SBATCH --error=cc_sweep_%j.err.out

# SLURM wrapper for `cell-classifier sweep`. The sweep YAML is passed in
# via --export=ALL,SWEEP_YAML=<path> so one script services every sweep.
#
# Usage:
#   cd /mnt/data/mliao/battery-ml-workbench/new_5cycle_ml/cell_classifier
#   sbatch -J cc_<sweep_name> \
#          --export=ALL,SWEEP_YAML=configs/sweeps/<sweep>.yaml \
#          scripts/slurm_sweep.sh
#
# Time-limit is 24h (override per-job with `--time=...`). RF and Optuna both
# use n_jobs=-1, so all 24 cores are exercised per inner fit.

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

command -v cell-classifier >/dev/null || {
    echo "ERROR: cell-classifier console entry not found on PATH." >&2
    echo "Run 'pip install -e .' inside the active env." >&2
    exit 1
}

# ----- inputs -----
: "${SWEEP_YAML:?usage: sbatch --export=ALL,SWEEP_YAML=<path> scripts/slurm_sweep.sh}"
[[ -f "$SWEEP_YAML" ]] || { echo "ERROR: $SWEEP_YAML not found" >&2; exit 1; }
echo "Sweep YAML: $SWEEP_YAML"

export BCC_PREPROCESS_ROOT=/mnt/data/mliao/battery-ml-workbench/new_5cycle_ml/ml_label_preprocess

# ----- run -----
# --force lets a sweep step into slugs whose {slug} symlink currently
# resolves to a prior run with a different config hash (e.g. tonight's
# smoke runs at the same axes). Prior runs are preserved on disk under
# their timestamped folder names; only the symlink is repointed.
cell-classifier sweep --sweep "$SWEEP_YAML" --force

echo
echo "=== Job finished at $(date) ==="

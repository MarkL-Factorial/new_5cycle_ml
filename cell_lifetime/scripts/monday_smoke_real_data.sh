#!/bin/bash
# Real-data smoke gate. Runs the appropriate cell-lifetime invocations for
# a given phase against the real A2.2_b1 bundle. Used both:
#   - in-session at the tail of Phase 1 (to validate before pushing)
#   - on Monday morning by the user after the cloud routines land Phase 2/3
#
# Usage:
#     bash monday_smoke_real_data.sh --phase 1   # XGBClass + XGBReg + EBMReg
#     bash monday_smoke_real_data.sh --phase 2   # adds XGB-AFT (Phase 2 only)
#     bash monday_smoke_real_data.sh --phase 3   # adds RSF (Phase 3 only)
#     bash monday_smoke_real_data.sh --phase all # everything wired

set -euo pipefail

# Workspace cap: max 10 cores across all parallel work (set by user).
# OMP_NUM_THREADS bounds XGBoost/sklearn/numpy's OpenMP layer; OPENBLAS too.
export OMP_NUM_THREADS=10
export OPENBLAS_NUM_THREADS=10
export MKL_NUM_THREADS=10
export NUMEXPR_NUM_THREADS=10

PHASE="all"
while [ $# -gt 0 ]; do
    case "$1" in
        --phase) PHASE="$2"; shift 2 ;;
        *) echo "unknown arg $1" >&2; exit 2 ;;
    esac
done

THIS_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PKG_DIR="$(dirname "$THIS_DIR")"
TS="$(date -u +%Y%m%dT%H%M%SZ)"
LOGDIR="$PKG_DIR/run_logs"
mkdir -p "$LOGDIR"
LOG="$LOGDIR/${TS}_smoke_phase${PHASE}.log"
exec > >(tee "$LOG") 2>&1

echo "=== monday_smoke_real_data.sh phase=$PHASE ==="
echo "  log: $LOG"
echo "  cwd: $(pwd)"

run_cmd() {
    local name="$1"; shift
    echo
    echo "--- $name ---"
    "$@" || { echo "[FAIL] $name"; return 1; }
}

cd "$PKG_DIR"

if [ "$PHASE" = "1" ] || [ "$PHASE" = "all" ]; then
    run_cmd "xgb_classifier (N=300)" cell-lifetime run \
        --task classification --model-config configs/xgb_classifier.yaml \
        --N 300 --db-version A2.2 --baseline-cycle 1 --feature-subset fs_cv \
        --tuning-protocol tune_inner_cv --tune.n-trials 20 --tune.inner-cv 3 \
        --seeds 1
    run_cmd "xgb_regressor log (N=300)" cell-lifetime run \
        --task regression --model-config configs/xgb_regressor.yaml \
        --N 300 --db-version A2.2 --baseline-cycle 1 --feature-subset fs_cv \
        --tuning-protocol tune_inner_cv --tune.n-trials 20 --tune.inner-cv 3 \
        --target-transform log --seeds 1
    run_cmd "ebm_regressor boxcox (N=300)" cell-lifetime run \
        --task regression --model-config configs/ebm_regressor.yaml \
        --N 300 --db-version A2.2 --baseline-cycle 1 --feature-subset fs_cv \
        --tuning-protocol tune_inner_cv --tune.n-trials 10 --tune.inner-cv 3 \
        --target-transform boxcox --seeds 1
fi

if [ "$PHASE" = "2" ] || [ "$PHASE" = "all" ]; then
    if [ -f configs/xgb_aft.yaml ]; then
        run_cmd "xgb_aft (N=300)" cell-lifetime run \
            --task survival --model-config configs/xgb_aft.yaml \
            --N 300 --db-version A2.2 --baseline-cycle 1 --feature-subset fs_cv \
            --tuning-protocol tune_inner_cv --tune.n-trials 20 --tune.inner-cv 3 \
            --seeds 1
    else
        echo "[skip] xgb_aft not yet implemented (Phase 2 routine hasn't run)"
    fi
fi

if [ "$PHASE" = "3" ] || [ "$PHASE" = "all" ]; then
    if [ -f configs/rsf.yaml ]; then
        run_cmd "rsf (N=300)" cell-lifetime run \
            --task survival --model-config configs/rsf.yaml \
            --N 300 --db-version A2.2 --baseline-cycle 1 --feature-subset fs_cv \
            --tuning-protocol tune_inner_cv --tune.n-trials 10 --tune.inner-cv 3 \
            --seeds 1
    else
        echo "[skip] rsf not yet implemented (Phase 3 routine hasn't run)"
    fi
fi

echo
echo "=== monday_smoke_real_data.sh DONE phase=$PHASE ==="

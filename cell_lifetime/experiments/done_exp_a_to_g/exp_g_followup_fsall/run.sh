#!/bin/bash
# Exp G follow-up — close the fs_all classifier gap.
#
# After Exp G compared xgb_classifier and ebm_classifier on fs_a_only
# and fs_cv across N ∈ {200,300,400}, the cross-experiment Q1 synthesis
# flagged that we have no fs_all classifier data at N=200 / N=400
# (only N=300 from Exp A's xgb_classifier × fs_all). To validate the
# headline F1 recommendation (currently ebm_classifier × fs_a_only at
# avg F1 = 0.862), we need:
#
#   - xgb_classifier × fs_all × N=200, N=400          (xgb already at N=300 from Exp A)
#   - ebm_classifier × fs_all × N=200, N=300, N=400    (ebm_classifier × fs_all never run)
#
# All AUC-tuned (Exp F/G showed F1-tuning doesn't help at these scales).
# Output goes to experiments/exp_g_followup_fsall/runs/ to keep clean.

set -euo pipefail

export OMP_NUM_THREADS=10
export OPENBLAS_NUM_THREADS=10
export MKL_NUM_THREADS=10
export NUMEXPR_NUM_THREADS=10

THIS_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PKG_DIR="$(dirname "$(dirname "$THIS_DIR")")"
LOGDIR="$THIS_DIR/logs"
OUT_ROOT="$THIS_DIR/runs"
mkdir -p "$LOGDIR" "$OUT_ROOT/runs/classification"
cd "$PKG_DIR"

SEEDS="1,2,3,4,5"

# Each entry: "model config N trials inner_cv"
declare -a JOBS=(
    "xgb_classifier  xgb_classifier  200  30  5"
    "xgb_classifier  xgb_classifier  400  30  5"
    "ebm_classifier  ebm_classifier  200  15  3"
    "ebm_classifier  ebm_classifier  300  15  3"
    "ebm_classifier  ebm_classifier  400  15  3"
)

for ENTRY in "${JOBS[@]}"; do
    read -r MODEL_NAME CONFIG_STEM N TRIALS INNER <<< "$ENTRY"
    SLUG_PREFIX="${MODEL_NAME}__classification__N${N}__A2.2_b1__fs_all"
    # Idempotent skip if a 5-seed AUC-tuned run already exists
    EXISTING=$(find "$OUT_ROOT/runs/classification" -maxdepth 1 -type d \
        -name "${SLUG_PREFIX}__*" 2>/dev/null \
        -exec test -s "{}/summary.json" \; -print 2>/dev/null | sort | tail -1)
    if [ -n "$EXISTING" ]; then
        N_S=$(python -c "import json; print(json.load(open('$EXISTING/summary.json')).get('n_seeds',0))" 2>/dev/null || echo 0)
        if [ "$N_S" -ge 5 ]; then
            echo "[skip] ${SLUG_PREFIX} already at $EXISTING"
            continue
        fi
    fi
    TS=$(date -u +%Y%m%dT%H%M%SZ)
    LOG="$LOGDIR/${TS}__${MODEL_NAME}__fs_all__N${N}.log"
    echo
    echo "=== [$TS] ${MODEL_NAME} × fs_all × N=${N} (AUC-tuned) ==="
    echo "  log: $LOG"
    START=$(date +%s)
    if cell-lifetime run \
            --task classification \
            --model-config "configs/${CONFIG_STEM}.yaml" \
            --N "$N" \
            --db-version A2.2 \
            --baseline-cycle 1 \
            --feature-subset fs_all \
            --tuning-protocol tune_inner_cv \
            --tune.n-trials "$TRIALS" \
            --tune.inner-cv "$INNER" \
            --seeds "$SEEDS" \
            --out-root "$OUT_ROOT" \
            >"$LOG" 2>&1; then
        ELAPSED=$(( $(date +%s) - START ))
        echo "  OK in ${ELAPSED}s"
    else
        ELAPSED=$(( $(date +%s) - START ))
        echo "  FAILED in ${ELAPSED}s — see $LOG"
        tail -5 "$LOG"
    fi
done

echo
echo "=== Exp G follow-up DONE ==="

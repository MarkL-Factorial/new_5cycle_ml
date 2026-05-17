#!/bin/bash
# Experiment D: feature tier ablation.
#
# Run rsf + xgb_classifier on each of 4 tier subsets at N=300, 5 seeds.
# Compare against Experiment A's fs_cv and fs_all baselines.
#
# Subsets:
#   fs_a_only — 3 cols, retention + CE
#   fs_b_only — 3 cols, nominal voltage
#   fs_c_only — 34 cols, CV-phase KWW + aggregates
#   fs_ab     — 6 cols, A + B (no CV phase)

set -euo pipefail

export OMP_NUM_THREADS=10
export OPENBLAS_NUM_THREADS=10
export MKL_NUM_THREADS=10
export NUMEXPR_NUM_THREADS=10

THIS_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PKG_DIR="$(dirname "$(dirname "$THIS_DIR")")"
LOGDIR="$THIS_DIR/logs"
mkdir -p "$LOGDIR"
cd "$PKG_DIR"

SEEDS="1,2,3,4,5"
N_TRIALS=30
INNER_CV=5
N=300

declare -a JOBS=(
    "xgb_classifier classification"
    "rsf            survival"
)

for FEATURES in fs_a_only fs_b_only fs_c_only fs_ab; do
    for ENTRY in "${JOBS[@]}"; do
        read -r MODEL_NAME TASK <<< "$ENTRY"
        SLUG="${MODEL_NAME}__${TASK}__${FEATURES}"
        # Idempotent skip if a 5-seed run already exists
        EXISTING=$(find "out/runs/${TASK}" -maxdepth 1 -type d \
            -name "${MODEL_NAME}__${TASK}__N${N}__A2.2_b1__${FEATURES}__*" 2>/dev/null \
            -exec test -s "{}/summary.json" \; -print 2>/dev/null | sort | tail -1)
        if [ -n "$EXISTING" ]; then
            N_SEEDS=$(python -c "import json; print(json.load(open('$EXISTING/summary.json')).get('n_seeds',0))" 2>/dev/null || echo 0)
            if [ "$N_SEEDS" -ge 5 ]; then
                echo "[skip] $SLUG already has $EXISTING with n_seeds=$N_SEEDS"
                continue
            fi
        fi
        TS=$(date -u +%Y%m%dT%H%M%SZ)
        LOG="$LOGDIR/${TS}__${SLUG}.log"
        echo
        echo "=== [$TS] $SLUG ==="
        echo "  log: $LOG"
        START=$(date +%s)
        if cell-lifetime run \
                --task "$TASK" \
                --model-config "configs/${MODEL_NAME}.yaml" \
                --N "$N" \
                --db-version A2.2 \
                --baseline-cycle 1 \
                --feature-subset "$FEATURES" \
                --tuning-protocol tune_inner_cv \
                --tune.n-trials "$N_TRIALS" \
                --tune.inner-cv "$INNER_CV" \
                --seeds "$SEEDS" \
                >"$LOG" 2>&1; then
            ELAPSED=$(( $(date +%s) - START ))
            echo "  OK in ${ELAPSED}s"
        else
            ELAPSED=$(( $(date +%s) - START ))
            echo "  FAILED in ${ELAPSED}s — see $LOG"
            tail -5 "$LOG"
        fi
    done
done

echo
echo "=== Experiment D DONE ==="

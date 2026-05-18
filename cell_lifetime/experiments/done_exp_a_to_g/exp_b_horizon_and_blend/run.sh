#!/bin/bash
# Experiment B: rsf + xgb_aft × N ∈ {200, 300, 400} on the winning feature set.
#
# Usage: bash run.sh <feature_subset>   # default fs_all
#
# Reuses the N=300 results from Experiment A when feature_subset matches.
# For N=200 and N=400, issues fresh `cell-lifetime run` invocations.
# 5 seeds × 30 trials × 5 inner CV each.

set -euo pipefail

export OMP_NUM_THREADS=10
export OPENBLAS_NUM_THREADS=10
export MKL_NUM_THREADS=10
export NUMEXPR_NUM_THREADS=10

FEATURES="${1:-fs_all}"
THIS_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PKG_DIR="$(dirname "$(dirname "$THIS_DIR")")"
LOGDIR="$THIS_DIR/logs"
mkdir -p "$LOGDIR"

cd "$PKG_DIR"

SEEDS="1,2,3,4,5"
N_TRIALS=30
INNER_CV=5

echo "=== Experiment B with feature_subset=$FEATURES ==="

for MODEL in xgb_aft rsf; do
    for N in 200 300 400; do
        SLUG="${MODEL}__survival__N${N}__${FEATURES}"
        # Idempotent skip if a 5-seed run already exists
        EXISTING=$(find "out/runs/survival" -maxdepth 1 -type d \
            -name "${MODEL}__survival__N${N}__A2.2_b1__${FEATURES}__*" 2>/dev/null \
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
        echo "--- [$TS] $SLUG ---"
        START=$(date +%s)
        if cell-lifetime run \
                --task survival \
                --model-config "configs/${MODEL}.yaml" \
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
echo "=== Experiment B DONE — feature_subset=$FEATURES ==="
echo "Next: python experiments/exp_b_horizon_and_blend/blend.py --features $FEATURES"

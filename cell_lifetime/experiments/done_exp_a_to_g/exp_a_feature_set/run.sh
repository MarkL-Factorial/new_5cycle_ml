#!/bin/bash
# Experiment A: 5 models × {fs_cv, fs_all} × 5 seeds, N=300.
#
# Each invocation does 30 Optuna trials × 5 inner CV; ~5-15 min/model.
# Outputs land in out/runs/{task}/<slug>__<ts>/ as usual.
# Logs tee'd into experiments/exp_a_feature_set/logs/.

set -euo pipefail

# Workspace cap
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

# Each entry: "model_config task target_transform_opt"
#   - target_transform_opt is empty for non-regression tasks
declare -a JOBS=(
    "xgb_classifier classification"
    "xgb_regressor  regression sqrt"
    "ebm_regressor  regression boxcox"
    "xgb_aft        survival"
    "rsf            survival"
)

for FEATURES in fs_cv fs_all; do
    for ENTRY in "${JOBS[@]}"; do
        read -r MODEL_NAME TASK TRANSFORM <<< "$ENTRY"
        SLUG="${MODEL_NAME}__${TASK}__${FEATURES}"
        # Idempotency: skip if a recent successful run exists with a non-empty summary.json
        EXISTING=$(find "out/runs/${TASK}" -maxdepth 1 -type d \
            -name "${MODEL_NAME}__${TASK}__N${N}__A2.2_b1__${FEATURES}__*" 2>/dev/null \
            -exec test -s "{}/summary.json" \; -print 2>/dev/null | sort | tail -1)
        if [ -n "$EXISTING" ]; then
            # Read n_seeds from existing summary; treat 5-seed runs as complete
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
        CMD=(cell-lifetime run
             --task "$TASK"
             --model-config "configs/${MODEL_NAME}.yaml"
             --N "$N"
             --db-version A2.2
             --baseline-cycle 1
             --feature-subset "$FEATURES"
             --tuning-protocol tune_inner_cv
             --tune.n-trials "$N_TRIALS"
             --tune.inner-cv "$INNER_CV"
             --seeds "$SEEDS"
        )
        if [ -n "${TRANSFORM:-}" ]; then
            CMD+=(--target-transform "$TRANSFORM")
        fi
        START=$(date +%s)
        if "${CMD[@]}" >"$LOG" 2>&1; then
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
echo "=== Experiment A DONE ==="

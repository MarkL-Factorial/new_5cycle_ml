#!/bin/bash
# Experiment F: re-run xgb_classifier with optimize=f1 on the same subsets
# Exp A and Exp D covered, then compare against the AUC-tuned baselines.
#
# Output goes to experiments/exp_f_optimize_target/runs/ to keep it cleanly
# separated from the AUC-tuned runs in cell_lifetime/out/runs/.

set -euo pipefail

export OMP_NUM_THREADS=10
export OPENBLAS_NUM_THREADS=10
export MKL_NUM_THREADS=10
export NUMEXPR_NUM_THREADS=10

THIS_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PKG_DIR="$(dirname "$(dirname "$THIS_DIR")")"
LOGDIR="$THIS_DIR/logs"
OUT_ROOT="$THIS_DIR/runs"
mkdir -p "$LOGDIR" "$OUT_ROOT"
cd "$PKG_DIR"

# Pre-create the search root so `find` doesn't error out on the first run
# (set -o pipefail + find-on-missing-dir would otherwise abort the script).
mkdir -p "$OUT_ROOT/runs/classification"

SEEDS="1,2,3,4,5"
N_TRIALS=50  # same as the AUC-tuned classifier
INNER_CV=5
N=300

for FEATURES in fs_a_only fs_b_only fs_cv fs_all; do
    SLUG="xgb_classifier__classification__${FEATURES}__optimize_f1"
    # Idempotent skip
    EXISTING=$(find "$OUT_ROOT/runs/classification" -maxdepth 1 -type d \
        -name "xgb_classifier__classification__N${N}__A2.2_b1__${FEATURES}__*" 2>/dev/null \
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
            --task classification \
            --model-config configs/xgb_classifier_f1.yaml \
            --N "$N" \
            --db-version A2.2 \
            --baseline-cycle 1 \
            --feature-subset "$FEATURES" \
            --tuning-protocol tune_inner_cv \
            --tune.n-trials "$N_TRIALS" \
            --tune.inner-cv "$INNER_CV" \
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
echo "=== Experiment F DONE ==="

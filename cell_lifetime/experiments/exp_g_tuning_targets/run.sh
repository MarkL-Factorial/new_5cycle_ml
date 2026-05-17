#!/bin/bash
# Experiment G: AUC-tuning vs F1-tuning across model families and horizons.
#
# Grid: 4 models × 2 tune targets × 2 feature subsets × 3 N = 48 cells.
# Of these:
#   - xgb_classifier × {auc, f1} × {fs_a, fs_cv} × N=300 were done in Exp F
#     (those live in cell_lifetime/out/runs/ and exp_f_optimize_target/runs/).
#     Re-run them here for a single-location dataset to simplify aggregation.
# So total fresh: 48 runs. Driver is idempotent (skip if 5-seed already exists).
#
# Output goes to experiments/exp_g_tuning_targets/runs/ to keep clean.

set -euo pipefail

export OMP_NUM_THREADS=10
export OPENBLAS_NUM_THREADS=10
export MKL_NUM_THREADS=10
export NUMEXPR_NUM_THREADS=10

THIS_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PKG_DIR="$(dirname "$(dirname "$THIS_DIR")")"
LOGDIR="$THIS_DIR/logs"
OUT_ROOT="$THIS_DIR/runs"
mkdir -p "$LOGDIR" "$OUT_ROOT/runs/classification" "$OUT_ROOT/runs/survival"
cd "$PKG_DIR"

SEEDS="1,2,3,4,5"
# Trial budgets — EBM lighter (it's the slow one)
EBM_TRIALS=15
EBM_INNER=3
DEFAULT_TRIALS=30
DEFAULT_INNER=5

# Each entry: "config-stem task tune_target_label"
# We use distinct config files for each tune target (defined in configs/).
declare -a JOBS=(
    "xgb_classifier        classification auc"
    "xgb_classifier_f1     classification f1"
    "ebm_classifier        classification auc"
    "ebm_classifier_f1     classification f1"
    "rsf_auc_at_N          survival       auc"
    "rsf_f1_at_N           survival       f1"
    "xgb_aft_auc_at_N      survival       auc"
    "xgb_aft_f1_at_N       survival       f1"
)

for FEATURES in fs_a_only fs_cv; do
    for N in 200 300 400; do
        for ENTRY in "${JOBS[@]}"; do
            read -r CONFIG_STEM TASK TUNE_LABEL <<< "$ENTRY"
            # The slug emitted by the CLI uses the YAML's `model:` field (not the config stem).
            # For idempotency we have to read the YAML's `model:` to know what subdir to check.
            MODEL_NAME=$(grep -E '^model:' "configs/${CONFIG_STEM}.yaml" | awk '{print $2}')
            SLUG_PREFIX="${MODEL_NAME}__${TASK}__N${N}__A2.2_b1__${FEATURES}"
            # Decide trial budget
            if [[ "$CONFIG_STEM" == ebm_* ]]; then
                TRIALS=$EBM_TRIALS
                INNER=$EBM_INNER
            else
                TRIALS=$DEFAULT_TRIALS
                INNER=$DEFAULT_INNER
            fi
            # Idempotent skip: if a 5-seed run exists for this slug AND was tuned
            # on the matching optimize target, skip. Otherwise the previous one
            # is from a different tune target and we should run again.
            EXISTING=$(find "$OUT_ROOT/runs/${TASK}" -maxdepth 1 -type d \
                -name "${SLUG_PREFIX}__*" 2>/dev/null \
                -exec test -s "{}/summary.json" \; -print 2>/dev/null \
                | while read d; do
                    M=$(python -c "import json; d=json.load(open('$d/summary.json')); print(d.get('n_seeds',0), d.get('tune_objective',''))" 2>/dev/null || echo "0 ?")
                    N_S=$(echo "$M" | awk '{print $1}')
                    OBJ=$(echo "$M" | awk '{print $2}')
                    EXPECTED_OBJ=$(grep -E '^  optimize:' "configs/${CONFIG_STEM}.yaml" | awk '{print $2}')
                    if [ "$N_S" -ge 5 ] && [ "$OBJ" = "$EXPECTED_OBJ" ]; then
                        echo "$d"
                        break
                    fi
                done | head -1)
            if [ -n "$EXISTING" ]; then
                echo "[skip] $SLUG_PREFIX (${TUNE_LABEL}-tuned) already exists at $EXISTING"
                continue
            fi
            TS=$(date -u +%Y%m%dT%H%M%SZ)
            LOG="$LOGDIR/${TS}__${MODEL_NAME}__${TUNE_LABEL}__${FEATURES}__N${N}.log"
            echo
            echo "=== [$TS] ${MODEL_NAME} tune=${TUNE_LABEL} subset=${FEATURES} N=${N} ==="
            echo "  log: $LOG"
            START=$(date +%s)
            if cell-lifetime run \
                    --task "$TASK" \
                    --model-config "configs/${CONFIG_STEM}.yaml" \
                    --N "$N" \
                    --db-version A2.2 \
                    --baseline-cycle 1 \
                    --feature-subset "$FEATURES" \
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
    done
done

echo
echo "=== Experiment G DONE ==="

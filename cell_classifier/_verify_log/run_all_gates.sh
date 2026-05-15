#!/bin/bash
# Master verification: runs gates 1-10 sequentially, logs each, writes STATUS.md.
# Each gate's stdout/stderr → _verify_log/{NN_name}.log
# Pass/fail per gate → STATUS.md (appended)
#
# Continues past failures so all gates produce a log entry. Gate dependencies
# noted inline.

set -u
cd "$(dirname "$0")/.."

PY=/home/mliao/miniconda3/envs/mldashboard/bin/python
CC=/home/mliao/miniconda3/envs/mldashboard/bin/cell-classifier
PYTEST=/home/mliao/miniconda3/envs/mldashboard/bin/pytest
OUT_ROOT=/tmp/cc_gates
LOG=_verify_log
STATUS=$LOG/STATUS.md

rm -rf $OUT_ROOT
mkdir -p $OUT_ROOT

echo "" >> $STATUS
echo "## Run $(date -u +%FT%TZ)" >> $STATUS
echo "" >> $STATUS

record() {
  local gate=$1
  local exit_code=$2
  local note=$3
  if [ $exit_code -eq 0 ]; then
    echo "- [x] **gate $gate** — PASS — $note" >> $STATUS
  else
    echo "- [ ] **gate $gate** — FAIL (exit=$exit_code) — $note" >> $STATUS
  fi
}

# ---------------- gate 1: pytest ----------------
echo "[$(date -u +%FT%TZ)] gate 1: pytest" | tee -a $STATUS
$PYTEST tests/ 2>&1 | tee $LOG/01_pytest.log >/dev/null
record 1 ${PIPESTATUS[0]} "pytest tests/"

# ---------------- gate 2: validation tune_inner_cv ----------------
echo "[$(date -u +%FT%TZ)] gate 2: validation tune_inner_cv"
$CC run --mode validation --model-config configs/rf.yaml \
  --N 300 --db-version A2.2 --baseline-cycle 1 --feature-subset fs_cv \
  --tuning-protocol tune_inner_cv --tune.n-trials 10 --tune.inner-cv 5 \
  --seeds 1,2,3 --out-root $OUT_ROOT \
  2>&1 | tee $LOG/02_val_tune_inner_cv.log >/dev/null
record 2 ${PIPESTATUS[0]} "validation tune_inner_cv (3 seeds, 10 trials)"

# ---------------- gate 3: validation nested_cv ----------------
echo "[$(date -u +%FT%TZ)] gate 3: validation nested_cv"
# Use baseline=3 to avoid colliding with gate 2's bundle
$CC run --mode validation --model-config configs/rf.yaml \
  --N 300 --db-version A2.2 --baseline-cycle 3 --feature-subset fs_cv \
  --tuning-protocol nested_cv --outer-k 3 \
  --tune.n-trials 5 --tune.inner-cv 3 \
  --seeds 1,2 --out-root $OUT_ROOT \
  2>&1 | tee $LOG/03_val_nested_cv.log >/dev/null
record 3 ${PIPESTATUS[0]} "validation nested_cv (2 seeds, outer_k=3, 5 trials)"

# ---------------- gate 4: idempotency + --force ----------------
echo "[$(date -u +%FT%TZ)] gate 4: idempotency"
# Re-run gate 2's command — must skip (exit 0, log says skip)
$CC run --mode validation --model-config configs/rf.yaml \
  --N 300 --db-version A2.2 --baseline-cycle 1 --feature-subset fs_cv \
  --tuning-protocol tune_inner_cv --tune.n-trials 10 --tune.inner-cv 5 \
  --seeds 1,2,3 --out-root $OUT_ROOT \
  2>&1 | tee $LOG/04a_idempotency_skip.log >/dev/null
skip_ok=${PIPESTATUS[0]}
grep -q "\[skip\]" $LOG/04a_idempotency_skip.log && skip_msg="OK (skip detected)" || skip_msg="FAIL (no skip)"

# Different n_trials without --force → must error
$CC run --mode validation --model-config configs/rf.yaml \
  --N 300 --db-version A2.2 --baseline-cycle 1 --feature-subset fs_cv \
  --tuning-protocol tune_inner_cv --tune.n-trials 20 --tune.inner-cv 5 \
  --seeds 1,2,3 --out-root $OUT_ROOT \
  2>&1 | tee $LOG/04b_idempotency_mismatch.log >/dev/null
mismatch_exit=${PIPESTATUS[0]}
[ $mismatch_exit -ne 0 ] && mismatch_msg="OK (errors)" || mismatch_msg="FAIL (should error)"

# Same with --force → must overwrite (exit 0)
$CC run --mode validation --model-config configs/rf.yaml \
  --N 300 --db-version A2.2 --baseline-cycle 1 --feature-subset fs_cv \
  --tuning-protocol tune_inner_cv --tune.n-trials 20 --tune.inner-cv 5 \
  --seeds 1,2,3 --out-root $OUT_ROOT --force \
  2>&1 | tee $LOG/04c_idempotency_force.log >/dev/null
force_exit=${PIPESTATUS[0]}
[ $force_exit -eq 0 ] && force_msg="OK (overwrites)" || force_msg="FAIL"

if [ "$skip_msg" = "OK (skip detected)" ] && [ "$mismatch_msg" = "OK (errors)" ] && [ "$force_msg" = "OK (overwrites)" ]; then
  record 4 0 "idempotency skip/mismatch/force all behave as expected"
else
  record 4 1 "skip=[$skip_msg], mismatch=[$mismatch_msg], force=[$force_msg]"
fi

# ---------------- gate 5: production from_validation_run ----------------
echo "[$(date -u +%FT%TZ)] gate 5: production from_validation_run"
# Note: gate 4c re-ran validation with n_trials=20 which produced best_params for seeds 1,2,3.
# Production uses those.
$CC run --mode production --model-config configs/rf.yaml \
  --N 300 --db-version A2.2 --baseline-cycle 1 --feature-subset fs_cv \
  --params-source from_validation_run \
  --seeds 1,2,3 --out-root $OUT_ROOT \
  2>&1 | tee $LOG/05_prod_from_val.log >/dev/null
record 5 ${PIPESTATUS[0]} "production from_validation_run (3 seeds)"

# ---------------- gate 6: production retune ----------------
echo "[$(date -u +%FT%TZ)] gate 6: production retune"
# Use baseline=3 to avoid colliding with gate 5
$CC run --mode production --model-config configs/rf.yaml \
  --N 300 --db-version A2.2 --baseline-cycle 3 --feature-subset fs_cv \
  --params-source retune --tune.n-trials 5 --tune.inner-cv 3 \
  --seeds 1,2,3 --out-root $OUT_ROOT \
  2>&1 | tee $LOG/06_prod_retune.log >/dev/null
record 6 ${PIPESTATUS[0]} "production retune (3 seeds, baseline=3)"

# ---------------- gate 7: sweep ----------------
echo "[$(date -u +%FT%TZ)] gate 7: sweep (2x1 grid, tiny config)"
# Place sweep yaml in configs/sweeps/ so `template: ../rf.yaml` resolves.
cat > configs/sweeps/_smoke_2x1.yaml <<EOF
template: ../rf.yaml
mode: validation
tuning_protocol: tune_inner_cv
sweep_id: smoke_2x1
axes:
  N: [200, 300]
  baseline_cycle: [1]
fixed:
  model: random_forest
  db_version: A2.2
  feature_subset: fs_cv
  out_root: $OUT_ROOT
  seeds: [1, 2, 3]
  tune:
    n_trials: 5
    inner_cv: 3
EOF
$CC sweep --sweep configs/sweeps/_smoke_2x1.yaml --force 2>&1 | tee $LOG/07_sweep.log >/dev/null
record 7 ${PIPESTATUS[0]} "sweep 2x1 grid (3 seeds, 5 trials)"

# ---------------- gate 8: discover ----------------
echo "[$(date -u +%FT%TZ)] gate 8: discover"
$PY - <<EOF 2>&1 | tee $LOG/08_discover.log >/dev/null
from pathlib import Path
from cell_classifier.utils.discover import find_runs
root = Path("$OUT_ROOT")
all_runs = find_runs(out_root=root)
print(f"total runs discovered: {len(all_runs)}")
for r in all_runs:
    print(f"  {r['mode']:<10s} {r['slug']}")
val_n300 = find_runs(out_root=root, mode="validation", N=300)
print(f"validation@N=300: {len(val_n300)}")
assert len(all_runs) >= 4, f"expected >=4, got {len(all_runs)}"
assert len(val_n300) >= 1, f"expected >=1 val@N=300, got {len(val_n300)}"
print("OK")
EOF
record 8 ${PIPESTATUS[0]} "discover finds runs from prior gates"

# ---------------- gate 9: AST audit (re-run targeted) ----------------
echo "[$(date -u +%FT%TZ)] gate 9: AST audit"
$PYTEST tests/test_no_mode_branching.py tests/test_no_cohort_features.py 2>&1 | tee $LOG/09_ast_audit.log >/dev/null
record 9 ${PIPESTATUS[0]} "AST audit + data-leakage canary"

# ---------------- gate 10: v2 cross-check 50 seeds ----------------
echo "[$(date -u +%FT%TZ)] gate 10: v2 cross-check (50 seeds, full tuning)"
$CC run --mode validation --model-config configs/rf.yaml \
  --N 300 --db-version A2.2 --baseline-cycle 1 --feature-subset fs_cv \
  --tuning-protocol tune_inner_cv --tune.n-trials 100 --tune.inner-cv 5 \
  --seeds-preset fresh \
  --out-root $OUT_ROOT --force \
  2>&1 | tee $LOG/10_v2_crosscheck.log >/dev/null
g10_exit=${PIPESTATUS[0]}
if [ $g10_exit -eq 0 ]; then
  $PY - <<EOF 2>&1 | tee -a $LOG/10_v2_crosscheck.log
import json
from pathlib import Path
v3_summary = json.loads(Path("$OUT_ROOT/runs/validation/rf__N300__A2.2_b1__fs_cv/summary.json").read_text())
v2_summary = json.loads(Path("/mnt/data/mliao/test_new_5cycle_classification/ml_classification_v2/out/rf_n300/summary.json").read_text())
delta_auc = v3_summary["test_roc_auc_mean"] - v2_summary["test_roc_auc_mean"]
delta_f1  = v3_summary["test_f1_mean"]      - v2_summary["test_f1_mean"]
print(f"v3 AUC = {v3_summary['test_roc_auc_mean']:.4f} ± {v3_summary['test_roc_auc_std']:.4f}")
print(f"v2 AUC = {v2_summary['test_roc_auc_mean']:.4f} ± {v2_summary['test_roc_auc_std']:.4f}")
print(f"Δ AUC = {delta_auc:+.4f}")
print(f"v3 F1  = {v3_summary['test_f1_mean']:.4f} ± {v3_summary['test_f1_std']:.4f}")
print(f"v2 F1  = {v2_summary['test_f1_mean']:.4f} ± {v2_summary['test_f1_std']:.4f}")
print(f"Δ F1  = {delta_f1:+.4f}")
ok = abs(delta_auc) < 0.01 and abs(delta_f1) < 0.01
print("CROSS-CHECK OK" if ok else "CROSS-CHECK FAILED")
EOF
  cross_ok=$?
  [ $cross_ok -eq 0 ] && record 10 0 "AUC/F1 vs v2 within 0.01" || record 10 1 "AUC/F1 drift > 0.01"
else
  record 10 $g10_exit "training failed"
fi

echo "" >> $STATUS
echo "Finished $(date -u +%FT%TZ)" >> $STATUS

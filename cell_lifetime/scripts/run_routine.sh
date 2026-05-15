#!/bin/bash
# Cloud /schedule routine wrapper.
#
# Usage (inside the routine's bash step):
#     bash cell_lifetime/scripts/run_routine.sh phase2_xgb_aft
#     bash cell_lifetime/scripts/run_routine.sh phase3_rsf_and_summary
#
# Each invocation:
#   1. checks out feature/cell_lifetime
#   2. installs deps
#   3. runs the routine-specific work (sourced from cell_lifetime/scripts/<phase>.sh
#      if it exists, otherwise expects the calling /schedule prompt to do the work
#      after this prologue returns)
#   4. tests, commits, pushes
#   5. appends to INDEX.md and writes a per-phase summary

set -euo pipefail

# Workspace cap: max 10 cores total across all automation.
export OMP_NUM_THREADS=10
export OPENBLAS_NUM_THREADS=10
export MKL_NUM_THREADS=10
export NUMEXPR_NUM_THREADS=10

PHASE="${1:?usage: bash run_routine.sh <phase2_xgb_aft|phase3_rsf_and_summary>}"
TS="$(date -u +%Y%m%dT%H%M%SZ)"
WORKSPACE_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
WORKSPACE_ROOT="$(dirname "$WORKSPACE_ROOT")"

# --- log capture ---
LOGDIR="$WORKSPACE_ROOT/cell_lifetime/run_logs"
mkdir -p "$LOGDIR"
LOG="$LOGDIR/${TS}_${PHASE}.log"
SUM="$LOGDIR/${TS}_${PHASE}.summary.md"
TESTXML="$LOGDIR/${TS}_${PHASE}.test.xml"
exec > >(tee "$LOG") 2>&1
trap 'echo -e "# $PHASE — FAILED at $(date -u +%FT%TZ)\n\nSee $LOG\n" > "$SUM"' ERR

cd "$WORKSPACE_ROOT"

echo "=== run_routine.sh ==="
echo "  PHASE = $PHASE"
echo "  TS    = $TS"
echo "  CWD   = $(pwd)"

# --- repo state ---
git fetch origin
git checkout feature/cell_lifetime
git pull --ff-only origin feature/cell_lifetime

# --- env install ---
bash cell_lifetime/scripts/install_env.sh

# --- idempotency check: skip if INDEX says we already did this phase ---
if grep -qE "^\| ${PHASE} \|.*\| OK \|" cell_lifetime/INDEX.md; then
    echo "[$PHASE] INDEX.md already shows status=OK — exiting (idempotent)"
    echo -e "# $PHASE — already complete (idempotent skip)\n" > "$SUM"
    exit 0
fi

# --- per-phase work: routine-specific script if present ---
PHASE_SCRIPT="cell_lifetime/scripts/${PHASE}.sh"
if [ -f "$PHASE_SCRIPT" ]; then
    echo "[$PHASE] executing $PHASE_SCRIPT"
    bash "$PHASE_SCRIPT"
else
    echo "[$PHASE] no $PHASE_SCRIPT — caller does work after run_routine.sh prologue"
fi

# --- tests ---
pytest cell_lifetime/tests -q --junitxml="$TESTXML" || TESTS_FAILED=1

# --- summary ---
N_PASS=$(grep -oE 'tests="[0-9]+"' "$TESTXML" 2>/dev/null | head -1 | grep -oE '[0-9]+' || echo "0")
N_FAIL=$(grep -oE 'failures="[0-9]+"' "$TESTXML" 2>/dev/null | head -1 | grep -oE '[0-9]+' || echo "0")
{
    echo "# $PHASE — $TS"
    echo
    echo "- tests run: $N_PASS"
    echo "- tests failed: $N_FAIL"
    echo "- log: \`run_logs/${TS}_${PHASE}.log\`"
    echo "- junit: \`run_logs/${TS}_${PHASE}.test.xml\`"
    echo
    echo "## Files in this commit"
    git status --short
} > "$SUM"

# --- INDEX append ---
STATUS="OK"
[ "${TESTS_FAILED:-0}" = "1" ] && STATUS="BROKEN"
python cell_lifetime/scripts/append_index.py \
    --phase "$PHASE" --ts "$TS" --status "$STATUS" \
    --n-pass "$N_PASS" --n-fail "$N_FAIL" \
    --summary "run_logs/${TS}_${PHASE}.summary.md"

# --- commit + push ---
PREFIX="[$PHASE]"
[ "$STATUS" = "BROKEN" ] && PREFIX="[BROKEN][$PHASE]"
git add cell_lifetime/
git commit -m "${PREFIX} ${TS} — see run_logs/${TS}_${PHASE}.summary.md" || echo "nothing to commit"
git push origin feature/cell_lifetime

echo "=== run_routine.sh DONE — status=$STATUS ==="

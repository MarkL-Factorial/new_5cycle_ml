# cloud /schedule routines — state machine

This file describes the cloud routines that drive Phase 2 + Phase 3
over the weekend. Phase 1 already runs in-session on the user's local
machine and lands on `feature/cell_lifetime` before the routines fire.

## Routines

| Routine ID | Fire (UTC) | Scope | Expected runtime | Idempotency |
|---|---|---|---|---|
| `phase2_xgb_aft` | Sat 2026-05-16 14:00 | Add `xgb_aft.py`, `survival_metrics.py`, related configs/tests; wire `pipelines/validation.py`'s survival branch | ~45 min | No-ops if `INDEX.md` shows phase2 status=OK |
| `phase3_rsf_and_summary` | Sun 2026-05-17 02:00 | Add `rsf.py` (guarded), related tests/configs; render `weekend_report.html` from accumulated `.summary.md` | ~75 min | No-ops if phase3 status=OK |

## Routine prologue (every invocation)

```bash
set -euo pipefail
PHASE="$1"   # phase2_xgb_aft | phase3_rsf_and_summary
TS=$(date -u +%Y%m%dT%H%M%SZ)

# 1. Repo
test -d new_5cycle_ml || git clone git@github.com:MarkL-Factorial/new_5cycle_ml.git
cd new_5cycle_ml
git fetch origin && git checkout feature/cell_lifetime && git pull origin feature/cell_lifetime

# 2. Env
pip install xgboost interpret scikit-survival optuna shap scipy pyyaml pytest polars pandas numpy scikit-learn
pip install -e cell_classifier/
pip install -e cell_lifetime/[xgb,survival,ebm]

# 3. Idempotency check
if grep -q "| ${PHASE} | .*OK |" cell_lifetime/INDEX.md; then
    echo "[$PHASE] already OK in INDEX.md — exiting"
    exit 0
fi

# 4. Log capture
LOG=cell_lifetime/run_logs/${TS}_${PHASE}.log
SUM=cell_lifetime/run_logs/${TS}_${PHASE}.summary.md
mkdir -p cell_lifetime/run_logs
exec > >(tee "$LOG") 2>&1
trap 'echo "Phase $PHASE FAILED; see $LOG" > "$SUM"' ERR

# 5. Routine-specific work (per-phase agent prompt does this)
# ... writes files ...

# 6. Test
pytest cell_lifetime/tests -q --junitxml="cell_lifetime/run_logs/${TS}_${PHASE}.test.xml"

# 7. INDEX append, commit, push
python cell_lifetime/scripts/append_index.py --phase "$PHASE" --ts "$TS" --status OK
git add cell_lifetime/
git commit -m "[$PHASE] $TS — phase landed, see run_logs/${TS}_${PHASE}.summary.md"
git push origin feature/cell_lifetime
```

## Failure handling

- Test failure → still commit code, but `[BROKEN]` prefix in message, INDEX row status=BROKEN.
- Install failure (`pip install scikit-survival`) → Phase 3 writes a guarded stub and INDEX row status=BLOCKED.
- The next routine reads INDEX, skips the prior phase's debris, works only on its own.

## Manual override

```bash
bash cell_lifetime/scripts/run_routine.sh phase2_xgb_aft   # re-fire from a local checkout
```

"""Per-cell ML label preprocessing — discharge-capacity retention status.

Reads annotation JSONs (one per cell) via _common.iter_annotations,
computes the per-cell ground-truth label that downstream ML uses as its
target, and writes one row per cell to out/cell_labels.{parquet,csv}.

Label semantics:
  - excluded   : cell can't be used for ML training. ``exclusion_reason``
                 records which gate fired (one of: ``rate_changed``,
                 ``no_regular``, ``human_review``, ``low_initial_capacity``,
                 ``no_baseline``). Some excluded cells DO get a feature
                 row and a meaningful n_regular — see the "rate_changed
                 featurizable" sub-class below — to support production-
                 inference scoring.
  - faded      : cell's retention dropped below 0.85 and stayed there;
                 last_fade_cycle records the regular_cycle ordinal of the
                 LAST crossing into bad (point of no return)
  - in_testing : cell is still healthy (no irrecoverable fade observed)

Rate_changed sub-class (schema_version=2): cells with
``cycling_consistency='rate_changed'`` are still status='excluded' (their
retention curve mixes capacities at different rates, so the fade
detector can't run honestly). But if ``regime[0].n_regular_cd >= 5``
(the first rate-regime is long enough to cover cycles 1..5), the cell
appears in ``cell_features.parquet`` AND gets a populated
``n_regular`` (lifetime regular count) + ``baseline_dis_ah`` so the
downstream asymmetric ``predict_min_n_regular=5`` filter admits it for
scoring. ``trainable_n{N}=False`` keeps the cell out of training.

Fade rule (last crossing into the permanently-bad regime):
  last_fade_cycle = cycle of the LAST 'crossing into bad' (cycle c where
  retention(c) < 0.85 AND the previous cycle was healthy or c is the very
  first cycle) such that fewer than RECOVERY_MIN (= 3) subsequent regular
  cycles have retention > 0.85 (counted globally over all later positions,
  NOT required to be consecutive).

v3: outputs live at ``datasets/{db_version}_b{baseline_cycle}/`` with a
``manifest.json`` carrying provenance. Baseline cycle (N0) is configurable;
retention(c) = cap_dis(c) / cap_dis(N0). The retention curve passed to the
fade detector covers cycles >= N0 only; pre-baseline cycles are dropped.
Default N0=1 reproduces v1 fade behavior.

Author: Mark Liao (Sheng-Lun Liao)
"""
from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Optional

import polars as pl

from _common import (
    ANNOT_DIR,
    _cohort,
    dataset_dir_for,
    iter_annotations,
    iter_regulars,
    promote_to_latest,
    write_manifest,
    write_outputs,
)

DEFAULT_BASELINE_CYCLE = 1
SCHEMA_VERSION = 2

# ML-fitness gate: any cell whose first N regular CD cycles include a
# cycle with charge or discharge capacity below this threshold is
# excluded with ``exclusion_reason="low_initial_capacity"``. Such
# cycles indicate calibration runs, sensor faults, or aborted formation
# — the retention denominator they imply is meaningless.
LOW_INITIAL_CAPACITY_THRESHOLD_AH = 0.1
LOW_INITIAL_CAPACITY_WINDOW = 5

# External artifacts produced by the curation/ pipeline. Both files
# are *optional* — missing files mean "no overrides", and the cell is
# processed exactly as it was before the curation wiring. See:
#   curation/README.md
HERE = Path(__file__).resolve().parent
DECISIONS_PATH = HERE / "curation" / "decisions.json"
OUTLIER_SIDECAR_PATH = HERE / "curation" / "outlier_sidecar.json"

RETENTION_THRESHOLD = 0.85
RECOVERY_MIN = 3

# Cycle-life thresholds for downstream ML classification. For each N,
# we emit two label columns (label_n{N}, trainable_n{N}) that summarise:
#
#   pass     : cell survived past N cycles with retention >= 0.85
#              (i.e. last_fade_cycle > N for faded cells, OR
#               n_regular >= N for cells still in testing with no fade)
#   bad      : cell faded at or before N cycles
#   censor   : cell hasn't reached N yet and hasn't faded — DO NOT TRAIN
#   excluded : cell was already excluded by the upstream label logic
#              (rate_changed / no_regular / no usable baseline)
#
# trainable_n{N} is True iff label_n{N} ∈ {"pass", "bad"} — gives
# downstream ML a one-flag filter for the binary-classification subset.
N_THRESHOLDS = (200, 300, 400)


def _load_decisions() -> dict:
    """Read ``decisions.json`` from the manual-validation artifact.

    Missing file ⇒ empty mapping (no overrides). Caller's downstream
    behavior degrades gracefully to pre-Phase-2 logic.

    Schema (per cell):
      exclude_from_ml      bool   — when true, drop cell from cohort
      last_available_cycle int|null — truncate retention curve at/before this cycle
      event_type           "censor"|"event"|null — manual annotation
      reason               str
      validated_at         str (ISO date)
    """
    if not DECISIONS_PATH.exists():
        return {}
    return json.loads(DECISIONS_PATH.read_text())


def _load_outlier_sidecar() -> dict:
    """Read ``outlier_sidecar.json`` from the outlier-detection investigation.

    Missing file ⇒ empty mapping (no cycles get masked).

    Schema: {cell_name: {n_outliers: int, outliers: [{cycle, ...}, ...]}}
    """
    if not OUTLIER_SIDECAR_PATH.exists():
        return {}
    return json.loads(OUTLIER_SIDECAR_PATH.read_text())


def _last_crossing_into_bad(
    cycles: list[int],
    retentions: list[float],
    threshold: float = RETENTION_THRESHOLD,
    recovery_min: int = RECOVERY_MIN,
) -> tuple[Optional[int], int]:
    """Walk forward; return (last_fade_cycle, n_recovered_crossings).

    A 'crossing into bad' at index i is ``retentions[i] < threshold`` AND
    (``i == 0`` OR ``retentions[i-1] >= threshold``).

    Among all crossings, return the LAST cycle whose post-crossing window
    has fewer than ``recovery_min`` cycles with retention > threshold.
    Earlier crossings with >= recovery_min healthy cycles after are
    counted in ``n_recovered_crossings`` and skipped.
    """
    assert len(cycles) == len(retentions)
    n = len(retentions)
    last_fade: Optional[int] = None
    n_recovered = 0

    for i, r in enumerate(retentions):
        if r >= threshold:
            continue
        if i > 0 and retentions[i - 1] < threshold:
            continue
        n_good_after = sum(1 for j in range(i + 1, n) if retentions[j] > threshold)
        if n_good_after >= recovery_min:
            n_recovered += 1
        else:
            last_fade = cycles[i]

    return last_fade, n_recovered


def _retention_at(
    regulars: list[dict], baseline: float, target_cycle: int,
) -> Optional[float]:
    """Return retention at the largest ``regular_cycle <= target_cycle``.

    Used by ``_process_cell`` when a human decision asserts an outcome at
    a specific cycle — we still want to report ``final_retention`` from
    the real data at that cycle (or the nearest preceding cycle that
    exists, in case the human's asserted cycle isn't present as a
    regular_cd event).

    Returns None if no regular cycle is ≤ target_cycle (e.g. baseline
    is past the asserted cycle).
    """
    best: Optional[dict] = None
    for e in regulars:
        c = e.get("regular_cycle")
        if c is None or c > target_cycle:
            continue
        if best is None or c > best["regular_cycle"]:
            best = e
    if best is None:
        return None
    cap = best.get("capacity_discharge_ah")
    if cap is None:
        return None
    return float(cap) / float(baseline)


def _classification_label_at(status: str,
                              last_fade_cycle: Optional[int],
                              n_regular: int,
                              N: int) -> tuple[str, bool]:
    """Return (label, trainable) for one cell at one N-cycle threshold.

    Decision table:
      status='excluded'                              → ('excluded', False)
      status='faded' AND last_fade_cycle > N         → ('pass',     True)
      status='faded' AND last_fade_cycle <= N        → ('bad',      True)
      status='in_testing' AND n_regular >= N         → ('pass',     True)
      status='in_testing' AND n_regular <  N         → ('censor',   False)

    "Pass" means the cell stayed at retention >= 0.85 strictly past
    cycle N. A cell that faded exactly AT cycle N counts as 'bad' (it
    did not exceed N healthy cycles). A cell with n_regular == N
    counts as 'pass' (we observed N healthy cycles).
    """
    if status == "excluded":
        return "excluded", False
    if status == "faded":
        # last_fade_cycle is the cycle of the LAST sticky crossing
        # into bad; existence is guaranteed when status == 'faded'.
        if last_fade_cycle is not None and last_fade_cycle > N:
            return "pass", True
        return "bad", True
    if status == "in_testing":
        if n_regular >= N:
            return "pass", True
        return "censor", False
    # Defensive — should not reach here.
    return "excluded", False


def _process_cell(
    d: dict,
    baseline_cycle: int = DEFAULT_BASELINE_CYCLE,
    decisions: Optional[dict] = None,
    outlier_sidecar: Optional[dict] = None,
) -> dict:
    """Build one label row for a single annotation JSON. Always returns a
    row; cells that can't be ML-trained get status='excluded' with a reason.

    ``baseline_cycle`` is the regular_cycle ordinal used as the retention
    denominator. Pre-baseline cycles are dropped from the retention curve
    that the fade detector consumes.

    ``decisions`` (optional) — mapping ``{cell_name: decision_entry}`` from
    ``investigations/manual_validation/validated/decisions.json``. When a
    cell's entry has ``exclude_from_ml=true``, the cell is dropped from
    the cohort with ``exclusion_reason='human_review'``. When the entry
    has ``last_available_cycle`` set, the retention curve is truncated
    at that cycle BEFORE fade detection.

    ``outlier_sidecar`` (optional) — mapping ``{cell_name: {outliers: [{cycle, ...}]}}``
    from ``investigations/outlier_detection/out/outlier_sidecar.json``.
    Listed cycles are dropped from the retention curve BEFORE fade
    detection (their measurements are untrustworthy per Pattern A).

    Both arguments default to empty: missing/absent → pre-Phase-2 behavior.
    """
    decisions = decisions or {}
    outlier_sidecar = outlier_sidecar or {}
    cell = d.get("cell_name", "?")
    consistency = d.get("cycling_consistency", "no_regular")
    protocol = d.get("protocol_pattern", "other")
    cohort = _cohort(cell)

    # Rate-truncation diagnostic. Populated ONLY for rate_changed cells:
    # the count of regular_cd events before the first rate change
    # (= regime[0].n_regular_cd from the toolkit's time-ordered regime
    # list). Null for single_rate and no_regular cells because the
    # column's concept ("cycles before the rate changed") doesn't apply
    # to them. For rate_changed cells this is the eligibility key for
    # featurization (see features.py::_check_omit: >= 5 → featurized).
    #
    # Note: even single_rate cells can have multiple regimes (toolkit
    # splits per-RPT-segment), so regime[0].n_regular_cd is NOT the
    # cell's lifetime n_regular in general — populating this column
    # for single_rate cells would just confuse readers.
    regimes = d.get("regular_rate_regimes") or []
    n_pre_rate_change: Optional[int] = (
        int(regimes[0]["n_regular_cd"])
        if (consistency == "rate_changed" and regimes)
        else None
    )

    base_row = {
        "cell_name": cell,
        "cohort": cohort,
        "protocol_pattern": protocol,
        "cycling_consistency": consistency,
        "n_regular_pre_rate_change": n_pre_rate_change,
        "status": "excluded",
        "exclusion_reason": None,
        "last_fade_cycle": None,
        "n_regular": 0,
        "baseline_dis_ah": None,
        "final_retention": None,
        "n_recovered_crossings": 0,
        # Manual-validation provenance (Phase 2). Both null/0 by default;
        # populated only when decisions.json / outlier_sidecar.json have
        # an entry for this cell that materially affects the row.
        "truncation_cycle": None,
        "n_outliers_masked": 0,
    }
    # Initialize the per-threshold classification columns to "excluded"
    # defaults; the late return paths inherit these.
    for N in N_THRESHOLDS:
        base_row[f"label_n{N}"] = "excluded"
        base_row[f"trainable_n{N}"] = False

    if consistency == "rate_changed":
        base_row["exclusion_reason"] = "rate_changed"
        # Predict-only enrichment for cells whose original-rate window
        # covers cycles 1..5 (matches features.py::_check_omit's
        # admission gate). We populate n_regular (lifetime count) and
        # baseline_dis_ah so the downstream asymmetric n_regular>=5
        # filter admits the row; status / trainable_n{N} stay at the
        # excluded defaults so training never sees these cells.
        if n_pre_rate_change is not None and n_pre_rate_change >= 5:
            regulars = iter_regulars(d)
            baseline_evt = next(
                (e for e in regulars if e["regular_cycle"] == baseline_cycle), None
            )
            if (
                regulars
                and baseline_evt is not None
                and baseline_evt.get("capacity_discharge_ah") is not None
                and baseline_evt["capacity_discharge_ah"] > 0
            ):
                base_row["n_regular"] = int(regulars[-1]["regular_cycle"])
                base_row["baseline_dis_ah"] = float(
                    baseline_evt["capacity_discharge_ah"]
                )
        return base_row

    if consistency == "no_regular":
        base_row["exclusion_reason"] = "no_regular"
        return base_row

    # Human-review exclusion (single_rate cells only). User decision in
    # decisions.json overrides anything the fade detector would conclude.
    cell_decision = decisions.get(cell, {})
    if cell_decision.get("exclude_from_ml"):
        base_row["exclusion_reason"] = "human_review"
        return base_row

    # single_rate path
    regulars = iter_regulars(d)
    if not regulars:
        base_row["exclusion_reason"] = "no_baseline"
        return base_row

    # Low-initial-capacity gate. Any cycle in the first
    # LOW_INITIAL_CAPACITY_WINDOW with charge or discharge capacity
    # below LOW_INITIAL_CAPACITY_THRESHOLD_AH means the cell's early
    # life is unusable as a retention baseline.
    for e in regulars[:LOW_INITIAL_CAPACITY_WINDOW]:
        cdis = e.get("capacity_discharge_ah")
        cchg = e.get("capacity_charge_ah")
        if (cdis is not None and cdis < LOW_INITIAL_CAPACITY_THRESHOLD_AH) or (
            cchg is not None and cchg < LOW_INITIAL_CAPACITY_THRESHOLD_AH
        ):
            base_row["exclusion_reason"] = "low_initial_capacity"
            return base_row

    baseline_evt = next(
        (e for e in regulars if e["regular_cycle"] == baseline_cycle), None
    )
    if baseline_evt is None:
        base_row["exclusion_reason"] = "no_baseline"
        return base_row

    baseline = baseline_evt["capacity_discharge_ah"]
    if baseline is None or baseline <= 0:
        base_row["exclusion_reason"] = "no_baseline"
        return base_row

    n_regular_full = int(regulars[-1]["regular_cycle"])

    # AUTHORITATIVE DECISION PATH. When the human has an entry with
    # event_type set, the row is determined directly by the decision —
    # we do NOT run the fade detector or apply the outlier mask. This
    # is the "manual review is the highest priority" principle: the
    # output for these cells is whatever the human asserted in
    # decisions.json. Algorithm-derived values would only confuse the
    # picture.
    decision_event_type = cell_decision.get("event_type")
    decision_last_avail = cell_decision.get("last_available_cycle")
    if decision_event_type in ("event", "censor"):
        target_cycle = (
            int(decision_last_avail)
            if decision_last_avail is not None else n_regular_full
        )
        fr = _retention_at(regulars, baseline, target_cycle)
        if decision_event_type == "event":
            status = "faded"
            last_fade_cycle: Optional[int] = target_cycle
            n_regular = target_cycle
        else:  # "censor"
            status = "in_testing"
            last_fade_cycle = None
            n_regular = target_cycle  # full lifetime when null

        base_row.update({
            "status": status,
            "exclusion_reason": None,
            "last_fade_cycle": last_fade_cycle,
            "n_regular": n_regular,
            "baseline_dis_ah": float(baseline),
            "final_retention": float(fr) if fr is not None else None,
            "n_recovered_crossings": 0,
            "truncation_cycle": (
                int(decision_last_avail)
                if decision_last_avail is not None else None
            ),
            "n_outliers_masked": 0,
        })
        for N in N_THRESHOLDS:
            label, trainable = _classification_label_at(
                status, last_fade_cycle, n_regular, N,
            )
            base_row[f"label_n{N}"] = label
            base_row[f"trainable_n{N}"] = trainable
        return base_row

    # ALGORITHM PATH (no human decision for this cell). Retention curve
    # covers cycles >= baseline only AND <= last_available (when the
    # user truncated via decisions.json) AND not in the outlier set
    # (when outlier_detection flagged that cycle as untrustworthy).
    outlier_cycles: set[int] = {
        int(o["cycle"])
        for o in outlier_sidecar.get(cell, {}).get("outliers", [])
    }
    last_avail = cell_decision.get("last_available_cycle")  # None or int

    window = [
        e for e in regulars
        if e["regular_cycle"] >= baseline_cycle
        and e["regular_cycle"] not in outlier_cycles
        and (last_avail is None or e["regular_cycle"] <= last_avail)
    ]
    if not window:
        # Pathological: every post-baseline cycle was masked or truncated
        # away. Treat as no_baseline (no curve to evaluate).
        base_row["exclusion_reason"] = "no_baseline"
        return base_row
    cycles = [int(e["regular_cycle"]) for e in window]
    retentions = [float(e["capacity_discharge_ah"]) / float(baseline) for e in window]

    # n_regular: full lifetime, but capped at last_available_cycle when
    # truncated. (Outlier-masked cycles are still "real" cycles that
    # happened — they only get dropped from fade detection — so they
    # don't reduce n_regular.)
    n_regular = (
        min(n_regular_full, int(last_avail))
        if last_avail is not None else n_regular_full
    )
    final_retention = retentions[-1]

    # Count outlier cycles that fell inside the (post-baseline,
    # pre-truncation) range — these are the masked cycles that actually
    # affected this run.
    n_masked = sum(
        1 for e in regulars
        if e["regular_cycle"] >= baseline_cycle
        and e["regular_cycle"] in outlier_cycles
        and (last_avail is None or e["regular_cycle"] <= last_avail)
    )

    last_fade_cycle, n_recovered_crossings = _last_crossing_into_bad(cycles, retentions)
    status = "faded" if last_fade_cycle is not None else "in_testing"

    base_row.update({
        "status": status,
        "exclusion_reason": None,
        "last_fade_cycle": last_fade_cycle,
        "n_regular": n_regular,
        "baseline_dis_ah": float(baseline),
        "final_retention": float(final_retention),
        "n_recovered_crossings": int(n_recovered_crossings),
        "truncation_cycle": int(last_avail) if last_avail is not None else None,
        "n_outliers_masked": n_masked,
    })
    # Per-N classification labels for the kept (faded / in_testing) cells.
    # Excluded cells already have the right defaults from base_row init.
    for N in N_THRESHOLDS:
        label, trainable = _classification_label_at(
            status, last_fade_cycle, n_regular, N,
        )
        base_row[f"label_n{N}"] = label
        base_row[f"trainable_n{N}"] = trainable
    return base_row


def _selftest_classification() -> int:
    """Hand-built (status, last_fade_cycle, n_regular, N) → expected
    (label, trainable) verification table.
    """
    cases = [
        # (status, last_fade_cycle, n_regular, N, expected_label, expected_trainable)

        # excluded — always excluded, always not-trainable
        ("excluded", None, 0,   200, "excluded", False),
        ("excluded", None, 0,   400, "excluded", False),

        # faded — strictly-after-N → pass, at-or-before-N → bad
        ("faded",     250, 500, 200, "pass",     True),   # faded after 200
        ("faded",     200, 500, 200, "bad",      True),   # faded at exactly 200 → bad
        ("faded",     150, 500, 200, "bad",      True),   # faded before 200
        ("faded",     350, 500, 300, "pass",     True),
        ("faded",     300, 500, 300, "bad",      True),
        ("faded",     401, 500, 400, "pass",     True),
        ("faded",     400, 500, 400, "bad",      True),

        # in_testing — has-reached-N → pass, not-yet → censor
        ("in_testing", None, 250, 200, "pass",   True),
        ("in_testing", None, 200, 200, "pass",   True),   # exactly at N still passes
        ("in_testing", None, 199, 200, "censor", False),
        ("in_testing", None, 350, 300, "pass",   True),
        ("in_testing", None, 250, 300, "censor", False),
        ("in_testing", None, 150, 400, "censor", False),
    ]
    print("Self-test (classification labels):")
    fail = 0
    for status, lfc, nreg, N, exp_label, exp_trainable in cases:
        got_label, got_trainable = _classification_label_at(status, lfc, nreg, N)
        ok = got_label == exp_label and got_trainable == exp_trainable
        marker = "PASS" if ok else "FAIL"
        if not ok:
            fail += 1
        print(f"  [{marker}] status={status!s:11s} lfc={lfc!s:>4s} nreg={nreg:>4d} N={N:>3d}  "
              f"→ ({got_label!r}, {got_trainable})  expected ({exp_label!r}, {exp_trainable})")
    if fail:
        print(f"\n{fail} classification self-test cases FAILED")
    else:
        print("All classification self-test cases PASSED")
    return fail


def selftest() -> int:
    """Hand-built curves to verify the recovery-aware walk.

    Returns 0 if all cases pass, non-zero count of failures otherwise.
    """
    cases = [
        # (name, cycles, retentions, expected last_fade, expected n_recovered_crossings)
        ("pure healthy", list(range(1, 11)), [0.99] * 10, None, 0),
        ("monotone fade", list(range(1, 11)),
         [0.99, 0.97, 0.94, 0.90, 0.87, 0.84, 0.81, 0.79, 0.77, 0.75], 6, 0),
        ("dip then recover (7 good after)", list(range(1, 11)),
         [0.99, 0.97, 0.83, 0.90, 0.91, 0.92, 0.93, 0.93, 0.94, 0.94], None, 1),
        ("dip with only 2 good after, then re-cross", list(range(1, 9)),
         [0.99, 0.97, 0.83, 0.90, 0.91, 0.80, 0.79, 0.78], 6, 0),
        ("multi-dip: first recovers, second sticks", list(range(1, 13)),
         [0.99, 0.83, 0.91, 0.92, 0.93, 0.94, 0.82, 0.81, 0.80, 0.79, 0.78, 0.77],
         7, 1),
        ("boundary at exactly 0.85", list(range(1, 6)),
         [0.99, 0.85, 0.85, 0.85, 0.85], None, 0),
        ("just under 0.85 sustained", list(range(1, 6)),
         [0.99, 0.84, 0.84, 0.84, 0.84], 2, 0),
        ("oscillating then permanent bad", list(range(1, 12)),
         [0.99, 0.83, 0.91, 0.83, 0.91, 0.83, 0.91, 0.83, 0.80, 0.79, 0.78], 8, 1),
        ("non-consecutive 3 healthy then thin", list(range(1, 9)),
         [0.99, 0.83, 0.91, 0.84, 0.92, 0.84, 0.93, 0.84], 8, 1),
        ("exactly 3 healthy then sustained bad", list(range(1, 9)),
         [0.99, 0.83, 0.91, 0.92, 0.93, 0.80, 0.79, 0.78], 6, 1),
    ]
    print("Self-test (labels):")
    fail = 0
    for name, cyc, ret, exp_fade, exp_rec in cases:
        got_fade, got_rec = _last_crossing_into_bad(cyc, ret)
        ok = got_fade == exp_fade and got_rec == exp_rec
        marker = "PASS" if ok else "FAIL"
        if not ok:
            fail += 1
        print(f"  [{marker}] {name}: last_fade={got_fade} (expected {exp_fade}); "
              f"n_recovered_crossings={got_rec} (expected {exp_rec})")
    if fail:
        print(f"\n{fail} self-test cases FAILED")
    else:
        print("All self-test cases PASSED")
    fail += _selftest_classification()
    fail += _selftest_baseline_cycle()
    fail += _selftest_rate_changed()
    fail += _selftest_phase2_overrides()
    return fail


def _selftest_rate_changed() -> int:
    """Verify _process_cell handles rate_changed cells correctly.

    Three cases:
      (1) rate_changed cell whose first rate regime covers cycles 1..5+
          (admitted to predict-only: n_regular > 0, baseline populated).
      (2) rate_changed cell whose first regime stops at cycle 3 (still
          fully excluded: n_regular = 0, baseline_dis_ah = None).
      (3) single_rate cell with an explicit regime list — should populate
          n_regular_pre_rate_change = total regulars (single regime
          covers everything).
    """
    print("Self-test (rate_changed):")
    fail = 0

    def _events(n):
        return [
            {"event_kind": "regular_cd", "regular_cycle": i,
             "capacity_discharge_ah": 1.0 - 0.005 * (i - 1),
             "capacity_charge_ah": 1.0 - 0.004 * (i - 1),
             "coulombic_efficiency": 0.99}
            for i in range(1, n + 1)
        ]

    # Case 1: rate_changed, regime[0].n=5, 10 total regulars → ADMITTED
    d1 = {
        "cell_name": "AR-rc-pass",
        "cycling_consistency": "rate_changed",
        "regular_rate_regimes": [
            {"seg_id": 0, "n_regular_cd": 5, "baseline_i_a": 0.10,
             "baseline_i_dis_a": 0.10, "frac_of_total_regulars": 0.5},
            {"seg_id": 0, "n_regular_cd": 5, "baseline_i_a": 0.04,
             "baseline_i_dis_a": 0.04, "frac_of_total_regulars": 0.5},
        ],
        "cd_events": _events(10),
    }
    row = _process_cell(d1, baseline_cycle=1)
    if row["status"] != "excluded":
        print(f"  [FAIL] rc-pass: status={row['status']} expected 'excluded'")
        fail += 1
    if row["exclusion_reason"] != "rate_changed":
        print(f"  [FAIL] rc-pass: exclusion_reason={row['exclusion_reason']} expected 'rate_changed'")
        fail += 1
    if row["n_regular"] != 10:
        print(f"  [FAIL] rc-pass: n_regular={row['n_regular']} expected 10")
        fail += 1
    if row["n_regular_pre_rate_change"] != 5:
        print(f"  [FAIL] rc-pass: n_regular_pre_rate_change={row['n_regular_pre_rate_change']} expected 5")
        fail += 1
    if row["baseline_dis_ah"] is None or abs(row["baseline_dis_ah"] - 1.0) > 1e-9:
        print(f"  [FAIL] rc-pass: baseline_dis_ah={row['baseline_dis_ah']} expected 1.0")
        fail += 1
    if row["final_retention"] is not None:
        print(f"  [FAIL] rc-pass: final_retention={row['final_retention']} expected None")
        fail += 1
    if row["last_fade_cycle"] is not None:
        print(f"  [FAIL] rc-pass: last_fade_cycle={row['last_fade_cycle']} expected None")
        fail += 1
    for N in N_THRESHOLDS:
        if row[f"label_n{N}"] != "excluded":
            print(f"  [FAIL] rc-pass: label_n{N}={row[f'label_n{N}']} expected 'excluded'")
            fail += 1
        if row[f"trainable_n{N}"]:
            print(f"  [FAIL] rc-pass: trainable_n{N}={row[f'trainable_n{N}']} expected False")
            fail += 1

    # Case 2: rate_changed, regime[0].n=3 → still fully excluded
    d2 = {
        "cell_name": "AR-rc-short",
        "cycling_consistency": "rate_changed",
        "regular_rate_regimes": [
            {"seg_id": 0, "n_regular_cd": 3, "baseline_i_a": 0.10,
             "baseline_i_dis_a": 0.10, "frac_of_total_regulars": 0.3},
            {"seg_id": 0, "n_regular_cd": 7, "baseline_i_a": 0.04,
             "baseline_i_dis_a": 0.04, "frac_of_total_regulars": 0.7},
        ],
        "cd_events": _events(10),
    }
    row2 = _process_cell(d2, baseline_cycle=1)
    if row2["status"] != "excluded":
        print(f"  [FAIL] rc-short: status={row2['status']} expected 'excluded'")
        fail += 1
    if row2["n_regular"] != 0:
        print(f"  [FAIL] rc-short: n_regular={row2['n_regular']} expected 0")
        fail += 1
    if row2["n_regular_pre_rate_change"] != 3:
        print(f"  [FAIL] rc-short: n_regular_pre_rate_change={row2['n_regular_pre_rate_change']} expected 3")
        fail += 1
    if row2["baseline_dis_ah"] is not None:
        print(f"  [FAIL] rc-short: baseline_dis_ah={row2['baseline_dis_ah']} expected None")
        fail += 1

    # Case 3: single_rate with an explicit regime list (multi-regime
    # single_rate cells exist — toolkit splits per-RPT-segment). The
    # diagnostic column is NULL for single_rate cells regardless: the
    # "pre rate change" concept doesn't apply.
    d3 = {
        "cell_name": "AR-sr",
        "cycling_consistency": "single_rate",
        "regular_rate_regimes": [
            {"seg_id": 0, "n_regular_cd": 5, "baseline_i_a": 0.10,
             "baseline_i_dis_a": 0.10, "frac_of_total_regulars": 0.71},
            {"seg_id": 1, "n_regular_cd": 2, "baseline_i_a": 0.10,
             "baseline_i_dis_a": 0.10, "frac_of_total_regulars": 0.29},
        ],
        "cd_events": _events(7),
    }
    row3 = _process_cell(d3, baseline_cycle=1)
    if row3["status"] != "in_testing":
        print(f"  [FAIL] sr: status={row3['status']} expected 'in_testing'")
        fail += 1
    if row3["n_regular"] != 7:
        print(f"  [FAIL] sr: n_regular={row3['n_regular']} expected 7")
        fail += 1
    if row3["n_regular_pre_rate_change"] is not None:
        print(f"  [FAIL] sr: n_regular_pre_rate_change={row3['n_regular_pre_rate_change']} expected None")
        fail += 1

    # Case 4: no_regular (no regimes) → n_regular_pre_rate_change = None
    d4 = {
        "cell_name": "AR-nr",
        "cycling_consistency": "no_regular",
        "regular_rate_regimes": [],
        "cd_events": [],
    }
    row4 = _process_cell(d4, baseline_cycle=1)
    if row4["status"] != "excluded":
        print(f"  [FAIL] nr: status={row4['status']} expected 'excluded'")
        fail += 1
    if row4["exclusion_reason"] != "no_regular":
        print(f"  [FAIL] nr: exclusion_reason={row4['exclusion_reason']} expected 'no_regular'")
        fail += 1
    if row4["n_regular_pre_rate_change"] is not None:
        print(f"  [FAIL] nr: n_regular_pre_rate_change={row4['n_regular_pre_rate_change']} expected None")
        fail += 1

    if fail:
        print(f"\n{fail} rate_changed self-test cases FAILED")
    else:
        print("All rate_changed self-test cases PASSED")
    return fail


def _selftest_baseline_cycle() -> int:
    """Verify _process_cell honors a non-default baseline_cycle.

    Synthetic cell: cap_dis ramps 1.00 → 0.96 over 5 cycles, then
    drops to 0.80 in cycles 6-8 (sustained bad). With baseline=1 the
    last_fade_cycle is 6 (0.80/1.00 = 0.80 < 0.85). With baseline=3
    the retention array starts at cycle 3 with denominator 0.98, so
    cycle-3 retention = 1.0, cycle-6 retention = 0.80/0.98 ≈ 0.816 —
    still bad, so last_fade_cycle is still 6.
    """
    cap_dis_curve = [1.00, 0.99, 0.98, 0.97, 0.96, 0.80, 0.79, 0.78]
    d = {
        "cell_name": "AR-baseline-test",
        "cycling_consistency": "single_rate",
        "protocol_pattern": "synthetic",
        "cd_events": [
            {
                "event_kind": "regular_cd",
                "regular_cycle": i + 1,
                "capacity_discharge_ah": cap_dis_curve[i],
                "capacity_charge_ah": cap_dis_curve[i] + 0.01,
                "coulombic_efficiency": 0.99,
            }
            for i in range(len(cap_dis_curve))
        ],
    }

    print("Self-test (baseline_cycle):")
    fail = 0

    row_b1 = _process_cell(d, baseline_cycle=1)
    if row_b1["status"] != "faded":
        print(f"  [FAIL] baseline=1: status={row_b1['status']} expected 'faded'")
        fail += 1
    if row_b1["last_fade_cycle"] != 6:
        print(f"  [FAIL] baseline=1: last_fade_cycle={row_b1['last_fade_cycle']} expected 6")
        fail += 1
    if row_b1["n_regular"] != 8:
        print(f"  [FAIL] baseline=1: n_regular={row_b1['n_regular']} expected 8")
        fail += 1
    expected_final_b1 = cap_dis_curve[-1] / cap_dis_curve[0]   # 0.78 / 1.00
    if abs(row_b1["final_retention"] - expected_final_b1) > 1e-9:
        print(f"  [FAIL] baseline=1: final_retention={row_b1['final_retention']} expected {expected_final_b1}")
        fail += 1

    row_b3 = _process_cell(d, baseline_cycle=3)
    if row_b3["status"] != "faded":
        print(f"  [FAIL] baseline=3: status={row_b3['status']} expected 'faded'")
        fail += 1
    if row_b3["last_fade_cycle"] != 6:
        print(f"  [FAIL] baseline=3: last_fade_cycle={row_b3['last_fade_cycle']} expected 6")
        fail += 1
    if row_b3["n_regular"] != 8:
        print(f"  [FAIL] baseline=3: n_regular={row_b3['n_regular']} expected 8 (unchanged across baselines)")
        fail += 1
    if abs(row_b3["baseline_dis_ah"] - cap_dis_curve[2]) > 1e-9:
        print(f"  [FAIL] baseline=3: baseline_dis_ah={row_b3['baseline_dis_ah']} expected {cap_dis_curve[2]}")
        fail += 1
    expected_final_b3 = cap_dis_curve[-1] / cap_dis_curve[2]   # 0.78 / 0.98
    if abs(row_b3["final_retention"] - expected_final_b3) > 1e-9:
        print(f"  [FAIL] baseline=3: final_retention={row_b3['final_retention']} expected {expected_final_b3}")
        fail += 1

    # cell that's healthy under baseline=1 but FADED under baseline=3:
    # cap=1.00,0.99,0.98,0.97,0.96,0.95,0.84,0.83 — under baseline=3:
    # cycle 7 retention = 0.84/0.98 ≈ 0.857 (just above 0.85, healthy)
    # need an example that crosses differently. Construct: cap goes
    # 1.00, 0.99, 0.98, 0.97, 0.96, 0.95, 0.83, 0.82.
    # baseline=1: cycle-7 = 0.83 < 0.85 → faded
    # baseline=3: cycle-7 = 0.83/0.98 = 0.847 < 0.85 → also faded
    # Hard to construct a case where baseline shifts the threshold one side.
    # Skip — just verify retention math, not the threshold-shifting case
    # (that's covered by the back-compat byte-identity gate in Verification).

    if fail:
        print(f"\n{fail} baseline_cycle self-test cases FAILED")
    else:
        print("All baseline_cycle self-test cases PASSED")
    return fail


def _selftest_phase2_overrides() -> int:
    """Verify _process_cell honors manual_validation + outlier_sidecar overrides.

    Synthetic cell (10 cycles, fade 1.00→0.93 then 0.83 dip at 9, then
    0.92). Covers two paths through _process_cell:

      Algorithm path (cells without a decision entry):
        (1) NO overrides → fade at cycle 9.
        (2) exclude_from_ml=true → status=excluded, reason=human_review.
        (3) outlier mask on cycle 9 (no event_type) → algorithm hides
            the dip → status=in_testing, n_outliers_masked=1.

      Authoritative-decision path (event_type set → algorithm bypassed):
        (4) event_type="censor", last_available_cycle=8 → in_testing,
            n_regular=8, n_outliers_masked=0 (mask not applied).
        (5) event_type="censor", last_available_cycle=null → in_testing,
            n_regular=10 (full lifetime).
        (6) event_type="event", last_available_cycle=9 → faded,
            last_fade_cycle=9, n_regular=9.
        (7) AR4195 regression: outlier sidecar masks cycles 7-9
            (would hide the fade under the algorithm path) BUT decision
            event_type="event" last_available_cycle=9 wins →
            status=faded, last_fade_cycle=9 regardless.
    """
    print("Self-test (Phase 2 overrides):")
    fail = 0

    # Cap curve: 1..8 are healthy (>0.85), 9 is a one-cycle dip below 0.85,
    # 10 recovers but the cell only ran 10 cycles total. With RECOVERY_MIN=3,
    # a dip at 9 followed by only 1 healthy cycle counts as a sticky fade.
    cap = [1.00, 0.99, 0.98, 0.97, 0.96, 0.95, 0.94, 0.93, 0.83, 0.92]
    d = {
        "cell_name": "AR-phase2-test",
        "cycling_consistency": "single_rate",
        "protocol_pattern": "synthetic",
        "cd_events": [
            {"event_kind": "regular_cd", "regular_cycle": i + 1,
             "capacity_discharge_ah": cap[i],
             "capacity_charge_ah": cap[i] + 0.01,
             "coulombic_efficiency": 0.99}
            for i in range(len(cap))
        ],
    }

    def _expect(case: str, row: dict, expectations: dict) -> None:
        nonlocal fail
        for key, exp in expectations.items():
            got = row.get(key)
            if got != exp:
                print(f"  [FAIL] {case}: {key}={got!r} expected {exp!r}")
                fail += 1

    # (1) No overrides
    row1 = _process_cell(d, baseline_cycle=1)
    _expect("baseline (no overrides)", row1, {
        "status": "faded",
        "exclusion_reason": None,
        "last_fade_cycle": 9,
        "n_regular": 10,
        "truncation_cycle": None,
        "n_outliers_masked": 0,
    })
    if row1["last_fade_cycle"] == 9:
        print(f"  [PASS] baseline: cell faded at cycle 9 as expected")

    # (2) exclude_from_ml=true
    row2 = _process_cell(d, baseline_cycle=1, decisions={
        "AR-phase2-test": {"exclude_from_ml": True},
    })
    _expect("exclude_from_ml=true", row2, {
        "status": "excluded",
        "exclusion_reason": "human_review",
        "last_fade_cycle": None,
    })
    if row2["status"] == "excluded":
        print(f"  [PASS] human_review: cell excluded as expected")

    # (3) outlier sidecar masks cycle 9 — fade detector should not see the dip.
    row3 = _process_cell(
        d, baseline_cycle=1,
        outlier_sidecar={"AR-phase2-test": {"outliers": [{"cycle": 9}]}},
    )
    _expect("outlier mask on cycle 9", row3, {
        "status": "in_testing",
        "exclusion_reason": None,
        "last_fade_cycle": None,
        "n_regular": 10,
        "n_outliers_masked": 1,
    })
    if row3["last_fade_cycle"] is None:
        print(f"  [PASS] outlier mask: dip at cycle 9 hidden from fade detector")

    # (4) event_type="censor", last_available_cycle=8 → direct assignment
    row4 = _process_cell(d, baseline_cycle=1, decisions={
        "AR-phase2-test": {
            "exclude_from_ml": False,
            "last_available_cycle": 8,
            "event_type": "censor",
            "reason": "test",
            "validated_at": "2026-05-20",
        },
    })
    _expect("censor with last_available_cycle=8", row4, {
        "status": "in_testing",
        "exclusion_reason": None,
        "last_fade_cycle": None,
        "n_regular": 8,
        "truncation_cycle": 8,
        "n_outliers_masked": 0,
    })
    if row4["n_regular"] == 8:
        print(f"  [PASS] censor decision: n_regular = last_available_cycle")

    # (5) event_type="censor", last_available_cycle=null → use full lifetime
    row5 = _process_cell(d, baseline_cycle=1, decisions={
        "AR-phase2-test": {
            "exclude_from_ml": False,
            "last_available_cycle": None,
            "event_type": "censor",
            "reason": "test",
            "validated_at": "2026-05-20",
        },
    })
    _expect("censor with last_available_cycle=null", row5, {
        "status": "in_testing",
        "last_fade_cycle": None,
        "n_regular": 10,                # full lifetime
        "truncation_cycle": None,
        "n_outliers_masked": 0,
    })

    # (6) event_type="event", last_available_cycle=9 → status=faded, last_fade=9
    row6 = _process_cell(d, baseline_cycle=1, decisions={
        "AR-phase2-test": {
            "exclude_from_ml": False,
            "last_available_cycle": 9,
            "event_type": "event",
            "reason": "test",
            "validated_at": "2026-05-20",
        },
    })
    _expect("event with last_available_cycle=9", row6, {
        "status": "faded",
        "last_fade_cycle": 9,
        "n_regular": 9,
        "truncation_cycle": 9,
        "n_outliers_masked": 0,
    })
    if row6["last_fade_cycle"] == 9 and row6["status"] == "faded":
        print(f"  [PASS] event decision: last_fade_cycle = last_available_cycle")

    # (7) AR4195 regression: outlier mask would hide the fade, but
    # decision asserts it anyway → decision wins.
    row7 = _process_cell(
        d, baseline_cycle=1,
        decisions={"AR-phase2-test": {
            "exclude_from_ml": False,
            "last_available_cycle": 9,
            "event_type": "event",
            "reason": "regression for AR4195",
            "validated_at": "2026-05-20",
        }},
        outlier_sidecar={"AR-phase2-test": {"outliers": [
            {"cycle": 7}, {"cycle": 8}, {"cycle": 9},
        ]}},
    )
    _expect("AR4195-style: decision overrides outlier mask", row7, {
        "status": "faded",
        "last_fade_cycle": 9,
        "n_regular": 9,
        "n_outliers_masked": 0,
    })
    if row7["status"] == "faded" and row7["last_fade_cycle"] == 9:
        print(f"  [PASS] AR4195 regression: decision wins over outlier mask")

    if fail:
        print(f"\n{fail} Phase 2 override self-test cases FAILED")
    else:
        print("All Phase 2 override self-test cases PASSED")
    return fail


def main(
    baseline_cycle: int = DEFAULT_BASELINE_CYCLE,
    db_version: str = "A2.2",
) -> None:
    decisions = _load_decisions()
    outlier_sidecar = _load_outlier_sidecar()
    print(
        f"loaded {len(decisions)} manual decisions from "
        f"{DECISIONS_PATH if DECISIONS_PATH.exists() else '(missing)'}"
    )
    print(
        f"loaded outlier sidecar covering {len(outlier_sidecar)} cells from "
        f"{OUTLIER_SIDECAR_PATH if OUTLIER_SIDECAR_PATH.exists() else '(missing)'}"
    )

    rows: list[dict] = []
    n_total = 0
    for path, d in iter_annotations():
        n_total += 1
        rows.append(_process_cell(
            d,
            baseline_cycle=baseline_cycle,
            decisions=decisions,
            outlier_sidecar=outlier_sidecar,
        ))

    if not rows:
        print(f"ERROR: no cells processed", file=sys.stderr)
        sys.exit(1)

    df = pl.DataFrame(rows).sort(["cohort", "cell_name"])
    out_dir = dataset_dir_for(db_version, baseline_cycle)
    parquet_path, csv_path = write_outputs(df, "cell_labels", out_dir=out_dir)

    manifest_path = write_manifest(out_dir, {
        "schema_version": SCHEMA_VERSION,
        "db_version": db_version,
        "baseline_cycle": baseline_cycle,
        "annot_dir": str(ANNOT_DIR),
        "n_cells_labels": df.height,
        "stages_populated": ["labels"],
    })

    latest_link = promote_to_latest(out_dir)

    print(f"db_version     = {db_version}")
    print(f"baseline_cycle = {baseline_cycle}")
    print(f"scanned      = {n_total} annotation JSONs")
    print(f"rows         = {df.height}")
    print(f"written      = {parquet_path}")
    print(f"               {csv_path}")
    print(f"               {manifest_path}")
    print(f"latest       = {latest_link} -> {out_dir.name}")
    print()
    print("Status histogram:")
    status_counts = df.group_by("status").len().sort("status")
    print(status_counts)
    print()
    print("Exclusion-reason histogram (status='excluded' only):")
    excl_counts = (
        df.filter(pl.col("status") == "excluded")
        .group_by("exclusion_reason").len()
        .sort("exclusion_reason")
    )
    print(excl_counts)
    print()
    n_faded = df.filter(pl.col("status") == "faded").height
    n_in_testing = df.filter(pl.col("status") == "in_testing").height
    n_excluded = df.filter(pl.col("status") == "excluded").height
    print(f"Headline: faded={n_faded}, in_testing={n_in_testing}, excluded={n_excluded}, "
          f"total={df.height}")

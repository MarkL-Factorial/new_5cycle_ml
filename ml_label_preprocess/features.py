"""Per-cell feature extraction — 12 FS_CV features + companions.

Writes out/cell_features.{parquet,csv}, one row per cell with at least 5
regular cycles (cells with fewer are OMITTED — Stage A onwards).

v3: outputs live at ``datasets/{db_version}_b{baseline_cycle}/`` with a
``manifest.json``. Baseline cycle (N0) is configurable: Tier A/B retention
features use cycle N0 as the denominator and aggregate over the
post-baseline window [N0, 5]. Tier C (KWW fit on cycles 3/4/5) is
unchanged — it does not depend on the baseline. Default N0=1.

Stage progression of this module (see preprocess_extension_feasibility.md
for rationale and rollout):
  - Stage A: Tier A populated, Tier B/C stubbed as null. ← THIS COMMIT
  - Stage B: Tier B populated (uses workbench-app data layer).
  - Stage C: Tier C populated (uses workbench-app KWW fitter).

Column-role manifest at column_roles.yaml is the single source of truth
for which columns are features vs labels vs meta. Downstream ML must
filter by role to avoid data leakage. A consistency check runs at
startup to flag any drift between this module's emitted columns and
the manifest.

Author: Mark Liao (Sheng-Lun Liao)
"""
from __future__ import annotations

import hashlib
import statistics
import sys
from pathlib import Path
from typing import Optional

import numpy as np
import polars as pl

from _common import (
    ANNOT_DIR,
    dataset_dir_for,
    iter_annotations,
    iter_regulars,
    write_manifest,
    write_outputs,
)

DEFAULT_BASELINE_CYCLE = 1
SCHEMA_VERSION = 1

# workbench-app surface used by Stage B/C. Imported at module top so an
# upstream API drift surfaces as ImportError at startup, not as a per-
# cell exception buried under 400+ "WARNING" lines.
from battery_workbench.core.data.annotations import (  # noqa: E402
    load_raw_tagged,
)
from battery_workbench.core.analysis.cv_fitting import (  # noqa: E402
    extract_cv_phase_by_cd,
    fit_kww_fast_exp,
)

# ---------------------------------------------------------------------------
# Output schema (the single source of truth for column order + dtype).
# Stage B/C will populate the currently-null columns; the schema itself
# does not change between stages.
# ---------------------------------------------------------------------------

SCHEMA: dict[str, pl.DataType] = {
    # --- meta keys ---
    # cell_name is the only meta column in this file — it's the join
    # key downstream uses to attach labels. Cohort, protocol_pattern,
    # baselines, etc. live in cell_labels.parquet so cell_features
    # stays a pure "(cell_id, X)" training matrix.
    "cell_name": pl.String,
    # --- Tier A ---
    "coulombic_efficiency_final": pl.Float64,
    "discharge_capacity_retention_final": pl.Float64,
    "charge_capacity_retention_min": pl.Float64,
    # --- Tier B (Stage B will populate) ---
    "discharge_nominal_voltage_retention_max": pl.Float64,
    "discharge_nominal_voltage_retention_std": pl.Float64,
    "charge_nominal_voltage_retention_max": pl.Float64,
    # --- Tier C per-cycle (Stage C will populate) ---
    "cv_A_fast_frac_c3": pl.Float64,
    "cv_A_fast_frac_c4": pl.Float64,
    "cv_A_fast_frac_c5": pl.Float64,
    "cv_tau_fast_c3": pl.Float64,
    "cv_tau_fast_c4": pl.Float64,
    "cv_tau_fast_c5": pl.Float64,
    "cv_tau_slow_c3": pl.Float64,
    "cv_tau_slow_c4": pl.Float64,
    "cv_tau_slow_c5": pl.Float64,
    "cv_beta_c3": pl.Float64,
    "cv_beta_c4": pl.Float64,
    "cv_beta_c5": pl.Float64,
    "cv_tau_ratio_c3": pl.Float64,
    "cv_tau_ratio_c4": pl.Float64,
    "cv_tau_ratio_c5": pl.Float64,
    # --- Tier C aggregated over (c3, c4, c5) (Stage C will populate) ---
    "cv_A_fast_frac_mean": pl.Float64,
    "cv_A_fast_frac_std": pl.Float64,
    "cv_A_fast_frac_delta": pl.Float64,
    "cv_tau_fast_mean": pl.Float64,
    "cv_tau_fast_std": pl.Float64,
    "cv_tau_fast_delta": pl.Float64,
    "cv_tau_slow_mean": pl.Float64,
    "cv_tau_slow_std": pl.Float64,
    "cv_tau_slow_delta": pl.Float64,
    "cv_beta_mean": pl.Float64,
    "cv_beta_std": pl.Float64,
    "cv_beta_delta": pl.Float64,
    "cv_tau_ratio_mean": pl.Float64,
    "cv_tau_ratio_std": pl.Float64,
    "cv_tau_ratio_delta": pl.Float64,
    # --- Tier C engineered (Stage C will populate) ---
    "A_slow_to_fast_mean": pl.Float64,
    "A_slow_to_fast_delta": pl.Float64,
    "log_A_slow_to_fast_delta": pl.Float64,
    "A_fast_frac_cv": pl.Float64,
    # cv_n_success (quality diagnostic) is computed internally for the
    # main() distribution print, but NOT emitted in this file — it's a
    # row-filter signal, not an ML input. Downstream can recover it by
    # aggregating cell_features_status.csv:
    #   status.group_by("cell_name").agg(pl.col("success").sum())
}

# Pre-built null row template for Tier B + C columns; merged with the
# Tier A row on a per-cell basis. Each tier's keys are overwritten by
# its _tier_<x>() helper once that stage's code is in. Stage B fills in
# the three Tier-B keys; Stage C fills the rest.
_NULL_TIER_BC: dict[str, Optional[float]] = {
    "discharge_nominal_voltage_retention_max": None,
    "discharge_nominal_voltage_retention_std": None,
    "charge_nominal_voltage_retention_max": None,
    **{f"cv_{f}_{c}": None
       for f in ("A_fast_frac", "tau_fast", "tau_slow", "beta", "tau_ratio")
       for c in ("c3", "c4", "c5")},
    **{f"cv_{f}_{agg}": None
       for f in ("A_fast_frac", "tau_fast", "tau_slow", "beta", "tau_ratio")
       for agg in ("mean", "std", "delta")},
    "A_slow_to_fast_mean": None,
    "A_slow_to_fast_delta": None,
    "log_A_slow_to_fast_delta": None,
    "A_fast_frac_cv": None,
}


# ---------------------------------------------------------------------------
# Tier A computation (per cell)
# ---------------------------------------------------------------------------

def _tier_a(cycles_window: list[dict], baseline_cap_dis: float,
            baseline_cap_chg: float) -> dict[str, float]:
    """Three Tier-A features from the post-baseline-cycle scalars.

    ``cycles_window`` is the slice ``[N0, 5]`` of regular_cd events (so
    ``cycles_window[-1]`` is cycle 5 regardless of N0, and ``cycles_window[0]``
    is the baseline cycle). All inputs come from the annotation JSON;
    no raw-parquet access.

    Note on units: ``coulombic_efficiency_final`` is emitted as PERCENT
    (× 100), matching the convention used by the legacy reference CSV
    (BOL_with_cv_features.csv) and by solstice_mlflow's feature
    extractor. The annotation JSON itself stores CE as a fraction
    (0.99), so we multiply here.
    """
    ce_final = cycles_window[-1]["coulombic_efficiency"] * 100.0
    dis_ret_final = cycles_window[-1]["capacity_discharge_ah"] / baseline_cap_dis
    chg_caps = [c["capacity_charge_ah"] for c in cycles_window]
    chg_ret_min = min(chg_caps) / baseline_cap_chg
    return {
        "coulombic_efficiency_final": float(ce_final),
        "discharge_capacity_retention_final": float(dis_ret_final),
        "charge_capacity_retention_min": float(chg_ret_min),
    }


# ---------------------------------------------------------------------------
# Tier B computation (per cell, per cycle, per phase)
# ---------------------------------------------------------------------------

def _nominal_v_phase(phase_df: pl.DataFrame) -> Optional[float]:
    """Capacity-weighted mean voltage of one phase slice (charge OR
    discharge). Computed by manual trapezoidal integration:

        V_nominal = ∫V·I·dt / ∫I·dt  =  abs(energy) / abs(capacity)

    The renumbered parquet's schema is heterogeneous across cells —
    some cells carry pre-computed ``energy`` and ``capacity`` columns
    (cumulative, signed; from MB / Digatron / Maccor sources), others
    only carry the 7-col core ``(cycle, step, elapsed_time, step_time,
    current, voltage, state)`` (Neware-source cells). Integrating
    V·I·dt and I·dt directly works for both.

    Returns None if the slice is empty or the capacity integral is
    effectively zero (degenerate phase — no current flow).
    """
    if phase_df.is_empty():
        return None
    V = phase_df["voltage"].to_numpy()
    I = phase_df["current"].to_numpy()
    t = phase_df["elapsed_time"].to_numpy()
    if len(t) < 2:
        return None
    energy = np.trapezoid(V * I, t)
    capacity = np.trapezoid(I, t)
    if abs(capacity) < 1e-12:
        return None
    return float(abs(energy) / abs(capacity))


def _tier_b(raw_tagged: pl.DataFrame,
            cycles_window: list[dict]) -> dict[str, Optional[float]]:
    """Three Tier-B retention features from per-cycle nominal voltages.

    Args:
        raw_tagged: Output of ``load_raw_tagged(cell_name)`` — full
            raw parquet for the cell, with ``cd_index`` and ``cd_phase``
            columns attached by the toolkit.
        cycles_window: post-baseline window of regular_cd events
            (length 6 − N0; ``cycles_window[0]`` is the baseline cycle).

    All three features are normalized to the baseline-cycle nominal
    voltage in the corresponding phase. Returns None for every Tier-B
    key if any cycle's nominal voltage is unavailable (degenerate
    phase, missing raw rows, etc.), or if the window has < 2 points
    (std is undefined) — partial Tier-B rows would invite
    mis-aggregation downstream.
    """
    null_b = {
        "discharge_nominal_voltage_retention_max": None,
        "discharge_nominal_voltage_retention_std": None,
        "charge_nominal_voltage_retention_max": None,
    }
    if len(cycles_window) < 2:
        return null_b
    v_nom_chg: list[Optional[float]] = []
    v_nom_dis: list[Optional[float]] = []
    for cyc in cycles_window:
        cd = cyc["cd_index"]
        chg_df = raw_tagged.filter(
            (pl.col("cd_index") == cd) & (pl.col("cd_phase") == "charge")
        )
        dis_df = raw_tagged.filter(
            (pl.col("cd_index") == cd) & (pl.col("cd_phase") == "discharge")
        )
        v_nom_chg.append(_nominal_v_phase(chg_df))
        v_nom_dis.append(_nominal_v_phase(dis_df))

    if any(v is None for v in v_nom_chg) or any(v is None for v in v_nom_dis):
        return null_b

    v0_chg = v_nom_chg[0]
    v0_dis = v_nom_dis[0]
    if v0_chg is None or v0_chg <= 0 or v0_dis is None or v0_dis <= 0:
        return null_b

    dis_retentions = [v / v0_dis for v in v_nom_dis]
    chg_retentions = [v / v0_chg for v in v_nom_chg]

    return {
        "discharge_nominal_voltage_retention_max": float(max(dis_retentions)),
        "discharge_nominal_voltage_retention_std": float(statistics.stdev(dis_retentions)),
        "charge_nominal_voltage_retention_max": float(max(chg_retentions)),
    }


# ---------------------------------------------------------------------------
# Tier C: per-cycle KWW fit on CV-phase current decay + aggregations
# ---------------------------------------------------------------------------
# Workbench-app's fit_kww_fast_exp uses these bounds. A "successful" fit
# that converged at a bound is suspect (the true optimum was outside the
# model's parameter space) — we treat pinned-to-bound as failure.
_BOUNDS_TAU_FAST = (10.0, 5000.0)
_BOUNDS_TAU_SLOW = (100.0, 50000.0)
_BOUNDS_BETA     = (0.3, 5.0)
_BOUND_TOL_ABS   = 1e-3     # value within this of a bound is "pinned"
_SAFE_RATIO_EPS  = 0.5      # min |denom| for safe_ratio (matches reference,
                            # 00_engineer_features.py:30 — 0.5 percent for
                            # A_fast_frac, the only consumer here)


def _within_bounds(fit) -> bool:
    """Reject fits that converged AT (or essentially at) a parameter
    bound. scipy.optimize.curve_fit with bounds can clamp a parameter
    to the boundary when the true optimum is outside the allowed range
    — the resulting parameters are degenerate and should not be trusted
    as features.
    """
    if not fit.success:
        return False
    lo, hi = _BOUNDS_TAU_FAST
    if not (lo + _BOUND_TOL_ABS < fit.tau_fast < hi - _BOUND_TOL_ABS):
        return False
    lo, hi = _BOUNDS_TAU_SLOW
    if not (lo + _BOUND_TOL_ABS < fit.tau_slow < hi - _BOUND_TOL_ABS):
        return False
    lo, hi = _BOUNDS_BETA
    if not (lo + _BOUND_TOL_ABS < fit.beta < hi - _BOUND_TOL_ABS):
        return False
    # A_fast_frac is a derived property (0–100 percent). Open interval:
    # exactly 0 or 100 means one component carried the entire signal.
    if not (0.0 < fit.A_fast_frac < 100.0):
        return False
    return True


def _safe_ratio_scalar(num: float, denom: float,
                       eps: float = _SAFE_RATIO_EPS) -> float:
    """Mirror of safe_ratio() from 00_engineer_features.py:36-40 for a
    single (num, denom) pair. Returns NaN when ``|denom| < eps`` or
    either input is non-finite. ``eps`` defaults to 0.5 to match the
    reference's percent-scale check for A_fast_frac.
    """
    if not (np.isfinite(num) and np.isfinite(denom)):
        return float("nan")
    if abs(denom) < eps:
        return float("nan")
    return float(num / denom)


def _safe_log_abs_scalar(x: float) -> float:
    """Mirror of safe_log_abs() from 00_engineer_features.py:43-47.
    Returns log(|x|) when x is finite and non-zero, else NaN.
    """
    if not np.isfinite(x) or x == 0:
        return float("nan")
    return float(np.log(abs(x)))


def _fit_one_cycle(cell_name: str, cd_index: int,
                   regular_cycle: int) -> tuple[Optional[object], dict]:
    """Run extract_cv_phase_by_cd + fit_kww_fast_exp for one cycle.

    Returns (fit_result_or_None, status_row_dict). The fit_result is
    None on any failure path; the status_row is always populated and
    intended for accumulation into cell_features_status.csv.

    A returned non-None fit MAY still be unsuccessful per bounds-sanity
    (caller should consult status_row["success"] not the fit object).
    """
    status = {
        "cell_name": cell_name,
        "regular_cycle": regular_cycle,
        "cd_index": cd_index,
        "success": False,
        "error_msg": "",
        "tau_fast": None,
        "tau_slow": None,
        "A_fast_frac": None,
        "beta": None,
        "r_squared": None,
    }
    try:
        t_s, I_A = extract_cv_phase_by_cd(cell_name, cd_index)
    except Exception as exc:
        status["error_msg"] = f"extract: {type(exc).__name__}: {exc}"
        return None, status
    try:
        fit = fit_kww_fast_exp(t_s, I_A * 1000.0)  # workbench-app expects mA
    except Exception as exc:
        status["error_msg"] = f"fit: {type(exc).__name__}: {exc}"
        return None, status
    # Always populate diagnostic fields, even on failure (helps triage).
    status["tau_fast"] = float(fit.tau_fast)
    status["tau_slow"] = float(fit.tau_slow)
    status["A_fast_frac"] = float(fit.A_fast_frac)
    status["beta"] = float(fit.beta)
    status["r_squared"] = float(fit.r_squared)
    if not fit.success:
        status["error_msg"] = f"non-converged: {fit.error_msg or 'unknown'}"
        return None, status
    if not _within_bounds(fit):
        status["error_msg"] = "pinned-to-bound"
        return None, status
    status["success"] = True
    return fit, status


def _engineered_a_ratio(f3: float, f4: float, f5: float) -> dict[str, float]:
    """Four engineered A-ratio features per
    [`00_engineer_features.py:50-76`](../experiment_cv_features/report_M2_vs_lean8_vs_CV_20260501/scripts/00_engineer_features.py).

    Inputs are per-cycle ``A_fast_frac`` values (in percent). NaN for
    failed cycles propagates per the reference's safe_ratio / safe_log_abs
    semantics. ``nanmean`` / ``nanstd`` mean partial fits still yield
    finite outputs as long as at least one cycle succeeded.
    """
    r3 = _safe_ratio_scalar(100.0 - f3, f3)
    r4 = _safe_ratio_scalar(100.0 - f4, f4)
    r5 = _safe_ratio_scalar(100.0 - f5, f5)
    rs = np.array([r3, r4, r5], dtype=float)
    fracs = np.array([f3, f4, f5], dtype=float)
    nan = float("nan")

    with np.errstate(invalid="ignore", all="ignore"):
        a_mean = float(np.nanmean(rs)) if np.any(np.isfinite(rs)) else nan
        if np.any(np.isfinite(fracs)):
            f_mean = float(np.nanmean(fracs))
            f_std = float(np.nanstd(fracs))    # ddof=0 (np default — matches reference)
        else:
            f_mean = f_std = nan

    a_delta = (
        float(r5 - r3) if np.isfinite(r3) and np.isfinite(r5) else nan
    )
    log_r3 = _safe_log_abs_scalar(r3)
    log_r5 = _safe_log_abs_scalar(r5)
    log_delta = (
        log_r5 - log_r3 if np.isfinite(log_r3) and np.isfinite(log_r5) else nan
    )
    cv = (
        _safe_ratio_scalar(f_std, abs(f_mean)) if np.isfinite(f_mean) else nan
    )
    return {
        "A_slow_to_fast_mean":      a_mean,
        "A_slow_to_fast_delta":     a_delta,
        "log_A_slow_to_fast_delta": log_delta,
        "A_fast_frac_cv":           cv,
    }


def _tier_c(cell_name: str,
            cycles_3_5: list[dict]) -> tuple[dict[str, Optional[float]], list[dict]]:
    """Per-cycle KWW fit on cycles 3, 4, 5 → 34 Tier-C columns + status.

    Returns (row_updates, status_rows). ``cycles_3_5`` must be a 3-element
    list of regular_cd events with ``regular_cycle`` in {3, 4, 5} and a
    ``cd_index`` key (workbench-app's slicer keys on cd_index).

    Behavior on partial fits: per-cycle scalars for failed cycles are
    NaN; aggregations use nanmean/nanstd so the mean/std exist if at
    least one cycle succeeded; delta is NaN unless BOTH c3 and c5 are
    finite (delta needs the endpoints).
    """
    fits: dict[int, object] = {}
    status_rows: list[dict] = []
    for cyc in cycles_3_5:
        rc = cyc["regular_cycle"]
        fit, status = _fit_one_cycle(cell_name, cyc["cd_index"], rc)
        status_rows.append(status)
        if fit is not None and status["success"]:
            fits[rc] = fit

    nan = float("nan")

    def _field(rc: int, attr: str) -> float:
        f = fits.get(rc)
        if f is None:
            return nan
        if attr == "tau_ratio":
            return float(f.tau_slow / f.tau_fast) if f.tau_fast > 0 else nan
        return float(getattr(f, attr))

    out: dict[str, Optional[float]] = {}
    # Per-cycle scalars (15 cols)
    for label, rc in (("c3", 3), ("c4", 4), ("c5", 5)):
        out[f"cv_A_fast_frac_{label}"] = _field(rc, "A_fast_frac")
        out[f"cv_tau_fast_{label}"]    = _field(rc, "tau_fast")
        out[f"cv_tau_slow_{label}"]    = _field(rc, "tau_slow")
        out[f"cv_beta_{label}"]        = _field(rc, "beta")
        out[f"cv_tau_ratio_{label}"]   = _field(rc, "tau_ratio")

    # Aggregations (15 cols): mean, std (ddof=0 — np default), delta = c5 − c3
    for attr in ("A_fast_frac", "tau_fast", "tau_slow", "beta", "tau_ratio"):
        c3 = out[f"cv_{attr}_c3"]
        c4 = out[f"cv_{attr}_c4"]
        c5 = out[f"cv_{attr}_c5"]
        vals = np.array([c3, c4, c5], dtype=float)
        with np.errstate(invalid="ignore", all="ignore"):
            if np.any(np.isfinite(vals)):
                out[f"cv_{attr}_mean"] = float(np.nanmean(vals))
                out[f"cv_{attr}_std"]  = float(np.nanstd(vals))
            else:
                out[f"cv_{attr}_mean"] = nan
                out[f"cv_{attr}_std"]  = nan
        out[f"cv_{attr}_delta"] = (
            float(c5 - c3) if np.isfinite(c3) and np.isfinite(c5) else nan
        )

    # Engineered A-ratio (4 cols)
    out.update(_engineered_a_ratio(
        out["cv_A_fast_frac_c3"],
        out["cv_A_fast_frac_c4"],
        out["cv_A_fast_frac_c5"],
    ))

    # NOTE: cv_n_success (quality) is NOT added to the output row — it
    # is computed in main() from status_rows and used only for the
    # diagnostic distribution print, not written to cell_features.csv.

    return out, status_rows


def _check_omit(
    d: dict,
    baseline_cycle: int = DEFAULT_BASELINE_CYCLE,
) -> Optional[tuple[list[dict], float, float]]:
    """Annotation-JSON-only omission gate.

    Returns ``(cycles_window, baseline_cap_dis_ah, baseline_cap_chg_ah)`` if
    the cell is keepable, or ``None`` if it must be omitted. Performed
    BEFORE loading raw parquet so we don't pay disk I/O for cells we'd
    immediately drop.

    ``cycles_window`` is the slice of cycles 1..5 whose ``regular_cycle``
    is >= baseline_cycle (length = 6 − N0).

    Omission rules (kept apples-to-apples across baselines):
      - cycling_consistency in {single_rate, rate_changed-with-first-regime>=5}:
        admitted. rate_changed cells are featurized on cycles 1..5 which
        are guaranteed to lie within regime[0] (toolkit-ordered regimes,
        regular_cycle being a 1-based linear counter over regular_cd
        events). These cells are still status='excluded' on the label
        side — they flow through for prediction-only scoring.
      - cycling_consistency == rate_changed but regime[0].n_regular_cd < 5:
        omitted (the feature window would span a rate change).
      - cycling_consistency == no_regular: omitted.
      - <5 regular_cd events: omitted.
      - no event with regular_cycle == 1: omitted (no v1 baseline anchor).
      - baseline-cycle event missing or has null/non-positive capacity: omitted.
      - any cycle in 1..5 has null CE or null capacity (Tier-A inputs): omitted.
    """
    consistency = d.get("cycling_consistency", "no_regular")
    if consistency == "rate_changed":
        rr = d.get("regular_rate_regimes", [])
        if not rr or (rr[0].get("n_regular_cd") or 0) < 5:
            return None
        # fall through: cycles 1..5 are guaranteed to be in regime[0]
    elif consistency != "single_rate":
        return None

    regulars = iter_regulars(d)
    if len(regulars) < 5:
        return None
    if regulars[0].get("regular_cycle") != 1:
        return None

    cycles_1_5 = regulars[:5]
    for c in cycles_1_5:
        if c.get("coulombic_efficiency") is None:
            return None
        if c.get("capacity_charge_ah") is None or c.get("capacity_discharge_ah") is None:
            return None

    baseline_evt = next(
        (c for c in cycles_1_5 if c.get("regular_cycle") == baseline_cycle), None
    )
    if baseline_evt is None:
        return None
    baseline_cap_dis = baseline_evt.get("capacity_discharge_ah")
    baseline_cap_chg = baseline_evt.get("capacity_charge_ah")
    if not baseline_cap_dis or baseline_cap_dis <= 0:
        return None
    if not baseline_cap_chg or baseline_cap_chg <= 0:
        return None

    cycles_window = [c for c in cycles_1_5 if c.get("regular_cycle") >= baseline_cycle]
    return cycles_window, float(baseline_cap_dis), float(baseline_cap_chg)


def _process_cell_features(d: dict,
                           raw_tagged: Optional[pl.DataFrame] = None,
                           run_tier_c: bool = False,
                           baseline_cycle: int = DEFAULT_BASELINE_CYCLE,
                           ) -> tuple[Optional[dict], list[dict]]:
    """Return (one feature row, status_rows) or (None, []) if the cell
    must be omitted.

    Args:
        d: parsed annotation JSON for one cell.
        raw_tagged: output of ``load_raw_tagged(cell)``. When None, Tier
            B columns stay null.
        run_tier_c: when True, run the KWW fit loop and populate Tier C
            columns. status_rows is the per-(cell, cycle) audit log to
            be appended to ``cell_features_status.csv``.
        baseline_cycle: N0 for Tier A/B retention. Tier C is unaffected.
    """
    check = _check_omit(d, baseline_cycle=baseline_cycle)
    if check is None:
        return None, []
    cycles_window, baseline_cap_dis, baseline_cap_chg = check
    cell = d["cell_name"]

    row: dict = {
        "cell_name": cell,
        **_tier_a(cycles_window, baseline_cap_dis, baseline_cap_chg),
        **_NULL_TIER_BC,
    }

    if raw_tagged is not None:
        row.update(_tier_b(raw_tagged, cycles_window))

    status_rows: list[dict] = []
    if run_tier_c:
        # Tier C is anchored to the calendar cycles 3/4/5 (KWW fit on
        # CV-phase current decay), independent of N0. Re-derive that
        # slice from the full cycles_1_5 (recomputed locally — cheaper
        # than threading another arg through).
        regulars = iter_regulars(d)
        cycles_1_5 = regulars[:5]
        cycles_3_5 = cycles_1_5[2:5]   # regular_cycle 3, 4, 5
        tier_c_updates, status_rows = _tier_c(cell, cycles_3_5)
        row.update(tier_c_updates)

    return row, status_rows


# ---------------------------------------------------------------------------
# Manifest consistency
# ---------------------------------------------------------------------------

def _manifest_columns() -> Optional[list[str]]:
    """Return the column-name list declared for cell_features in the
    manifest, or None if PyYAML / manifest unavailable. Failure to load
    is a soft warning — we don't want to gate the pipeline on a missing
    yaml dep — but a drift between schema and manifest IS an error.
    """
    manifest_path = Path(__file__).resolve().parent / "column_roles.yaml"
    if not manifest_path.exists():
        print(f"WARNING: manifest missing at {manifest_path}", file=sys.stderr)
        return None
    try:
        import yaml  # type: ignore
    except ImportError:
        print("WARNING: PyYAML not available; skipping manifest check",
              file=sys.stderr)
        return None
    m = yaml.safe_load(manifest_path.read_text())
    return [c["name"] for c in m["datasets"]["cell_features"]["columns"]]


def _check_manifest_consistency() -> None:
    """Hard-fail on any drift between SCHEMA and the manifest column list."""
    manifest_cols = _manifest_columns()
    if manifest_cols is None:
        return  # soft skip (warning already printed)
    schema_cols = list(SCHEMA.keys())
    missing_in_manifest = [c for c in schema_cols if c not in manifest_cols]
    missing_in_schema = [c for c in manifest_cols if c not in schema_cols]
    if missing_in_manifest or missing_in_schema:
        print("ERROR: manifest / SCHEMA mismatch", file=sys.stderr)
        if missing_in_manifest:
            print(f"  columns in SCHEMA but not in manifest: {missing_in_manifest}",
                  file=sys.stderr)
        if missing_in_schema:
            print(f"  columns in manifest but not in SCHEMA: {missing_in_schema}",
                  file=sys.stderr)
        sys.exit(3)


# ---------------------------------------------------------------------------
# Selftest
# ---------------------------------------------------------------------------

def _selftest_tier_a() -> int:
    """Hand-built per-cycle scalars → assert Tier-A aggregations."""
    fail = 0

    # Case 1: monotone fade with nice round numbers
    cycles_1_5 = [
        {"regular_cycle": 1, "capacity_discharge_ah": 1.000, "capacity_charge_ah": 1.000, "coulombic_efficiency": 1.000},
        {"regular_cycle": 2, "capacity_discharge_ah": 0.980, "capacity_charge_ah": 0.990, "coulombic_efficiency": 0.990},
        {"regular_cycle": 3, "capacity_discharge_ah": 0.970, "capacity_charge_ah": 0.985, "coulombic_efficiency": 0.985},
        {"regular_cycle": 4, "capacity_discharge_ah": 0.960, "capacity_charge_ah": 0.975, "coulombic_efficiency": 0.985},
        {"regular_cycle": 5, "capacity_discharge_ah": 0.950, "capacity_charge_ah": 0.970, "coulombic_efficiency": 0.979},
    ]
    expected = {
        "coulombic_efficiency_final": 97.9,  # percent
        "discharge_capacity_retention_final": 0.95,
        "charge_capacity_retention_min": 0.970,
    }
    got = _tier_a(cycles_1_5, 1.000, 1.000)
    for k, v in expected.items():
        if abs(got[k] - v) > 1e-9:
            print(f"  [FAIL] monotone fade: {k} got {got[k]} expected {v}")
            fail += 1

    # Case 2: charge cap dips at cycle 3 (min should be at c3)
    cycles_dip = [
        {"regular_cycle": 1, "capacity_discharge_ah": 1.0, "capacity_charge_ah": 1.0, "coulombic_efficiency": 1.0},
        {"regular_cycle": 2, "capacity_discharge_ah": 0.99, "capacity_charge_ah": 0.99, "coulombic_efficiency": 0.99},
        {"regular_cycle": 3, "capacity_discharge_ah": 0.98, "capacity_charge_ah": 0.85, "coulombic_efficiency": 0.95},
        {"regular_cycle": 4, "capacity_discharge_ah": 0.97, "capacity_charge_ah": 0.97, "coulombic_efficiency": 0.985},
        {"regular_cycle": 5, "capacity_discharge_ah": 0.96, "capacity_charge_ah": 0.96, "coulombic_efficiency": 0.99},
    ]
    got2 = _tier_a(cycles_dip, 1.0, 1.0)
    if abs(got2["charge_capacity_retention_min"] - 0.85) > 1e-9:
        print(f"  [FAIL] charge_capacity_retention_min: got {got2['charge_capacity_retention_min']} expected 0.85")
        fail += 1
    if abs(got2["coulombic_efficiency_final"] - 99.0) > 1e-9:
        print(f"  [FAIL] CE_final at c5: got {got2['coulombic_efficiency_final']} expected 99.0")
        fail += 1

    return fail


def _selftest_tier_b() -> int:
    """Synthetic phase frames → exercise _nominal_v_phase + _tier_b."""
    fail = 0

    # _nominal_v_phase: charge — constant 1 A, constant 4 V, 3600 s
    # → ∫V·I dt = 14400, ∫I dt = 3600, ratio = 4.0
    chg = pl.DataFrame({
        "elapsed_time": [0.0, 3600.0],
        "current": [1.0, 1.0],
        "voltage": [4.0, 4.0],
    })
    got = _nominal_v_phase(chg)
    if got is None or abs(got - 4.0) > 1e-9:
        print(f"  [FAIL] _nominal_v_phase charge: got {got} expected 4.0")
        fail += 1

    # _nominal_v_phase: discharge — signed-negative current, abs ratio = 3.5
    dis = pl.DataFrame({
        "elapsed_time": [0.0, 3600.0],
        "current": [-1.0, -1.0],
        "voltage": [3.5, 3.5],
    })
    got = _nominal_v_phase(dis)
    if got is None or abs(got - 3.5) > 1e-9:
        print(f"  [FAIL] _nominal_v_phase discharge: got {got} expected 3.5")
        fail += 1

    # _nominal_v_phase: voltage ramps 3 → 4 V at constant 1 A → 3.5 V mean
    ramp = pl.DataFrame({
        "elapsed_time": [0.0, 3600.0],
        "current": [1.0, 1.0],
        "voltage": [3.0, 4.0],
    })
    got = _nominal_v_phase(ramp)
    if got is None or abs(got - 3.5) > 1e-9:
        print(f"  [FAIL] _nominal_v_phase ramp: got {got} expected 3.5")
        fail += 1

    # Empty df → None
    empty = pl.DataFrame(
        {"elapsed_time": [], "current": [], "voltage": []},
        schema={"elapsed_time": pl.Float64, "current": pl.Float64, "voltage": pl.Float64},
    )
    if _nominal_v_phase(empty) is not None:
        print(f"  [FAIL] _nominal_v_phase empty: not None")
        fail += 1

    # Single row (< 2 samples for trapezoid) → None
    single = pl.DataFrame({"elapsed_time": [0.0], "current": [1.0], "voltage": [4.0]})
    if _nominal_v_phase(single) is not None:
        print(f"  [FAIL] _nominal_v_phase single-row: not None")
        fail += 1

    # Zero current (∫I dt = 0) → None (degenerate)
    zero_i = pl.DataFrame({
        "elapsed_time": [0.0, 100.0],
        "current": [0.0, 0.0],
        "voltage": [4.0, 4.0],
    })
    if _nominal_v_phase(zero_i) is not None:
        print(f"  [FAIL] _nominal_v_phase zero-current: not None")
        fail += 1

    # _tier_b: 5 cycles. Charge V_nom = 4.20 every cycle (no degradation).
    # Discharge V_nom = 3.50, 3.49, 3.48, 3.47, 3.46.
    rows = []
    for cd in range(1, 6):
        chg_V = 4.20
        dis_V = 3.50 - 0.01 * (cd - 1)
        # 2 rows per phase: constant I and V → trapezoid gives V_nom = V
        # Use distinct elapsed_time across cycles so the synthetic is realistic
        t_base = cd * 1e5
        rows.append({"cd_index": cd, "cd_phase": "charge",
                     "elapsed_time": t_base, "current": 1.0, "voltage": chg_V})
        rows.append({"cd_index": cd, "cd_phase": "charge",
                     "elapsed_time": t_base + 3600, "current": 1.0, "voltage": chg_V})
        rows.append({"cd_index": cd, "cd_phase": "discharge",
                     "elapsed_time": t_base + 3700, "current": -1.0, "voltage": dis_V})
        rows.append({"cd_index": cd, "cd_phase": "discharge",
                     "elapsed_time": t_base + 7300, "current": -1.0, "voltage": dis_V})
    raw_tagged = pl.DataFrame(rows)
    cycles_1_5 = [{"cd_index": i} for i in range(1, 6)]
    got = _tier_b(raw_tagged, cycles_1_5)

    expected_max_chg = 1.0   # all 4.20 → all ratios are 1.0
    expected_max_dis = 1.0   # cycle 1 is the max (degrading downward)
    dis_ratios = [(3.50 - 0.01 * (c - 1)) / 3.50 for c in range(1, 6)]
    expected_std_dis = statistics.stdev(dis_ratios)

    if abs(got["charge_nominal_voltage_retention_max"] - expected_max_chg) > 1e-9:
        print(f"  [FAIL] _tier_b chg_max: got {got['charge_nominal_voltage_retention_max']} expected {expected_max_chg}")
        fail += 1
    if abs(got["discharge_nominal_voltage_retention_max"] - expected_max_dis) > 1e-9:
        print(f"  [FAIL] _tier_b dis_max: got {got['discharge_nominal_voltage_retention_max']} expected {expected_max_dis}")
        fail += 1
    if abs(got["discharge_nominal_voltage_retention_std"] - expected_std_dis) > 1e-9:
        print(f"  [FAIL] _tier_b dis_std: got {got['discharge_nominal_voltage_retention_std']} expected {expected_std_dis}")
        fail += 1

    # Missing cycle (cd_index 99 has no rows) → all-None return
    cycles_missing = [{"cd_index": i} for i in (1, 2, 99, 4, 5)]
    got_bad = _tier_b(raw_tagged, cycles_missing)
    if any(v is not None for v in got_bad.values()):
        print(f"  [FAIL] _tier_b missing-cycle should return all-None: got {got_bad}")
        fail += 1

    return fail


def _selftest_tier_c() -> int:
    """Exercise Tier-C helpers (no real KWW fit; that's workbench-app's job)."""
    fail = 0

    # --- _safe_ratio_scalar ---
    if abs(_safe_ratio_scalar(99.0, 1.0) - 99.0) > 1e-12:
        print("  [FAIL] safe_ratio: 99/1 != 99"); fail += 1
    if not np.isnan(_safe_ratio_scalar(99.0, 0.3)):
        print("  [FAIL] safe_ratio: |denom|<0.5 should NaN"); fail += 1
    if not np.isnan(_safe_ratio_scalar(float("nan"), 1.0)):
        print("  [FAIL] safe_ratio: NaN num should NaN"); fail += 1
    if not np.isnan(_safe_ratio_scalar(99.0, float("nan"))):
        print("  [FAIL] safe_ratio: NaN denom should NaN"); fail += 1
    if abs(_safe_ratio_scalar(-99.0, 1.0) - (-99.0)) > 1e-12:
        print("  [FAIL] safe_ratio: signed num"); fail += 1

    # --- _safe_log_abs_scalar ---
    if abs(_safe_log_abs_scalar(np.e) - 1.0) > 1e-12:
        print("  [FAIL] safe_log_abs: log(e) != 1"); fail += 1
    if abs(_safe_log_abs_scalar(-np.e) - 1.0) > 1e-12:
        print("  [FAIL] safe_log_abs: log(|-e|) != 1"); fail += 1
    if not np.isnan(_safe_log_abs_scalar(0.0)):
        print("  [FAIL] safe_log_abs: log(0) should NaN"); fail += 1
    if not np.isnan(_safe_log_abs_scalar(float("nan"))):
        print("  [FAIL] safe_log_abs: NaN input"); fail += 1

    # --- _within_bounds: tiny stub objects mimicking CVFitResult ---
    class _Stub:
        def __init__(self, **kw):
            self.success = kw.pop("success", True)
            self.tau_fast = kw.pop("tau_fast", 200.0)
            self.tau_slow = kw.pop("tau_slow", 1000.0)
            self.beta = kw.pop("beta", 1.3)
            self.A_fast_frac = kw.pop("A_fast_frac", 25.0)
    if not _within_bounds(_Stub()):
        print("  [FAIL] _within_bounds: nominal good fit rejected"); fail += 1
    if _within_bounds(_Stub(success=False)):
        print("  [FAIL] _within_bounds: success=False accepted"); fail += 1
    if _within_bounds(_Stub(tau_fast=10.0)):
        print("  [FAIL] _within_bounds: tau_fast pinned-low accepted"); fail += 1
    if _within_bounds(_Stub(tau_fast=5000.0)):
        print("  [FAIL] _within_bounds: tau_fast pinned-high accepted"); fail += 1
    if _within_bounds(_Stub(beta=0.3)):
        print("  [FAIL] _within_bounds: beta pinned-low accepted"); fail += 1
    if _within_bounds(_Stub(A_fast_frac=0.0)):
        print("  [FAIL] _within_bounds: A_fast_frac=0 accepted"); fail += 1
    if _within_bounds(_Stub(A_fast_frac=100.0)):
        print("  [FAIL] _within_bounds: A_fast_frac=100 accepted"); fail += 1

    # --- _engineered_a_ratio: all 3 cycles valid ---
    # f = 25 → r = (100-25)/25 = 3.0 every cycle
    got = _engineered_a_ratio(25.0, 25.0, 25.0)
    if abs(got["A_slow_to_fast_mean"] - 3.0) > 1e-12:
        print(f"  [FAIL] engineered mean: got {got['A_slow_to_fast_mean']} expected 3.0"); fail += 1
    if abs(got["A_slow_to_fast_delta"]) > 1e-12:
        print(f"  [FAIL] engineered delta: got {got['A_slow_to_fast_delta']} expected 0"); fail += 1
    if abs(got["log_A_slow_to_fast_delta"]) > 1e-12:
        print(f"  [FAIL] engineered log delta: got {got['log_A_slow_to_fast_delta']} expected 0"); fail += 1
    if abs(got["A_fast_frac_cv"]) > 1e-12:
        print(f"  [FAIL] engineered cv (uniform): got {got['A_fast_frac_cv']} expected 0"); fail += 1

    # --- _engineered_a_ratio: drift from c3 to c5 ---
    # f3=10, f5=50  →  r3 = 9.0, r5 = 1.0, delta = -8.0
    # log delta = log(1) - log(9) = -ln(9)
    got2 = _engineered_a_ratio(10.0, 30.0, 50.0)
    if abs(got2["A_slow_to_fast_delta"] - (-8.0)) > 1e-12:
        print(f"  [FAIL] engineered delta drift: got {got2['A_slow_to_fast_delta']} expected -8.0"); fail += 1
    if abs(got2["log_A_slow_to_fast_delta"] - (-np.log(9.0))) > 1e-12:
        print(f"  [FAIL] engineered log delta drift: got {got2['log_A_slow_to_fast_delta']}"); fail += 1

    # --- _engineered_a_ratio: 1 cycle failed (NaN A_fast_frac) → partial result ---
    got3 = _engineered_a_ratio(25.0, float("nan"), 25.0)
    if abs(got3["A_slow_to_fast_mean"] - 3.0) > 1e-12:
        print(f"  [FAIL] engineered mean partial: got {got3['A_slow_to_fast_mean']}"); fail += 1
    # delta still OK because c3 and c5 are finite
    if abs(got3["A_slow_to_fast_delta"]) > 1e-12:
        print(f"  [FAIL] engineered delta partial: got {got3['A_slow_to_fast_delta']}"); fail += 1

    # --- _engineered_a_ratio: A_fast_frac below eps → NaN ratios ---
    got4 = _engineered_a_ratio(0.1, 0.2, 0.3)   # all < 0.5
    for k in ("A_slow_to_fast_mean", "A_slow_to_fast_delta",
              "log_A_slow_to_fast_delta"):
        if not np.isnan(got4[k]):
            print(f"  [FAIL] engineered {k} on tiny frac: got {got4[k]} expected NaN")
            fail += 1
    # A_fast_frac_cv: f_mean = 0.2, |f_mean| = 0.2 < 0.5 → NaN
    if not np.isnan(got4["A_fast_frac_cv"]):
        print(f"  [FAIL] engineered cv on tiny frac: got {got4['A_fast_frac_cv']} expected NaN")
        fail += 1

    # --- _engineered_a_ratio: all NaN → all NaN ---
    nan = float("nan")
    got5 = _engineered_a_ratio(nan, nan, nan)
    for k, v in got5.items():
        if not np.isnan(v):
            print(f"  [FAIL] engineered all-NaN: {k}={v} expected NaN"); fail += 1

    return fail


def _selftest_omission() -> int:
    """Cells with <5 cycles, missing baseline, etc. → None."""
    fail = 0

    # < 5 regulars
    d_short = {
        "cell_name": "AR-short",
        "cycling_consistency": "single_rate",
        "cd_events": [
            {"event_kind": "regular_cd", "regular_cycle": i,
             "capacity_discharge_ah": 1.0 - 0.01 * i,
             "capacity_charge_ah": 1.0 - 0.005 * i,
             "coulombic_efficiency": 0.99}
            for i in range(1, 4)
        ],
    }
    row, _ = _process_cell_features(d_short)
    if row is not None:
        print("  [FAIL] short cell (3 cycles) not omitted")
        fail += 1

    # rate_changed with empty regimes (no cycling at all) → omitted
    d_rc_empty = {"cell_name": "AR-rc-empty", "cycling_consistency": "rate_changed",
                  "regular_rate_regimes": [], "cd_events": []}
    row, _ = _process_cell_features(d_rc_empty)
    if row is not None:
        print("  [FAIL] rate_changed with no regimes not omitted")
        fail += 1

    # rate_changed with regime[0].n_regular_cd < 5 → omitted (would feature
    # across the rate boundary)
    d_rc_short_first = {
        "cell_name": "AR-rc-short-first",
        "cycling_consistency": "rate_changed",
        "regular_rate_regimes": [
            {"seg_id": 0, "n_regular_cd": 3, "baseline_i_a": 0.1,
             "baseline_i_dis_a": 0.1, "frac_of_total_regulars": 0.3},
            {"seg_id": 0, "n_regular_cd": 7, "baseline_i_a": 0.04,
             "baseline_i_dis_a": 0.04, "frac_of_total_regulars": 0.7},
        ],
        "cd_events": [
            {"event_kind": "regular_cd", "regular_cycle": i,
             "capacity_discharge_ah": 1.0 - 0.01 * i,
             "capacity_charge_ah": 1.0 - 0.005 * i,
             "coulombic_efficiency": 0.99}
            for i in range(1, 11)
        ],
    }
    row, _ = _process_cell_features(d_rc_short_first)
    if row is not None:
        print("  [FAIL] rate_changed with regime[0].n=3 not omitted")
        fail += 1

    # rate_changed with regime[0].n_regular_cd >= 5 → ADMITTED. The feature
    # window covers cycles 1..5, all at the original rate. The cell still
    # gets status='excluded' on the LABEL side, but the FEATURE row exists
    # for downstream production-inference scoring.
    d_rc_admitted = {
        "cell_name": "AR-rc-admitted",
        "cycling_consistency": "rate_changed",
        "regular_rate_regimes": [
            {"seg_id": 0, "n_regular_cd": 5, "baseline_i_a": 0.1,
             "baseline_i_dis_a": 0.1, "frac_of_total_regulars": 0.5},
            {"seg_id": 0, "n_regular_cd": 5, "baseline_i_a": 0.04,
             "baseline_i_dis_a": 0.04, "frac_of_total_regulars": 0.5},
        ],
        "cd_events": [
            {"event_kind": "regular_cd", "regular_cycle": i,
             "capacity_discharge_ah": 1.0 - 0.01 * i,
             "capacity_charge_ah": 1.0 - 0.005 * i,
             "coulombic_efficiency": 0.99}
            for i in range(1, 11)
        ],
    }
    row, _ = _process_cell_features(d_rc_admitted)
    if row is None:
        print("  [FAIL] rate_changed with regime[0].n=5 omitted (should be admitted)")
        fail += 1

    # missing cycle 1
    d_no_c1 = {
        "cell_name": "AR-no-c1",
        "cycling_consistency": "single_rate",
        "cd_events": [
            {"event_kind": "regular_cd", "regular_cycle": i,
             "capacity_discharge_ah": 1.0, "capacity_charge_ah": 1.0,
             "coulombic_efficiency": 0.99}
            for i in (2, 3, 4, 5, 6)
        ],
    }
    row, _ = _process_cell_features(d_no_c1)
    if row is not None:
        print("  [FAIL] no-cycle-1 cell not omitted")
        fail += 1

    # cycle-1 baseline <= 0
    d_bad_baseline = {
        "cell_name": "AR-bad-base",
        "cycling_consistency": "single_rate",
        "cd_events": [
            {"event_kind": "regular_cd", "regular_cycle": 1,
             "capacity_discharge_ah": 0.0, "capacity_charge_ah": 0.0,
             "coulombic_efficiency": 0.99},
            *[
                {"event_kind": "regular_cd", "regular_cycle": i,
                 "capacity_discharge_ah": 1.0, "capacity_charge_ah": 1.0,
                 "coulombic_efficiency": 0.99}
                for i in (2, 3, 4, 5)
            ],
        ],
    }
    row, _ = _process_cell_features(d_bad_baseline)
    if row is not None:
        print("  [FAIL] zero-baseline cell not omitted")
        fail += 1

    return fail


def _selftest_baseline_cycle() -> int:
    """Verify Tier A and the omission gate honor a non-default
    baseline_cycle. Tier C and Tier B do not get a baseline-specific
    case here — Tier C is independent of baseline, Tier B requires a
    workbench-app raw frame and is exercised by the main run.
    """
    fail = 0

    cap_dis = [1.00, 0.99, 0.98, 0.97, 0.96]
    cap_chg = [1.00, 0.99, 0.85, 0.97, 0.96]   # min over c3..c5 = 0.85 (= cap_chg[c3])
    ce = [1.00, 0.99, 0.985, 0.985, 0.979]

    # Window for baseline=3 is cycles 3-5 → [0.98, 0.97, 0.96]
    cycles_window_b3 = [
        {"regular_cycle": rc, "capacity_discharge_ah": cap_dis[i],
         "capacity_charge_ah": cap_chg[i], "coulombic_efficiency": ce[i]}
        for i, rc in enumerate(range(1, 6)) if rc >= 3
    ]
    got = _tier_a(cycles_window_b3, cap_dis[2], cap_chg[2])
    expected = {
        "coulombic_efficiency_final": 97.9,           # cycle 5 CE, unchanged across N0
        "discharge_capacity_retention_final": 0.96 / 0.98,
        "charge_capacity_retention_min": 0.85 / 0.85,  # min(c3..c5) / cap_chg(c3)
    }
    for k, v in expected.items():
        if abs(got[k] - v) > 1e-9:
            print(f"  [FAIL] tier_a baseline=3: {k} got {got[k]} expected {v}")
            fail += 1

    # _check_omit with baseline=3: cell must still have cycles 1..5, and
    # cycle 3 must have positive baseline.
    d_good = {
        "cell_name": "AR-b3-good",
        "cycling_consistency": "single_rate",
        "cd_events": [
            {"event_kind": "regular_cd", "regular_cycle": rc,
             "capacity_discharge_ah": cap_dis[i],
             "capacity_charge_ah": cap_chg[i],
             "coulombic_efficiency": ce[i]}
            for i, rc in enumerate(range(1, 6))
        ],
    }
    res = _check_omit(d_good, baseline_cycle=3)
    if res is None:
        print("  [FAIL] _check_omit baseline=3: keepable cell rejected")
        fail += 1
    else:
        window, b_dis, b_chg = res
        if len(window) != 3:
            print(f"  [FAIL] _check_omit baseline=3: window len {len(window)} expected 3")
            fail += 1
        if abs(b_dis - cap_dis[2]) > 1e-9:
            print(f"  [FAIL] _check_omit baseline=3: b_dis {b_dis} expected {cap_dis[2]}")
            fail += 1

    # Same JSON but missing cycle 3 (still has cycles 1, 2, 4, 5, 6 — 5 regulars
    # total, but no cycle 3). Should be omitted under baseline=3.
    cd_events = [
        {"event_kind": "regular_cd", "regular_cycle": rc,
         "capacity_discharge_ah": 1.0 - 0.01 * i,
         "capacity_charge_ah": 1.0 - 0.01 * i,
         "coulombic_efficiency": 0.99}
        for i, rc in enumerate((1, 2, 4, 5, 6))
    ]
    d_no_c3 = {"cell_name": "AR-b3-no-c3", "cycling_consistency": "single_rate",
               "cd_events": cd_events}
    # With baseline=1 this cell is keepable (cycle 1 exists, 5 regulars).
    if _check_omit(d_no_c3, baseline_cycle=1) is None:
        print("  [FAIL] _check_omit baseline=1: cell with 5 regulars (1,2,4,5,6) wrongly omitted")
        fail += 1
    # With baseline=3 it must be omitted (cycle 3 missing).
    if _check_omit(d_no_c3, baseline_cycle=3) is not None:
        print("  [FAIL] _check_omit baseline=3: cell missing cycle 3 not omitted")
        fail += 1

    if fail:
        print(f"\n{fail} features baseline_cycle self-test cases FAILED")
    else:
        print("  [PASS] features baseline_cycle (Tier A + _check_omit)")
    return fail


def selftest() -> int:
    """Run features selftests; return count of failures."""
    print("Self-test (features):")
    fail = 0
    fail += _selftest_tier_a()
    fail += _selftest_tier_b()
    fail += _selftest_tier_c()
    fail += _selftest_omission()
    fail += _selftest_baseline_cycle()

    # Manifest consistency check (separate from per-stage math)
    manifest_cols = _manifest_columns()
    if manifest_cols is not None:
        schema_cols = list(SCHEMA.keys())
        if manifest_cols != schema_cols and set(manifest_cols) != set(schema_cols):
            print(f"  [FAIL] manifest column set mismatch")
            fail += 1
        else:
            print(f"  [PASS] manifest consistency ({len(schema_cols)} cols)")

    if fail:
        print(f"\n{fail} features self-test cases FAILED")
    else:
        print("All features self-test cases PASSED")
    return fail


# ---------------------------------------------------------------------------
# Pipeline entry-point
# ---------------------------------------------------------------------------

def _column_roles_sha256() -> Optional[str]:
    """SHA-256 hex digest of column_roles.yaml, or None if not found.

    Stamped into manifest.json so downstream can detect schema drift
    without parsing the YAML.
    """
    p = Path(__file__).resolve().parent / "column_roles.yaml"
    if not p.exists():
        return None
    return hashlib.sha256(p.read_bytes()).hexdigest()


def main(
    cells: Optional[list[str]] = None,
    baseline_cycle: int = DEFAULT_BASELINE_CYCLE,
    db_version: str = "A2.2",
) -> None:
    """Iterate annotations, build feature rows, write outputs.

    Args:
        cells: Optional whitelist of cell_name strings (debugging). When
            None, processes every annotation JSON in ANNOT_DIR.
        baseline_cycle: N0 for Tier A/B retention features. Default 1
            reproduces v1 behavior; pass 3 (or other) to test alternate
            baselines.
        db_version: DB version tag for the output bundle path
            (datasets/{db_version}_b{baseline_cycle}/).
    """
    _check_manifest_consistency()

    cell_filter = set(cells) if cells else None
    rows: list[dict] = []
    all_status_rows: list[dict] = []
    n_scanned = 0
    n_omitted = 0
    n_raw_missing = 0
    for path, d in iter_annotations():
        n_scanned += 1
        cell = d.get("cell_name")
        if cell_filter is not None and cell not in cell_filter:
            continue

        # Cheap omission decision (annotation JSON only). Avoids loading
        # raw parquets for cells that we'd discard anyway.
        if _check_omit(d, baseline_cycle=baseline_cycle) is None:
            n_omitted += 1
            continue

        # Tier B/C need the cell's raw parquet tagged with cd_index +
        # cd_phase. Cells with annotation-but-no-parquet (uncommon)
        # produce a row with null Tier B/C columns.
        raw_tagged: Optional[pl.DataFrame] = None
        try:
            raw_tagged = load_raw_tagged(cell)
        except FileNotFoundError:
            n_raw_missing += 1

        row, status_rows = _process_cell_features(
            d, raw_tagged=raw_tagged, run_tier_c=True,
            baseline_cycle=baseline_cycle,
        )
        if row is None:
            # Should not happen — _check_omit already accepted this cell.
            n_omitted += 1
            continue
        rows.append(row)
        all_status_rows.extend(status_rows)
        if n_scanned % 25 == 0:
            print(f"  [{n_scanned}] scanned; kept={len(rows)}, omitted={n_omitted}",
                  file=sys.stderr)

    if not rows:
        print(f"ERROR: no feature rows produced "
              f"(scanned={n_scanned}, omitted={n_omitted})", file=sys.stderr)
        sys.exit(1)

    # cell_name alone is enough for sort: cell_name has cohort as its
    # prefix ("0MC..." sorts before "AR..."), so the row order matches
    # the previous (cohort, cell_name) sort without needing a cohort
    # column in the output.
    out_dir = dataset_dir_for(db_version, baseline_cycle)
    df = pl.DataFrame(rows, schema=SCHEMA).sort("cell_name")
    parquet_path, csv_path = write_outputs(df, "cell_features", out_dir=out_dir)

    # Status CSV (Tier C diagnostics). Per-(cell, cycle) row of fit
    # success / error / param diagnostics.
    status_csv_path: Optional[Path] = None
    if all_status_rows:
        status_df = pl.DataFrame(all_status_rows).sort(["cell_name", "regular_cycle"])
        status_csv_path = out_dir / "cell_features_status.csv"
        status_df.write_csv(status_csv_path)

    manifest_path = write_manifest(out_dir, {
        "schema_version": SCHEMA_VERSION,
        "db_version": db_version,
        "baseline_cycle": baseline_cycle,
        "annot_dir": str(ANNOT_DIR),
        "n_cells_features": df.height,
        "column_roles_sha256": _column_roles_sha256(),
        "stages_populated": ["features"],
    })

    print(f"db_version     = {db_version}")
    print(f"baseline_cycle = {baseline_cycle}")
    print(f"scanned        = {n_scanned}")
    print(f"omitted        = {n_omitted} (rate_changed / no_regular / n_regular<5 / no baseline)")
    print(f"raw missing    = {n_raw_missing} (annotation present, parquet absent — Tier B+ null)")
    print(f"rows           = {df.height}")
    print(f"columns        = {len(df.columns)} (cell_name + features only; cohort + cv_n_success "
          f"intentionally excluded — join cell_labels for cohort, aggregate status CSV for cv_n_success)")
    print(f"written        = {parquet_path}")
    print(f"                 {csv_path}")
    if status_csv_path is not None:
        print(f"                 {status_csv_path}  ({len(all_status_rows)} fit-attempt rows)")
    print(f"                 {manifest_path}")
    print()
    tier_b_cols = [
        "discharge_nominal_voltage_retention_max",
        "discharge_nominal_voltage_retention_std",
        "charge_nominal_voltage_retention_max",
    ]
    print("Tier B sanity (nulls per column):")
    for c in tier_b_cols:
        n_null = df[c].null_count()
        print(f"  {c:<48s} nulls={n_null}/{df.height}")
    print()
    print("Tier C fit-success distribution (from status CSV):")
    if all_status_rows:
        succ_per_cell: dict[str, int] = {}
        for s in all_status_rows:
            cn = s["cell_name"]
            succ_per_cell[cn] = succ_per_cell.get(cn, 0) + (1 if s["success"] else 0)
        for k in range(4):
            n = sum(1 for v in succ_per_cell.values() if v == k)
            print(f"  cv_n_success == {k}: {n}")
    print()
    print("Tier A sanity (CE_final):")
    print(df["coulombic_efficiency_final"].describe())

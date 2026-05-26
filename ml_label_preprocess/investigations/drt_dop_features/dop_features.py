"""dop_features — DOP peak θ from first 10 min charge / 5 min discharge.

Parallel to investigations/dqdv_features/dqdv_features.py. For every
cell we fit four chronoamperometry transients with hybrid-drt:

    (cycle 1, charge,    10 min window)
    (cycle 5, charge,    10 min window)
    (cycle 1, discharge,  5 min window)
    (cycle 5, discharge,  5 min window)

Each fit yields a DOP distribution ρ(θ). We keep only the θ of the
dominant peak (largest ρ_max) — see ``pick_dominant_peak_theta`` — and
emit two signed c5 − c1 shifts as derived features.

Output schema (6 columns):
    dop_peak_theta_c1_chg
    dop_peak_theta_c5_chg
    dop_peak_theta_c1_dis
    dop_peak_theta_c5_dis
    dop_peak_theta_shift_chg_c1c5   = c5_chg − c1_chg  (signed)
    dop_peak_theta_shift_dis_c1c5   = c5_dis − c1_dis  (signed)

Dependency: hybrid-drt via battery_workbench.core.analysis.drt_wrapper.
The wrapper lazy-imports hybdrt, so this module can be IMPORTED without
hybrid-drt installed — only the actual fit_transient call needs it.

Author: Mark Liao (Sheng-Lun Liao)
"""
from __future__ import annotations

import math
import time
from typing import Optional

from battery_workbench.core.analysis.drt import (
    DIRECTION_CHARGE,
    DIRECTION_DISCHARGE,
    extract_cd_transient,
)
from battery_workbench.core.analysis.drt_wrapper import (
    DOPPeakInfo,
    DRTAnalyzer,
    DRTResult,
)


CHARGE_WINDOW_MIN: float = 10.0
DISCHARGE_WINDOW_MIN: float = 5.0
_CYCLES_NEEDED: tuple[int, ...] = (1, 5)

FEATURE_COLUMNS: tuple[str, ...] = (
    "dop_peak_theta_c1_chg",
    "dop_peak_theta_c5_chg",
    "dop_peak_theta_c1_dis",
    "dop_peak_theta_c5_dis",
    "dop_peak_theta_shift_chg_c1c5",
    "dop_peak_theta_shift_dis_c1c5",
)

# Segment specs. Order matches the column order above so the result-
# collection dict can be iterated deterministically by the plotter.
_SEGMENTS: tuple[tuple[str, int, str, float], ...] = (
    # (segment_key, cycle, direction, minutes)
    ("c1_chg", 1, DIRECTION_CHARGE,    CHARGE_WINDOW_MIN),
    ("c5_chg", 5, DIRECTION_CHARGE,    CHARGE_WINDOW_MIN),
    ("c1_dis", 1, DIRECTION_DISCHARGE, DISCHARGE_WINDOW_MIN),
    ("c5_dis", 5, DIRECTION_DISCHARGE, DISCHARGE_WINDOW_MIN),
)


# ---------------------------------------------------------------------------
# Pure helpers — testable without hybrid-drt.
# ---------------------------------------------------------------------------


def pick_dominant_peak_theta(peaks: list[DOPPeakInfo]) -> float:
    """θ (degrees) of the peak with the largest ρ_max.

    Returns NaN when ``peaks`` is empty. Ties are broken by first
    occurrence (``max`` with a key function is stable on ties).
    """
    if not peaks:
        return math.nan
    dominant = max(peaks, key=lambda p: p.rho_max)
    return float(dominant.theta_center)


def peak_shift(theta_c1: float, theta_c5: float) -> float:
    """Signed θ_c5 − θ_c1, NaN-propagating.

    Returns NaN if either input is non-finite.
    """
    if not (math.isfinite(theta_c1) and math.isfinite(theta_c5)):
        return math.nan
    return float(theta_c5 - theta_c1)


# ---------------------------------------------------------------------------
# Annotation -> cd_index. Duplicated from dqdv_features to keep this
# investigation self-contained.
# ---------------------------------------------------------------------------


def _regular_cycle_cd_index(annot: dict, regular_cycle: int) -> Optional[int]:
    for ev in annot.get("cd_events", []):
        if (
            ev.get("event_kind") == "regular_cd"
            and ev.get("regular_cycle") == regular_cycle
        ):
            return int(ev["cd_index"])
    return None


# ---------------------------------------------------------------------------
# One-segment driver.
# ---------------------------------------------------------------------------


def _fit_dop_one_segment(
    cell_name: str,
    cd_index: int,
    direction: str,
    minutes: float,
    analyzer: DRTAnalyzer,
) -> tuple[float, bool, str | None, Optional[DRTResult]]:
    """Run one chronoamperometry fit + dominant-peak extraction.

    Returns (theta_deg, dop_ok, error_msg, result):
        theta_deg : float, dominant DOP peak's θ (deg). NaN on any failure.
        dop_ok    : True iff ≥1 DOP peak found.
        error_msg : None on success; short tag on failure
                    ('extract_failed:*', 'fit_failed:*', 'no_dop_peaks').
        result    : DRTResult on fit success (regardless of dop_ok), else None.
                    Held so the plotter can show ρ(θ) without re-fitting.
    """
    try:
        times, current, voltage, delta_I = extract_cd_transient(
            cell_name, cd_index, direction, minutes=minutes,
        )
    except ValueError as exc:
        return math.nan, False, f"extract_failed:{exc.__class__.__name__}", None
    except Exception as exc:
        return math.nan, False, f"extract_failed:{type(exc).__name__}", None

    try:
        result = analyzer.fit_transient(
            times, current, voltage, delta_I, direction=direction,
        )
    except Exception as exc:
        return math.nan, False, f"fit_failed:{type(exc).__name__}", None

    if not result.dop_peaks:
        return math.nan, False, "no_dop_peaks", result

    theta = pick_dominant_peak_theta(result.dop_peaks)
    return theta, True, None, result


# ---------------------------------------------------------------------------
# Orchestration — one cell to 6 features + status row + segment results.
# ---------------------------------------------------------------------------


def featurize_cell(
    cell_name: str,
    annot: dict,
    analyzer: Optional[DRTAnalyzer] = None,
) -> tuple[dict, dict, dict[str, Optional[DRTResult]]]:
    """Compute DOP features for one cell.

    Returns ``(feature_row, status_row, segment_results)``:
        feature_row     : cell_name + 6 feature cols (caller adds cohort).
        status_row      : cell_name + per-segment success flags + timing.
        segment_results : segment_key -> DRTResult (or None). The plotter
                          consumes these without re-fitting.

    All 6 features default to NaN. If the annotation lacks regular
    cycle 1 or 5, the function returns immediately with all NaN and a
    populated ``error_msg``.
    """
    feat: dict = {"cell_name": cell_name}
    for col in FEATURE_COLUMNS:
        feat[col] = math.nan

    status: dict = {
        "cell_name": cell_name,
        "has_c1_chg": False, "has_c5_chg": False,
        "has_c1_dis": False, "has_c5_dis": False,
        "dop_ok_c1_chg": False, "dop_ok_c5_chg": False,
        "dop_ok_c1_dis": False, "dop_ok_c5_dis": False,
        "fit_time_s_total": 0.0,
        "n_features_success": 0,
        "error_msg": "",
    }

    segment_results: dict[str, Optional[DRTResult]] = {
        key: None for key, *_ in _SEGMENTS
    }

    cd1 = _regular_cycle_cd_index(annot, 1)
    cd5 = _regular_cycle_cd_index(annot, 5)
    cd_lookup = {1: cd1, 5: cd5}
    if cd1 is None or cd5 is None:
        status["error_msg"] = f"missing regular cycle: cd1={cd1} cd5={cd5}"
        return feat, status, segment_results

    if analyzer is None:
        analyzer = DRTAnalyzer(fit_dop=True)

    first_err: Optional[str] = None
    t_start = time.perf_counter()
    for seg_key, cycle, direction, minutes in _SEGMENTS:
        cd_idx = cd_lookup[cycle]
        # Direction abbrev for column name: "chg" / "dis".
        direction_abbrev = "chg" if direction == DIRECTION_CHARGE else "dis"
        feat_key = f"dop_peak_theta_c{cycle}_{direction_abbrev}"
        has_key = f"has_c{cycle}_{direction_abbrev}"
        ok_key = f"dop_ok_c{cycle}_{direction_abbrev}"

        theta, dop_ok, err, result = _fit_dop_one_segment(
            cell_name, cd_idx, direction, minutes, analyzer,
        )
        feat[feat_key] = theta
        # "has" = extract succeeded (we got data). Anything past extract
        # counts as "has data". Only an extract_failed err clears it.
        status[has_key] = err is None or not err.startswith("extract_failed")
        status[ok_key] = dop_ok
        segment_results[seg_key] = result
        if err and first_err is None:
            first_err = f"{seg_key}:{err}"

    status["fit_time_s_total"] = float(time.perf_counter() - t_start)

    # Derived shifts (NaN-propagating).
    feat["dop_peak_theta_shift_chg_c1c5"] = peak_shift(
        feat["dop_peak_theta_c1_chg"], feat["dop_peak_theta_c5_chg"],
    )
    feat["dop_peak_theta_shift_dis_c1c5"] = peak_shift(
        feat["dop_peak_theta_c1_dis"], feat["dop_peak_theta_c5_dis"],
    )

    status["n_features_success"] = sum(
        1 for col in FEATURE_COLUMNS
        if isinstance(feat[col], float) and math.isfinite(feat[col])
    )
    status["error_msg"] = first_err or ""

    return feat, status, segment_results

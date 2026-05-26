"""dQ/dV feature extraction — 8 features per cell, two design groups.

Pure functions on numpy arrays. Each per-feature helper returns NaN
on degenerate inputs (too few samples, empty arrays, non-overlapping
voltage ranges) so a single bad half-cycle doesn't kill the whole row.

Two feature groups are emitted side-by-side. Each is a self-contained
4-column design — downstream consumers select whichever set they want.

v1 — physically-interpretable peak/shape features (V units, dimensionless):

    dqdv_peak_v_c5_dis                  V of dominant discharge dQ/dV peak at c5
    dqdv_peak_v_shift_c1c5_dis          signed V_peak_c5_dis − V_peak_c1_dis
    dqdv_charge_discharge_hysteresis_c5 signed V_peak_charge − V_peak_discharge at c5
    dqdv_cosine_sim_c1c5_dis            cosine sim of c1 and c5 discharge dQ/dV(V)

v2 — Severson-style ΔQ(V) statistics (all dimensionless):

    dqv_norm_c5_c1_var_log10    log10( var( ΔQ_norm(V) ) )
    dqv_norm_c5_c1_min          min(  ΔQ_norm(V) )
    dqv_norm_c5_c1_mean         mean( ΔQ_norm(V) )
    dqv_norm_c5_c1_skew         sample skewness of ΔQ_norm(V)

Where

    ΔQ_norm(V) = ( |Q_c5(V)| − |Q_c1(V)| ) / |c1_discharge_capacity|

and ``c1_discharge_capacity`` is computed inline from the cell's own
cycle-1 raw discharge curve (NOT from cell_labels.parquet's
``baseline_dis_ah`` — the investigation stays self-contained). The
normalization makes ΔQ a fraction of the cell's own initial capacity,
which removes the 47× cohort capacity scale (0MC ≈ 6.65 Ah baseline
vs AR ≈ 0.14 Ah) so a classifier sees degradation, not cell size.

History:
  - 8-feature draft (peak + Ah-scale ΔQ): 4 absolute-Ah features
    collapsed onto cohort capacity (47× ratio). Dropped.
  - v1: physically-interpretable peak/shape features (this module's
    first revision). Empirically didn't help downstream classification.
  - v2: Severson statistics with c1-capacity normalization (this
    module's second revision). Replaced v1, which was then deleted.
  - Current revision: both v1 and v2 are emitted from one pass so each
    can be regenerated on fresh annotation snapshots. Downstream
    experiments (exp_o, exp_p, exp_q) join whichever set they need.

This module is the "library"; `run_investigation.py` is the CLI that
iterates cells, writes parquet/CSV, and renders diagnostic plots.

Author: Mark Liao (Sheng-Lun Liao)
"""
from __future__ import annotations

import math
from typing import Optional

import numpy as np
import polars as pl
from scipy.signal import find_peaks

# Workbench-app primitives — wired through the .pth file in eis env (same
# as features.py Stage B/C). Imported at module top so a missing dep
# fails fast.
#
# We intentionally do NOT use extract_cc_voltage_capacity here: that
# function reads ``capacity_step`` directly from the renumbered parquet,
# but ~76% of A2.2 cells lack that column. The workbench-app normally
# synthesizes it inside ``CellData.__post_init__`` via cumulative I·dt
# integration; we replicate that integration locally so we work on every
# cell uniformly.
from battery_workbench.core.analysis.dqdv import (  # noqa: E402
    DIRECTION_CHARGE,
    DIRECTION_DISCHARGE,
    DIRECTION_BOTH,
    compute_dqdv,
)
from battery_workbench.core.analysis.cv_fitting import detect_cv_start  # noqa: E402
from battery_workbench.core.data.annotations import load_raw_tagged  # noqa: E402


# ---------------------------------------------------------------------------
# Constants — tunable defaults, frozen here so behavior is reproducible
# without passing kwargs through every function.
# ---------------------------------------------------------------------------

COMMON_GRID_STEP_V: float = 0.005      # 5 mV — coarser than 1 mV savgol grid
PEAK_PROMINENCE_FRAC: float = 0.05     # peak prominence ≥ 5% of curve max
PEAK_MIN_DISTANCE_SAMPLES: int = 20    # ≥20 mV between peaks on 1 mV grid
MIN_SAMPLES_FOR_DQDV: int = 10         # mirrors compute_dqdv_savgol guard

# Cycles we touch: the two endpoints (1, 5) for the differential features +
# cycle 3 for diagnostic plotting only.
_CYCLES_NEEDED = (1, 5)
_CYCLES_FOR_PLOTS = (1, 3, 5)


# ---------------------------------------------------------------------------
# Pure feature primitives — operate on numpy arrays, no I/O.
# ---------------------------------------------------------------------------

def common_voltage_grid(
    V_a: np.ndarray,
    V_b: np.ndarray,
    step_v: float = COMMON_GRID_STEP_V,
) -> np.ndarray:
    """Uniform voltage grid spanning the intersection of V_a and V_b.

    Returns empty array if either input is empty or the ranges don't
    overlap with at least 2 grid points.
    """
    if V_a.size == 0 or V_b.size == 0:
        return np.array([])
    v_lo = max(float(np.min(V_a)), float(np.min(V_b)))
    v_hi = min(float(np.max(V_a)), float(np.max(V_b)))
    if v_hi - v_lo < step_v:
        return np.array([])
    return np.arange(v_lo, v_hi + step_v / 2, step_v)


def find_dominant_peak(
    V: np.ndarray,
    dqdv: np.ndarray,
    *,
    direction: str,
    prominence_frac: float = PEAK_PROMINENCE_FRAC,
    min_distance: int = PEAK_MIN_DISTANCE_SAMPLES,
) -> Optional[tuple[float, float, int]]:
    """Locate the dominant peak in a dQ/dV curve.

    Used as a primitive by v1 features (peak voltage, peak shift,
    charge↔discharge hysteresis) and as the marker source for the
    overlay diagnostic plot.

    For discharge (negative dqdv) the search runs on ``|dqdv|``;
    ``height`` is reported signed so downstream consumers can keep the
    sign convention if they want it.
    """
    if direction not in (DIRECTION_CHARGE, DIRECTION_DISCHARGE):
        raise ValueError(f"direction must be charge|discharge, got {direction!r}")
    if V.size < MIN_SAMPLES_FOR_DQDV or dqdv.size != V.size:
        return None

    signal = np.abs(dqdv)
    s_max = float(np.max(signal))
    if not math.isfinite(s_max) or s_max <= 0:
        return None

    peaks, _props = find_peaks(
        signal,
        prominence=s_max * prominence_frac,
        distance=min_distance,
    )
    if peaks.size == 0:
        return None

    dominant = peaks[int(np.argmax(signal[peaks]))]
    return float(V[dominant]), float(dqdv[dominant]), int(peaks.size)


def _sample_skewness(x: np.ndarray) -> float:
    """Population skewness E[((x-μ)/σ)^3], NaN on degenerate inputs.

    Matches scipy.stats.skew default (bias=True) without taking on the
    scipy.stats dependency.
    """
    if x.size < 3:
        return math.nan
    m = float(np.mean(x))
    s = float(np.std(x, ddof=0))
    if s == 0 or not math.isfinite(s):
        return math.nan
    return float(np.mean(((x - m) / s) ** 3))


# ---------------------------------------------------------------------------
# v1 feature primitives — physically-interpretable peak/shape statistics.
# ---------------------------------------------------------------------------

def peak_voltage_discharge(V_dqdv: np.ndarray, dqdv: np.ndarray) -> float:
    """V of the dominant discharge dQ/dV peak (V). NaN if no peak."""
    peak = find_dominant_peak(V_dqdv, dqdv, direction=DIRECTION_DISCHARGE)
    return math.nan if peak is None else float(peak[0])


def peak_voltage_shift_c1c5_dis(
    V1d: np.ndarray, d1d: np.ndarray,
    V5d: np.ndarray, d5d: np.ndarray,
) -> float:
    """Signed shift V_peak_c5_dis − V_peak_c1_dis (V).

    Positive means the c5 peak sits at higher voltage than c1's. NaN
    if either peak detection fails. Sign convention matches the frozen
    v1 parquet (out/20260521_1406/), where ~48% of cells are negative.
    """
    p1 = find_dominant_peak(V1d, d1d, direction=DIRECTION_DISCHARGE)
    p5 = find_dominant_peak(V5d, d5d, direction=DIRECTION_DISCHARGE)
    if p1 is None or p5 is None:
        return math.nan
    return float(p5[0] - p1[0])


def charge_discharge_hysteresis(
    V_c: np.ndarray, d_c: np.ndarray,
    V_d: np.ndarray, d_d: np.ndarray,
) -> float:
    """Signed hysteresis V_peak_charge − V_peak_discharge (V) at one cycle.

    Almost always positive (charge peak sits at higher V than discharge
    peak due to overpotential). NaN if either peak missing. Sign
    convention matches the frozen v1 parquet — only ~2/470 cells have
    a negative value there.
    """
    pc = find_dominant_peak(V_c, d_c, direction=DIRECTION_CHARGE)
    pd = find_dominant_peak(V_d, d_d, direction=DIRECTION_DISCHARGE)
    if pc is None or pd is None:
        return math.nan
    return float(pc[0] - pd[0])


def cosine_similarity_c1c5_dis(
    V1d: np.ndarray, d1d: np.ndarray,
    V5d: np.ndarray, d5d: np.ndarray,
    step_v: float = COMMON_GRID_STEP_V,
) -> float:
    """Cosine similarity of c1 and c5 discharge dQ/dV(V) on a common grid.

    Both curves are interpolated onto the V-range intersection at
    ``step_v`` spacing. Returns NaN if either input is too short, the
    voltage ranges don't overlap, or either curve has zero norm on the
    common grid.
    """
    if V1d.size < 2 or V5d.size < 2:
        return math.nan
    grid = common_voltage_grid(V1d, V5d, step_v=step_v)
    if grid.size < 2:
        return math.nan
    s1 = np.argsort(V1d)
    s5 = np.argsort(V5d)
    a = np.interp(grid, V1d[s1], d1d[s1])
    b = np.interp(grid, V5d[s5], d5d[s5])
    na = float(np.linalg.norm(a))
    nb = float(np.linalg.norm(b))
    if na == 0 or nb == 0 or not (math.isfinite(na) and math.isfinite(nb)):
        return math.nan
    return float(np.dot(a, b) / (na * nb))


# ---------------------------------------------------------------------------
# v2 feature primitive — Severson ΔQ(V) statistics.
# ---------------------------------------------------------------------------

def severson_delta_q_normalized(
    V_c1: np.ndarray,
    Q_c1: np.ndarray,
    V_c5: np.ndarray,
    Q_c5: np.ndarray,
    step_v: float = COMMON_GRID_STEP_V,
) -> dict[str, float]:
    """4 Severson statistics of ΔQ_norm(V), normalized by c1 capacity.

    ΔQ_norm(V) = (|Q_c5(V)| - |Q_c1(V)|) / |c1_total_discharge|

    The denominator is computed inline from the c1 discharge curve as
    ``|Q_c1[-1] - Q_c1[0]|`` (cumulative-capacity span) — NOT looked
    up from cell_labels.parquet, so this module stays standalone.
    With the standard baseline_cycle=1 convention in
    ../../features.py, the two values are identical, modulo numerical
    precision of the local I·dt integration.

    Returns dict with all 4 keys NaN when any of:
      - either input has < 2 samples
      - c1 capacity is 0 or non-finite
      - voltage ranges don't overlap (no common grid)
      - variance is 0 (log10 undefined)
    """
    null = {
        "dqv_norm_c5_c1_var_log10": math.nan,
        "dqv_norm_c5_c1_min":  math.nan,
        "dqv_norm_c5_c1_mean": math.nan,
        "dqv_norm_c5_c1_skew": math.nan,
    }
    if V_c1.size < 2 or V_c5.size < 2:
        return null

    # c1 discharge total |Q| — the cell's own initial capacity. Q arrays
    # from _half_cycle_VQI start at 0 and accumulate, so the span is
    # |Q[-1] - Q[0]| regardless of cycler sign convention.
    c1_cap = float(abs(Q_c1[-1] - Q_c1[0]))
    if c1_cap <= 0 or not math.isfinite(c1_cap):
        return null

    grid = common_voltage_grid(V_c1, V_c5, step_v=step_v)
    if grid.size < 2:
        return null

    s1 = np.argsort(V_c1)
    s5 = np.argsort(V_c5)
    q1 = np.interp(grid, V_c1[s1], np.abs(Q_c1[s1]))
    q5 = np.interp(grid, V_c5[s5], np.abs(Q_c5[s5]))
    delta = (q5 - q1) / c1_cap

    var = float(np.var(delta))
    var_log10 = float(math.log10(var)) if var > 0 else math.nan
    return {
        "dqv_norm_c5_c1_var_log10": var_log10,
        "dqv_norm_c5_c1_min":  float(np.min(delta)),
        "dqv_norm_c5_c1_mean": float(np.mean(delta)),
        "dqv_norm_c5_c1_skew": _sample_skewness(delta),
    }


# ---------------------------------------------------------------------------
# Per-cell orchestration: load raw curves, compute features, return rows.
# ---------------------------------------------------------------------------

def _regular_cycle_cd_index(annot: dict, regular_cycle: int) -> Optional[int]:
    """Find cd_index for the requested regular_cycle, or None if absent."""
    for ev in annot.get("cd_events", []):
        if (
            ev.get("event_kind") == "regular_cd"
            and ev.get("regular_cycle") == regular_cycle
        ):
            return int(ev["cd_index"])
    return None


def _integrate_capacity_step(I: np.ndarray, t: np.ndarray) -> np.ndarray:
    """Cumulative ∫ I·dt/3600 in Ah, starting from 0 at the first sample.

    Mirrors CellData._compute_capacity in
    battery-workbench-app: (I_avg × dt) / 3600, accumulated. Used as a
    fallback for renumbered parquets missing the ``capacity_step``
    column (most of A2.2).
    """
    if I.size < 2:
        return np.zeros_like(I, dtype=float)
    dt = np.diff(t)
    I_avg = (I[:-1] + I[1:]) / 2.0
    dq = I_avg * dt / 3600.0
    return np.concatenate([[0.0], np.cumsum(dq)])


def _half_cycle_VQI(
    raw_tagged: pl.DataFrame,
    cd_index: int,
    direction: str,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Slice one half-cycle from a load_raw_tagged frame.

    Returns ``(V, Q, I)`` for the requested (cd_index, direction).
    ``Q`` is computed locally via cumulative I·dt integration; if the
    parquet happens to carry a ``capacity_step`` column we ignore it
    in favour of the locally-integrated version, so behaviour is
    uniform across cells regardless of which writer produced the
    parquet.
    """
    sub = raw_tagged.filter(
        (pl.col("cd_index") == cd_index) & (pl.col("cd_phase") == direction)
    ).sort("step_time")
    if sub.is_empty():
        return np.array([]), np.array([]), np.array([])
    V = sub["voltage"].to_numpy()
    I = sub["current"].to_numpy()
    t = sub["step_time"].to_numpy()
    Q = _integrate_capacity_step(I, t)
    return V, Q, I


def _load_dqdv(raw_tagged: pl.DataFrame, cd_index: int, direction: str
               ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Return (V_raw, Q_raw, V_dqdv, dqdv) for one (cell, cd, direction).

    For charge: applies ``detect_cv_start`` and trims the CV plateau so
    dQ/dV doesn't blow up at V_max. Pure-CC charges (no CV detected)
    keep the full arrays. For discharge: returns the full half-cycle —
    standard cycling discharges have no CV phase.

    Inputs come from ``load_raw_tagged(cell_name)`` so we only hit the
    parquet once per cell. ``capacity_step`` is recomputed locally; see
    ``_integrate_capacity_step``.
    """
    V, Q, I = _half_cycle_VQI(raw_tagged, cd_index, direction)
    if V.size < MIN_SAMPLES_FOR_DQDV:
        return V, Q, np.array([]), np.array([])

    if direction == DIRECTION_CHARGE:
        try:
            cv_idx = detect_cv_start(V, 0.0005, current=I)
        except ValueError:
            cv_idx = V.size  # pure-CC charge — no CV plateau
        V = V[:cv_idx]
        Q = Q[:cv_idx]
        if V.size < MIN_SAMPLES_FOR_DQDV:
            return V, Q, np.array([]), np.array([])

    V_d, d = compute_dqdv(V, Q, method="savgol")
    return V, Q, V_d, d


def featurize_cell(
    cell_name: str,
    annot: dict,
    raw_tagged: Optional[pl.DataFrame] = None,
) -> tuple[dict, dict]:
    """Compute the 8 dQ/dV features (4 v1 + 4 v2) + status for one cell.

    Returns (feature_row, status_row). The feature row always has all
    keys; missing values are ``math.nan``. status_row carries per-stage
    diagnostics so downstream QC can tell whether a NaN came from
    missing data, peak detection failure, or an exception.

    ``raw_tagged`` may be pre-loaded (e.g. by the caller for sharing
    with diagnostic plotting); otherwise it's loaded here.
    """
    features: dict[str, float] = {k: math.nan for k in FEATURE_COLUMNS}
    status: dict[str, object] = {
        "cell_name": cell_name,
        "has_c1_dis": False, "has_c5_dis": False, "has_c5_chg": False,
        "c1_discharge_cap_ah": math.nan,
        "dqdv_v1_n_success": 0,
        "dqdv_v2_n_success": 0,
        "error_msg": "",
    }

    try:
        # Hard gate: skip cells without at least 5 regular cd cycles —
        # mirrors the omission rule in ../../features.py so the dQ/dV
        # cohort matches the main cell_features cohort.
        n_regular = sum(
            1 for e in annot.get("cd_events", [])
            if e.get("event_kind") == "regular_cd"
            and e.get("regular_cycle") is not None
        )
        if n_regular < 5:
            status["error_msg"] = f"n_regular={n_regular} < 5 (skipped)"
            return {"cell_name": cell_name, **features}, status

        cd1 = _regular_cycle_cd_index(annot, 1)
        cd5 = _regular_cycle_cd_index(annot, 5)
        if cd1 is None or cd5 is None:
            status["error_msg"] = f"missing regular_cycle: cd1={cd1}, cd5={cd5}"
            return {"cell_name": cell_name, **features}, status

        # One parquet read per cell; everything below slices it.
        if raw_tagged is None:
            raw_tagged = load_raw_tagged(cell_name)

        # Load curves: c1 discharge, c5 discharge, c5 charge.
        # _load_dqdv returns (V_raw, Q_raw, V_dqdv, dqdv) — raw needed
        # for v2 (Severson Q-domain stats), dqdv needed for v1 (peaks +
        # cosine sim on the smoothed dQ/dV curves).
        V1d_raw, Q1d_raw, V1d_dq, d1d_dq = _load_dqdv(raw_tagged, cd1, DIRECTION_DISCHARGE)
        V5d_raw, Q5d_raw, V5d_dq, d5d_dq = _load_dqdv(raw_tagged, cd5, DIRECTION_DISCHARGE)
        _, _, V5c_dq, d5c_dq = _load_dqdv(raw_tagged, cd5, DIRECTION_CHARGE)

        status["has_c1_dis"] = V1d_raw.size > 0
        status["has_c5_dis"] = V5d_raw.size > 0
        status["has_c5_chg"] = V5c_dq.size > 0
        if Q1d_raw.size >= 2:
            status["c1_discharge_cap_ah"] = float(abs(Q1d_raw[-1] - Q1d_raw[0]))

        # v1 features (peak/shape, on smoothed dQ/dV curves).
        if V5d_dq.size:
            features["dqdv_peak_v_c5_dis"] = peak_voltage_discharge(V5d_dq, d5d_dq)
        if V1d_dq.size and V5d_dq.size:
            features["dqdv_peak_v_shift_c1c5_dis"] = peak_voltage_shift_c1c5_dis(
                V1d_dq, d1d_dq, V5d_dq, d5d_dq,
            )
            features["dqdv_cosine_sim_c1c5_dis"] = cosine_similarity_c1c5_dis(
                V1d_dq, d1d_dq, V5d_dq, d5d_dq,
            )
        if V5c_dq.size and V5d_dq.size:
            features["dqdv_charge_discharge_hysteresis_c5"] = charge_discharge_hysteresis(
                V5c_dq, d5c_dq, V5d_dq, d5d_dq,
            )

        # v2 features (Severson ΔQ stats, on raw V, Q curves).
        if V1d_raw.size and V5d_raw.size:
            features.update(severson_delta_q_normalized(
                V1d_raw, Q1d_raw, V5d_raw, Q5d_raw,
            ))

    except Exception as exc:
        status["error_msg"] = f"{type(exc).__name__}: {exc}"

    status["dqdv_v1_n_success"] = sum(
        1 for k in FEATURE_COLUMNS_V1
        if not (isinstance(features[k], float) and math.isnan(features[k]))
    )
    status["dqdv_v2_n_success"] = sum(
        1 for k in FEATURE_COLUMNS_V2
        if not (isinstance(features[k], float) and math.isnan(features[k]))
    )
    return {"cell_name": cell_name, **features}, status


# Convenience exports for the CLI runner. v1 and v2 are kept as
# separate tuples so run_investigation.py can write one parquet per set
# without re-splitting strings; FEATURE_COLUMNS is their union.
FEATURE_COLUMNS_V1: tuple[str, ...] = (
    "dqdv_peak_v_c5_dis",
    "dqdv_peak_v_shift_c1c5_dis",
    "dqdv_charge_discharge_hysteresis_c5",
    "dqdv_cosine_sim_c1c5_dis",
)
FEATURE_COLUMNS_V2: tuple[str, ...] = (
    "dqv_norm_c5_c1_var_log10",
    "dqv_norm_c5_c1_min",
    "dqv_norm_c5_c1_mean",
    "dqv_norm_c5_c1_skew",
)
FEATURE_COLUMNS: tuple[str, ...] = FEATURE_COLUMNS_V1 + FEATURE_COLUMNS_V2

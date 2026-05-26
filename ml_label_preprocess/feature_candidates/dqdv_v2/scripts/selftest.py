"""Synthetic-case unit tests for dqdv_features pure functions.

Run before any CLI run to verify peak detection, cosine similarity,
and ΔQ statistics behave correctly on hand-built inputs. Returns
non-zero exit code on any failure.

Author: Mark Liao (Sheng-Lun Liao)
"""
from __future__ import annotations

import math
import sys

import numpy as np

from dqdv_features import (
    common_voltage_grid,
    find_dominant_peak,
    peak_voltage_discharge,
    peak_voltage_shift_c1c5_dis,
    charge_discharge_hysteresis,
    cosine_similarity_c1c5_dis,
    severson_delta_q_normalized,
    _sample_skewness,
)
from battery_workbench.core.analysis.dqdv import (
    DIRECTION_CHARGE,
    DIRECTION_DISCHARGE,
)


# ---------------------------------------------------------------------------
# Helpers for synthetic curves.
# ---------------------------------------------------------------------------

def _gaussian_dqdv(V_min=3.0, V_max=4.2, mu=3.7, sigma=0.05,
                   amplitude=1.0, sign=+1.0, step=0.001):
    """Synthetic Gaussian-shaped dQ/dV curve."""
    V = np.arange(V_min, V_max + step / 2, step)
    dqdv = sign * amplitude * np.exp(-0.5 * ((V - mu) / sigma) ** 2)
    return V, dqdv


def _shifted_gaussian_pair(mu_a=3.70, mu_b=3.71, **kw):
    """Two Gaussians differing only in peak position."""
    Va, da = _gaussian_dqdv(mu=mu_a, **kw)
    Vb, db = _gaussian_dqdv(mu=mu_b, **kw)
    return Va, da, Vb, db


# ---------------------------------------------------------------------------
# Tests.
# ---------------------------------------------------------------------------

def _check(condition, label, *, info=""):
    status = "PASS" if condition else "FAIL"
    line = f"  [{status}] {label}"
    if info and not condition:
        line += f" — {info}"
    print(line)
    return 0 if condition else 1


def test_common_grid() -> int:
    fail = 0
    g = common_voltage_grid(np.array([3.0, 4.0]), np.array([3.5, 4.5]), step_v=0.01)
    fail += _check(g.size > 0 and abs(g[0] - 3.5) < 1e-6 and abs(g[-1] - 4.0) < 0.01,
                   "common_voltage_grid intersects correctly")
    fail += _check(common_voltage_grid(np.array([]), np.array([1.0, 2.0])).size == 0,
                   "common_voltage_grid empty input → empty")
    fail += _check(common_voltage_grid(np.array([0.0, 1.0]), np.array([2.0, 3.0])).size == 0,
                   "common_voltage_grid disjoint ranges → empty")
    return fail


def test_find_dominant_peak() -> int:
    fail = 0
    V, dqdv = _gaussian_dqdv(mu=3.70, sigma=0.05)
    peak = find_dominant_peak(V, dqdv, direction=DIRECTION_CHARGE)
    fail += _check(peak is not None, "Gaussian peak found")
    if peak is not None:
        v_peak, h, n = peak
        fail += _check(abs(v_peak - 3.70) < 0.002,
                       f"Gaussian peak V within 2 mV of 3.70 (got {v_peak:.4f})")
        fail += _check(h > 0.9, f"Gaussian peak height ≈ 1.0 (got {h:.3f})")
        fail += _check(n == 1, f"exactly one peak detected (got n={n})")

    # Discharge: signed peak should be negative.
    V, d = _gaussian_dqdv(mu=3.65, sigma=0.05, sign=-1.0)
    peak = find_dominant_peak(V, d, direction=DIRECTION_DISCHARGE)
    fail += _check(peak is not None and peak[1] < 0,
                   "discharge peak: height is negative (signed)")

    # No peak: flat curve.
    V = np.linspace(3.0, 4.2, 1201)
    d = np.zeros_like(V)
    fail += _check(find_dominant_peak(V, d, direction=DIRECTION_CHARGE) is None,
                   "flat curve → no peak")

    # Bad direction raises.
    raised = False
    try:
        find_dominant_peak(V, d, direction="sideways")
    except ValueError:
        raised = True
    fail += _check(raised, "bad direction raises ValueError")
    return fail


def test_skewness() -> int:
    fail = 0
    fail += _check(math.isnan(_sample_skewness(np.array([]))),
                   "skew of empty array → NaN")
    fail += _check(math.isnan(_sample_skewness(np.array([1.0, 1.0]))),
                   "skew of <3 samples → NaN")
    fail += _check(math.isnan(_sample_skewness(np.ones(100))),
                   "skew of constant array → NaN (σ=0)")
    # Symmetric Gaussian sample → ~0
    rng = np.random.default_rng(42)
    sk = _sample_skewness(rng.normal(size=10_000))
    fail += _check(abs(sk) < 0.1,
                   f"skew of N(0,1) sample ≈ 0 (got {sk:.3f})")
    # Right-skewed: exponential → positive
    sk = _sample_skewness(rng.exponential(size=10_000))
    fail += _check(sk > 1.0,
                   f"skew of Exp(1) sample > 1 (got {sk:.3f})")
    return fail


def test_severson_normalized() -> int:
    fail = 0
    # Synthetic discharge: Q ramps 0 → Q_total as V drops V_max → V_min.
    # Use the same parametrization _half_cycle_VQI would produce
    # (Q starts at 0 in time-order, accumulates to Q_total).
    def discharge_curve(V_max=4.2, V_min=3.0, Q_total=1.0, step=0.001):
        V = np.arange(V_max, V_min - step / 2, -step)        # time-ordered
        Q = np.linspace(0.0, Q_total, V.size)
        return V, Q

    # Identical curves → var=0 → var_log10=NaN; min, mean ≈ 0
    V, Q = discharge_curve(Q_total=1.0)
    feat = severson_delta_q_normalized(V, Q, V, Q)
    fail += _check(math.isnan(feat["dqv_norm_c5_c1_var_log10"]),
                   "identical curves → var_log10 NaN (var=0)")
    fail += _check(abs(feat["dqv_norm_c5_c1_mean"]) < 1e-12,
                   "identical curves → mean ≈ 0")
    fail += _check(abs(feat["dqv_norm_c5_c1_min"]) < 1e-12,
                   "identical curves → min ≈ 0")

    # c5 has 5% less capacity uniformly. ΔQ_norm = (0.95*Q - Q) / 1.0 ramps
    # from 0 to -0.05; on the V grid (which is sorted ascending), the
    # smaller V values correspond to the END of the discharge (highest Q),
    # so the largest negative drop sits at V_min.
    V1, Q1 = discharge_curve(Q_total=1.0)
    V5, Q5 = discharge_curve(Q_total=0.95)
    feat = severson_delta_q_normalized(V1, Q1, V5, Q5)
    fail += _check(feat["dqv_norm_c5_c1_mean"] < 0,
                   f"5% loss → mean < 0 (got {feat['dqv_norm_c5_c1_mean']:.4f})")
    fail += _check(abs(feat["dqv_norm_c5_c1_min"] - (-0.05)) < 0.01,
                   f"5% uniform loss → min ≈ -0.05 (got {feat['dqv_norm_c5_c1_min']:.4f})")
    fail += _check(math.isfinite(feat["dqv_norm_c5_c1_var_log10"]),
                   "non-degenerate → var_log10 finite")

    # Capacity normalization: doubling both Q arrays should not change
    # the normalized statistics (they're fractions of c1 capacity).
    V1, Q1 = discharge_curve(Q_total=1.0)
    V5, Q5 = discharge_curve(Q_total=0.95)
    feat_small = severson_delta_q_normalized(V1, Q1, V5, Q5)
    V1, Q1 = discharge_curve(Q_total=10.0)
    V5, Q5 = discharge_curve(Q_total=9.5)
    feat_big = severson_delta_q_normalized(V1, Q1, V5, Q5)
    for k in feat_small:
        if math.isnan(feat_small[k]):
            fail += _check(math.isnan(feat_big[k]), f"{k} NaN-equivariant under scale")
        else:
            fail += _check(abs(feat_small[k] - feat_big[k]) < 1e-9,
                           f"{k} scale-invariant (small={feat_small[k]:.6f}, "
                           f"big={feat_big[k]:.6f})")

    # Empty input → all NaN
    feat = severson_delta_q_normalized(np.array([]), np.array([]), V5, Q5)
    fail += _check(all(math.isnan(v) for v in feat.values()),
                   "empty input → all 4 features NaN")

    # Zero c1 capacity → all NaN
    feat = severson_delta_q_normalized(V1, np.zeros_like(V1), V5, Q5)
    fail += _check(all(math.isnan(v) for v in feat.values()),
                   "zero c1 capacity → all 4 features NaN")

    return fail


def test_peak_shift() -> int:
    """End-to-end: a known 10 mV shift between two Gaussian curves is
    recovered to within 2 mV."""
    fail = 0
    Va, da, Vb, db = _shifted_gaussian_pair(mu_a=3.700, mu_b=3.710)
    pa = find_dominant_peak(Va, da, direction=DIRECTION_CHARGE)
    pb = find_dominant_peak(Vb, db, direction=DIRECTION_CHARGE)
    fail += _check(pa is not None and pb is not None, "both shifted peaks found")
    if pa is not None and pb is not None:
        shift = pb[0] - pa[0]
        fail += _check(abs(shift - 0.010) < 0.002,
                       f"recovered shift ≈ +10 mV (got {shift*1000:.2f} mV)")
    return fail


def test_peak_voltage_discharge_v1() -> int:
    """v1: dominant discharge peak voltage is recovered to ≤2 mV."""
    fail = 0
    V, d = _gaussian_dqdv(mu=3.55, sigma=0.05, sign=-1.0)
    pv = peak_voltage_discharge(V, d)
    fail += _check(math.isfinite(pv) and abs(pv - 3.55) < 0.002,
                   f"discharge Gaussian peak V ≈ 3.55 (got {pv:.4f})")
    # Empty/short input → NaN
    pv = peak_voltage_discharge(np.array([]), np.array([]))
    fail += _check(math.isnan(pv), "empty input → NaN")
    return fail


def test_peak_voltage_shift_v1() -> int:
    """v1: signed peak shift c1→c5 is recovered (both signs)."""
    fail = 0
    # +10 mV: c5 peak at higher V than c1
    Va, da, Vb, db = _shifted_gaussian_pair(mu_a=3.700, mu_b=3.710, sign=-1.0)
    sh = peak_voltage_shift_c1c5_dis(Va, da, Vb, db)
    fail += _check(math.isfinite(sh) and abs(sh - 0.010) < 0.002,
                   f"+10 mV shift recovered (got {sh*1000:.2f} mV)")
    # −10 mV: c5 peak at lower V than c1
    Va, da, Vb, db = _shifted_gaussian_pair(mu_a=3.710, mu_b=3.700, sign=-1.0)
    sh = peak_voltage_shift_c1c5_dis(Va, da, Vb, db)
    fail += _check(math.isfinite(sh) and abs(sh - (-0.010)) < 0.002,
                   f"−10 mV shift recovered (got {sh*1000:.2f} mV)")
    # Missing peak on one side → NaN
    V_flat = np.linspace(3.0, 4.2, 1201)
    d_flat = np.zeros_like(V_flat)
    V, d = _gaussian_dqdv(mu=3.65, sigma=0.05, sign=-1.0)
    sh = peak_voltage_shift_c1c5_dis(V_flat, d_flat, V, d)
    fail += _check(math.isnan(sh), "flat c1 → NaN shift")
    return fail


def test_charge_discharge_hysteresis_v1() -> int:
    """v1: charge↔discharge hysteresis at one cycle, signed."""
    fail = 0
    # Charge peak at 3.75 V, discharge at 3.65 V → hysteresis = +0.10 V
    V_c, d_c = _gaussian_dqdv(mu=3.75, sigma=0.05, sign=+1.0)
    V_d, d_d = _gaussian_dqdv(mu=3.65, sigma=0.05, sign=-1.0)
    h = charge_discharge_hysteresis(V_c, d_c, V_d, d_d)
    fail += _check(math.isfinite(h) and abs(h - 0.10) < 0.002,
                   f"hysteresis +0.10 V (got {h:.4f})")
    # Inverted case → negative hysteresis (rare but possible)
    V_c, d_c = _gaussian_dqdv(mu=3.55, sigma=0.05, sign=+1.0)
    h = charge_discharge_hysteresis(V_c, d_c, V_d, d_d)
    fail += _check(math.isfinite(h) and h < 0,
                   f"charge below discharge → hysteresis < 0 (got {h:.4f})")
    return fail


def test_cosine_similarity_v1() -> int:
    """v1: cosine sim of c1/c5 discharge dQ/dV curves."""
    fail = 0
    V, d = _gaussian_dqdv(mu=3.65, sigma=0.05, sign=-1.0)
    # Identical curves → 1.0
    cs = cosine_similarity_c1c5_dis(V, d, V, d)
    fail += _check(math.isfinite(cs) and abs(cs - 1.0) < 1e-9,
                   f"identical → 1.0 (got {cs:.6f})")
    # Scale invariance: 2× one curve still 1.0
    cs = cosine_similarity_c1c5_dis(V, d, V, 2.0 * d)
    fail += _check(math.isfinite(cs) and abs(cs - 1.0) < 1e-9,
                   f"scale invariance (got {cs:.6f})")
    # Zero-norm curve → NaN
    V_flat = np.linspace(3.0, 4.2, 1201)
    cs = cosine_similarity_c1c5_dis(V_flat, np.zeros_like(V_flat), V, d)
    fail += _check(math.isnan(cs), "zero c1 curve → NaN")
    # Shifted Gaussians: still highly similar but < 1
    Va, da, Vb, db = _shifted_gaussian_pair(mu_a=3.70, mu_b=3.72, sign=-1.0)
    cs = cosine_similarity_c1c5_dis(Va, da, Vb, db)
    fail += _check(math.isfinite(cs) and 0.5 < cs < 0.999,
                   f"20 mV-shifted Gaussians → 0.5 < cs < 0.999 (got {cs:.4f})")
    # Disjoint voltage ranges → NaN
    V_lo = np.linspace(3.0, 3.2, 200)
    V_hi = np.linspace(3.8, 4.0, 200)
    d_lo = -np.exp(-((V_lo - 3.1) / 0.02) ** 2)
    d_hi = -np.exp(-((V_hi - 3.9) / 0.02) ** 2)
    cs = cosine_similarity_c1c5_dis(V_lo, d_lo, V_hi, d_hi)
    fail += _check(math.isnan(cs), "disjoint V ranges → NaN")
    return fail


def main() -> int:
    print("dqdv_features selftest:")
    fail = 0
    for name, fn in [
        ("common_voltage_grid", test_common_grid),
        ("find_dominant_peak", test_find_dominant_peak),
        ("_sample_skewness", test_skewness),
        ("severson_delta_q_normalized (v2)", test_severson_normalized),
        ("peak_shift_end_to_end (v1 primitive)", test_peak_shift),
        ("peak_voltage_discharge (v1)", test_peak_voltage_discharge_v1),
        ("peak_voltage_shift_c1c5_dis (v1)", test_peak_voltage_shift_v1),
        ("charge_discharge_hysteresis (v1)", test_charge_discharge_hysteresis_v1),
        ("cosine_similarity_c1c5_dis (v1)", test_cosine_similarity_v1),
    ]:
        print(f"[{name}]")
        fail += fn()
    print()
    if fail:
        print(f"FAILED {fail} case(s)", file=sys.stderr)
        return 2
    print("All selftest cases PASSED")
    return 0


if __name__ == "__main__":
    sys.exit(main())

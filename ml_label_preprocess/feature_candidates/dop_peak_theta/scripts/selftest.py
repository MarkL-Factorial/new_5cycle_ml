"""Synthetic tests for dop_features pure helpers.

Construct ``DOPPeakInfo`` instances directly so hybrid-drt is NOT
required to run these tests — they validate the deterministic peak-
selection and shift-arithmetic logic only. The end-to-end fit pipeline
is exercised by ``run_investigation.py --pilot``.

Run with:
    python selftest.py

Exits 0 on success, 1 on any failure.
"""
from __future__ import annotations

import math
import sys
from pathlib import Path

_THIS_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(_THIS_DIR))

from dop_features import pick_dominant_peak_theta, peak_shift  # noqa: E402
from battery_workbench.core.analysis.drt_wrapper import DOPPeakInfo  # noqa: E402


def _peak(theta_center: float, rho_max: float) -> DOPPeakInfo:
    """Helper: minimal DOPPeakInfo with only the fields we exercise."""
    return DOPPeakInfo(
        theta_center=theta_center,
        rho_max=rho_max,
        fwhm_degrees=5.0,
        theta_left=theta_center - 2.5,
        theta_right=theta_center + 2.5,
        area=rho_max * 5.0,
    )


# ---------------------------------------------------------------------------
# pick_dominant_peak_theta
# ---------------------------------------------------------------------------


def test_pick_dominant_empty() -> None:
    out = pick_dominant_peak_theta([])
    assert math.isnan(out), f"empty list should be NaN, got {out}"


def test_pick_dominant_single() -> None:
    out = pick_dominant_peak_theta([_peak(45.0, 0.05)])
    assert out == 45.0, f"single peak should return its theta, got {out}"


def test_pick_dominant_picks_largest_rho() -> None:
    peaks = [
        _peak(30.0, 0.02),  # smaller
        _peak(60.0, 0.07),  # LARGER
        _peak(45.0, 0.04),  # middle
    ]
    out = pick_dominant_peak_theta(peaks)
    assert out == 60.0, f"should pick θ of largest-ρ peak, got {out}"


def test_pick_dominant_negative_theta_ok() -> None:
    # Drt_wrapper filters θ ≤ 5° internally, but the helper itself
    # must accept any signed input. (Defense in depth for future
    # changes to the upstream filter.)
    peaks = [_peak(-12.0, 0.10), _peak(40.0, 0.05)]
    out = pick_dominant_peak_theta(peaks)
    assert out == -12.0, f"expected -12.0, got {out}"


# ---------------------------------------------------------------------------
# peak_shift
# ---------------------------------------------------------------------------


def test_shift_positive() -> None:
    out = peak_shift(30.0, 42.0)
    assert out == 12.0, f"expected +12.0, got {out}"


def test_shift_negative() -> None:
    out = peak_shift(50.0, 35.0)
    assert out == -15.0, f"expected -15.0, got {out}"


def test_shift_zero() -> None:
    out = peak_shift(40.0, 40.0)
    assert out == 0.0, f"expected 0.0, got {out}"


def test_shift_nan_propagation() -> None:
    out1 = peak_shift(math.nan, 40.0)
    assert math.isnan(out1), f"NaN c1 must propagate, got {out1}"
    out2 = peak_shift(40.0, math.nan)
    assert math.isnan(out2), f"NaN c5 must propagate, got {out2}"
    out3 = peak_shift(math.nan, math.nan)
    assert math.isnan(out3), f"both NaN must propagate, got {out3}"


def test_shift_infinity_propagation() -> None:
    out = peak_shift(math.inf, 40.0)
    assert math.isnan(out), f"infinite input must give NaN, got {out}"


# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------

TESTS = [
    test_pick_dominant_empty,
    test_pick_dominant_single,
    test_pick_dominant_picks_largest_rho,
    test_pick_dominant_negative_theta_ok,
    test_shift_positive,
    test_shift_negative,
    test_shift_zero,
    test_shift_nan_propagation,
    test_shift_infinity_propagation,
]


def main() -> int:
    n_pass = 0
    n_fail = 0
    for fn in TESTS:
        try:
            fn()
            print(f"  PASS  {fn.__name__}")
            n_pass += 1
        except AssertionError as exc:
            print(f"  FAIL  {fn.__name__}: {exc}")
            n_fail += 1
        except Exception as exc:
            print(f"  ERROR {fn.__name__}: {type(exc).__name__}: {exc}")
            n_fail += 1
    print()
    print(f"{n_pass}/{len(TESTS)} passed")
    return 0 if n_fail == 0 else 1


if __name__ == "__main__":
    sys.exit(main())

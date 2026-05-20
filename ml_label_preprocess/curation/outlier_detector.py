"""Outlier detector — pure-function algorithm core (Pattern A).

Per-cell algorithm that finds **isolated 1-3-cycle measurement glitches**
in the discharge-retention series, while staying orthogonal to the
sustained-step / regime-shift detection done in
``curation.jump_detection``.

Algorithm: local-residual MAD with **two-sided OLS fit** (LTS-trimmed).

For each tested cycle ``i``:

  pre  := LTS-OLS on cycles  [i - W, i - 1]   →  predict_pre  at cycle i
  post := LTS-OLS on cycles  [i + 1, i + W]   →  predict_post at cycle i

If ``|predict_pre - predict_post| > discontinuity_max``: the curve has a
step at ``i`` — leave this to the jump detector, do NOT flag as outlier.

Otherwise, use ``(predict_pre + predict_post) / 2`` as the smooth-trend
prediction, compute the residual, and flag if ``|residual| > N * sigma``
where ``sigma = max(1.4826 * MAD, sigma_floor)`` over all tested-cycle
residuals in the cell.

Tail handling: when the post window is too short (last few cycles of a
cell), the algorithm degrades to a **pre-only** prediction (no step-edge
guard available). Catches end-of-life outliers that would otherwise
slip through.

Skipped from classification:

  - cycles within ±``boundary_skip`` of a recorded regime boundary
    (those are Pattern B territory)
  - cycles where the pre-window doesn't satisfy ``min_pre_len + n_trim``
    after boundary cycles are dropped

Pure functions only — no I/O. The CLI counterpart is
``curation.outlier_detection``, which iterates annotations and writes
the sidecar + plots that ``labels.py`` and ``curation.sustained_step``
both consume.

Author: Mark Liao (Sheng-Lun Liao)
"""
from __future__ import annotations

import sys
from dataclasses import dataclass
from itertools import combinations
from typing import Optional


# ---------------- parameters (initial defaults; tunable via CLI) ----------------

WINDOW_HALF = 5          # ±W neighbors used for each one-sided OLS fit
MIN_PRE_LEN = 3          # need ≥ this many usable pre-side neighbors (after trim)
MIN_POST_LEN = 3         # need ≥ this many usable post-side neighbors (after trim)
BOUNDARY_SKIP = 2        # ignore cycles within ±this of a regime boundary
SKIP_LAST_N = 0          # don't classify the last N cycles (0 = let pre-only handle tail)
DISCONTINUITY_MAX = 0.03 # |predict_pre - predict_post| above this ⇒ step edge, skip
MAD_MULTIPLIER = 4.0     # |residual| > N * 1.4826 * MAD ⇒ outlier
N_TRIM = 2               # iteratively drop this many largest-residual points before
                         # finalizing each pre/post side fit (handles burst leak)
SIGMA_FLOOR = 0.005      # floor on sigma to suppress floating-point/noise-free curves


@dataclass
class OutlierParams:
    window_half: int = WINDOW_HALF
    min_pre_len: int = MIN_PRE_LEN
    min_post_len: int = MIN_POST_LEN
    boundary_skip: int = BOUNDARY_SKIP
    skip_last_n: int = SKIP_LAST_N
    discontinuity_max: float = DISCONTINUITY_MAX
    mad_multiplier: float = MAD_MULTIPLIER
    n_trim: int = N_TRIM
    sigma_floor: float = SIGMA_FLOOR


@dataclass
class OutlierReport:
    """One row per (cell, outlier-cycle)."""
    list_index: int                # position in the per-cell regulars list
    cycle: int                     # regular_cycle ordinal
    retention: float               # actual retention at that cycle
    predicted: float               # mean of two-sided fits at that cycle
    residual: float                # actual - predicted
    z_score: float                 # (residual - median_residual) / sigma
    pre_post_disagreement: float   # |predict_pre - predict_post|
    pre_n_points: int              # actual pre-window length used after filtering
    post_n_points: int             # actual post-window length used after filtering


# ---------------- pure-Python helpers (mirror style of jump_detection.py) ----------------

def _lstsq_fit(xs: list[int], ys: list[float]) -> tuple[float, float]:
    """OLS: return (slope, intercept) for y = slope*x + intercept."""
    n = len(xs)
    if n < 2:
        return 0.0, (ys[0] if ys else 0.0)
    sx = sum(xs)
    sy = sum(ys)
    sxx = sum(x * x for x in xs)
    sxy = sum(x * y for x, y in zip(xs, ys))
    denom = n * sxx - sx * sx
    if denom == 0:
        return 0.0, sy / n
    slope = (n * sxy - sx * sy) / denom
    intercept = (sy - slope * sx) / n
    return slope, intercept


def _lts_fit(
    xs: list[int], ys: list[float], n_trim: int,
) -> tuple[float, float, int]:
    """Least Trimmed Squares: OLS on the best size-(n-n_trim) subset.

    Enumerates all C(n, n_keep) subsets where n_keep = n - n_trim, fits
    OLS on each, and returns the (slope, intercept) of the subset with
    smallest sum-of-squared-residuals *on its own points*. This is the
    standard high-breakdown-point regression — robust to up to n_trim
    outliers in the window without the tie-breaking pathology of
    iterative single-point trimming.

    For the parameter set used here (window ≤ 10, n_trim = 2), the
    subset enumeration is ≤ C(10, 8) = 45 candidates per call — trivial
    compute even at full-cohort scale.

    Returns (slope, intercept, n_used) where n_used = n_keep.
    """
    n = len(xs)
    n_keep = max(2, n - n_trim)
    if n <= n_keep:
        slope, intercept = _lstsq_fit(xs, ys)
        return slope, intercept, n
    best_ssr = float("inf")
    best_slope = 0.0
    best_intercept = 0.0
    for idx in combinations(range(n), n_keep):
        sub_xs = [xs[k] for k in idx]
        sub_ys = [ys[k] for k in idx]
        slope, intercept = _lstsq_fit(sub_xs, sub_ys)
        ssr = sum((y - (slope * x + intercept)) ** 2
                  for x, y in zip(sub_xs, sub_ys))
        if ssr < best_ssr:
            best_ssr = ssr
            best_slope, best_intercept = slope, intercept
    return best_slope, best_intercept, n_keep


def _median(vals: list[float]) -> float:
    s = sorted(vals)
    n = len(s)
    mid = n // 2
    if n % 2:
        return s[mid]
    return (s[mid - 1] + s[mid]) / 2.0


def _is_near_any(cycle: int, boundaries: list[int], tol: int) -> bool:
    return any(abs(cycle - b) <= tol for b in boundaries)


def _filter_boundary_aware(
    xs: list[int], ys: list[float],
    boundaries: list[int], boundary_skip: int,
) -> tuple[list[int], list[float]]:
    """Drop (x, y) pairs whose x is within ±boundary_skip of any boundary."""
    if not boundaries:
        return list(xs), list(ys)
    keep_xs: list[int] = []
    keep_ys: list[float] = []
    for x, y in zip(xs, ys):
        if not _is_near_any(x, boundaries, boundary_skip):
            keep_xs.append(x)
            keep_ys.append(y)
    return keep_xs, keep_ys


# ---------------- detector ----------------

def detect_outliers(
    cycles: list[int],
    retentions: list[float],
    boundaries: list[int],
    params: Optional[OutlierParams] = None,
) -> list[OutlierReport]:
    """Return one OutlierReport per flagged outlier cycle.

    See module docstring for algorithm details.
    """
    p = params or OutlierParams()
    n = len(cycles)
    if n < 2:
        return []

    # Phase 1: collect (i, predict, residual, disagreement, pre_n, post_n)
    # for every cycle eligible for classification.
    tested: list[tuple[int, float, float, float, int, int]] = []

    last_eligible_idx = n - p.skip_last_n  # exclusive upper bound
    for i in range(last_eligible_idx):
        ci = cycles[i]

        # (a) Skip cycles near a recorded regime boundary
        if _is_near_any(ci, boundaries, p.boundary_skip):
            continue

        # (b) Pre-window: cycles strictly before i
        pre_lo = max(0, i - p.window_half)
        pre_xs_raw = cycles[pre_lo:i]
        pre_ys_raw = retentions[pre_lo:i]
        pre_xs, pre_ys = _filter_boundary_aware(
            pre_xs_raw, pre_ys_raw, boundaries, p.boundary_skip)
        # Need enough points to survive trimming AND still fit a line
        if len(pre_xs) < p.min_pre_len + p.n_trim:
            continue

        # (c) Pre-side LTS fit (always available — pre-window check passed).
        slope_pre, intercept_pre, pre_used = _lts_fit(
            pre_xs, pre_ys, p.n_trim)
        predict_pre = slope_pre * ci + intercept_pre

        # (d) Post-window: cycles strictly after i
        post_hi = min(n, i + p.window_half + 1)
        post_xs_raw = cycles[i + 1:post_hi]
        post_ys_raw = retentions[i + 1:post_hi]
        post_xs, post_ys = _filter_boundary_aware(
            post_xs_raw, post_ys_raw, boundaries, p.boundary_skip)

        # (e) Two-sided fit if enough post points; otherwise pre-only.
        # Pre-only mode handles tail cycles where the cell ends before
        # post_window can be filled. Without a post side, we can't run
        # the step-edge guard — but the last few cycles of a cell are
        # the *end* of the curve, so "step edge" interpretation is
        # irrelevant there anyway. Gradual end-of-life decline shows
        # up across many cycles → MAD captures it → not flagged.
        if len(post_xs) >= p.min_post_len + p.n_trim:
            slope_post, intercept_post, post_used = _lts_fit(
                post_xs, post_ys, p.n_trim)
            predict_post = slope_post * ci + intercept_post
            disagreement = abs(predict_pre - predict_post)
            # Step-edge guard: skip if the two sides disagree too much
            # (curve has a discontinuity at i — that's the jump
            # detector's job, not this one).
            if disagreement > p.discontinuity_max:
                continue
            predict_combined = 0.5 * (predict_pre + predict_post)
        else:
            # Pre-only: no post side, no disagreement check possible.
            post_used = 0
            disagreement = float("nan")
            predict_combined = predict_pre

        residual = retentions[i] - predict_combined
        tested.append((i, predict_combined, residual, disagreement,
                       pre_used, post_used))

    if not tested:
        return []

    # Phase 2: MAD over all tested-cycle residuals (per-cell scale).
    # Apply sigma_floor so that floating-point-quiet curves don't generate
    # absurdly large z-scores from sub-noise-level residuals.
    residuals = [t[2] for t in tested]
    med = _median(residuals)
    mad = _median([abs(r - med) for r in residuals])
    sigma = max(1.4826 * mad, p.sigma_floor)
    threshold = p.mad_multiplier * sigma

    # Phase 3: flag
    reports: list[OutlierReport] = []
    for (i, predict, residual, disagreement, pre_n, post_n) in tested:
        if abs(residual - med) <= threshold:
            continue
        z = (residual - med) / sigma
        reports.append(OutlierReport(
            list_index=i,
            cycle=cycles[i],
            retention=retentions[i],
            predicted=float(predict),
            residual=float(residual),
            z_score=float(z),
            pre_post_disagreement=float(disagreement),
            pre_n_points=pre_n,
            post_n_points=post_n,
        ))
    return reports


# ---------------- selftest ----------------

def _make_curve(cycles_n: int, baseline: float = 1.0,
                fade_per_cycle: float = 0.0005) -> tuple[list[int], list[float]]:
    """Synthetic baseline: linear fade starting at retention 1.0."""
    cycles = list(range(1, cycles_n + 1))
    rets = [baseline - fade_per_cycle * (c - 1) for c in cycles]
    return cycles, rets


def selftest() -> int:
    """Synthetic test cases to verify orthogonality with jump detector.

    Returns 0 on full pass, non-zero count of failures.
    """
    print("Self-test (outlier detector):")
    fail = 0
    p = OutlierParams()

    def _check(name: str, cycles: list[int], rets: list[float],
               boundaries: list[int],
               want_cycles: list[int]) -> None:
        """Assert flagged cycles match ``want_cycles`` exactly."""
        nonlocal fail
        got = detect_outliers(cycles, rets, boundaries, p)
        got_cycles = sorted(r.cycle for r in got)
        ok = (got_cycles == sorted(want_cycles))
        marker = "PASS" if ok else "FAIL"
        if not ok:
            fail += 1
        msg = (f"  [{marker}] {name}: flagged={got_cycles} "
               f"expected={sorted(want_cycles)}")
        for r in got:
            msg += (f"\n           cycle={r.cycle:4d} "
                    f"ret={r.retention:.4f} pred={r.predicted:.4f} "
                    f"resid={r.residual:+.4f} z={r.z_score:+.2f} "
                    f"disagree={r.pre_post_disagreement:.4f}")
        print(msg)

    # 1) Pure smooth fade — zero outliers.
    cyc, ret = _make_curve(100, fade_per_cycle=0.002)
    _check("smooth fade → no outliers", cyc, ret, [], [])

    # 2) Single up-outlier at cycle 50.
    cyc, ret = _make_curve(100, fade_per_cycle=0.002)
    ret[49] += 0.20
    _check("single up-outlier at cycle 50", cyc, ret, [], [50])

    # 3) Single down-outlier at cycle 50.
    cyc, ret = _make_curve(100, fade_per_cycle=0.002)
    ret[49] -= 0.20
    _check("single down-outlier at cycle 50", cyc, ret, [], [50])

    # 4) Two-cycle outlier burst at cycles 50-51.
    cyc, ret = _make_curve(100, fade_per_cycle=0.002)
    ret[49] -= 0.20
    ret[50] -= 0.18
    _check("two-cycle burst at 50-51", cyc, ret, [], [50, 51])

    # 5) Three-cycle outlier burst at cycles 50-52.
    cyc, ret = _make_curve(100, fade_per_cycle=0.002)
    ret[49] += 0.15
    ret[50] += 0.18
    ret[51] += 0.13
    _check("three-cycle burst at 50-52", cyc, ret, [], [50, 51, 52])

    # 6) Sharp upward step at cycle 100 (Pattern C/D shape) — no outliers.
    cyc, ret = _make_curve(200, fade_per_cycle=0.002)
    for k in range(99, 200):
        ret[k] += 0.12
    _check("sharp upward step → no outliers", cyc, ret, [], [])

    # 7) Sharp downward step at cycle 100 — no outliers.
    cyc, ret = _make_curve(200, fade_per_cycle=0.002)
    for k in range(99, 200):
        ret[k] -= 0.12
    _check("sharp downward step → no outliers", cyc, ret, [], [])

    # 8) Outlier at the last cycle — flagged via pre-only fallback.
    cyc, ret = _make_curve(100, fade_per_cycle=0.002)
    ret[-1] += 0.20  # cycle 100
    _check("outlier at last cycle → flagged (pre-only)", cyc, ret, [], [100])

    # 8b) Gradual end-of-life acceleration — NOT flagged. Last 10 cycles
    # fade twice as fast as the rest; pre-only mode would extrapolate the
    # mild trend and find residuals, but the residuals are spread across
    # ~10 cycles so MAD captures them and threshold rises accordingly.
    cyc, ret = _make_curve(100, fade_per_cycle=0.002)
    for k in range(90, 100):
        ret[k] -= 0.002 * (k - 89)  # extra fade in the last 10 cycles
    _check("gradual end-of-life acceleration → not flagged", cyc, ret, [], [])

    # 9) Outlier adjacent to a recorded regime boundary — NOT flagged
    # (boundary_skip=2). Use cycle 50 as boundary; add an outlier at cycle 51.
    cyc, ret = _make_curve(100, fade_per_cycle=0.002)
    ret[50] += 0.20  # cycle 51
    _check("outlier adjacent to boundary → skipped", cyc, ret, [50], [])

    # 10) AR4313-shape: step at cycle 268, no outliers anywhere.
    n = 472
    cyc = list(range(1, n + 1))
    ret = [1.0 - 0.0008 * (c - 1) for c in cyc]
    for k in range(267, n):
        ret[k] += 0.12
    # No glitch points; just the sustained step. The step-edge guard
    # must keep this clean.
    got = detect_outliers(cyc, ret, [], p)
    ok = (len(got) == 0)
    marker = "PASS" if ok else "FAIL"
    if not ok:
        fail += 1
    msg = (f"  [{marker}] AR4313-shape synthetic → no outliers (orthogonality): "
           f"got {len(got)} flagged")
    if got:
        for r in got:
            msg += (f"\n           cycle={r.cycle:4d} "
                    f"resid={r.residual:+.4f} z={r.z_score:+.2f} "
                    f"disagree={r.pre_post_disagreement:.4f}")
    print(msg)

    # 11) Combined: AR4313-shape PLUS a single glitch at cycle 100 — the
    # glitch should be flagged, the step at 268 should NOT.
    n = 472
    cyc = list(range(1, n + 1))
    ret = [1.0 - 0.0008 * (c - 1) for c in cyc]
    for k in range(267, n):
        ret[k] += 0.12
    ret[99] += 0.20  # glitch at cycle 100
    _check("step + isolated glitch → glitch flagged, step not", cyc, ret,
           [], [100])

    if fail:
        print(f"\n{fail} outlier-detector self-test cases FAILED")
    else:
        print("All outlier-detector self-test cases PASSED")
    return fail


def _parse_args(argv: list[str]) -> bool:
    return "--selftest" in argv


if __name__ == "__main__":
    if _parse_args(sys.argv[1:]):
        sys.exit(selftest())
    print("Usage: python -m curation.outlier_detector --selftest", file=sys.stderr)
    sys.exit(2)

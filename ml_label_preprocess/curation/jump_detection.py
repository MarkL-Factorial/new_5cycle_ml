"""Capacity-jump detector — pure-function algorithm core.

Per-cell algorithm that finds step changes in the discharge-retention
curve and discriminates *pathological regime shifts* (sustained
post-jump offset) from *normal RPT-style recovery* (single-cycle blip
that returns to the pre-jump trend).

Pure functions only — no I/O. Consumed by:
  - ``curation.sustained_step`` (CLI: detect_jumps with strict params on
    outlier-masked retentions, ``rate_changed`` excluded).
  - ``../investigations/jump_detection/run_investigation.py`` (the
    historical diagnostic harness with permissive params).

Both consumers feed cell candidates into ``curation.validation``'s
pending review queue, where the human decision in
``curation/decisions.json`` becomes authoritative for
``../labels.py``.

Author: Mark Liao (Sheng-Lun Liao)
"""
from __future__ import annotations

import sys
from dataclasses import dataclass, asdict
from typing import Optional


# ---------------- parameters (initial defaults; tunable via CLI) ----------------

BUMP_MIN = 0.03      # min |Δretention| between adjacent cycles to trigger
PRE_WINDOW = 20      # regular_cycles before candidate, used for the pre-trend fit
POST_WINDOW = 10     # regular_cycles from candidate forward, used for persistence
PERSIST_MIN = 0.03   # min |median post-window residual| to call it sustained
MIN_PRE_LEN = 10     # need at least this many points before to fit a trend
MIN_POST_LEN = 5     # need at least this many points after to judge persistence


@dataclass
class DetectorParams:
    bump_min: float = BUMP_MIN
    pre_window: int = PRE_WINDOW
    post_window: int = POST_WINDOW
    persist_min: float = PERSIST_MIN
    min_pre_len: int = MIN_PRE_LEN
    min_post_len: int = MIN_POST_LEN


@dataclass
class JumpReport:
    """One row per (cell, candidate jump).

    Cells with zero candidates are not reported here; CLI consumers
    (e.g. ``curation.sustained_step``) decide how to represent the
    empty-candidates case in their own output (typically as a sentinel
    row in their summary CSV).
    """
    jump_cycle_idx: int            # list index into the per-cell regulars
    jump_cycle_ordinal: int        # the regular_cycle ordinal (1-based)
    jump_magnitude: float          # signed Δret at the candidate
    jump_direction: str            # 'up' or 'down'
    pre_slope: float               # slope of pre-window linear fit (Δret/cycle)
    pre_intercept: float           # intercept of pre-window linear fit
    pre_n_points: int              # actual pre-window length used (<= PRE_WINDOW)
    post_n_points: int             # actual post-window length used (<= POST_WINDOW)
    persistence_score: float       # signed median residual over the post window
    classification: str            # 'sustained' | 'transient' | 'edge_skip'


# ---------------- retention extraction ----------------

def compute_retentions(regulars: list[dict]) -> tuple[list[int], list[float]]:
    """Convert iter_regulars() output into (cycle_ordinals, retentions).

    Baseline is the first available regular cycle (matches labels.py's
    default baseline_cycle=1 behavior). If the first event's
    capacity_discharge_ah is missing or non-positive, returns ([], []).
    The caller decides what to do with empty cells.
    """
    if not regulars:
        return [], []
    baseline = regulars[0].get("capacity_discharge_ah")
    if baseline is None or baseline <= 0:
        return [], []
    cycles = [int(e["regular_cycle"]) for e in regulars]
    retentions = [float(e["capacity_discharge_ah"]) / float(baseline) for e in regulars]
    return cycles, retentions


# ---------------- linear-trend fit (no external deps) ----------------

def _lstsq_fit(xs: list[int], ys: list[float]) -> tuple[float, float]:
    """Ordinary least squares: return (slope, intercept) for y = slope*x + b.

    Pure-Python so this module has no numpy dependency for the core
    algorithm (the CLI consumers use numpy/matplotlib at the edges).
    """
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


def _median(vals: list[float]) -> float:
    s = sorted(vals)
    n = len(s)
    mid = n // 2
    if n % 2:
        return s[mid]
    return (s[mid - 1] + s[mid]) / 2.0


# ---------------- detector ----------------

def detect_jumps(
    cycles: list[int],
    retentions: list[float],
    params: Optional[DetectorParams] = None,
) -> list[JumpReport]:
    """Return one JumpReport per candidate jump.

    A candidate is any list-index ``i >= 1`` whose
    ``|retentions[i] - retentions[i-1]| >= params.bump_min``. For each
    candidate, fit a linear trend on the pre-window, extrapolate
    forward, and classify based on the median residual.

    Returns ``[]`` if there are no candidates. Returns one or more
    reports otherwise (one per candidate, list-ordered by jump_cycle_idx).
    """
    p = params or DetectorParams()
    n = len(retentions)
    if n < 2:
        return []
    reports: list[JumpReport] = []

    for i in range(1, n):
        delta = retentions[i] - retentions[i - 1]
        if abs(delta) < p.bump_min:
            continue

        pre_lo = max(0, i - p.pre_window)
        pre_xs = cycles[pre_lo:i]
        pre_ys = retentions[pre_lo:i]
        post_hi = min(n, i + p.post_window)
        post_xs = cycles[i:post_hi]
        post_ys = retentions[i:post_hi]

        if len(pre_xs) < p.min_pre_len or len(post_xs) < p.min_post_len:
            reports.append(JumpReport(
                jump_cycle_idx=i,
                jump_cycle_ordinal=cycles[i],
                jump_magnitude=float(delta),
                jump_direction="up" if delta > 0 else "down",
                pre_slope=0.0,
                pre_intercept=0.0,
                pre_n_points=len(pre_xs),
                post_n_points=len(post_xs),
                persistence_score=0.0,
                classification="edge_skip",
            ))
            continue

        slope, intercept = _lstsq_fit(pre_xs, pre_ys)
        residuals = [post_ys[j] - (slope * post_xs[j] + intercept)
                     for j in range(len(post_xs))]
        persist = _median(residuals)
        classification = "sustained" if abs(persist) >= p.persist_min else "transient"

        reports.append(JumpReport(
            jump_cycle_idx=i,
            jump_cycle_ordinal=cycles[i],
            jump_magnitude=float(delta),
            jump_direction="up" if delta > 0 else "down",
            pre_slope=float(slope),
            pre_intercept=float(intercept),
            pre_n_points=len(pre_xs),
            post_n_points=len(post_xs),
            persistence_score=float(persist),
            classification=classification,
        ))

    return reports


# ---------------- selftest ----------------

def _make_curve(cycles_n: int, baseline: float = 1.0,
                fade_per_cycle: float = 0.0005) -> tuple[list[int], list[float]]:
    """Synthetic baseline curve: linear fade, retention starts at 1.0."""
    cycles = list(range(1, cycles_n + 1))
    rets = [baseline - fade_per_cycle * (c - 1) for c in cycles]
    return cycles, rets


def selftest() -> int:
    """Hand-built synthetic curves to verify detection logic.

    Returns 0 on full pass, non-zero count of failures.
    """
    print("Self-test (jump detector):")
    fail = 0
    p = DetectorParams()

    def _check(name: str, cycles: list[int], rets: list[float],
               want_n: int, want: Optional[list[tuple[int, str, str]]] = None) -> int:
        """Assert n candidates + (optional) (ordinal, direction, class) per candidate."""
        nonlocal fail
        got = detect_jumps(cycles, rets, p)
        ok = (len(got) == want_n)
        details = []
        if want is not None and ok:
            for r, exp in zip(got, want):
                triplet_got = (r.jump_cycle_ordinal, r.jump_direction, r.classification)
                if triplet_got != exp:
                    ok = False
                    details.append(f"got {triplet_got!r} expected {exp!r}")
        marker = "PASS" if ok else "FAIL"
        if not ok:
            fail += 1
        msg = f"  [{marker}] {name}: n_candidates={len(got)} (expected {want_n})"
        for d in details:
            msg += f"\n           {d}"
        for r in got:
            msg += (f"\n           cycle={r.jump_cycle_ordinal:4d} "
                    f"Δ={r.jump_magnitude:+.4f} dir={r.jump_direction} "
                    f"persist={r.persistence_score:+.4f} class={r.classification}")
        print(msg)
        return 0

    # 1) pure healthy curve (zero fade) — no candidates
    cyc, ret = _make_curve(100, fade_per_cycle=0.0)
    _check("pure healthy", cyc, ret, want_n=0)

    # 2) monotone fade (smooth) — no candidates
    cyc, ret = _make_curve(100, fade_per_cycle=0.002)
    _check("monotone fade", cyc, ret, want_n=0)

    # 3) RPT-style transient bump: +0.04 at cycle 50, returns within 3 cycles.
    # An RPT bump produces an up-trigger at cycle 50 and a symmetric
    # down-trigger at cycle 51 as the curve falls back. Both are
    # correctly classified 'transient' because the post-window median
    # residual is ~0 in each case (curve returns to trend).
    cyc, ret = _make_curve(100, fade_per_cycle=0.002)
    ret[49] += 0.04           # cycle 50 (index 49) gets bumped
    ret[50] += 0.012          # decaying back
    ret[51] += 0.004
    _check("RPT-style transient bump", cyc, ret, want_n=2,
           want=[(50, "up", "transient"), (51, "down", "transient")])

    # 4) sustained upward step: +0.12 offset starting at cycle 100, persists
    cyc, ret = _make_curve(200, fade_per_cycle=0.002)
    for k in range(99, 200):  # index 99 = cycle 100
        ret[k] += 0.12
    _check("sustained upward step", cyc, ret, want_n=1,
           want=[(100, "up", "sustained")])

    # 5) sustained downward step: -0.10 offset starting at cycle 100
    cyc, ret = _make_curve(200, fade_per_cycle=0.002)
    for k in range(99, 200):
        ret[k] -= 0.10
    _check("sustained downward step", cyc, ret, want_n=1,
           want=[(100, "down", "sustained")])

    # 6) edge skip: jump within first PRE_WINDOW cycles → edge_skip
    cyc, ret = _make_curve(100, fade_per_cycle=0.002)
    ret[4] += 0.10            # cycle 5 — only 4 pre-points
    for k in range(5, 100):
        ret[k] += 0.10
    _check("jump near start → edge_skip", cyc, ret, want_n=1,
           want=[(5, "up", "edge_skip")])

    # 7) edge skip: jump within last POST_WINDOW cycles → edge_skip
    cyc, ret = _make_curve(50, fade_per_cycle=0.002)
    ret[47] -= 0.10           # cycle 48 — only 2 post-points (48,49,50 = 3 incl trigger)
    ret[48] -= 0.10
    ret[49] -= 0.10
    _check("jump near end → edge_skip", cyc, ret, want_n=1,
           want=[(48, "down", "edge_skip")])

    # 8) AR4313-shape: jump from ~0.80 to ~0.92 at index 267 (cycle 268),
    #    persisting through cycles 268..472.
    n = 472
    cyc = list(range(1, n + 1))
    ret = [1.0 - 0.0008 * (c - 1) for c in cyc]  # linear fade through cycle 1..267
    # at cycle 268 (index 267), retention jumps up by ~0.12 and stays elevated
    for k in range(267, n):
        ret[k] += 0.12
    reports = detect_jumps(cyc, ret, p)
    sustained_up = [r for r in reports if r.classification == "sustained"
                    and r.jump_direction == "up"]
    ok = (len(sustained_up) >= 1 and sustained_up[0].jump_cycle_ordinal == 268)
    if not ok:
        fail += 1
    marker = "PASS" if ok else "FAIL"
    print(f"  [{marker}] AR4313-shape synthetic: sustained_up at cycle 268 "
          f"(found {len(sustained_up)} sustained-up, first at "
          f"{sustained_up[0].jump_cycle_ordinal if sustained_up else 'none'})")

    if fail:
        print(f"\n{fail} jump-detector self-test cases FAILED")
    else:
        print("All jump-detector self-test cases PASSED")
    return fail


def _parse_args(argv: list[str]) -> bool:
    """Tiny arg parsing — only --selftest for this module."""
    return "--selftest" in argv


if __name__ == "__main__":
    if _parse_args(sys.argv[1:]):
        sys.exit(selftest())
    print("Usage: python -m curation.jump_detection --selftest", file=sys.stderr)
    sys.exit(2)

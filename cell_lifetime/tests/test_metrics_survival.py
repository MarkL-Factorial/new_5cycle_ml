"""Survival metrics sanity tests (C-index, time-dep AUC)."""

import numpy as np

from cell_lifetime.evaluation.survival_metrics import survival_metrics


def test_perfect_ranking_gives_high_c_index():
    """When risk_scores rank-correlate perfectly with -time on observed events, C-index = 1.0."""
    rng = np.random.default_rng(0)
    event = np.ones(50, dtype=bool)
    time = np.arange(50, dtype=float) + 1.0  # 1..50
    risk = -time  # higher risk = sooner failure
    out = survival_metrics(event, time, risk)
    assert out["c_index"] >= 0.99


def test_anti_ranking_gives_low_c_index():
    event = np.ones(50, dtype=bool)
    time = np.arange(50, dtype=float) + 1.0
    risk = time  # wrong direction
    out = survival_metrics(event, time, risk)
    assert out["c_index"] <= 0.01


def test_random_ranking_near_half():
    rng = np.random.default_rng(42)
    n = 500
    event = rng.random(n) > 0.4
    time = rng.uniform(10, 800, size=n)
    risk = rng.normal(size=n)
    out = survival_metrics(event, time, risk)
    assert abs(out["c_index"] - 0.5) < 0.07


def test_per_cohort_breakdown():
    rng = np.random.default_rng(0)
    n = 200
    event = rng.random(n) > 0.5
    time = rng.uniform(10, 800, size=n)
    risk = -time + rng.normal(scale=10, size=n)  # mostly correct ranking
    cohorts = np.where(rng.random(n) > 0.5, "AR", "0MC")
    out = survival_metrics(event, time, risk, cohorts=cohorts)
    assert "c_index_AR" in out
    assert "c_index_0MC" in out


def test_time_dep_auc_keys_present():
    rng = np.random.default_rng(0)
    n = 300
    event = rng.random(n) > 0.5
    time = rng.uniform(10, 800, size=n)
    risk = rng.normal(size=n)
    out = survival_metrics(event, time, risk)
    for N in (200, 300, 400):
        assert f"auc_at_{N}" in out
        val = out[f"auc_at_{N}"]
        assert np.isnan(val) or (0.0 <= val <= 1.0)


def test_auc_at_horizon_perfect_separation():
    """Manual case: 4 cells, 2 fail before N=300, 2 pass; perfect risk → AUC=1."""
    event = np.array([True, True, False, False])
    time = np.array([100.0, 200.0, 500.0, 600.0])
    risk = np.array([10.0, 8.0, 1.0, 0.5])  # high risk for the failers
    out = survival_metrics(event, time, risk)
    assert out["auc_at_300"] == 1.0

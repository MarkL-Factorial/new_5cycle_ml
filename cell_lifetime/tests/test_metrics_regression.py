"""Regression metrics sanity checks."""

import numpy as np

from cell_lifetime.evaluation.regression_metrics import regression_metrics


def test_perfect_prediction():
    y = np.array([100.0, 200.0, 300.0, 400.0])
    m = regression_metrics(y, y)
    assert m["mae"] == 0.0
    assert m["rmse"] == 0.0
    assert m["medae"] == 0.0
    assert m["r2"] == 1.0
    assert m["n"] == 4


def test_constant_prediction():
    y = np.array([100.0, 200.0, 300.0, 400.0])
    y_pred = np.full_like(y, y.mean())
    m = regression_metrics(y, y_pred)
    assert m["mae"] > 0
    assert m["r2"] == 0.0   # constant prediction → R²=0 (predicts the mean)


def test_per_cohort_breakdown():
    y = np.array([100.0, 200.0, 300.0, 400.0, 500.0, 600.0])
    y_pred = np.array([110.0, 195.0, 320.0, 380.0, 510.0, 590.0])
    cohorts = np.array(["AR", "AR", "AR", "0MC", "0MC", "0MC"])
    m = regression_metrics(y, y_pred, cohorts)
    assert "mae_AR" in m
    assert "mae_0MC" in m
    assert m["mae_AR"] > 0
    assert m["mae_0MC"] > 0

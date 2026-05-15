"""Metrics emit all 5 + per-cohort AUC; NaN-safe."""

import numpy as np

from cell_classifier.evaluation.metrics import metrics_from_predictions, prefix


def test_metrics_from_predictions_complete():
    y = np.array([0, 1, 0, 1, 1, 0])
    y_pred = np.array([0, 1, 0, 0, 1, 1])
    y_proba = np.array([0.1, 0.9, 0.2, 0.4, 0.7, 0.6])
    cohorts = np.array(["AR", "AR", "0MC", "0MC", "AR", "0MC"])
    m = metrics_from_predictions(y, y_pred, y_proba, cohorts)
    for key in ("n", "accuracy", "precision", "recall", "f1", "roc_auc",
                "n_AR", "auc_AR", "n_0MC", "auc_0MC"):
        assert key in m


def test_safe_auc_nan_on_single_class():
    y = np.array([1, 1, 1])
    y_pred = np.array([1, 1, 1])
    y_proba = np.array([0.9, 0.8, 0.7])
    cohorts = np.array(["AR", "AR", "AR"])
    m = metrics_from_predictions(y, y_pred, y_proba, cohorts)
    assert np.isnan(m["roc_auc"])


def test_prefix():
    assert prefix({"a": 1, "b": 2}, "x_") == {"x_a": 1, "x_b": 2}

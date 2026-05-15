"""Tests for ml_classification.metrics."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from ml_classification.metrics import evaluate, prefix


class _FakeModel:
    """Returns fixed predictions/probabilities for deterministic test."""

    def __init__(self, preds, probs):
        self._preds = np.asarray(preds)
        self._probs = np.asarray(probs)

    def predict(self, X):
        return self._preds

    def predict_proba(self, X):
        return np.column_stack([1 - self._probs, self._probs])


def _X(n: int) -> pd.DataFrame:
    return pd.DataFrame({"f": np.arange(n)})


def test_overall_metrics_perfect():
    y = np.array([0, 0, 1, 1])
    model = _FakeModel(preds=y, probs=[0.1, 0.2, 0.8, 0.9])
    out = evaluate(model, _X(4), y, cohorts=np.array(["AR"] * 4))
    assert out["accuracy"] == 1.0
    assert out["f1"] == 1.0
    assert out["roc_auc"] == 1.0


def test_per_cohort_breakdown_split():
    y = np.array([0, 1, 0, 1])
    cohorts = np.array(["AR", "AR", "0MC", "0MC"])
    model = _FakeModel(preds=y, probs=[0.1, 0.9, 0.2, 0.8])
    out = evaluate(model, _X(4), y, cohorts=cohorts)
    assert out["n_AR"] == 2
    assert out["n_0MC"] == 2
    assert out["auc_AR"] == 1.0
    assert out["auc_0MC"] == 1.0


def test_single_class_cohort_yields_nan_auc():
    y = np.array([1, 1, 0, 1])
    cohorts = np.array(["AR", "AR", "0MC", "0MC"])  # 0MC has classes {0,1}; AR is all 1
    model = _FakeModel(preds=[1, 1, 0, 1], probs=[0.9, 0.8, 0.2, 0.7])
    out = evaluate(model, _X(4), y, cohorts=cohorts)
    assert np.isnan(out["auc_AR"])  # single class — undefined
    assert not np.isnan(out["auc_0MC"])  # mixed — defined


def test_prefix_namespacing():
    p = prefix({"a": 1, "b": 2.0}, "train_")
    assert p == {"train_a": 1, "train_b": 2.0}

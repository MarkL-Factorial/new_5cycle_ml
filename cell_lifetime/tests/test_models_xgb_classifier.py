"""XGBoost classifier smoke test on synthetic data."""

import optuna
import numpy as np
import pytest

from cell_lifetime.data.synthetic import make_synthetic_dataset
from cell_lifetime.models.xgb_classifier import XGBClassifierModel


def test_fit_predict_predict_proba():
    ds = make_synthetic_dataset(n_faded=80, n_censored=120, seed=0)
    view = ds.view_for_task("classification")
    params = {"n_estimators": 50, "max_depth": 3, "learning_rate": 0.1}
    model = XGBClassifierModel(params).fit(view.X, view.y_class)
    pred = model.predict(view.X)
    proba = model.predict_proba(view.X)
    assert pred.shape == (len(view),)
    assert set(np.unique(pred)).issubset({0, 1})
    assert proba.shape == (len(view), 2)
    np.testing.assert_allclose(proba.sum(axis=1), 1.0, atol=1e-6)


def test_suggest_params_returns_dict():
    study = optuna.create_study()
    trial = study.ask()
    params = XGBClassifierModel.suggest_params(trial)
    assert "n_estimators" in params
    assert "max_depth" in params


def test_feature_importance():
    ds = make_synthetic_dataset(seed=0)
    view = ds.view_for_task("classification")
    model = XGBClassifierModel({"n_estimators": 30, "max_depth": 3}).fit(view.X, view.y_class)
    fi = model.feature_importance(view.feature_names)
    assert set(fi.keys()) == set(view.feature_names)
    assert all(v >= 0 for v in fi.values())

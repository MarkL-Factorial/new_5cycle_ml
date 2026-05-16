"""Random Survival Forest smoke tests on synthetic data.

Guarded: skipped if scikit-survival isn't installed.
"""

import numpy as np
import optuna
import pytest

pytest.importorskip("sksurv")

from cell_lifetime.data.synthetic import make_synthetic_dataset
from cell_lifetime.models.rsf import RSFModel


def test_task_and_orientation_attrs():
    assert RSFModel.task == "survival"
    assert RSFModel.risk_orientation == "risk_high"


def test_fit_predict_signature():
    ds = make_synthetic_dataset(n_faded=80, n_censored=120, seed=0)
    view = ds.view_for_task("survival")
    params = {"n_estimators": 30, "max_depth": 6, "min_samples_split": 6,
              "min_samples_leaf": 6, "max_features": "sqrt"}
    model = RSFModel(params).fit(view.X, view.time, view.event)
    pred = model.predict(view.X)
    assert pred.shape == (len(view),)
    assert np.isfinite(pred).all()


def test_fit_requires_time_and_event():
    params = {"n_estimators": 10, "max_depth": 4, "min_samples_split": 5,
              "min_samples_leaf": 5, "max_features": "sqrt"}
    ds = make_synthetic_dataset(seed=0)
    view = ds.view_for_task("survival")
    model = RSFModel(params)
    with pytest.raises(TypeError):
        model.fit(view.X)


def test_predict_proba_raises():
    params = {"n_estimators": 10, "max_depth": 4, "min_samples_split": 5,
              "min_samples_leaf": 5, "max_features": "sqrt"}
    ds = make_synthetic_dataset(seed=0)
    view = ds.view_for_task("survival")
    model = RSFModel(params).fit(view.X, view.time, view.event)
    with pytest.raises(NotImplementedError):
        model.predict_proba(view.X)


def test_suggest_params_returns_dict():
    study = optuna.create_study()
    trial = study.ask()
    p = RSFModel.suggest_params(trial)
    for key in ("n_estimators", "max_depth", "min_samples_split",
                "min_samples_leaf", "max_features"):
        assert key in p


def test_c_index_above_random_on_synthetic():
    """RSF with signal-bearing synthetic data → C-index well above 0.5."""
    from cell_lifetime.evaluation.survival_metrics import survival_metrics
    ds = make_synthetic_dataset(n_faded=100, n_censored=100, seed=1)
    view = ds.view_for_task("survival")
    params = {"n_estimators": 80, "max_depth": 8, "min_samples_split": 6,
              "min_samples_leaf": 6, "max_features": "sqrt"}
    model = RSFModel(params).fit(view.X, view.time, view.event)
    # risk_high orientation: pass predict() through unchanged
    risk = model.predict(view.X)
    out = survival_metrics(view.event, view.time, risk)
    assert out["c_index"] > 0.6   # in-sample, with signal, easy threshold


def test_feature_importance_keys():
    params = {"n_estimators": 20, "max_depth": 5, "min_samples_split": 5,
              "min_samples_leaf": 5, "max_features": "sqrt"}
    ds = make_synthetic_dataset(seed=0)
    view = ds.view_for_task("survival")
    model = RSFModel(params).fit(view.X, view.time, view.event)
    fi = model.feature_importance(view.feature_names)
    assert set(fi.keys()) == set(view.feature_names)

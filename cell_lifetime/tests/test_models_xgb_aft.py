"""XGBoost-AFT survival model smoke tests on synthetic data."""

import numpy as np
import optuna
import pytest

from cell_lifetime.data.synthetic import make_synthetic_dataset
from cell_lifetime.models.xgb_aft import XGBAFTModel


def test_fit_predict_signature():
    ds = make_synthetic_dataset(n_faded=80, n_censored=120, seed=0)
    view = ds.view_for_task("survival")
    params = {"n_estimators": 50, "max_depth": 3, "learning_rate": 0.1,
              "subsample": 0.8, "colsample_bytree": 0.8,
              "aft_loss_distribution": "normal",
              "aft_loss_distribution_scale": 1.0}
    model = XGBAFTModel(params).fit(view.X, view.time, view.event)
    pred = model.predict(view.X)
    assert pred.shape == (len(view),)
    assert np.isfinite(pred).all()
    # predict_cycle_life returns positive values (exp(log_time))
    cycle = model.predict_cycle_life(view.X)
    assert (cycle > 0).all()


def test_task_and_orientation_attrs():
    assert XGBAFTModel.task == "survival"
    assert XGBAFTModel.risk_orientation == "time_high"


def test_fit_requires_time_and_event():
    ds = make_synthetic_dataset(seed=0)
    view = ds.view_for_task("survival")
    params = {"n_estimators": 10, "max_depth": 2, "learning_rate": 0.1,
              "subsample": 0.8, "colsample_bytree": 0.8,
              "aft_loss_distribution": "normal", "aft_loss_distribution_scale": 1.0}
    model = XGBAFTModel(dict(params))
    with pytest.raises(TypeError):
        model.fit(view.X)  # missing time + event


def test_predict_proba_raises():
    ds = make_synthetic_dataset(seed=0)
    view = ds.view_for_task("survival")
    params = {"n_estimators": 10, "max_depth": 2, "learning_rate": 0.1,
              "subsample": 0.8, "colsample_bytree": 0.8,
              "aft_loss_distribution": "normal", "aft_loss_distribution_scale": 1.0}
    model = XGBAFTModel(params).fit(view.X, view.time, view.event)
    with pytest.raises(NotImplementedError):
        model.predict_proba(view.X)


def test_suggest_params_returns_dict():
    study = optuna.create_study()
    trial = study.ask()
    p = XGBAFTModel.suggest_params(trial)
    for key in ("n_estimators", "max_depth", "learning_rate",
                "aft_loss_distribution", "aft_loss_distribution_scale"):
        assert key in p


def test_c_index_above_random_on_synthetic():
    """With synthetic data that has signal, the C-index should beat 0.55."""
    from cell_lifetime.evaluation.survival_metrics import survival_metrics
    ds = make_synthetic_dataset(n_faded=100, n_censored=100, seed=1)
    view = ds.view_for_task("survival")
    params = {"n_estimators": 80, "max_depth": 3, "learning_rate": 0.1,
              "subsample": 0.8, "colsample_bytree": 0.8,
              "aft_loss_distribution": "normal", "aft_loss_distribution_scale": 1.0}
    model = XGBAFTModel(params).fit(view.X, view.time, view.event)
    # time_high orientation: negate predict() for risk-positive scores
    risk = -model.predict(view.X)
    out = survival_metrics(view.event, view.time, risk)
    # In-sample on signal-bearing synthetic data — should be clearly above random
    assert out["c_index"] > 0.55

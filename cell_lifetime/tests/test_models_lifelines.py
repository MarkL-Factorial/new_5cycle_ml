"""Smoke tests for the lifelines-based survival models."""

import numpy as np
import optuna
import pytest

pytest.importorskip("lifelines")

from cell_lifetime.data.synthetic import make_synthetic_dataset
from cell_lifetime.models.lifelines_weibull_aft import LifelinesWeibullAFTModel
from cell_lifetime.models.lifelines_cox import LifelinesCoxModel


@pytest.mark.parametrize("cls", [LifelinesWeibullAFTModel, LifelinesCoxModel])
def test_task_attr(cls):
    assert cls.task == "survival"


def test_weibull_fit_predict_signature():
    ds = make_synthetic_dataset(n_faded=60, n_censored=80, seed=0)
    view = ds.view_for_task("survival")
    params = {"penalizer": 0.1, "l1_ratio": 0.0, "fit_intercept": True}
    model = LifelinesWeibullAFTModel(params).fit(view.X, view.time, view.event)
    pred = model.predict(view.X)
    assert pred.shape == (len(view),)
    assert np.isfinite(pred).all()


def test_cox_fit_predict_signature():
    ds = make_synthetic_dataset(n_faded=60, n_censored=80, seed=0)
    view = ds.view_for_task("survival")
    params = {"penalizer": 0.1, "l1_ratio": 0.0}
    model = LifelinesCoxModel(params).fit(view.X, view.time, view.event)
    pred = model.predict(view.X)
    assert pred.shape == (len(view),)
    assert (pred > 0).all()  # partial hazard is always positive


@pytest.mark.parametrize("cls", [LifelinesWeibullAFTModel, LifelinesCoxModel])
def test_fit_requires_time_and_event(cls):
    ds = make_synthetic_dataset(seed=0)
    view = ds.view_for_task("survival")
    model = cls({"penalizer": 0.1})
    with pytest.raises(TypeError):
        model.fit(view.X)


@pytest.mark.parametrize("cls", [LifelinesWeibullAFTModel, LifelinesCoxModel])
def test_predict_proba_raises(cls):
    ds = make_synthetic_dataset(seed=0)
    view = ds.view_for_task("survival")
    model = cls({"penalizer": 0.1}).fit(view.X, view.time, view.event)
    with pytest.raises(NotImplementedError):
        model.predict_proba(view.X)


@pytest.mark.parametrize("cls", [LifelinesWeibullAFTModel, LifelinesCoxModel])
def test_suggest_params_returns_dict(cls):
    study = optuna.create_study()
    trial = study.ask()
    p = cls.suggest_params(trial)
    assert "penalizer" in p


def test_weibull_c_index_above_random():
    """Synthetic data has signal; Weibull AFT C-index should beat 0.55."""
    from cell_lifetime.evaluation.survival_metrics import survival_metrics
    ds = make_synthetic_dataset(n_faded=100, n_censored=100, seed=1)
    view = ds.view_for_task("survival")
    model = LifelinesWeibullAFTModel({"penalizer": 0.01, "l1_ratio": 0.0,
                                       "fit_intercept": True}).fit(view.X, view.time, view.event)
    # time_high orientation: negate for risk-positive
    risk = -model.predict(view.X)
    out = survival_metrics(view.event, view.time, risk)
    assert out["c_index"] > 0.55


def test_cox_c_index_above_random():
    from cell_lifetime.evaluation.survival_metrics import survival_metrics
    ds = make_synthetic_dataset(n_faded=100, n_censored=100, seed=1)
    view = ds.view_for_task("survival")
    model = LifelinesCoxModel({"penalizer": 0.01, "l1_ratio": 0.0}).fit(
        view.X, view.time, view.event
    )
    risk = model.predict(view.X)  # risk_high
    out = survival_metrics(view.event, view.time, risk)
    assert out["c_index"] > 0.55


@pytest.mark.parametrize("cls", [LifelinesWeibullAFTModel, LifelinesCoxModel])
def test_orientation_attr(cls):
    expected = "time_high" if cls is LifelinesWeibullAFTModel else "risk_high"
    assert cls.risk_orientation == expected

"""XGBoost regressor smoke test on synthetic data."""

import numpy as np

from cell_lifetime.data.synthetic import make_synthetic_dataset
from cell_lifetime.models.xgb_regressor import XGBRegressorModel


def test_fit_predict_log_transform():
    ds = make_synthetic_dataset(seed=0)
    view = ds.view_for_task("regression")
    params = {"n_estimators": 50, "max_depth": 3, "learning_rate": 0.1}
    model = XGBRegressorModel(params, target_transform="log").fit(view.X, view.y_cycle)
    pred = model.predict(view.X)
    assert pred.shape == (len(view),)
    # Predictions on untransformed scale, must be positive (log → exp)
    assert (pred > 0).all()
    # Some training fit (not asserting on test set — that's smoke for validation)
    assert pred.mean() == pred.mean()  # not NaN


def test_fit_predict_boxcox_transform():
    ds = make_synthetic_dataset(seed=0)
    view = ds.view_for_task("regression")
    params = {"n_estimators": 50, "max_depth": 3, "learning_rate": 0.1}
    model = XGBRegressorModel(params, target_transform="boxcox").fit(view.X, view.y_cycle)
    pred = model.predict(view.X)
    assert pred.shape == (len(view),)
    assert (pred > 0).all()


def test_predict_proba_raises():
    ds = make_synthetic_dataset(seed=0)
    view = ds.view_for_task("regression")
    model = XGBRegressorModel({"n_estimators": 10}).fit(view.X, view.y_cycle)
    try:
        model.predict_proba(view.X)
    except NotImplementedError:
        return
    raise AssertionError("predict_proba should raise NotImplementedError")

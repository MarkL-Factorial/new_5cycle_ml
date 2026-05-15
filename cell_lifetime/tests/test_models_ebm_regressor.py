"""EBM regressor smoke test on synthetic data."""

import numpy as np

from cell_lifetime.data.synthetic import make_synthetic_dataset
from cell_lifetime.models.ebm_regressor import EBMRegressorModel


def test_fit_predict():
    ds = make_synthetic_dataset(n_faded=50, n_censored=50, seed=0)  # smaller for speed
    view = ds.view_for_task("regression")
    params = {"max_bins": 64, "interactions": 0, "learning_rate": 0.05, "min_samples_leaf": 4, "max_leaves": 3, "max_interaction_bins": 8}
    model = EBMRegressorModel(params, target_transform="log").fit(view.X, view.y_cycle)
    pred = model.predict(view.X)
    assert pred.shape == (len(view),)
    assert (pred > 0).all()


def test_feature_importance():
    ds = make_synthetic_dataset(n_faded=50, n_censored=50, seed=0)
    view = ds.view_for_task("regression")
    params = {"max_bins": 64, "interactions": 0, "learning_rate": 0.05, "min_samples_leaf": 4, "max_leaves": 3, "max_interaction_bins": 8}
    model = EBMRegressorModel(params, target_transform="log").fit(view.X, view.y_cycle)
    fi = model.feature_importance(view.feature_names)
    assert set(fi.keys()) == set(view.feature_names)

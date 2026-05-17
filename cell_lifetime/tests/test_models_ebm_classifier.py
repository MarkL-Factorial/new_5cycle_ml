"""EBM classifier smoke tests on synthetic data."""

import numpy as np
import optuna

from cell_lifetime.data.synthetic import make_synthetic_dataset
from cell_lifetime.models.ebm_classifier import EBMClassifierModel


def test_fit_predict_signature():
    ds = make_synthetic_dataset(n_faded=50, n_censored=50, seed=0)
    view = ds.view_for_task("classification")
    params = {"max_bins": 64, "max_interaction_bins": 8, "interactions": 0,
              "learning_rate": 0.1, "min_samples_leaf": 4, "max_leaves": 3}
    model = EBMClassifierModel(params).fit(view.X, view.y_class)
    pred = model.predict(view.X)
    proba = model.predict_proba(view.X)
    assert pred.shape == (len(view),)
    assert set(np.unique(pred)).issubset({0, 1})
    assert proba.shape == (len(view), 2)
    np.testing.assert_allclose(proba.sum(axis=1), 1.0, atol=1e-6)


def test_task_attr():
    assert EBMClassifierModel.task == "classification"


def test_suggest_params_returns_dict():
    study = optuna.create_study()
    trial = study.ask()
    p = EBMClassifierModel.suggest_params(trial)
    for key in ("max_bins", "max_interaction_bins", "interactions",
                "learning_rate", "min_samples_leaf", "max_leaves"):
        assert key in p
    # Interactions capped at 3
    assert p["interactions"] <= 3


def test_feature_importance_keys():
    ds = make_synthetic_dataset(n_faded=40, n_censored=40, seed=0)
    view = ds.view_for_task("classification")
    params = {"max_bins": 64, "max_interaction_bins": 8, "interactions": 0,
              "learning_rate": 0.1, "min_samples_leaf": 4, "max_leaves": 3}
    model = EBMClassifierModel(params).fit(view.X, view.y_class)
    fi = model.feature_importance(view.feature_names)
    assert set(fi.keys()) == set(view.feature_names)

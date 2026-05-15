"""RandomForestModel fit/predict/predict_proba/feature_importance/shap."""

import numpy as np
import pandas as pd

from cell_classifier.models.random_forest import RandomForestModel


def _toy_data(n=120, seed=0):
    rng = np.random.default_rng(seed)
    X = pd.DataFrame({
        "a": rng.normal(size=n),
        "b": rng.normal(size=n),
        "c": rng.normal(size=n),
    })
    y = ((X["a"] + 0.5 * X["b"] + rng.normal(scale=0.5, size=n)) > 0).astype(np.int8).to_numpy()
    return X, y


def test_fit_predict_predict_proba():
    X, y = _toy_data()
    model = RandomForestModel(
        {"n_estimators": 50, "max_depth": 5, "random_state": 0,
         "min_samples_split": 2, "min_samples_leaf": 1,
         "max_features": "sqrt", "ccp_alpha": 0.0, "class_weight": None}
    ).fit(X, y)
    preds = model.predict(X)
    proba = model.predict_proba(X)
    assert preds.shape == (len(y),)
    assert proba.shape == (len(y), 2)
    np.testing.assert_allclose(proba.sum(axis=1), 1.0, atol=1e-6)


def test_feature_importance():
    X, y = _toy_data()
    model = RandomForestModel(
        {"n_estimators": 50, "max_depth": 5, "random_state": 0,
         "min_samples_split": 2, "min_samples_leaf": 1,
         "max_features": "sqrt", "ccp_alpha": 0.0, "class_weight": None}
    ).fit(X, y)
    fi = model.feature_importance(list(X.columns))
    assert set(fi.keys()) == set(X.columns)
    assert all(v >= 0 for v in fi.values())


def test_compute_shap_shape():
    X, y = _toy_data(n=40)
    model = RandomForestModel(
        {"n_estimators": 25, "max_depth": 4, "random_state": 0,
         "min_samples_split": 2, "min_samples_leaf": 1,
         "max_features": "sqrt", "ccp_alpha": 0.0, "class_weight": None}
    ).fit(X, y)
    sv = model.compute_shap(X)
    assert sv is not None
    assert sv.shape == (len(X), X.shape[1])


def test_handles_nan_via_imputer():
    X, y = _toy_data()
    X.iloc[0, 0] = np.nan
    model = RandomForestModel(
        {"n_estimators": 25, "random_state": 0, "max_depth": 4,
         "min_samples_split": 2, "min_samples_leaf": 1,
         "max_features": "sqrt", "ccp_alpha": 0.0, "class_weight": None}
    ).fit(X, y)
    # No exception → pipeline-internal imputer worked
    model.predict(X)

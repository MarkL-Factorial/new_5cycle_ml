"""tune() returns valid params for the RF search space."""

import numpy as np
import pandas as pd
import pytest

from cell_classifier.models.random_forest import RandomForestModel
from cell_classifier.training.tuning import tune


@pytest.fixture
def toy():
    rng = np.random.default_rng(0)
    X = pd.DataFrame({
        "a": rng.normal(size=80),
        "b": rng.normal(size=80),
        "c": rng.normal(size=80),
    })
    y = (X["a"] > 0).astype(np.int8).to_numpy()
    return X, y


def test_tune_smoke(toy):
    X, y = toy
    params, study = tune(
        RandomForestModel, X, y,
        n_trials=3, inner_cv=3, seed=0, optimize="f1",
    )
    assert isinstance(params, dict)
    assert "n_estimators" in params
    assert study.best_value > 0

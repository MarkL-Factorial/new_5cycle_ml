"""Nested CV partitions test set exactly once and aggregates fold results."""

import numpy as np
import pandas as pd

from cell_classifier.models.random_forest import RandomForestModel
from cell_classifier.training.nested import nested_cv


def test_nested_cv_partition():
    rng = np.random.default_rng(0)
    X = pd.DataFrame({"a": rng.normal(size=60), "b": rng.normal(size=60)})
    y = (X["a"] > 0).astype(np.int8).to_numpy()
    result = nested_cv(
        RandomForestModel, X, y,
        outer_k=3, inner_cv=3, n_trials=2, optimize="f1", seed=0,
    )
    assert result.y_pred.shape == (60,)
    assert result.y_proba.shape == (60,)
    # Each fold contributes one chunk; concatenated covers all rows exactly once
    counts = np.bincount(result.fold_id)
    assert counts.sum() == 60
    assert len(result.per_fold_best_params) == 3

"""Nested CV outer folds form a partition — every labeled cell appears in
exactly one outer-test slot per seed.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from cell_classifier.models.random_forest import RandomForestModel
from cell_classifier.training.nested import nested_cv


@pytest.fixture
def toy():
    rng = np.random.default_rng(0)
    n = 90
    X = pd.DataFrame({"a": rng.normal(size=n), "b": rng.normal(size=n)})
    y = (X["a"] > 0).astype(np.int8).to_numpy()
    return X, y


def test_full_coverage_one_seed(toy):
    X, y = toy
    outer_k = 5
    result = nested_cv(
        RandomForestModel, X, y,
        outer_k=outer_k, inner_cv=3, n_trials=2,
        optimize="f1", seed=0,
    )
    assert result.y_pred.shape == (len(y),)
    assert result.y_proba.shape == (len(y),)
    # Every row got assigned exactly one fold id in [0, K)
    assert set(np.unique(result.fold_id).tolist()) == set(range(outer_k))
    counts = np.bincount(result.fold_id)
    assert counts.sum() == len(y), "fold_id must cover every cell exactly once"


def test_fold_partition_is_disjoint(toy):
    X, y = toy
    outer_k = 5
    result = nested_cv(
        RandomForestModel, X, y,
        outer_k=outer_k, inner_cv=3, n_trials=2,
        optimize="f1", seed=42,
    )
    seen = set()
    for k in range(outer_k):
        idx = np.flatnonzero(result.fold_id == k)
        # No index appears in two folds
        assert seen.isdisjoint(idx.tolist())
        seen.update(idx.tolist())
    assert seen == set(range(len(y)))


def test_per_fold_studies_captured(toy):
    """The studies are kept so optuna_history.csv can be populated."""
    X, y = toy
    outer_k = 3
    result = nested_cv(
        RandomForestModel, X, y,
        outer_k=outer_k, inner_cv=3, n_trials=2,
        optimize="f1", seed=0,
    )
    assert len(result.per_fold_studies) == outer_k
    for study in result.per_fold_studies:
        assert len(study.trials) == 2

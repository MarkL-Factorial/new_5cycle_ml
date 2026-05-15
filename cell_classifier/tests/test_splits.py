"""Validation vs production splitters."""

import numpy as np

from cell_classifier.data.splits import (
    split_production,
    split_validation_nested_cv,
    split_validation_tune_inner_cv,
)


def _y(n_pass: int, n_bad: int) -> np.ndarray:
    return np.array([1] * n_pass + [0] * n_bad, dtype=np.int8)


def test_tune_inner_cv_shape():
    y = _y(80, 20)
    tr, te = split_validation_tune_inner_cv(y, test_frac=0.2, seed=0)
    assert len(tr) + len(te) == len(y)
    assert abs(len(te) - 20) <= 1
    # Stratification: both classes present in train and test
    assert set(np.unique(y[tr])) == {0, 1}
    assert set(np.unique(y[te])) == {0, 1}


def test_tune_inner_cv_reproducible():
    y = _y(80, 20)
    tr1, te1 = split_validation_tune_inner_cv(y, test_frac=0.2, seed=42)
    tr2, te2 = split_validation_tune_inner_cv(y, test_frac=0.2, seed=42)
    np.testing.assert_array_equal(tr1, tr2)
    np.testing.assert_array_equal(te1, te2)


def test_nested_cv_partition():
    y = _y(80, 20)
    folds = list(split_validation_nested_cv(y, outer_k=5, seed=0))
    assert len(folds) == 5
    all_test = np.concatenate([te for _, te in folds])
    assert len(all_test) == len(y)
    assert len(set(all_test.tolist())) == len(y)  # each row appears once in test


def test_production_split():
    mask = np.array([True, True, False, True, False], dtype=bool)
    train, inf = split_production(mask)
    assert list(train) == [0, 1, 3]
    assert list(inf) == [0, 1, 2, 3, 4]

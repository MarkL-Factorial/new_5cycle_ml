"""Tests for ml_classification.splits."""

from __future__ import annotations

import numpy as np
import pytest

from ml_classification.splits import stratified_split


def _y(n_pos: int, n_neg: int) -> np.ndarray:
    return np.array([1] * n_pos + [0] * n_neg)


def test_disjoint_and_covers_all():
    y = _y(150, 100)
    i_tr, i_te = stratified_split(y, seed=42)
    assert len(set(i_tr) & set(i_te)) == 0
    assert sorted(np.concatenate([i_tr, i_te])) == list(range(len(y)))


def test_approximate_fractions():
    y = _y(200, 200)
    i_tr, i_te = stratified_split(y, test_frac=0.2, seed=42)
    n = len(y)
    assert abs(len(i_tr) / n - 0.8) < 0.02
    assert abs(len(i_te) / n - 0.2) < 0.02


def test_stratified_by_target():
    y = _y(150, 100)
    i_tr, i_te = stratified_split(y, seed=42)
    full_pos_frac = y.mean()
    for idx in (i_tr, i_te):
        assert abs(y[idx].mean() - full_pos_frac) < 0.05


def test_seed_determinism():
    y = _y(150, 100)
    a = stratified_split(y, seed=42)
    b = stratified_split(y, seed=42)
    for x, z in zip(a, b):
        np.testing.assert_array_equal(x, z)


def test_seed_variance():
    y = _y(150, 100)
    a = stratified_split(y, seed=42)
    b = stratified_split(y, seed=1729)
    assert not np.array_equal(a[1], b[1])


def test_bad_fractions_raise():
    with pytest.raises(ValueError):
        stratified_split(_y(50, 50), test_frac=0.0)
    with pytest.raises(ValueError):
        stratified_split(_y(50, 50), test_frac=1.0)

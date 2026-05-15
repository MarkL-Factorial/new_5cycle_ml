"""Tests for ml_classification.data."""

from __future__ import annotations

import numpy as np
import pytest

from ml_classification.data import _LABEL_COLUMNS_DENYLIST, load_dataset


@pytest.mark.parametrize("N", [200, 300, 400])
def test_load_fs_cv_shape(N):
    ds = load_dataset(N=N, feature_subset="fs_cv")
    assert len(ds.feature_names) == 12, ds.feature_names
    assert ds.X.shape == (len(ds.y), 12)
    assert ds.X.shape[0] > 0
    assert ds.y.dtype in (np.int8, np.int32, np.int64)
    assert set(np.unique(ds.y)).issubset({0, 1})


@pytest.mark.parametrize("N", [200, 300, 400])
def test_no_label_leakage(N):
    ds = load_dataset(N=N, feature_subset="fs_cv")
    leakage = set(ds.X.columns) & _LABEL_COLUMNS_DENYLIST
    assert leakage == set(), f"label columns in X: {leakage}"


def test_trainable_counts_match_manifest():
    expected = {200: 291, 300: 248, 400: 233}
    # Some cells in cell_labels.parquet may have trainable_n{N}=True but lack
    # features (n_regular < 5). Inner-join may reduce the count slightly.
    for N, ref in expected.items():
        ds = load_dataset(N=N, feature_subset="fs_cv")
        assert len(ds) <= ref
        assert len(ds) >= ref - 5  # at most ~5 cells lost to feature-missingness


def test_cohorts_alignment():
    ds = load_dataset(N=300, feature_subset="fs_cv")
    assert len(ds.cohorts) == len(ds)
    assert set(np.unique(ds.cohorts)).issubset({"AR", "0MC"})


def test_unknown_subset_raises():
    with pytest.raises(KeyError, match="not in column_roles.yaml"):
        load_dataset(N=300, feature_subset="not_a_real_subset")


def test_bad_N_raises():
    with pytest.raises(ValueError):
        load_dataset(N=100, feature_subset="fs_cv")

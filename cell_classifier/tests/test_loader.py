"""Loader against the ml_label_preprocess bundles."""

from pathlib import Path

import numpy as np
import pytest

from cell_classifier.data.loader import (
    SUPPORTED_BASELINE, SUPPORTED_N, _LABEL_COLUMNS_DENYLIST, load_dataset,
)


@pytest.mark.parametrize("N", SUPPORTED_N)
def test_load_fs_cv_shape(N):
    ds = load_dataset(N=N, feature_subset="fs_cv")
    assert len(ds.feature_names) == 12
    assert ds.X.shape[0] == len(ds.y)
    assert ds.y.dtype == np.int8
    assert ds.label_mask.dtype == bool
    assert ds.X.shape[0] > 0


def test_no_label_leakage():
    ds = load_dataset(N=300, feature_subset="fs_cv")
    leakage = set(ds.X.columns) & _LABEL_COLUMNS_DENYLIST
    assert leakage == set()


def test_labeled_view_consistency():
    ds = load_dataset(N=300, feature_subset="fs_cv")
    labeled = ds.labeled_view()
    assert int(ds.label_mask.sum()) == len(labeled)
    assert set(np.unique(labeled.y)).issubset({0, 1})
    # labeled_view's label_mask is all-True
    assert labeled.label_mask.all()


def test_default_db_baseline():
    ds = load_dataset(N=300, feature_subset="fs_cv")
    assert ds.baseline_cycle == 1
    assert ds.db_version == "A2.2"
    assert ds.source_dir.name == "A2.2_b1"


def test_baseline_3_loads():
    target = ds_path("A2.2", 3)
    if not target.exists():
        pytest.skip(f"baseline=3 bundle missing at {target}")
    ds = load_dataset(N=300, feature_subset="fs_cv", baseline_cycle=3)
    assert ds.baseline_cycle == 3
    assert ds.source_dir.name == "A2.2_b3"


def test_bad_N_raises():
    with pytest.raises(ValueError):
        load_dataset(N=100, feature_subset="fs_cv")


def test_unknown_subset_raises():
    with pytest.raises(KeyError):
        load_dataset(N=300, feature_subset="not_a_subset")


def test_missing_bundle_raises(tmp_path):
    with pytest.raises(FileNotFoundError, match="--baseline-cycle 1 --db-version ZZZ"):
        load_dataset(N=300, feature_subset="fs_cv", db_version="ZZZ")


def ds_path(db: str, baseline: int) -> Path:
    """Helper for skip checks."""
    from cell_classifier.data.loader import _default_preprocess_root
    return _default_preprocess_root() / "datasets" / f"{db}_b{baseline}" / "cell_features.parquet"

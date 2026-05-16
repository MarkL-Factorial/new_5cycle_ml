"""Verify the loader's `fs_all` special subset returns all feature-role columns."""

from pathlib import Path

import pytest

from cell_lifetime.data.loader import _load_feature_subset


_BUNDLE = (
    Path(__file__).resolve().parents[2]
    / "ml_label_preprocess/datasets/A2.2_b1/cell_labels.parquet"
)


@pytest.mark.skipif(not _BUNDLE.exists(), reason=f"bundle missing at {_BUNDLE}")
def test_fs_all_returns_more_than_fs_cv():
    fs_cv = _load_feature_subset("fs_cv")
    fs_all = _load_feature_subset("fs_all")
    assert len(fs_cv) == 12   # current manifest definition
    assert len(fs_all) >= 30  # should be ~40 on A2.2_b1
    # fs_cv is a strict subset of fs_all
    assert set(fs_cv).issubset(set(fs_all))


@pytest.mark.skipif(not _BUNDLE.exists(), reason=f"bundle missing at {_BUNDLE}")
def test_unknown_subset_raises():
    with pytest.raises(KeyError):
        _load_feature_subset("not_a_real_subset")


@pytest.mark.skipif(not _BUNDLE.exists(), reason=f"bundle missing at {_BUNDLE}")
def test_fs_cv_still_works():
    fs_cv = _load_feature_subset("fs_cv")
    assert "coulombic_efficiency_final" in fs_cv or "discharge_capacity_retention_final" in fs_cv

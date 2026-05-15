"""Real-data loader test against ml_label_preprocess/datasets/A2.2_b1/.

Skipped if the bundle is unavailable (e.g. inside a cloud sandbox).
"""

from pathlib import Path

import numpy as np
import pytest

from cell_lifetime.data.loader import load_dataset


_BUNDLE = (
    Path(__file__).resolve().parents[2]
    / "ml_label_preprocess/datasets/A2.2_b1/cell_labels.parquet"
)


@pytest.mark.skipif(not _BUNDLE.exists(), reason=f"bundle missing at {_BUNDLE}")
def test_real_bundle_shape_and_targets():
    ds = load_dataset(N=300, feature_subset="fs_cv", baseline_cycle=1, db_version="A2.2")
    # With n_regular>=6 filter on A2.2_b1 (May-15 rerun): ~415 cells (was
    # 439 before the filter); ~24 boundary cells with n_regular=5 dropped.
    assert 410 <= len(ds) <= 425
    # Faded count after n_regular>=6: 187 (one cell with last_fade_cycle=5 dropped)
    assert ds.event.sum() == 187
    # 12 fs_cv features
    assert len(ds.feature_names) == 12
    # event True → y_cycle is the cycle life (not NaN)
    assert (~np.isnan(ds.y_cycle[ds.event])).all()
    # event False → y_cycle is NaN
    assert np.isnan(ds.y_cycle[~ds.event]).all()
    # No mismatch between event mask and faded mask
    assert (ds.event == ds.faded_mask).all()
    # time >= 6 (the cutoff)
    assert (ds.time >= 6).all()


@pytest.mark.skipif(not _BUNDLE.exists(), reason=f"bundle missing at {_BUNDLE}")
def test_real_bundle_view_for_each_task():
    ds = load_dataset(N=300, feature_subset="fs_cv", baseline_cycle=1, db_version="A2.2")
    v_class = ds.view_for_task("classification")
    v_reg = ds.view_for_task("regression")
    v_surv = ds.view_for_task("survival")
    # Regression: only faded cells = 187
    assert len(v_reg) == 187
    # Survival: all rows with features and n_regular>=6
    assert len(v_surv) == len(ds)
    # Classification: trainable at N=300 ~250 after the filter
    assert 230 <= len(v_class) <= 270


@pytest.mark.skipif(not _BUNDLE.exists(), reason=f"bundle missing at {_BUNDLE}")
def test_real_bundle_min_n_regular_override():
    # Sanity: lowering min_n_regular to 5 gives the original 439
    ds_5 = load_dataset(
        N=300, feature_subset="fs_cv", baseline_cycle=1, db_version="A2.2",
        min_n_regular=5,
    )
    ds_6 = load_dataset(
        N=300, feature_subset="fs_cv", baseline_cycle=1, db_version="A2.2",
        min_n_regular=6,
    )
    assert len(ds_5) > len(ds_6)
    assert len(ds_5) - len(ds_6) <= 30  # bounded number of boundary cells

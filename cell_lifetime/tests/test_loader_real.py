"""Real-data loader test against ml_label_preprocess/datasets/A2.2_b1/.

Skipped if the bundle is unavailable (e.g. inside a cloud sandbox).
Resolves through the loader's bundle-resolution helper so it follows
the ``A2.2_b1_latest`` symlink (v3 layout) or the legacy flat layout.
"""

import numpy as np
import pytest

from cell_classifier.data.loader import _resolve_bundle_dir, _resolve_preprocess_root
from cell_lifetime.data.loader import load_dataset


_BUNDLE = (
    _resolve_bundle_dir(_resolve_preprocess_root(), "A2.2", 1) / "cell_labels.parquet"
)


@pytest.mark.skipif(not _BUNDLE.exists(), reason=f"bundle missing at {_BUNDLE}")
def test_real_bundle_shape_and_targets():
    ds = load_dataset(N=300, feature_subset="fs_cv", baseline_cycle=1, db_version="A2.2")
    # With n_regular>=6 + drop_excluded=True on A2.2_b1 the bundle has
    # ~420 cells and ~190 faded events. Exact counts drift across upstream
    # regenerations as more annotations land; ranges are intentionally
    # loose to absorb that drift.
    assert 400 <= len(ds) <= 440
    assert 180 <= int(ds.event.sum()) <= 210
    # 12 fs_cv features
    assert len(ds.feature_names) == 12
    # event True → y_cycle is the cycle life (not NaN)
    assert (~np.isnan(ds.y_cycle[ds.event])).all()
    # event False → y_cycle is NaN
    assert np.isnan(ds.y_cycle[~ds.event]).all()
    # No mismatch between event mask and faded mask
    assert (ds.event == ds.faded_mask).all()
    # time >= 5: censored cells inherit n_regular (>=6 by filter), but a
    # faded cell may have last_fade_cycle=5 even when n_regular>=6 (e.g.
    # AR4431 in A2.2_b1: faded at cycle 5, ran 6 cycles total).
    assert (ds.time >= 5).all()


@pytest.mark.skipif(not _BUNDLE.exists(), reason=f"bundle missing at {_BUNDLE}")
def test_real_bundle_view_for_each_task():
    ds = load_dataset(N=300, feature_subset="fs_cv", baseline_cycle=1, db_version="A2.2")
    v_class = ds.view_for_task("classification")
    v_reg = ds.view_for_task("regression")
    v_surv = ds.view_for_task("survival")
    # Regression view = faded cells; matches event.sum() exactly.
    assert len(v_reg) == int(ds.event.sum())
    # Survival: all rows with features and n_regular>=6 (drop_excluded=True
    # has already removed status='excluded' rows at load time)
    assert len(v_surv) == len(ds)
    # Classification: trainable at N=300 ~250 after the filter
    assert 220 <= len(v_class) <= 280


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


@pytest.mark.skipif(not _BUNDLE.exists(), reason=f"bundle missing at {_BUNDLE}")
def test_real_bundle_rate_changed_predict_only_invariant():
    """Production-mode load (drop_excluded=False, min_n_regular=5) must
    admit rate_changed cells for INFERENCE but never for TRAINING.

    This pins the load-bearing invariant of the rate_changed predict-only
    pipeline: if `task_target('survival')` ever reverts to an all-True
    mask, this test fails — preventing silent regression of the bug fix
    that put status='excluded' into the survival mask in the first place.
    """
    ds = load_dataset(
        N=300, feature_subset="fs_cv", baseline_cycle=1, db_version="A2.2",
        min_n_regular=5, drop_excluded=False,
    )
    excluded_mask = (ds.status == "excluded")
    n_excluded = int(excluded_mask.sum())

    # At least one rate_changed cell must come through (the bundle has 15
    # admitted rate_changed cells in A2.2_b1 schema_v2).
    assert n_excluded > 0, (
        "drop_excluded=False on A2.2_b1 should admit rate_changed cells; "
        "if this assertion fails, upstream may have stopped emitting them."
    )
    # Every excluded cell admitted here must have exclusion_reason set.
    assert all(
        ds.exclusion_reason[i] is not None for i in np.where(excluded_mask)[0]
    )

    # 1) view_for_task('survival') must drop them.
    v_surv = ds.view_for_task("survival")
    assert len(v_surv) == len(ds) - n_excluded
    assert (v_surv.status != "excluded").all()

    # 2) task_target('survival') mask must match.
    _, surv_mask = ds.task_target("survival")
    assert surv_mask.sum() == len(ds) - n_excluded
    assert (surv_mask == (ds.status != "excluded")).all()

    # 3) Classification and regression masks must already exclude them
    #    upstream (preprocessor sets trainable_n{N}=False; status='excluded'
    #    is by definition not 'faded'). Belt-and-suspenders check.
    assert ds.label_mask[excluded_mask].sum() == 0
    assert ds.faded_mask[excluded_mask].sum() == 0

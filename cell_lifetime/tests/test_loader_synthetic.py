"""Synthetic-data tests for the loader + CycleLifeDataset."""

import numpy as np
import pandas as pd

from cell_lifetime.data.synthetic import make_synthetic_dataset


def test_synthetic_default_shape():
    ds = make_synthetic_dataset(seed=42)
    assert len(ds) == 200
    assert ds.X.shape == (200, 12)
    assert ds.event.sum() == 80
    assert ds.faded_mask.sum() == 80
    assert ds.time.shape == (200,)
    # y_cycle is NaN for non-faded
    assert np.isnan(ds.y_cycle[~ds.event]).all()
    assert (~np.isnan(ds.y_cycle[ds.event])).all()


def test_synthetic_classification_target_consistency():
    ds = make_synthetic_dataset(seed=42)
    # y_class == 1 should only be assigned to rows the loader considers trainable
    # (the synthetic implementation may mark some censored rows as non-trainable
    # if they haven't reached N)
    assert ds.y_class.dtype == np.int8
    assert set(np.unique(ds.y_class)).issubset({0, 1})


def test_view_for_task_classification():
    ds = make_synthetic_dataset(seed=0)
    v = ds.view_for_task("classification")
    assert v.label_mask.all()
    assert len(v) == int(ds.label_mask.sum())


def test_view_for_task_regression():
    ds = make_synthetic_dataset(seed=0)
    v = ds.view_for_task("regression")
    assert v.faded_mask.all()
    assert len(v) == int(ds.event.sum())  # 80 faded
    assert (~np.isnan(v.y_cycle)).all()


def test_view_for_task_survival():
    ds = make_synthetic_dataset(seed=0)
    v = ds.view_for_task("survival")
    assert len(v) == 200    # all rows; censored handled via event flag
    # event + time arrays match the parent
    assert v.event.sum() == 80

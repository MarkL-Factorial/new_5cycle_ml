"""Imputer factory + Pipeline integration."""

import numpy as np
import pandas as pd
import pytest

from cell_classifier.preprocessing.imputer import make_imputer


def test_median_strategy():
    imputer = make_imputer("median")
    X = pd.DataFrame({"a": [1.0, np.nan, 3.0]})
    out = imputer.fit_transform(X)
    assert out[1, 0] == 2.0   # median of [1, 3]


def test_mean_strategy():
    imputer = make_imputer("mean")
    X = pd.DataFrame({"a": [1.0, np.nan, 3.0]})
    out = imputer.fit_transform(X)
    assert out[1, 0] == 2.0


def test_constant_strategy():
    imputer = make_imputer("constant")
    X = pd.DataFrame({"a": [1.0, np.nan, 3.0]})
    out = imputer.fit_transform(X)
    assert out[1, 0] == 0.0


def test_unknown_strategy_raises():
    with pytest.raises(ValueError, match="unknown imputer strategy"):
        make_imputer("knn")

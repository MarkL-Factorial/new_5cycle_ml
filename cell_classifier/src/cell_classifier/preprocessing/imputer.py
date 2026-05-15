"""Imputer factory.

Each model class composes itself with an sklearn Pipeline of (imputer, estimator).
The imputer strategy is config-driven so we can swap median → KNN → iterative
without touching any model class.
"""

from __future__ import annotations

from sklearn.impute import SimpleImputer
from sklearn.base import BaseEstimator


_STRATEGIES = ("median", "mean", "most_frequent", "constant")


def make_imputer(strategy: str = "median") -> BaseEstimator:
    """Return a fresh imputer instance for the given strategy.

    Supported: "median", "mean", "most_frequent", "constant" (fill 0).
    """
    if strategy not in _STRATEGIES:
        raise ValueError(
            f"unknown imputer strategy {strategy!r}; supported: {_STRATEGIES}"
        )
    if strategy == "constant":
        return SimpleImputer(strategy="constant", fill_value=0.0)
    return SimpleImputer(strategy=strategy)

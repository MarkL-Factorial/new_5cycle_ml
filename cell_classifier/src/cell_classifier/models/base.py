"""Abstract base for a classification model family.

A model wraps an sklearn `Pipeline([imputer, estimator])` so the pipeline
code can call `fit / predict / predict_proba` without knowing which algorithm
is underneath, and so preprocessing stats are guaranteed to be fit on training
data only (sklearn Pipeline semantics).

Optional hooks (`predict_proba_samples`, `feature_importance`, `compute_shap`)
let Bayesian / tree / glassbox models expose extras without inflating the
required interface.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

import numpy as np
import optuna
import pandas as pd


class BaseModel(ABC):
    name: str = "abstract"
    fixed_params: dict[str, Any] = {}

    def __init__(self, params: dict[str, Any], *, imputer_strategy: str = "median"):
        self.params = params
        self.imputer_strategy = imputer_strategy

    @classmethod
    @abstractmethod
    def suggest_params(cls, trial: optuna.Trial) -> dict[str, Any]:
        """Hyperparameter search space for Optuna."""

    @abstractmethod
    def fit(self, X: pd.DataFrame, y: np.ndarray) -> "BaseModel":
        """Fit on labeled data. Must return self."""

    @abstractmethod
    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Hard class labels, shape (n,)."""

    @abstractmethod
    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """Class probabilities, shape (n, 2). Columns ordered [0, 1]."""

    # --- optional hooks ---

    def predict_proba_samples(self, X: pd.DataFrame) -> np.ndarray | None:
        """Posterior samples, shape (n_draws, n, 2). Default None (non-Bayesian)."""
        return None

    def feature_importance(self, feature_names: list[str]) -> dict[str, float]:
        """Native importance, {feature_name: importance}."""
        raise NotImplementedError(f"{self.name} has no native importance")

    def compute_shap(self, X: pd.DataFrame) -> np.ndarray | None:
        """SHAP for class 1, shape (n, n_features). Default None."""
        return None

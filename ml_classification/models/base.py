"""Abstract base class for a model specification.

A `ModelSpec` is the contract every supported model implements. It bundles three
concerns:
  1. `build(params)` — construct an sklearn-compatible estimator
  2. `suggest_params(trial)` — Optuna search space (called per trial)
  3. `feature_importance(fitted, X, feature_names)` — native importances dict

Pipeline code only ever talks to this interface, so adding a new model (EBM, BART,
etc.) does not require changes outside `models/`.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

import optuna
import pandas as pd


class ModelSpec(ABC):
    name: str = "abstract"
    fixed_params: dict[str, Any] = {}

    @abstractmethod
    def build(self, params: dict[str, Any]):
        """Return a fitted-ready sklearn-compatible estimator.

        Must support .fit(X, y), .predict(X), .predict_proba(X).
        Caller is responsible for merging fixed_params and tuned params.
        """

    @abstractmethod
    def suggest_params(self, trial: optuna.Trial) -> dict[str, Any]:
        """Return a hyperparameter dict drawn from the Optuna trial."""

    @abstractmethod
    def feature_importance(
        self,
        fitted,
        X: pd.DataFrame,
        feature_names: list[str],
    ) -> dict[str, float]:
        """Return {feature_name: importance} for the fitted estimator."""

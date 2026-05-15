"""Random Forest model spec.

Search space loosely follows the reference report's `RandomizedSearchCV` grid in
`experiment_cv_features/scripts/04_train_with_cv.py`, expanded for Optuna's continuous
sampling.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import optuna
import pandas as pd
from sklearn.ensemble import RandomForestClassifier

from .base import ModelSpec


class RFModelSpec(ModelSpec):
    name = "random_forest"
    fixed_params = {"n_jobs": -1}

    def build(self, params: dict[str, Any]) -> RandomForestClassifier:
        return RandomForestClassifier(**params)

    def suggest_params(self, trial: optuna.Trial) -> dict[str, Any]:
        return {
            "n_estimators": trial.suggest_int("n_estimators", 25, 400, step=25),
            "max_depth": trial.suggest_int("max_depth", 3, 20),
            "min_samples_split": trial.suggest_int("min_samples_split", 2, 30),
            "min_samples_leaf": trial.suggest_int("min_samples_leaf", 1, 20),
            "max_features": trial.suggest_categorical(
                "max_features", ["sqrt", "log2", None]
            ),
            "ccp_alpha": trial.suggest_float("ccp_alpha", 1e-4, 1e-1, log=True),
            "class_weight": trial.suggest_categorical(
                "class_weight", ["balanced", None]
            ),
        }

    def feature_importance(
        self,
        fitted: RandomForestClassifier,
        X: pd.DataFrame,
        feature_names: list[str],
    ) -> dict[str, float]:
        importances = np.asarray(fitted.feature_importances_, dtype=float)
        return dict(zip(feature_names, importances.tolist()))

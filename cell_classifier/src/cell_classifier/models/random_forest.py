"""Random Forest classifier wrapped in an sklearn Pipeline with imputation.

Hyperparameter search space mirrors ml_classification_v2/models/random_forest.py
(loosely follows the report's RandomizedSearchCV grid, expanded for Optuna's
continuous sampling).
"""

from __future__ import annotations

from typing import Any

import numpy as np
import optuna
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline

from cell_classifier.models.base import BaseModel
from cell_classifier.preprocessing.imputer import make_imputer


class RandomForestModel(BaseModel):
    name = "random_forest"
    fixed_params: dict[str, Any] = {"n_jobs": -1}

    def __init__(self, params: dict[str, Any], *, imputer_strategy: str = "median"):
        super().__init__(params, imputer_strategy=imputer_strategy)
        merged = {**self.fixed_params, **params}
        self.pipeline = Pipeline(
            steps=[
                ("imputer", make_imputer(imputer_strategy)),
                ("estimator", RandomForestClassifier(**merged)),
            ]
        )

    @classmethod
    def suggest_params(cls, trial: optuna.Trial) -> dict[str, Any]:
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

    def fit(self, X: pd.DataFrame, y: np.ndarray) -> "RandomForestModel":
        self.pipeline.fit(X, y)
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        return self.pipeline.predict(X)

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        return self.pipeline.predict_proba(X)

    def feature_importance(self, feature_names: list[str]) -> dict[str, float]:
        estimator: RandomForestClassifier = self.pipeline.named_steps["estimator"]
        importances = np.asarray(estimator.feature_importances_, dtype=float)
        return dict(zip(feature_names, importances.tolist()))

    def compute_shap(self, X: pd.DataFrame) -> np.ndarray | None:
        import shap

        estimator: RandomForestClassifier = self.pipeline.named_steps["estimator"]
        imputer = self.pipeline.named_steps["imputer"]
        X_imp = pd.DataFrame(
            imputer.transform(X), columns=X.columns, index=X.index
        )
        explainer = shap.TreeExplainer(
            estimator, feature_perturbation="tree_path_dependent"
        )
        raw = explainer.shap_values(X_imp, check_additivity=False)
        if isinstance(raw, list):
            return np.asarray(raw[1])
        arr = np.asarray(raw)
        if arr.ndim == 3:
            return arr[:, :, 1]
        return arr

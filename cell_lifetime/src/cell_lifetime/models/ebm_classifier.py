"""EBM (Explainable Boosting Machine) binary classifier.

Mirrors ebm_regressor.py but wraps `ExplainableBoostingClassifier`
instead of the regressor variant. Used for the cycle-life pass/bad
classification task.

`interactions` upper bound capped at 3 (same rationale as ebm_regressor):
higher values exploded wall-clock per fit on the small (~200-cell)
trainable set with diminishing accuracy gain.
"""

from __future__ import annotations

from typing import Any, ClassVar

import numpy as np
import optuna
import pandas as pd
from interpret.glassbox import ExplainableBoostingClassifier
from sklearn.pipeline import Pipeline

from cell_classifier.preprocessing.imputer import make_imputer
from cell_lifetime.models.base import CycleLifeModel


class EBMClassifierModel(CycleLifeModel):
    name = "ebm_classifier"
    task: ClassVar[str] = "classification"
    fixed_params: dict[str, Any] = {
        # n_jobs=10 — workspace cap (max 10 cores across all automation).
        "n_jobs": 10,
    }

    def __init__(self, params: dict[str, Any], *, imputer_strategy: str = "median"):
        super().__init__(params, imputer_strategy=imputer_strategy)
        merged = {**self.fixed_params, **params}
        self.pipeline = Pipeline(
            steps=[
                ("imputer", make_imputer(imputer_strategy)),
                ("estimator", ExplainableBoostingClassifier(**merged)),
            ]
        )

    @classmethod
    def suggest_params(cls, trial: optuna.Trial) -> dict[str, Any]:
        # `interactions` capped at 3 — see ebm_regressor.py for rationale.
        return {
            "max_bins": trial.suggest_int("max_bins", 64, 512, step=64),
            "max_interaction_bins": trial.suggest_int("max_interaction_bins", 8, 64, step=8),
            "interactions": trial.suggest_int("interactions", 0, 3),
            "learning_rate": trial.suggest_float("learning_rate", 1e-3, 0.3, log=True),
            "min_samples_leaf": trial.suggest_int("min_samples_leaf", 2, 20),
            "max_leaves": trial.suggest_int("max_leaves", 2, 8),
        }

    def fit(self, X: pd.DataFrame, y: np.ndarray) -> "EBMClassifierModel":
        self.pipeline.fit(X, y)
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        return self.pipeline.predict(X)

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        return self.pipeline.predict_proba(X)

    def feature_importance(self, feature_names: list[str]) -> dict[str, float]:
        est: ExplainableBoostingClassifier = self.pipeline.named_steps["estimator"]
        try:
            imp = est.term_importances()
        except Exception:
            return {name: float("nan") for name in feature_names}
        # term_importances includes pair interactions when used; take the
        # first len(feature_names) (the per-feature main effects).
        main = list(imp[: len(feature_names)])
        return dict(zip(feature_names, [float(v) for v in main]))

    def compute_shap(self, X: pd.DataFrame) -> np.ndarray | None:
        # EBM has native explanations; defer SHAP wrapping. The pipeline's
        # shap-summary stage no-ops cleanly on None.
        return None

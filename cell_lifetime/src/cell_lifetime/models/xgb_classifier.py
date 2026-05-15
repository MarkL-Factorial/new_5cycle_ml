"""XGBoost binary classifier — mirrors cell_classifier's RandomForestModel shape."""

from __future__ import annotations

from typing import Any, ClassVar

import numpy as np
import optuna
import pandas as pd
from sklearn.pipeline import Pipeline
from xgboost import XGBClassifier

from cell_classifier.preprocessing.imputer import make_imputer
from cell_lifetime.models.base import CycleLifeModel


class XGBClassifierModel(CycleLifeModel):
    name = "xgb_classifier"
    task: ClassVar[str] = "classification"
    fixed_params: dict[str, Any] = {
        # n_jobs=10 — workspace cap (max 10 cores across all automation).
        # -1 with 80 cores caused OMP thread contention on a 200-row dataset.
        "n_jobs": 10,
        "eval_metric": "logloss",
        "tree_method": "hist",
    }

    def __init__(self, params: dict[str, Any], *, imputer_strategy: str = "median"):
        super().__init__(params, imputer_strategy=imputer_strategy)
        merged = {**self.fixed_params, **params}
        self.pipeline = Pipeline(
            steps=[
                ("imputer", make_imputer(imputer_strategy)),
                ("estimator", XGBClassifier(**merged)),
            ]
        )

    @classmethod
    def suggest_params(cls, trial: optuna.Trial) -> dict[str, Any]:
        return {
            "n_estimators": trial.suggest_int("n_estimators", 50, 600, step=25),
            "max_depth": trial.suggest_int("max_depth", 2, 10),
            "learning_rate": trial.suggest_float("learning_rate", 1e-3, 0.3, log=True),
            "subsample": trial.suggest_float("subsample", 0.5, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
            "min_child_weight": trial.suggest_int("min_child_weight", 1, 10),
            "gamma": trial.suggest_float("gamma", 1e-4, 1.0, log=True),
            "reg_lambda": trial.suggest_float("reg_lambda", 1e-3, 10.0, log=True),
        }

    def fit(self, X: pd.DataFrame, y: np.ndarray) -> "XGBClassifierModel":
        self.pipeline.fit(X, y)
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        return self.pipeline.predict(X)

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        return self.pipeline.predict_proba(X)

    def feature_importance(self, feature_names: list[str]) -> dict[str, float]:
        est: XGBClassifier = self.pipeline.named_steps["estimator"]
        imp = np.asarray(est.feature_importances_, dtype=float)
        return dict(zip(feature_names, imp.tolist()))

    def compute_shap(self, X: pd.DataFrame) -> np.ndarray | None:
        import shap

        est: XGBClassifier = self.pipeline.named_steps["estimator"]
        imputer = self.pipeline.named_steps["imputer"]
        X_imp = pd.DataFrame(imputer.transform(X), columns=X.columns, index=X.index)
        explainer = shap.TreeExplainer(est, feature_perturbation="tree_path_dependent")
        raw = explainer.shap_values(X_imp, check_additivity=False)
        arr = np.asarray(raw)
        # XGBClassifier shap: shape (n, p) for binary (log-odds for class 1)
        return arr

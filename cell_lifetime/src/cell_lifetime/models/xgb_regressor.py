"""XGBoost regressor with target_transform around the sklearn pipeline.

The pipeline imputes + fits on the *transformed* y; `predict()` returns
predictions on the *untransformed* cycle-count scale by applying the
target_transform's inverse.
"""

from __future__ import annotations

from typing import Any, ClassVar

import numpy as np
import optuna
import pandas as pd
from sklearn.pipeline import Pipeline
from xgboost import XGBRegressor

from cell_classifier.preprocessing.imputer import make_imputer
from cell_lifetime.models.base import CycleLifeModel
from cell_lifetime.preprocessing.target_transform import TargetTransform


class XGBRegressorModel(CycleLifeModel):
    name = "xgb_regressor"
    task: ClassVar[str] = "regression"
    fixed_params: dict[str, Any] = {
        # n_jobs=10 — workspace cap (max 10 cores across all automation).
        # -1 with 80 cores caused 8000% CPU thrash on a 200-row dataset.
        "n_jobs": 10,
        "objective": "reg:squarederror",
        "tree_method": "hist",
    }

    def __init__(
        self,
        params: dict[str, Any],
        *,
        imputer_strategy: str = "median",
        target_transform: str = "sqrt",
    ):
        super().__init__(params, imputer_strategy=imputer_strategy)
        self.target_transform_kind = target_transform
        self.target_transform = TargetTransform(kind=target_transform)
        merged = {**self.fixed_params, **params}
        self.pipeline = Pipeline(
            steps=[
                ("imputer", make_imputer(imputer_strategy)),
                ("estimator", XGBRegressor(**merged)),
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

    def fit(self, X: pd.DataFrame, y: np.ndarray) -> "XGBRegressorModel":
        y_t = self.target_transform.fit_transform(np.asarray(y, dtype=float))
        self.pipeline.fit(X, y_t)
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Return predictions on the untransformed cycle-count scale."""
        y_t = self.pipeline.predict(X)
        return self.target_transform.inverse(y_t)

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        raise NotImplementedError("regression has no probabilistic output")

    def feature_importance(self, feature_names: list[str]) -> dict[str, float]:
        est: XGBRegressor = self.pipeline.named_steps["estimator"]
        imp = np.asarray(est.feature_importances_, dtype=float)
        return dict(zip(feature_names, imp.tolist()))

    def compute_shap(self, X: pd.DataFrame) -> np.ndarray | None:
        import shap

        est: XGBRegressor = self.pipeline.named_steps["estimator"]
        imputer = self.pipeline.named_steps["imputer"]
        X_imp = pd.DataFrame(imputer.transform(X), columns=X.columns, index=X.index)
        explainer = shap.TreeExplainer(est, feature_perturbation="tree_path_dependent")
        return np.asarray(explainer.shap_values(X_imp, check_additivity=False))

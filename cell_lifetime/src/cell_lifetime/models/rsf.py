"""Random Survival Forest from scikit-survival.

Wraps `sksurv.ensemble.RandomSurvivalForest`. The structured `(event, time)`
target is constructed inside fit() so the package's BaseModel contract
(`fit(X, time, event)` for survival) stays uniform with xgb_aft.

predict(X) returns the risk score (ensemble-averaged cumulative hazard
summed over training-set event times). Higher = sooner failure →
`risk_orientation = "risk_high"` → pipeline does NOT negate before
C-index.

SHAP is not natively supported on sksurv's RSF; compute_shap returns
None and the pipeline's shap stage handles that cleanly.
"""

from __future__ import annotations

from typing import Any, ClassVar

import numpy as np
import optuna
import pandas as pd
from sksurv.ensemble import RandomSurvivalForest
from sksurv.util import Surv

from cell_classifier.preprocessing.imputer import make_imputer
from cell_lifetime.models.base import CycleLifeModel


class RSFModel(CycleLifeModel):
    name = "rsf"
    task: ClassVar[str] = "survival"
    risk_orientation: ClassVar[str] = "risk_high"
    fixed_params: dict[str, Any] = {
        # Workspace cap (max 10 cores across all automation)
        "n_jobs": 10,
        "oob_score": False,
        "low_memory": True,
    }

    def __init__(self, params: dict[str, Any], *, imputer_strategy: str = "median"):
        super().__init__(params, imputer_strategy=imputer_strategy)
        self.imputer = make_imputer(imputer_strategy)
        merged = {**self.fixed_params, **params}
        self.estimator = RandomSurvivalForest(**merged)
        self.feature_names_: list[str] | None = None

    @classmethod
    def suggest_params(cls, trial: optuna.Trial) -> dict[str, Any]:
        return {
            "n_estimators": trial.suggest_int("n_estimators", 50, 300, step=25),
            "max_depth": trial.suggest_int("max_depth", 5, 20),
            "min_samples_split": trial.suggest_int("min_samples_split", 5, 20),
            "min_samples_leaf": trial.suggest_int("min_samples_leaf", 5, 20),
            "max_features": trial.suggest_categorical(
                "max_features", ["sqrt", "log2", None]
            ),
        }

    def fit(  # type: ignore[override]
        self,
        X: pd.DataFrame,
        time: np.ndarray | None = None,
        event: np.ndarray | None = None,
        y: np.ndarray | None = None,
    ) -> "RSFModel":
        if time is None or event is None:
            raise TypeError(
                "RSFModel.fit requires (X, time, event); "
                f"got time={time is None}, event={event is None}"
            )
        self.feature_names_ = list(X.columns)
        X_imp = self.imputer.fit_transform(X)
        y_struct = Surv.from_arrays(
            event=np.asarray(event, dtype=bool),
            time=np.asarray(time, dtype=float),
        )
        self.estimator.fit(X_imp, y_struct)
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Return risk score (higher = sooner failure)."""
        X_imp = self.imputer.transform(X)
        return np.asarray(self.estimator.predict(X_imp), dtype=float)

    def predict_survival_curve(self, X: pd.DataFrame) -> list:
        """Return per-sample step-function survival curves.

        Not part of the BaseModel ABC; a helper for downstream analysis.
        """
        X_imp = self.imputer.transform(X)
        return list(self.estimator.predict_survival_function(X_imp))

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        raise NotImplementedError("survival has no probabilistic output; use predict()")

    def feature_importance(self, feature_names: list[str]) -> dict[str, float]:
        try:
            imp = np.asarray(self.estimator.feature_importances_, dtype=float)
        except (AttributeError, NotImplementedError):
            return {name: float("nan") for name in feature_names}
        return dict(zip(feature_names, imp.tolist()))

    def compute_shap(self, X: pd.DataFrame) -> np.ndarray | None:
        # shap does not natively support sksurv RandomSurvivalForest;
        # documented as deferred in the roadmap. Permutation importance
        # via the pipeline's importance.py still works.
        return None

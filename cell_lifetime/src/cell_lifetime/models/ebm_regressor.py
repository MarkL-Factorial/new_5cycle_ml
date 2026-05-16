"""EBM (Explainable Boosting Machine) regressor with target_transform.

interpret.glassbox.ExplainableBoostingRegressor handles NaNs natively,
so no sklearn imputer is needed (we still wrap in a Pipeline for shape
parity with the other models). The target_transform is applied around
the EBM in the same way as xgb_regressor.
"""

from __future__ import annotations

from typing import Any, ClassVar

import numpy as np
import optuna
import pandas as pd
from interpret.glassbox import ExplainableBoostingRegressor
from sklearn.pipeline import Pipeline

from cell_classifier.preprocessing.imputer import make_imputer
from cell_lifetime.models.base import CycleLifeModel
from cell_lifetime.preprocessing.target_transform import TargetTransform


class EBMRegressorModel(CycleLifeModel):
    name = "ebm_regressor"
    task: ClassVar[str] = "regression"
    fixed_params: dict[str, Any] = {
        # n_jobs=10 — workspace cap (max 10 cores across all automation).
        "n_jobs": 10,
        # NOTE: `interactions` was previously fixed at 0 here, but the
        # tuner also returns `interactions` from suggest_params, so the
        # fixed value silently won. Removed so Optuna can actually tune
        # the pair-interaction count (small expected gain for EBM).
    }

    def __init__(
        self,
        params: dict[str, Any],
        *,
        imputer_strategy: str = "median",
        target_transform: str = "boxcox",
    ):
        super().__init__(params, imputer_strategy=imputer_strategy)
        self.target_transform_kind = target_transform
        self.target_transform = TargetTransform(kind=target_transform)
        merged = {**self.fixed_params, **params}
        self.pipeline = Pipeline(
            steps=[
                ("imputer", make_imputer(imputer_strategy)),
                ("estimator", ExplainableBoostingRegressor(**merged)),
            ]
        )

    @classmethod
    def suggest_params(cls, trial: optuna.Trial) -> dict[str, Any]:
        # `interactions` upper bound capped at 3 (was 10): higher values
        # explode wall-clock per fit on the 187-cell faded subset, and
        # multi-seed Exp A showed Optuna's TPE biased toward the high
        # end with diminishing accuracy gain. EBM is the interpretability
        # play here; we don't need expensive interactions.
        return {
            "max_bins": trial.suggest_int("max_bins", 64, 512, step=64),
            "max_interaction_bins": trial.suggest_int("max_interaction_bins", 8, 64, step=8),
            "interactions": trial.suggest_int("interactions", 0, 3),
            "learning_rate": trial.suggest_float("learning_rate", 1e-3, 0.3, log=True),
            "min_samples_leaf": trial.suggest_int("min_samples_leaf", 2, 20),
            "max_leaves": trial.suggest_int("max_leaves", 2, 8),
        }

    def fit(self, X: pd.DataFrame, y: np.ndarray) -> "EBMRegressorModel":
        y_t = self.target_transform.fit_transform(np.asarray(y, dtype=float))
        self.pipeline.fit(X, y_t)
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        y_t = self.pipeline.predict(X)
        return self.target_transform.inverse(np.asarray(y_t))

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        raise NotImplementedError("regression has no probabilistic output")

    def feature_importance(self, feature_names: list[str]) -> dict[str, float]:
        est: ExplainableBoostingRegressor = self.pipeline.named_steps["estimator"]
        # EBM gives mean absolute score per term; use that as importance
        try:
            imp = est.term_importances()
        except Exception:
            return {name: float("nan") for name in feature_names}
        # term_importances includes pair interactions when used; take the
        # first len(feature_names) (the per-feature main effects) for the
        # dict we return. If interactions > 0 the trailing entries are pairs.
        main = list(imp[: len(feature_names)])
        return dict(zip(feature_names, [float(v) for v in main]))

    def compute_shap(self, X: pd.DataFrame) -> np.ndarray | None:
        # EBM has native explanations; SHAP wrapping is more nuanced and we
        # defer to permutation_importance for ranking. Return None so the
        # pipeline's shap-summary stage no-ops cleanly.
        return None

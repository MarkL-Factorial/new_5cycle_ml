"""Weibull AFT survival model via lifelines.

Parametric AFT with a Weibull baseline: assumes log-time follows a
location-scale family with Weibull errors. Faster to fit and more
interpretable than tree-based survival, at the cost of the parametric
assumption (which holds reasonably well for battery cycle life).

risk_orientation = "time_high": predict() returns -predicted_median_time,
so "higher = sooner failure" matches the C-index convention via the
pipeline's `_normalize_risk()`. (Lifelines' WeibullAFTFitter exposes
`predict_median()` directly.)
"""

from __future__ import annotations

from typing import Any, ClassVar

import numpy as np
import optuna
import pandas as pd
from lifelines import WeibullAFTFitter
from sklearn.preprocessing import StandardScaler

from cell_classifier.preprocessing.imputer import make_imputer
from cell_lifetime.models.base import CycleLifeModel


class LifelinesWeibullAFTModel(CycleLifeModel):
    name = "lifelines_weibull_aft"
    task: ClassVar[str] = "survival"
    # WeibullAFTFitter.predict_median returns time-positive; pipeline negates
    risk_orientation: ClassVar[str] = "time_high"
    # lifelines is single-threaded; n_jobs not exposed
    fixed_params: dict[str, Any] = {}

    def __init__(self, params: dict[str, Any], *, imputer_strategy: str = "median"):
        super().__init__(params, imputer_strategy=imputer_strategy)
        self.imputer = make_imputer(imputer_strategy)
        self.scaler = StandardScaler()
        # Pop only the keys WeibullAFTFitter accepts
        self.penalizer = float(params.get("penalizer", 0.01))
        self.l1_ratio = float(params.get("l1_ratio", 0.0))
        self.fit_intercept = bool(params.get("fit_intercept", True))
        self.fitter: WeibullAFTFitter | None = None
        self.feature_names_: list[str] | None = None

    @classmethod
    def suggest_params(cls, trial: optuna.Trial) -> dict[str, Any]:
        return {
            # Log-uniform penalizer is the standard sweep
            "penalizer": trial.suggest_float("penalizer", 1e-4, 10.0, log=True),
            "l1_ratio": trial.suggest_float("l1_ratio", 0.0, 1.0),
            "fit_intercept": trial.suggest_categorical("fit_intercept", [True]),
        }

    def fit(  # type: ignore[override]
        self,
        X: pd.DataFrame,
        time: np.ndarray | None = None,
        event: np.ndarray | None = None,
        y: np.ndarray | None = None,
    ) -> "LifelinesWeibullAFTModel":
        if time is None or event is None:
            raise TypeError(
                "LifelinesWeibullAFTModel.fit requires (X, time, event)"
            )
        self.feature_names_ = list(X.columns)
        # Impute + scale (lifelines is sensitive to feature scale via the
        # penalizer; standardizing makes the regularization uniform).
        X_imp = self.imputer.fit_transform(X)
        X_scaled = self.scaler.fit_transform(X_imp)
        df = pd.DataFrame(X_scaled, columns=self.feature_names_, index=X.index)
        df["__time"] = np.maximum(np.asarray(time, dtype=float), 1e-6)  # lifelines requires t > 0
        df["__event"] = np.asarray(event, dtype=bool).astype(int)

        self.fitter = WeibullAFTFitter(
            penalizer=self.penalizer,
            l1_ratio=self.l1_ratio,
            fit_intercept=self.fit_intercept,
        )
        # Suppress convergence warnings for routine runs
        try:
            self.fitter.fit(df, duration_col="__time", event_col="__event",
                            show_progress=False)
        except Exception:
            # Some penalizer/l1_ratio combos diverge; retry with a stronger ridge
            self.fitter = WeibullAFTFitter(penalizer=1.0, l1_ratio=0.0,
                                            fit_intercept=self.fit_intercept)
            self.fitter.fit(df, duration_col="__time", event_col="__event",
                            show_progress=False)
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Predicted median cycle life (time-positive).

        risk_orientation='time_high' tells the pipeline to negate this
        for C-index computation.
        """
        if self.fitter is None:
            raise RuntimeError("call .fit() before .predict()")
        X_imp = self.imputer.transform(X)
        X_scaled = self.scaler.transform(X_imp)
        df = pd.DataFrame(X_scaled, columns=self.feature_names_, index=X.index)
        med = self.fitter.predict_median(df)
        # predict_median may return inf for cells the model thinks
        # never fail; clip to a large finite value so downstream metrics
        # don't choke
        return np.clip(np.asarray(med, dtype=float), 0.0, 1e6)

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        raise NotImplementedError("survival has no probabilistic output; use predict()")

    def feature_importance(self, feature_names: list[str]) -> dict[str, float]:
        """Absolute lambda_-coefficient magnitude as importance.

        lifelines exposes coefficients via fitter.params_; for the AFT
        parameterization we use the lambda_ (location) coefficients,
        which directly modulate the predicted median time.
        """
        if self.fitter is None:
            return {name: float("nan") for name in feature_names}
        try:
            params = self.fitter.params_  # MultiIndex (param, covariate)
            lam = params.loc["lambda_"] if "lambda_" in params.index.get_level_values(0) else params
            imp = {name: float(abs(lam.get(name, 0.0))) for name in feature_names}
            return imp
        except Exception:
            return {name: float("nan") for name in feature_names}

    def compute_shap(self, X: pd.DataFrame) -> np.ndarray | None:
        # lifelines fits aren't trees; SHAP would require a custom kernel
        # explainer with O(n²) cost. Defer.
        return None

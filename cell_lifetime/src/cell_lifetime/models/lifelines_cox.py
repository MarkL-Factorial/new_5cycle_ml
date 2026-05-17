"""Cox proportional-hazards survival model via lifelines.

Semi-parametric — no assumption on the baseline hazard shape; only the
proportional-hazards assumption (covariate effects are time-invariant
multiplicative shifts on the hazard). The classical battery survival
baseline.

risk_orientation = "risk_high": predict() returns partial hazard
(exp(β·x)), which is directly the risk score in the right orientation
(higher = sooner failure). The pipeline's `_normalize_risk()` passes
it through unchanged.
"""

from __future__ import annotations

from typing import Any, ClassVar

import numpy as np
import optuna
import pandas as pd
from lifelines import CoxPHFitter
from sklearn.preprocessing import StandardScaler

from cell_classifier.preprocessing.imputer import make_imputer
from cell_lifetime.models.base import CycleLifeModel


class LifelinesCoxModel(CycleLifeModel):
    name = "lifelines_cox"
    task: ClassVar[str] = "survival"
    # CoxPHFitter.predict_partial_hazard returns risk-positive
    risk_orientation: ClassVar[str] = "risk_high"
    fixed_params: dict[str, Any] = {}

    def __init__(self, params: dict[str, Any], *, imputer_strategy: str = "median"):
        super().__init__(params, imputer_strategy=imputer_strategy)
        self.imputer = make_imputer(imputer_strategy)
        self.scaler = StandardScaler()
        self.penalizer = float(params.get("penalizer", 0.1))
        self.l1_ratio = float(params.get("l1_ratio", 0.0))
        self.fitter: CoxPHFitter | None = None
        self.feature_names_: list[str] | None = None

    @classmethod
    def suggest_params(cls, trial: optuna.Trial) -> dict[str, Any]:
        return {
            "penalizer": trial.suggest_float("penalizer", 1e-3, 10.0, log=True),
            "l1_ratio": trial.suggest_float("l1_ratio", 0.0, 1.0),
        }

    def fit(  # type: ignore[override]
        self,
        X: pd.DataFrame,
        time: np.ndarray | None = None,
        event: np.ndarray | None = None,
        y: np.ndarray | None = None,
    ) -> "LifelinesCoxModel":
        if time is None or event is None:
            raise TypeError("LifelinesCoxModel.fit requires (X, time, event)")
        self.feature_names_ = list(X.columns)
        X_imp = self.imputer.fit_transform(X)
        X_scaled = self.scaler.fit_transform(X_imp)
        df = pd.DataFrame(X_scaled, columns=self.feature_names_, index=X.index)
        df["__time"] = np.maximum(np.asarray(time, dtype=float), 1e-6)
        df["__event"] = np.asarray(event, dtype=bool).astype(int)

        self.fitter = CoxPHFitter(penalizer=self.penalizer, l1_ratio=self.l1_ratio)
        try:
            self.fitter.fit(df, duration_col="__time", event_col="__event",
                            show_progress=False)
        except Exception:
            # Convergence fallback: stronger ridge
            self.fitter = CoxPHFitter(penalizer=1.0, l1_ratio=0.0)
            self.fitter.fit(df, duration_col="__time", event_col="__event",
                            show_progress=False)
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Partial hazard exp(β·x) — risk-positive, ready for C-index."""
        if self.fitter is None:
            raise RuntimeError("call .fit() before .predict()")
        X_imp = self.imputer.transform(X)
        X_scaled = self.scaler.transform(X_imp)
        df = pd.DataFrame(X_scaled, columns=self.feature_names_, index=X.index)
        hazard = self.fitter.predict_partial_hazard(df)
        return np.asarray(hazard, dtype=float)

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        raise NotImplementedError("survival has no probabilistic output; use predict()")

    def feature_importance(self, feature_names: list[str]) -> dict[str, float]:
        """Absolute coefficient as importance (covariates are pre-standardized)."""
        if self.fitter is None:
            return {name: float("nan") for name in feature_names}
        try:
            coefs = self.fitter.params_
            return {name: float(abs(coefs.get(name, 0.0))) for name in feature_names}
        except Exception:
            return {name: float("nan") for name in feature_names}

    def compute_shap(self, X: pd.DataFrame) -> np.ndarray | None:
        return None

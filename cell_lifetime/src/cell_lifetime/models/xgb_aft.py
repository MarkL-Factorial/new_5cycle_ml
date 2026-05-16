"""XGBoost-AFT (Accelerated Failure Time) survival model.

Uses XGBoost's native `survival:aft` objective. Handles censoring via the
(y_lower, y_upper) DMatrix label pair: y_lower==y_upper==time for
observed events; y_lower==time, y_upper=+inf for right-censored.

We use the low-level xgboost.train + DMatrix API rather than the
XGBRegressor sklearn wrapper because AFT's label-pair format isn't a
plain (n,) y vector. The sklearn Pipeline-around-imputer pattern from
other models is preserved by composing manually (imputer fit_transform
+ booster.train); we don't get sklearn's cross_val_score-style hooks
but those aren't used in cell_lifetime's pipelines anyway.

predict(X) returns log-cycle-life predictions (n,) — the raw AFT
output, which is the natural scale for ranking. predict_cycle_life(X)
returns exp(predict(X)) on the original cycle-count scale.

risk_orientation = "time_high" — pipeline negates predict() before
computing C-index (concordance_index_censored expects risk-positive).
"""

from __future__ import annotations

from typing import Any, ClassVar

import numpy as np
import optuna
import pandas as pd
import xgboost as xgb

from cell_classifier.preprocessing.imputer import make_imputer
from cell_lifetime.models.base import CycleLifeModel


class XGBAFTModel(CycleLifeModel):
    name = "xgb_aft"
    task: ClassVar[str] = "survival"
    risk_orientation: ClassVar[str] = "time_high"
    fixed_params: dict[str, Any] = {
        # Workspace cap: max 10 cores across all automation
        "nthread": 10,
        "tree_method": "hist",
        "objective": "survival:aft",
        "eval_metric": "aft-nloglik",
        "verbosity": 0,
    }

    def __init__(self, params: dict[str, Any], *, imputer_strategy: str = "median"):
        super().__init__(params, imputer_strategy=imputer_strategy)
        self.imputer = make_imputer(imputer_strategy)
        # Separate booster hyperparameters from xgb.train control params
        self._n_estimators = int(params.pop("n_estimators", 100))
        self._params = {**self.fixed_params, **params}
        self.booster: xgb.Booster | None = None
        self.feature_names_: list[str] | None = None

    @classmethod
    def suggest_params(cls, trial: optuna.Trial) -> dict[str, Any]:
        return {
            "n_estimators": trial.suggest_int("n_estimators", 50, 600, step=25),
            "max_depth": trial.suggest_int("max_depth", 2, 10),
            "learning_rate": trial.suggest_float("learning_rate", 1e-3, 0.3, log=True),
            "subsample": trial.suggest_float("subsample", 0.5, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
            "aft_loss_distribution": trial.suggest_categorical(
                "aft_loss_distribution", ["normal", "logistic", "extreme"]
            ),
            "aft_loss_distribution_scale": trial.suggest_float(
                "aft_loss_distribution_scale", 0.5, 2.0
            ),
        }

    def fit(  # type: ignore[override]
        self,
        X: pd.DataFrame,
        time: np.ndarray | None = None,
        event: np.ndarray | None = None,
        y: np.ndarray | None = None,
    ) -> "XGBAFTModel":
        """Survival fit. Signature accepts EITHER (X, time, event) OR (X, y)
        where y is interpreted as (time_lower, time_upper) — but the canonical
        invocation uses time + event.
        """
        if time is None or event is None:
            raise TypeError(
                "XGBAFTModel.fit requires (X, time, event); got "
                f"time={time is None}, event={event is None}"
            )
        time = np.asarray(time, dtype=float)
        event = np.asarray(event, dtype=bool)
        if len(time) != len(event) or len(time) != len(X):
            raise ValueError("X, time, event must have the same length")

        self.feature_names_ = list(X.columns)
        X_imp = self.imputer.fit_transform(X)

        y_lower = time.copy()
        y_upper = np.where(event, time, np.inf)

        dtrain = xgb.DMatrix(X_imp, feature_names=self.feature_names_)
        dtrain.set_float_info("label_lower_bound", y_lower)
        dtrain.set_float_info("label_upper_bound", y_upper)

        self.booster = xgb.train(
            self._params,
            dtrain,
            num_boost_round=self._n_estimators,
            verbose_eval=False,
        )
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Return log-cycle-life predictions (n,).

        NOTE: this is `time_high` (higher value = later failure). The
        validation pipeline reads `self.risk_orientation` and negates
        before computing C-index.
        """
        if self.booster is None:
            raise RuntimeError("call .fit() before .predict()")
        X_imp = self.imputer.transform(X)
        dmat = xgb.DMatrix(X_imp, feature_names=self.feature_names_)
        # XGBoost's AFT predict returns the AFT model's location parameter
        # (mu, in log-time space for the default normal/logistic dists).
        return np.asarray(self.booster.predict(dmat), dtype=float)

    def predict_cycle_life(self, X: pd.DataFrame) -> np.ndarray:
        """Convenience: predict cycle life on the untransformed scale.

        Clips the log-time predictions to [-50, 50] before exp so an
        extreme HP (very large n_estimators × deep trees) doesn't make
        the booster emit predictions outside float64's exp range
        (~709), which would silently produce inf and break downstream
        metrics. Clipping at 50 caps cycle life at e^50 ≈ 5.2e21 —
        comfortably above any plausible cell lifetime.
        """
        log_time = np.clip(self.predict(X), -50.0, 50.0)
        return np.exp(log_time)

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        raise NotImplementedError("survival has no probabilistic output; use predict()")

    def feature_importance(self, feature_names: list[str]) -> dict[str, float]:
        if self.booster is None:
            return {name: float("nan") for name in feature_names}
        score = self.booster.get_score(importance_type="gain")
        return {name: float(score.get(name, 0.0)) for name in feature_names}

    def compute_shap(self, X: pd.DataFrame) -> np.ndarray | None:
        # SHAP for survival:aft requires interpretation in log-time space
        # plus careful sign conventions. Defer to a future phase; the
        # shap_per_seed pipeline stage no-ops cleanly on None.
        return None

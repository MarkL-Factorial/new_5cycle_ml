"""Hyperparameter tuning via Optuna TPE with k-fold inner CV.

Adapted from ml_classification_v2/tuning.py. The model class itself owns
its imputer (sklearn Pipeline inside BaseModel), so no separate pipeline
assembly is needed here — we just instantiate a fresh model per trial and
hand it to `cross_val_score` (which knows how to fit/predict it because
the model exposes sklearn-compatible methods through its inner Pipeline).
"""

from __future__ import annotations

from typing import Any

import numpy as np
import optuna
import pandas as pd
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.model_selection import StratifiedKFold, cross_val_score

from cell_classifier.models.base import BaseModel


class _SklearnAdapter(BaseEstimator, ClassifierMixin):
    """Tiny adapter so `cross_val_score` can clone/fit a BaseModel.

    `cross_val_score` wants an unfitted estimator that can be cloned and
    refit per fold. We re-instantiate the BaseModel subclass for each fold
    via the captured `ModelClass` + `params` + `imputer_strategy`.
    """

    def __init__(
        self,
        ModelClass: type[BaseModel] | None = None,
        params: dict[str, Any] | None = None,
        imputer_strategy: str = "median",
    ):
        self.ModelClass = ModelClass
        self.params = params if params is not None else {}
        self.imputer_strategy = imputer_strategy

    def _build(self) -> BaseModel:
        assert self.ModelClass is not None
        return self.ModelClass(self.params, imputer_strategy=self.imputer_strategy)

    def fit(self, X, y):
        self._model = self._build().fit(X, y)
        # classes_ is required by some sklearn scorers (e.g., 'roc_auc')
        self.classes_ = np.unique(y)
        return self

    def predict(self, X):
        return self._model.predict(X)

    def predict_proba(self, X):
        return self._model.predict_proba(X)


def tune(
    ModelClass: type[BaseModel],
    X_train: pd.DataFrame,
    y_train: np.ndarray,
    *,
    n_trials: int = 100,
    inner_cv: int = 5,
    seed: int = 42,
    optimize: str = "f1",
    imputer_strategy: str = "median",
) -> tuple[dict[str, Any], optuna.Study]:
    cv = StratifiedKFold(n_splits=inner_cv, shuffle=True, random_state=seed)

    def objective(trial: optuna.Trial) -> float:
        params = ModelClass.suggest_params(trial)
        params = {**params, "random_state": seed}
        est = _SklearnAdapter(
            ModelClass=ModelClass, params=params, imputer_strategy=imputer_strategy
        )
        scores = cross_val_score(
            est, X_train, y_train, cv=cv, scoring=optimize, n_jobs=-1
        )
        return float(np.mean(scores))

    sampler = optuna.samplers.TPESampler(seed=seed)
    study = optuna.create_study(direction="maximize", sampler=sampler)
    study.optimize(objective, n_trials=n_trials, show_progress_bar=False)
    return dict(study.best_params), study

"""Hyperparameter tuning via Optuna (TPE).

Runs `n_trials` TPE-sampled trials, each evaluating a hyperparameter set by mean
ROC-AUC over a `k`-fold stratified inner CV on the training set. Returns the
best-params dict and the full Optuna study (for persistence).

The pipeline assembles `final_params = best_params ∪ fixed_params ∪ {random_state}`
before fitting on the full training split.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import optuna
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.pipeline import Pipeline

from .models import ModelSpec


def _make_estimator(model_spec: ModelSpec, params: dict[str, Any], seed: int):
    final = {**model_spec.fixed_params, **params, "random_state": seed}
    base = model_spec.build(final)
    return Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("model", base),
        ]
    )


def tune(
    model_spec: ModelSpec,
    X_train: pd.DataFrame,
    y_train: np.ndarray,
    n_trials: int = 100,
    inner_cv: int = 5,
    seed: int = 42,
    optimize: str = "roc_auc",
) -> tuple[dict[str, Any], optuna.Study]:
    cv = StratifiedKFold(n_splits=inner_cv, shuffle=True, random_state=seed)

    def objective(trial: optuna.Trial) -> float:
        params = model_spec.suggest_params(trial)
        est = _make_estimator(model_spec, params, seed=seed)
        scores = cross_val_score(
            est, X_train, y_train, cv=cv, scoring=optimize, n_jobs=-1
        )
        return float(np.mean(scores))

    sampler = optuna.samplers.TPESampler(seed=seed)
    study = optuna.create_study(direction="maximize", sampler=sampler)
    study.optimize(objective, n_trials=n_trials, show_progress_bar=False)
    return dict(study.best_params), study

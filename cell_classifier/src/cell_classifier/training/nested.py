"""Nested cross-validation — statistically clean evaluation protocol.

For each of K outer folds: tune (with inner CV) on the outer-train slice,
fit on outer-train, predict on outer-test. Concatenate predictions across
all K folds so every cell appears exactly once in the test set per seed.

Returns the concatenated predictions plus per-fold metadata (best_params,
inner CV score) for diagnostics.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd

from cell_classifier.data.splits import split_validation_nested_cv
from cell_classifier.models.base import BaseModel
from cell_classifier.training.core import train
from cell_classifier.training.tuning import tune


@dataclass
class NestedCVResult:
    """Folded predictions for one seed."""
    y_true: np.ndarray
    y_pred: np.ndarray
    y_proba: np.ndarray            # shape (n,) — class-1 probability
    fold_id: np.ndarray            # 0..K-1, which outer fold each row came from
    per_fold_best_params: list[dict[str, Any]] = field(default_factory=list)
    per_fold_inner_cv_score: list[float] = field(default_factory=list)
    per_fold_studies: list[Any] = field(default_factory=list)  # list[optuna.Study]
    cell_index_order: np.ndarray = field(default_factory=lambda: np.array([]))


def nested_cv(
    ModelClass: type[BaseModel],
    X: pd.DataFrame,
    y: np.ndarray,
    *,
    outer_k: int,
    inner_cv: int,
    n_trials: int,
    optimize: str,
    seed: int,
    imputer_strategy: str = "median",
) -> NestedCVResult:
    n = len(y)
    y_pred = np.empty(n, dtype=np.int8)
    y_proba = np.empty(n, dtype=np.float64)
    fold_id = np.empty(n, dtype=np.int32)
    cell_index_order = np.empty(n, dtype=np.int64)

    fold_params: list[dict[str, Any]] = []
    fold_scores: list[float] = []
    fold_studies: list[Any] = []

    cursor = 0
    for k, (train_idx, test_idx) in enumerate(
        split_validation_nested_cv(y, outer_k=outer_k, seed=seed)
    ):
        fold_t0 = time.time()
        X_tr = X.iloc[train_idx]
        y_tr = y[train_idx]
        X_te = X.iloc[test_idx]

        best_params, study = tune(
            ModelClass, X_tr, y_tr,
            n_trials=n_trials, inner_cv=inner_cv, seed=seed,
            optimize=optimize, imputer_strategy=imputer_strategy,
        )
        model = train(
            ModelClass, best_params, X_tr, y_tr, seed=seed,
            imputer_strategy=imputer_strategy,
        )

        # Fill into the original-index positions
        y_pred[test_idx] = model.predict(X_te)
        y_proba[test_idx] = model.predict_proba(X_te)[:, 1]
        fold_id[test_idx] = k
        cell_index_order[cursor:cursor + len(test_idx)] = test_idx
        cursor += len(test_idx)

        fold_params.append(dict(best_params))
        fold_scores.append(float(study.best_value))
        fold_studies.append(study)
        print(
            f"[nested_cv] seed={seed} fold={k + 1}/{outer_k} done in "
            f"{time.time() - fold_t0:.1f}s "
            f"(inner_cv={optimize}={study.best_value:.4f})",
            flush=True,
        )

    return NestedCVResult(
        y_true=y.copy(),
        y_pred=y_pred,
        y_proba=y_proba,
        fold_id=fold_id,
        per_fold_best_params=fold_params,
        per_fold_inner_cv_score=fold_scores,
        per_fold_studies=fold_studies,
        cell_index_order=cell_index_order,
    )

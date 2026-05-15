"""Feature importance — model-native + sklearn permutation importance.

Permutation importance uses ROC-AUC scoring (n_repeats=30, n_jobs=-1).
ROC-AUC is threshold-agnostic; permutation importance ranks features by
their effect on ranking quality, independent of the model's decision rule.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.inspection import permutation_importance

from cell_classifier.models.base import BaseModel


def compute_importance(
    model: BaseModel,
    X_test: pd.DataFrame,
    y_test: np.ndarray,
    feature_names: list[str],
    *,
    n_repeats: int = 30,
    seed: int = 42,
) -> pd.DataFrame:
    try:
        native = model.feature_importance(feature_names)
    except NotImplementedError:
        native = {f: np.nan for f in feature_names}

    if len(np.unique(y_test)) < 2:
        perm_mean = np.full(len(feature_names), np.nan)
        perm_std = np.full(len(feature_names), np.nan)
    else:
        # The model exposes sklearn-compatible fit/predict_proba via its
        # internal Pipeline, which is what permutation_importance wants.
        perm = permutation_importance(
            model.pipeline if hasattr(model, "pipeline") else model,
            X_test, y_test,
            n_repeats=n_repeats,
            random_state=seed,
            scoring="roc_auc",
            n_jobs=-1,
        )
        perm_mean = perm.importances_mean
        perm_std = perm.importances_std

    return pd.DataFrame({
        "feature": feature_names,
        "native_importance": [native.get(f, np.nan) for f in feature_names],
        "perm_importance_mean": perm_mean,
        "perm_importance_std": perm_std,
    })

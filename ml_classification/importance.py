"""Feature importance — model-native + sklearn permutation importance.

Permutation importance uses ROC-AUC scoring on the test set (10 repeats), matching
the reference report (`05_perm_importance.py`).
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.inspection import permutation_importance

from .models import ModelSpec


def compute_importance(
    model_spec: ModelSpec,
    fitted,
    X_test: pd.DataFrame,
    y_test: np.ndarray,
    feature_names: list[str],
    n_repeats: int = 30,
    seed: int = 42,
) -> pd.DataFrame:
    native = model_spec.feature_importance(fitted, X_test, feature_names)

    if len(np.unique(y_test)) < 2:
        perm_mean = np.full(len(feature_names), np.nan)
        perm_std = np.full(len(feature_names), np.nan)
    else:
        perm = permutation_importance(
            fitted,
            X_test,
            y_test,
            n_repeats=n_repeats,
            random_state=seed,
            scoring="roc_auc",
            n_jobs=-1,
        )
        perm_mean = perm.importances_mean
        perm_std = perm.importances_std

    return pd.DataFrame(
        {
            "feature": feature_names,
            "native_importance": [native.get(f, np.nan) for f in feature_names],
            "perm_importance_mean": perm_mean,
            "perm_importance_std": perm_std,
        }
    )

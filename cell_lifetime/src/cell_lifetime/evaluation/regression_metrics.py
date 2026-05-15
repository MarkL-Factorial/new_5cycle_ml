"""Regression metrics on the untransformed cycle-count scale.

Per-cohort breakdowns mirror cell_classifier.evaluation.metrics's
auc_AR/auc_0MC pattern.
"""

from __future__ import annotations

import numpy as np
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    median_absolute_error,
    r2_score,
)


def regression_metrics(
    y_true: np.ndarray, y_pred: np.ndarray, cohorts: np.ndarray | None = None
) -> dict[str, float]:
    """MAE / RMSE / R² / MedAE on cycle-count scale, plus per-cohort MAE/R²."""
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    if len(y_true) != len(y_pred):
        raise ValueError(f"shape mismatch: {y_true.shape} vs {y_pred.shape}")
    if len(y_true) == 0:
        return {k: float("nan") for k in ("mae", "rmse", "r2", "medae", "n")}

    mae = float(mean_absolute_error(y_true, y_pred))
    rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
    medae = float(median_absolute_error(y_true, y_pred))
    r2 = float(r2_score(y_true, y_pred)) if len(y_true) > 1 else float("nan")
    out: dict[str, float] = {
        "mae": mae, "rmse": rmse, "r2": r2, "medae": medae,
        "n": float(len(y_true)),
    }

    if cohorts is not None:
        cohorts = np.asarray(cohorts)
        for c in np.unique(cohorts):
            mask = cohorts == c
            if mask.sum() < 2:
                out[f"mae_{c}"] = float("nan")
                out[f"r2_{c}"] = float("nan")
                continue
            out[f"mae_{c}"] = float(mean_absolute_error(y_true[mask], y_pred[mask]))
            out[f"r2_{c}"] = float(r2_score(y_true[mask], y_pred[mask]))
    return out


def prefix(d: dict[str, float], prefix_: str) -> dict[str, float]:
    return {prefix_ + k: v for k, v in d.items()}

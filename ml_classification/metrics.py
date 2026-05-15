"""Classification metrics — overall and per-cohort.

Returns a flat dict so it composes cleanly into per-seed rows. Per-cohort AUC is
NaN when a slice has <2 samples or only one class (sklearn cannot compute AUC).
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)


def _safe_auc(y_true: np.ndarray, y_proba: np.ndarray) -> float:
    if len(y_true) < 2 or len(np.unique(y_true)) < 2:
        return float("nan")
    return float(roc_auc_score(y_true, y_proba))


def evaluate(
    model,
    X: pd.DataFrame,
    y: np.ndarray,
    cohorts: np.ndarray,
) -> dict[str, Any]:
    y_pred = model.predict(X)
    y_proba = model.predict_proba(X)[:, 1]

    out: dict[str, Any] = {
        "n": int(len(y)),
        "accuracy": float(accuracy_score(y, y_pred)),
        "precision": float(precision_score(y, y_pred, zero_division=0)),
        "recall": float(recall_score(y, y_pred, zero_division=0)),
        "f1": float(f1_score(y, y_pred, zero_division=0)),
        "roc_auc": _safe_auc(y, y_proba),
    }

    for cohort_name in ("AR", "0MC"):
        mask = cohorts == cohort_name
        out[f"n_{cohort_name}"] = int(mask.sum())
        out[f"auc_{cohort_name}"] = _safe_auc(y[mask], y_proba[mask])

    return out


def prefix(d: dict[str, Any], pref: str) -> dict[str, Any]:
    return {f"{pref}{k}": v for k, v in d.items()}

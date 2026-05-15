"""EBM (Explainable Boosting Machine) — v0.2 stub.

The body raises NotImplementedError so the registry can guard import-time
failure while still exposing a clear error if a user passes --model ebm.

Real implementation lands in v0.2 (see ROADMAP.md).
"""

from __future__ import annotations

from typing import Any

import numpy as np
import optuna
import pandas as pd

# Will raise ImportError at module load if `interpret` not installed; the
# registry catches that and skips registration of "ebm".
from interpret.glassbox import ExplainableBoostingClassifier  # noqa: F401

from cell_classifier.models.base import BaseModel


class EBMModel(BaseModel):
    name = "ebm"
    fixed_params: dict[str, Any] = {}

    @classmethod
    def suggest_params(cls, trial: optuna.Trial) -> dict[str, Any]:
        raise NotImplementedError("EBM body is a v0.2 stub (see ROADMAP.md)")

    def fit(self, X: pd.DataFrame, y: np.ndarray) -> "EBMModel":
        raise NotImplementedError("EBM body is a v0.2 stub (see ROADMAP.md)")

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        raise NotImplementedError("EBM body is a v0.2 stub")

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        raise NotImplementedError("EBM body is a v0.2 stub")

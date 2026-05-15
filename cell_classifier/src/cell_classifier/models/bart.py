"""BART (Bayesian Additive Regression Trees) — v0.2 stub.

When implemented, `predict_proba_samples()` will return the full posterior
draws, exposing BART's epistemic uncertainty without collapsing to a point
estimate.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import optuna
import pandas as pd

# Will raise ImportError at module load if the chosen BART backend isn't
# installed; the registry catches that and skips registration of "bart".
import pymc_bart  # noqa: F401

from cell_classifier.models.base import BaseModel


class BARTModel(BaseModel):
    name = "bart"
    fixed_params: dict[str, Any] = {}

    @classmethod
    def suggest_params(cls, trial: optuna.Trial) -> dict[str, Any]:
        raise NotImplementedError("BART body is a v0.2 stub (see ROADMAP.md)")

    def fit(self, X: pd.DataFrame, y: np.ndarray) -> "BARTModel":
        raise NotImplementedError("BART body is a v0.2 stub")

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        raise NotImplementedError("BART body is a v0.2 stub")

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        raise NotImplementedError("BART body is a v0.2 stub")

    def predict_proba_samples(self, X: pd.DataFrame) -> np.ndarray | None:
        raise NotImplementedError("BART body is a v0.2 stub")

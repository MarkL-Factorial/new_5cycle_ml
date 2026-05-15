"""BART (Bayesian Additive Regression Trees) model spec — Stage 3 stub.

To implement:
  1. Pick a backend: `pymc-bart` (PyMC-based) or `bartpy` (lightweight).
     Both need a thin sklearn-compat shim that exposes
     `.fit(X, y) / .predict(X) / .predict_proba(X)`.
  2. Replace each `_not_implemented` body with the real implementation:
       - build:  wrap the chosen library in a small adapter class.
       - suggest_params: typical knobs are
                  num_trees (20–200), alpha (0.5–0.99),
                  beta (1–4), k (1–5), number of MCMC draws / chains.
       - feature_importance: BART exposes per-feature inclusion proportions
                 (`m.inclusion_proportions()` in bartpy) — use those.
  3. Add `configs/bart_n{200,300,400}.yaml`.

The pipeline / config loader / output schema do not change.
"""

from __future__ import annotations

from typing import Any

import optuna
import pandas as pd

from .base import ModelSpec

_STAGE = "BART support — Stage 3; see ml_classification/README.md extension guide."


class BARTModelSpec(ModelSpec):
    name = "bart"

    def build(self, params: dict[str, Any]):
        raise NotImplementedError(_STAGE)

    def suggest_params(self, trial: optuna.Trial) -> dict[str, Any]:
        raise NotImplementedError(_STAGE)

    def feature_importance(
        self,
        fitted,
        X: pd.DataFrame,
        feature_names: list[str],
    ) -> dict[str, float]:
        raise NotImplementedError(_STAGE)

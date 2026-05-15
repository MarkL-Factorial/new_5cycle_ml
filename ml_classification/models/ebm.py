"""EBM (Explainable Boosting Machine) model spec — Stage 2 stub.

To implement:
  1. `pip install interpret` in the eis env.
  2. Replace each `_not_implemented` body with the real implementation:
       - build:  `from interpret.glassbox import ExplainableBoostingClassifier`
                 `return ExplainableBoostingClassifier(**params)`
       - suggest_params: define an Optuna search space over interactions,
                 learning_rate, max_bins, max_leaves, min_samples_leaf, etc.
       - feature_importance: `fitted.explain_global().data()['scores']` keyed
                 by `fitted.term_names_`. Filter to single-feature terms
                 (no pairs) so the dict matches `feature_names`.
  3. Add `configs/ebm_n{200,300,400}.yaml` cloning the rf_ versions with
     `model: ebm`.

The pipeline / config loader / output schema do not change.
"""

from __future__ import annotations

from typing import Any

import optuna
import pandas as pd

from .base import ModelSpec

_STAGE = "EBM support — Stage 2; see ml_classification/README.md extension guide."


class EBMModelSpec(ModelSpec):
    name = "ebm"

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

"""Shared training step — mode-agnostic.

Both validation and production pipelines call `train()` after their split.
The mode never appears here.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from cell_classifier.models.base import BaseModel


def train(
    ModelClass: type[BaseModel],
    params: dict[str, Any],
    X_train: pd.DataFrame,
    y_train: np.ndarray,
    seed: int,
    *,
    imputer_strategy: str = "median",
) -> BaseModel:
    """Instantiate, fit, and return the model with random_state pinned to `seed`."""
    final_params = {**params, "random_state": seed}
    model = ModelClass(final_params, imputer_strategy=imputer_strategy)
    return model.fit(X_train, y_train)

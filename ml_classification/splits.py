"""Train / test split for the classifier pipeline.

Stratified 80 / 20 by target only. Returns two numpy index arrays, disjoint
and exhaustive over the input.

Hyperparameter tuning is done by Optuna with 5-fold stratified inner CV on the
train slice — see `tuning.py`. There is no separate validation slice.
"""

from __future__ import annotations

import numpy as np
from sklearn.model_selection import train_test_split


def stratified_split(
    y: np.ndarray,
    test_frac: float = 0.2,
    seed: int = 42,
) -> tuple[np.ndarray, np.ndarray]:
    if not 0.0 < test_frac < 1.0:
        raise ValueError(f"test_frac must be in (0, 1) (got {test_frac})")

    all_idx = np.arange(len(y))
    idx_train, idx_test = train_test_split(
        all_idx,
        test_size=test_frac,
        stratify=y,
        random_state=seed,
    )
    return idx_train, idx_test

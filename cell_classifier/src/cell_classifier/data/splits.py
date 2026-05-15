"""Mode-aware split layer.

This is the only module that knows about modes. The CLI dispatches based
on `--mode`, and the matching pipeline orchestrator calls one of these
splitter functions. Training, models, preprocessing, and evaluation code
has no mode awareness.

Splitters:
  - split_validation_tune_inner_cv(y, test_frac, seed) → one (train, test) split
  - split_validation_nested_cv(y, outer_k, seed) → K (train, test) folds
  - split_production(label_mask) → (train_idx, inference_idx)
"""

from __future__ import annotations

from typing import Iterator

import numpy as np
from sklearn.model_selection import StratifiedKFold, StratifiedShuffleSplit


def split_validation_tune_inner_cv(
    y: np.ndarray, test_frac: float, seed: int
) -> tuple[np.ndarray, np.ndarray]:
    """Single stratified train/test split for tune-inner-cv protocol."""
    splitter = StratifiedShuffleSplit(n_splits=1, test_size=test_frac, random_state=seed)
    train_idx, test_idx = next(splitter.split(np.zeros(len(y)), y))
    return train_idx, test_idx


def split_validation_nested_cv(
    y: np.ndarray, outer_k: int, seed: int
) -> Iterator[tuple[np.ndarray, np.ndarray]]:
    """Yield K (train_idx, test_idx) pairs from a stratified K-fold."""
    kf = StratifiedKFold(n_splits=outer_k, shuffle=True, random_state=seed)
    for train_idx, test_idx in kf.split(np.zeros(len(y)), y):
        yield train_idx, test_idx


def split_production(
    label_mask: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """train_idx = labeled (trainable) rows; inference_idx = ALL rows.

    Per user choice: score every cell with features, regardless of label
    status. label-availability of the train set is required; the inference
    set's labels (if any) are ignored at scoring time.
    """
    n = len(label_mask)
    train_idx = np.flatnonzero(label_mask)
    inference_idx = np.arange(n)
    return train_idx, inference_idx

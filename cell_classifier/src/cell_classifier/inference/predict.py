"""Ensemble inference — combine per-seed predictions into final scores.

Inputs are lists of per-seed predict_proba outputs (or posterior sample arrays
when a Bayesian model is in use). Returns:
  - predictions DataFrame (one row per cell): cell_name, mean_proba_pass,
    std_proba_pass, predicted_class (= 1 iff mean_proba_pass > 0.5)
  - per-seed long-format frame for the parquet
  - optional posterior frame (only when samples are provided)
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def ensemble_predictions(
    cell_names: np.ndarray,
    per_seed_proba_pass: list[np.ndarray],
    *,
    threshold: float = 0.5,
) -> pd.DataFrame:
    """Mean / std of per-seed P(pass); class = mean > threshold."""
    if not per_seed_proba_pass:
        raise ValueError("no per-seed predictions to ensemble")
    arr = np.vstack(per_seed_proba_pass)              # (n_seeds, n_cells)
    mean = arr.mean(axis=0)
    std = arr.std(axis=0, ddof=0)
    pred = (mean > threshold).astype(np.int8)
    return pd.DataFrame({
        "cell_name": cell_names,
        "mean_proba_pass": mean,
        "std_proba_pass": std,
        "predicted_class": pred,
    })


def per_seed_long(
    cell_names: np.ndarray,
    seeds: list[int],
    per_seed_proba_pass: list[np.ndarray],
) -> pd.DataFrame:
    """Long format: (seed, cell_name, proba_pass). Suitable for parquet."""
    frames = []
    for seed, probas in zip(seeds, per_seed_proba_pass):
        frames.append(pd.DataFrame({
            "seed": int(seed),
            "cell_name": cell_names,
            "proba_pass": probas,
        }))
    return pd.concat(frames, ignore_index=True)


def per_seed_posterior_long(
    cell_names: np.ndarray,
    seeds: list[int],
    per_seed_samples: list[np.ndarray],   # each entry shape (n_draws, n_cells, 2)
) -> pd.DataFrame:
    """Long format for posterior samples: (seed, draw, cell_name, proba_pass)."""
    frames = []
    for seed, samples in zip(seeds, per_seed_samples):
        n_draws, n_cells, _ = samples.shape
        proba_pass = samples[:, :, 1]      # (n_draws, n_cells)
        draw_idx = np.repeat(np.arange(n_draws), n_cells)
        cell_idx = np.tile(np.arange(n_cells), n_draws)
        frames.append(pd.DataFrame({
            "seed": int(seed),
            "draw": draw_idx,
            "cell_name": cell_names[cell_idx],
            "proba_pass": proba_pass.ravel(),
        }))
    return pd.concat(frames, ignore_index=True)

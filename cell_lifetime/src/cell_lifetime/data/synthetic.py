"""Synthetic CycleLifeDataset for tests that can't access the real bundle.

Used by cloud /schedule routines (which run in a sandbox without
ml_label_preprocess/datasets/) and by fast local unit tests.

Cycle life is drawn lognormal so target transforms have something to do.
Features are correlated noise so models can actually fit.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from cell_lifetime.data.loader import CycleLifeDataset


def make_synthetic_dataset(
    *,
    n_faded: int = 80,
    n_censored: int = 120,
    n_features: int = 12,
    cycle_life_mu_sigma: tuple[float, float] = (5.5, 0.7),  # mean ln(cycle), std
    censor_horizon: int = 600,
    seed: int = 0,
    N: int = 300,
    feature_subset_name: str = "fs_cv",
) -> CycleLifeDataset:
    """Generate a CycleLifeDataset shaped like A2.2_b1's `fs_cv` subset.

    With defaults: 200 cells total (80 faded + 120 censored), 12 features.
    Faded cell lifetimes ~ Lognormal(5.5, 0.7) → median ~245, IQR-ish 150-400.
    Features are 12-dim correlated noise, mildly predictive of cycle life
    so models converge to >0 R² in tests.
    """
    rng = np.random.default_rng(seed)
    n_total = n_faded + n_censored

    # Cycle life lognormal
    mu, sigma = cycle_life_mu_sigma
    cycle_life_faded = rng.lognormal(mean=mu, sigma=sigma, size=n_faded)
    cycle_life_faded = np.clip(cycle_life_faded, 5, 2000).astype(np.int64)

    # Censored cells: observed up to some horizon (>=6 to match loader's filter)
    n_regular_censored = rng.integers(6, censor_horizon + 1, size=n_censored)

    # Features: a low-rank latent driver of cycle life + noise
    # Latent: z ~ Normal; cycle_life depends on z; features = mixture of z + noise
    latent_faded = (np.log(cycle_life_faded) - mu) / sigma  # standardized
    latent_censored = rng.normal(loc=0.5, scale=0.8, size=n_censored)  # slightly higher z (these survived longer)
    latent = np.concatenate([latent_faded, latent_censored])

    # 12 features: 4 strongly correlated with latent, 8 noise
    F = rng.normal(size=(n_total, n_features))
    for j in range(min(4, n_features)):
        F[:, j] += 0.7 * latent  # signal
    feature_names = [f"feat_{j:02d}" for j in range(n_features)]
    X = pd.DataFrame(F, columns=feature_names)

    # Insert a few NaNs so imputer is exercised
    nan_idx = rng.choice(n_total, size=max(1, n_total // 20), replace=False)
    nan_col = rng.choice(n_features, size=len(nan_idx))
    for i, j in zip(nan_idx, nan_col):
        X.iat[int(i), int(j)] = np.nan

    # Build the per-row arrays
    event = np.concatenate(
        [np.ones(n_faded, dtype=bool), np.zeros(n_censored, dtype=bool)]
    )
    time = np.concatenate([cycle_life_faded, n_regular_censored])
    y_cycle = np.where(event, time.astype(float), np.nan)

    # Classification: pass iff (event=False and time>=N) or (event=True and time>N)
    is_pass = np.where(event, time > N, time >= N)
    y_class = is_pass.astype(np.int8)
    # Censored rows that haven't reached N are non-trainable
    trainable = np.where(event, True, time >= N)
    label_mask = trainable.astype(bool)

    cohorts = np.where(rng.random(n_total) < 0.85, "AR", "0MC").astype(object)
    cell_names = np.array([f"SYN-{i:04d}" for i in range(n_total)])

    # n_regular: number of regular cycles observed. For synthetic faded
    # cells we use their cycle_life as a proxy; for censored cells we use
    # the censoring time (n_regular_censored). These match the upstream
    # ml_label_preprocess semantics closely enough for tests.
    n_regular = np.concatenate(
        [cycle_life_faded, n_regular_censored]
    ).astype(np.int64)

    return CycleLifeDataset(
        X=X,
        y_class=y_class,
        y_cycle=y_cycle,
        event=event,
        time=time,
        label_mask=label_mask,
        faded_mask=event.copy(),
        cohorts=cohorts,
        cell_names=cell_names,
        n_regular=n_regular,
        feature_names=feature_names,
        N=N,
        baseline_cycle=1,
        db_version="SYN",
        source_dir=Path("/synthetic"),
    )

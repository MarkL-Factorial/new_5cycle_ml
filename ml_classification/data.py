"""Data loading and assembly for the classifier pipeline.

Joins `cell_features.parquet` and `cell_labels.parquet` from `ml_label_preprocess/out/`,
filters to trainable cells at a given cycle threshold N, and selects a named feature
subset (e.g. `fs_cv`) from `column_roles.yaml`.

The loader is strict about leakage:
  * `X` columns are exactly the named feature subset (no labels or meta sneak in).
  * Missing feature names in the parquet abort with a clear error.
  * Cells without features (n_regular < 5) are dropped by the inner join.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import polars as pl
import yaml

REPO_ROOT = Path(__file__).resolve().parent.parent
PREPROCESS_DIR = REPO_ROOT / "ml_label_preprocess"
FEATURES_PATH = PREPROCESS_DIR / "out" / "cell_features.parquet"
LABELS_PATH = PREPROCESS_DIR / "out" / "cell_labels.parquet"
COLUMN_ROLES_PATH = PREPROCESS_DIR / "column_roles.yaml"

_LABEL_COLUMNS_DENYLIST = {
    "status",
    "exclusion_reason",
    "last_fade_cycle",
    "n_regular",
    "final_retention",
    "n_recovered_crossings",
    "label_n200",
    "label_n300",
    "label_n400",
    "trainable_n200",
    "trainable_n300",
    "trainable_n400",
}


@dataclass
class Dataset:
    X: pd.DataFrame
    y: np.ndarray
    cohorts: np.ndarray
    cell_names: np.ndarray
    feature_names: list[str]
    N: int

    def __len__(self) -> int:
        return len(self.y)


def _load_feature_subset(subset_name: str) -> list[str]:
    manifest = yaml.safe_load(COLUMN_ROLES_PATH.read_text())
    subsets = manifest.get("subsets", {})
    if subset_name not in subsets:
        raise KeyError(
            f"subset {subset_name!r} not in column_roles.yaml::subsets "
            f"(available: {list(subsets)})"
        )
    return list(subsets[subset_name]["members"])


def load_dataset(N: int, feature_subset: str = "fs_cv") -> Dataset:
    if N not in (200, 300, 400):
        raise ValueError(f"N must be one of 200, 300, 400 (got {N})")

    feature_names = _load_feature_subset(feature_subset)
    features = pl.read_parquet(FEATURES_PATH)
    labels = pl.read_parquet(LABELS_PATH)

    missing = [f for f in feature_names if f not in features.columns]
    if missing:
        raise KeyError(
            f"requested features missing from {FEATURES_PATH.name}: {missing}"
        )

    leakage = set(feature_names) & _LABEL_COLUMNS_DENYLIST
    if leakage:
        raise ValueError(
            f"feature subset {feature_subset!r} contains label-like columns: {leakage}"
        )

    joined = labels.join(features, on="cell_name", how="inner").filter(
        pl.col(f"trainable_n{N}")
    )

    pdf = joined.to_pandas()
    X = pdf[feature_names].copy()
    y = (pdf[f"label_n{N}"] == "pass").to_numpy().astype(np.int8)
    cohorts = pdf["cohort"].to_numpy()
    cell_names = pdf["cell_name"].to_numpy()

    if X.isna().any().any():
        nan_cols = X.columns[X.isna().any()].tolist()
        n_rows_with_nan = int(X.isna().any(axis=1).sum())
        print(
            f"[data] WARNING: {n_rows_with_nan} cells have NaN in features "
            f"{nan_cols}; rows kept (downstream imputation expected)."
        )

    return Dataset(
        X=X,
        y=y,
        cohorts=cohorts,
        cell_names=cell_names,
        feature_names=feature_names,
        N=N,
    )

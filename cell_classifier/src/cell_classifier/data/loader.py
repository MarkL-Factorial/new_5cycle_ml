"""Load (features + labels) for a (db_version, baseline_cycle, N, feature_subset).

Reads ml_label_preprocess bundles. Path resolution:

  1. ``BCC_PREPROCESS_ROOT`` env var, if set
  2. ``preprocess_root`` arg from caller (config), if set
  3. Default: ``Path(__file__).resolve().parents[4] / "ml_label_preprocess"``

The dataclass `Dataset` carries everything downstream needs:
  - X (features), y (binary 0/1 labels), cohorts, cell_names
  - feature_names, N, baseline_cycle, db_version
  - label_mask (trainable_n{N} == True; for production-mode split)

`y` is defined for every row of `X`, but rows with `label_mask == False`
are not safe to use for training (their labels are 'censor' or 'excluded',
not 'pass' / 'bad'). Production mode uses the labeled subset for training
and the full X for inference.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import polars as pl
import yaml


SUPPORTED_N = (200, 300, 400)
SUPPORTED_BASELINE = (1, 2, 3, 4)


def _default_preprocess_root() -> Path:
    """Five parents up from this file lands at new_5cycle_ml/, then add the bundle dir."""
    return Path(__file__).resolve().parents[4] / "ml_label_preprocess"


def _resolve_preprocess_root(override: Optional[str] = None) -> Path:
    if override:
        return Path(override)
    env = os.getenv("BCC_PREPROCESS_ROOT")
    if env:
        return Path(env)
    return _default_preprocess_root()


def column_roles_path(preprocess_root: Optional[str] = None) -> Path:
    return _resolve_preprocess_root(preprocess_root) / "column_roles.yaml"


_REQUIRED_BUNDLE_FILES = ("cell_features.parquet", "cell_labels.parquet")


def _bundle_is_complete(dir_: Path) -> bool:
    return all((dir_ / name).is_file() for name in _REQUIRED_BUNDLE_FILES)


def _resolve_bundle_dir(
    root: Path,
    db_version: str,
    baseline_cycle: int,
    snapshot: Optional[str] = None,
) -> Path:
    """Resolve the dir holding both cell_features.parquet AND cell_labels.parquet.

    Upstream layout (ml_label_preprocess v3):

        datasets/{db_version}_b{baseline_cycle}/
            {db_version}_b{baseline_cycle}_{ts}/      <- timestamped snapshot
                cell_features.parquet
                cell_labels.parquet
                manifest.json
            {db_version}_b{baseline_cycle}_latest     <- symlink to a snapshot

    Resolution order:

      1. ``snapshot`` arg (e.g. ``'A2.2_b1_20260520_1352_legacy'``) — pin
         a specific snapshot for reproducibility. No completeness check —
         caller asked for it explicitly.
      2. ``{bundle}_latest`` symlink, IF the target is complete (both
         parquets present). Upstream's ``preprocess.py --labels`` writes
         a labels-only snapshot and re-points ``_latest`` at it, so the
         symlink alone is not enough.
      3. Most recent complete snapshot by directory name (timestamp-
         sortable: ``{bundle}_YYYYMMDD_HHMM[_legacy]``).
      4. Legacy flat layout (pre-v3): parquets directly under ``{bundle}/``.
    """
    bundle_parent = root / "datasets" / f"{db_version}_b{baseline_cycle}"
    if snapshot is not None:
        return bundle_parent / snapshot

    latest = bundle_parent / f"{db_version}_b{baseline_cycle}_latest"
    if latest.exists() and _bundle_is_complete(latest):
        return latest

    if bundle_parent.is_dir():
        prefix = f"{db_version}_b{baseline_cycle}_"
        candidates = sorted(
            (p for p in bundle_parent.iterdir()
             if p.is_dir() and p.name.startswith(prefix) and p.name != f"{prefix}latest"),
            key=lambda p: p.name,
            reverse=True,
        )
        for snap in candidates:
            if _bundle_is_complete(snap):
                return snap

    if _bundle_is_complete(bundle_parent):
        return bundle_parent  # pre-v3 flat layout

    return latest  # let caller's FileNotFoundError surface the broken default


def _load_feature_subset(subset_name: str, preprocess_root: Optional[str] = None) -> list[str]:
    path = column_roles_path(preprocess_root)
    manifest = yaml.safe_load(path.read_text())
    subsets = manifest.get("subsets", {})
    if subset_name not in subsets:
        raise KeyError(
            f"subset {subset_name!r} not in {path}::subsets "
            f"(available: {list(subsets)})"
        )
    return list(subsets[subset_name]["members"])


_LABEL_COLUMNS_DENYLIST = {
    "status", "exclusion_reason", "last_fade_cycle", "n_regular",
    "final_retention", "n_recovered_crossings",
    "label_n200", "label_n300", "label_n400",
    "trainable_n200", "trainable_n300", "trainable_n400",
}


@dataclass
class Dataset:
    X: pd.DataFrame
    y: np.ndarray              # int8 in {0, 1}; 0 for non-pass rows including censor/excluded
    label_mask: np.ndarray     # bool; True iff trainable_n{N} == True
    cohorts: np.ndarray
    cell_names: np.ndarray
    feature_names: list[str]
    N: int
    baseline_cycle: int
    db_version: str
    source_dir: Path

    def __len__(self) -> int:
        return len(self.y)

    def labeled_view(self) -> "Dataset":
        """Restrict X/y/cohorts/cell_names to the trainable subset.

        label_mask becomes all-True on the restricted view.
        """
        mask = self.label_mask
        return Dataset(
            X=self.X.loc[mask].reset_index(drop=True),
            y=self.y[mask],
            label_mask=np.ones(int(mask.sum()), dtype=bool),
            cohorts=self.cohorts[mask],
            cell_names=self.cell_names[mask],
            feature_names=self.feature_names,
            N=self.N, baseline_cycle=self.baseline_cycle,
            db_version=self.db_version, source_dir=self.source_dir,
        )


def load_dataset(
    N: int,
    feature_subset: str = "fs_cv",
    baseline_cycle: int = 1,
    db_version: str = "A2.2",
    preprocess_root: Optional[str] = None,
    snapshot: Optional[str] = None,
) -> Dataset:
    """Read an ml_label_preprocess bundle for (db_version, baseline_cycle).

    ``snapshot`` (default None) pins to a specific snapshot dir name
    (e.g. ``'A2.2_b1_20260520_1352_legacy'``). When None, follows the
    ``{bundle}_latest`` symlink. See :func:`_resolve_bundle_dir`.
    """
    if N not in SUPPORTED_N:
        raise ValueError(f"N must be one of {SUPPORTED_N} (got {N})")
    if baseline_cycle not in SUPPORTED_BASELINE:
        raise ValueError(
            f"baseline_cycle must be one of {SUPPORTED_BASELINE} (got {baseline_cycle})"
        )

    root = _resolve_preprocess_root(preprocess_root)
    bundle = _resolve_bundle_dir(root, db_version, baseline_cycle, snapshot)
    features_path = bundle / "cell_features.parquet"
    labels_path = bundle / "cell_labels.parquet"
    if not features_path.exists() or not labels_path.exists():
        raise FileNotFoundError(
            f"preprocess bundle not found at {bundle}. "
            f"Generate it via: "
            f"python {root}/preprocess.py --all "
            f"--baseline-cycle {baseline_cycle} --db-version {db_version}"
        )

    feature_names = _load_feature_subset(feature_subset, preprocess_root)
    features = pl.read_parquet(features_path)
    labels = pl.read_parquet(labels_path)

    missing = [f for f in feature_names if f not in features.columns]
    if missing:
        raise KeyError(
            f"requested features missing from {features_path.name}: {missing}"
        )
    leakage = set(feature_names) & _LABEL_COLUMNS_DENYLIST
    if leakage:
        raise ValueError(
            f"feature subset {feature_subset!r} contains label-like columns: {leakage}"
        )

    joined = labels.join(features, on="cell_name", how="inner").to_pandas()
    X = joined[feature_names].copy().reset_index(drop=True)
    label_mask = joined[f"trainable_n{N}"].to_numpy().astype(bool)
    y_str = joined[f"label_n{N}"].to_numpy()
    # y = 1 iff "pass"; 0 for "bad", "censor", "excluded". The label_mask
    # excludes non-trainable rows from training-set use.
    y = (y_str == "pass").astype(np.int8)
    cohorts = joined["cohort"].to_numpy()
    cell_names = joined["cell_name"].to_numpy()

    return Dataset(
        X=X, y=y, label_mask=label_mask,
        cohorts=cohorts, cell_names=cell_names,
        feature_names=feature_names,
        N=N, baseline_cycle=baseline_cycle, db_version=db_version,
        source_dir=bundle,
    )

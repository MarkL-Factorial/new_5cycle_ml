"""Load (features + cycle-life targets) for a (db_version, baseline_cycle, N, feature_subset).

Reads the same ml_label_preprocess bundles as cell_classifier, but emits a
CycleLifeDataset with regression and survival targets in addition to the
binary classification y. The classification path matches cell_classifier
exactly so cross-comparison is direct.

Three target shapes per row:
  - y_class: int8 in {0, 1} — pass (=1) iff `label_n{N}` == "pass"
  - y_cycle: float — `last_fade_cycle` if status=="faded", else NaN
  - event:   bool  — True iff status=="faded"
  - time:    int   — `last_fade_cycle` if faded, else `n_regular`

`label_mask` (classification trainable: `trainable_n{N}`) and
`faded_mask` (regression trainable: status=="faded") gate which rows
each model sees. Survival uses (event, time) directly with no masking
beyond `excluded` rows dropped at load time.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Literal, Optional

import numpy as np
import pandas as pd
import polars as pl
import yaml

from cell_classifier.data.loader import column_roles_path


SUPPORTED_N = (200, 300, 400)
SUPPORTED_BASELINE = (1, 2, 3, 4)
SUPPORTED_TASKS = ("classification", "regression", "survival")

_LABEL_COLUMNS_DENYLIST = {
    # cell_classifier's own denylist
    "status", "exclusion_reason", "last_fade_cycle", "n_regular",
    "final_retention", "n_recovered_crossings",
    "label_n200", "label_n300", "label_n400",
    "trainable_n200", "trainable_n300", "trainable_n400",
    # cell_lifetime additions
    "cohort",  # never use cohort as a feature (chemistry shortcut)
}


def _default_preprocess_root() -> Path:
    """Parents: data/loader.py -> data -> cell_lifetime -> src -> cell_lifetime -> new_5cycle_ml."""
    return Path(__file__).resolve().parents[4] / "ml_label_preprocess"


def _resolve_preprocess_root(override: Optional[str] = None) -> Path:
    if override:
        return Path(override)
    env = os.getenv("BCC_PREPROCESS_ROOT") or os.getenv("CL_PREPROCESS_ROOT")
    if env:
        return Path(env)
    return _default_preprocess_root()


def _load_feature_subset(subset_name: str, preprocess_root: Optional[str] = None) -> list[str]:
    path = column_roles_path(preprocess_root)
    manifest = yaml.safe_load(path.read_text())
    subsets = manifest.get("subsets", {})
    if subset_name not in subsets:
        raise KeyError(
            f"subset {subset_name!r} not in {path}::subsets "
            f"(available: {sorted(subsets)})"
        )
    return list(subsets[subset_name]["members"])


@dataclass
class CycleLifeDataset:
    """Dataset with classification, regression, and survival targets."""

    X: pd.DataFrame
    # Targets — one per task
    y_class: np.ndarray             # int8 in {0,1}; defined for all rows
    y_cycle: np.ndarray             # float; NaN for non-faded rows
    event: np.ndarray               # bool
    time: np.ndarray                # int (regular_cycle ordinal)
    # Masks
    label_mask: np.ndarray          # bool — classification trainable (trainable_n{N})
    faded_mask: np.ndarray          # bool — regression trainable (status=='faded')
    # Metadata
    cohorts: np.ndarray             # str
    cell_names: np.ndarray          # str
    feature_names: list[str]
    N: int
    baseline_cycle: int
    db_version: str
    source_dir: Path

    def __len__(self) -> int:
        return len(self.y_class)

    def task_target(self, task: str) -> tuple[np.ndarray, np.ndarray]:
        """Return (y, mask) for a task. Caller filters X by mask before fitting.

        - classification → (y_class, label_mask)
        - regression     → (y_cycle, faded_mask)
        - survival       → (time, event-as-mask)  — but survival models consume
                           (event, time) directly; this is the "scalar y" view.
        """
        if task == "classification":
            return self.y_class, self.label_mask
        if task == "regression":
            return self.y_cycle, self.faded_mask
        if task == "survival":
            return self.time, np.ones_like(self.event, dtype=bool)
        raise ValueError(f"unknown task {task!r}; supported: {SUPPORTED_TASKS}")

    def view_for_task(self, task: str) -> "CycleLifeDataset":
        """Restrict X/targets/cohorts/cell_names to rows usable for the task.

        Mirrors cell_classifier.Dataset.labeled_view(): returns a new dataset
        whose `*_mask` is all-True on the restricted rows.
        """
        if task == "classification":
            mask = self.label_mask
        elif task == "regression":
            mask = self.faded_mask
        elif task == "survival":
            mask = np.ones_like(self.event, dtype=bool)  # all 444 trainable rows
        else:
            raise ValueError(f"unknown task {task!r}")
        n = int(mask.sum())
        all_true = np.ones(n, dtype=bool)
        return CycleLifeDataset(
            X=self.X.loc[mask].reset_index(drop=True),
            y_class=self.y_class[mask],
            y_cycle=self.y_cycle[mask],
            event=self.event[mask],
            time=self.time[mask],
            label_mask=all_true if task == "classification" else self.label_mask[mask],
            faded_mask=all_true if task == "regression" else self.faded_mask[mask],
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
    drop_excluded: bool = True,
    min_n_regular: int = 6,
) -> CycleLifeDataset:
    """Read an ml_label_preprocess bundle and return targets for all three tasks.

    Setting `drop_excluded=True` (default) removes 17 cells with
    status=='excluded' from every output array — they have no usable
    target under any task. Set to False if you specifically want them
    returned with NaN/False targets.

    `min_n_regular` (default 6) is a hard filter: cells with fewer than
    this many regular cycles are dropped before any masks or targets are
    computed. Rationale: with very few cycles the cell hasn't yet
    expressed its degradation behaviour, so it's neither a reliable
    "in-testing" example nor a meaningful "faded" one. Upstream features
    pipeline filters at n_regular >= 5; cell_lifetime tightens this to 6
    to remove the boundary cases.
    """
    if N not in SUPPORTED_N:
        raise ValueError(f"N must be one of {SUPPORTED_N} (got {N})")
    if baseline_cycle not in SUPPORTED_BASELINE:
        raise ValueError(
            f"baseline_cycle must be one of {SUPPORTED_BASELINE} (got {baseline_cycle})"
        )

    root = _resolve_preprocess_root(preprocess_root)
    bundle = root / "datasets" / f"{db_version}_b{baseline_cycle}"
    features_path = bundle / "cell_features.parquet"
    labels_path = bundle / "cell_labels.parquet"
    if not features_path.exists() or not labels_path.exists():
        raise FileNotFoundError(
            f"preprocess bundle not found at {bundle}. Generate it via "
            f"`python {root}/preprocess.py --all --baseline-cycle {baseline_cycle} "
            f"--db-version {db_version}`."
        )

    feature_names = _load_feature_subset(feature_subset, preprocess_root)
    features = pl.read_parquet(features_path)
    labels = pl.read_parquet(labels_path)

    missing = [f for f in feature_names if f not in features.columns]
    if missing:
        raise KeyError(f"requested features missing from {features_path.name}: {missing}")
    leakage = set(feature_names) & _LABEL_COLUMNS_DENYLIST
    if leakage:
        raise ValueError(
            f"feature subset {feature_subset!r} contains label-like columns: {leakage}"
        )

    joined = labels.join(features, on="cell_name", how="inner").to_pandas()
    if drop_excluded:
        joined = joined.loc[joined["status"] != "excluded"].reset_index(drop=True)
    if min_n_regular > 0:
        joined = joined.loc[joined["n_regular"] >= min_n_regular].reset_index(drop=True)

    X = joined[feature_names].copy().reset_index(drop=True)
    status = joined["status"].to_numpy()
    last_fade = joined["last_fade_cycle"].to_numpy()
    n_regular = joined["n_regular"].to_numpy()

    label_mask = joined[f"trainable_n{N}"].to_numpy().astype(bool)
    y_class = (joined[f"label_n{N}"].to_numpy() == "pass").astype(np.int8)

    event = (status == "faded")
    faded_mask = event.copy()
    # time = last_fade_cycle if faded, else n_regular (right-censored at follow-up).
    # last_fade is float (NaN for non-faded); coerce to int via fillna(n_regular).
    time = np.where(event, last_fade, n_regular).astype(np.int64)

    y_cycle = np.where(event, last_fade.astype(float), np.nan)

    cohorts = joined["cohort"].to_numpy()
    cell_names = joined["cell_name"].to_numpy()

    return CycleLifeDataset(
        X=X, y_class=y_class, y_cycle=y_cycle, event=event, time=time,
        label_mask=label_mask, faded_mask=faded_mask,
        cohorts=cohorts, cell_names=cell_names,
        feature_names=feature_names,
        N=N, baseline_cycle=baseline_cycle, db_version=db_version,
        source_dir=bundle,
    )

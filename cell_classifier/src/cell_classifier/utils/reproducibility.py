"""Snapshot upstream inputs + resolved config into the run folder.

Goal: a single run folder is independently reproducible — given the
folder, you can re-train the same model without reaching back into
``ml_label_preprocess/``. We copy:

  - ``cell_features.parquet`` and ``cell_labels.parquet`` from the
    preprocess bundle (the exact rows the trainer saw)
  - the bundle's ``manifest.json`` (preprocess provenance)
  - ``column_roles.yaml`` (records which columns are features vs labels)
  - ``resolved_config.yaml`` (the hashed config dict — every CLI / sweep
    override is already folded into this)
"""

from __future__ import annotations

import shutil
from pathlib import Path
from typing import Any

import yaml


_BUNDLE_FILES = ("cell_features.parquet", "cell_labels.parquet", "manifest.json")


def snapshot_inputs(
    out_dir: Path, source_bundle: Path, column_roles_yaml: Path,
) -> Path:
    """Copy parquets + manifests into ``out_dir/inputs/``."""
    inputs_dir = out_dir / "inputs"
    inputs_dir.mkdir(parents=True, exist_ok=True)
    for name in _BUNDLE_FILES:
        src = source_bundle / name
        if src.exists():
            shutil.copy2(src, inputs_dir / name)
    if column_roles_yaml.exists():
        shutil.copy2(column_roles_yaml, inputs_dir / "column_roles.yaml")
    return inputs_dir


def write_resolved_config(out_dir: Path, config: dict[str, Any]) -> Path:
    """Persist the resolved config as YAML alongside ``manifest.json``."""
    path = out_dir / "resolved_config.yaml"
    path.write_text(
        yaml.safe_dump(config, sort_keys=True, default_flow_style=False)
    )
    return path

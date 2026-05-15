"""Manifest read/write helpers + resolved-config SHA-256.

`hash_resolved_config(config)` is the canonical idempotency hash. Two runs
with the same hash should be expected to produce the same outputs (up to
non-deterministic floating-point sums). The hash covers:

  - mode
  - model + model_fixed_params (model class name is part of the hash; the
    hyperparameter search space is implied by the class)
  - all four data axes (N, db_version, baseline_cycle, feature_subset)
  - preprocessing
  - tuning (protocol + n_trials + inner_cv_folds + outer_cv_folds + test_frac
    + optimize_metric, as a single nested dict)
  - hp_provenance (source + source_run_slug + representative_strategy)
  - resolved seeds list
  - package version + sklearn version + optuna version

Any change to `_HASH_FIELDS` — rename, add, remove, reorder — MUST bump
`schema_version` in `build_manifest`, regenerate `_HASH_FIELDS_FINGERPRINT`,
and document the change in CLAUDE.md. The module-load assertion below is the
guard that catches silent drift.
"""

from __future__ import annotations

import hashlib
import importlib.metadata
import json
import platform
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


def _versions() -> dict[str, str]:
    out: dict[str, str] = {"python": platform.python_version()}
    for pkg in ("cell_classifier", "scikit-learn", "optuna", "shap"):
        try:
            out[pkg] = importlib.metadata.version(pkg)
        except importlib.metadata.PackageNotFoundError:
            out[pkg] = "unknown"
    return out


_HASH_FIELDS: tuple[str, ...] = (
    "mode",
    "model",
    "model_fixed_params",
    "N",
    "db_version",
    "baseline_cycle",
    "feature_subset",
    "preprocessing",
    "tuning",
    "hp_provenance",
    "seeds",
)
_HASH_VERSION_FIELDS = ("versions",)


def _compute_hash_fields_fingerprint(fields: tuple[str, ...]) -> str:
    canon = json.dumps(list(fields), separators=(",", ":"))
    return "sha256:" + hashlib.sha256(canon.encode("utf-8")).hexdigest()


_HASH_FIELDS_FINGERPRINT = (
    "sha256:0ca97172a3d93c9d3acf7cfe7ade8cfca47662e90e0c7d789c16d30da25aae20"
)

assert _compute_hash_fields_fingerprint(_HASH_FIELDS) == _HASH_FIELDS_FINGERPRINT, (
    "_HASH_FIELDS changed without updating fingerprint. "
    "Bump schema_version in build_manifest(), regenerate "
    "_HASH_FIELDS_FINGERPRINT, and document the change in CLAUDE.md."
)


def hash_resolved_config(config: dict[str, Any]) -> str:
    """SHA-256 of canonical JSON of the hashable subset of the resolved config."""
    payload = {k: config.get(k) for k in _HASH_FIELDS}
    payload["versions"] = config.get("versions") or _versions()
    canon = json.dumps(payload, sort_keys=True, separators=(",", ":"), default=str)
    return hashlib.sha256(canon.encode("utf-8")).hexdigest()


def hash_resolved_config_ignoring_versions(config: dict[str, Any]) -> str:
    """Same as hash_resolved_config but drops the `versions` block."""
    payload = {k: config.get(k) for k in _HASH_FIELDS}
    canon = json.dumps(payload, sort_keys=True, separators=(",", ":"), default=str)
    return hashlib.sha256(canon.encode("utf-8")).hexdigest()


def build_manifest(
    *,
    config: dict[str, Any],
    runtime_seconds: float,
    n_cells_labeled_trainable: int,
    n_cells_scored: int,
    preprocess_manifest: dict[str, Any] | None = None,
    shap_summary_scope: str = "test_set",
) -> dict[str, Any]:
    """Build the manifest dict for a completed run.

    Reads `tuning` and `hp_provenance` directly from the resolved config —
    both blocks are populated by the CLI / pipeline orchestrator before this
    function is called. No flat `tuning_protocol` / `params_source` fields
    are produced (schema_version 2).
    """
    versions = _versions()
    config_for_hash = {**config, "versions": versions}
    tuning_block: dict[str, Any] = dict(config.get("tuning") or {})
    hp_prov_block: dict[str, Any] = dict(config.get("hp_provenance") or {})
    out: dict[str, Any] = {
        "schema_version": 2,
        "slug": config["slug"],
        "mode": config["mode"],
        "model": config["model"],
        "model_fixed_params": config.get("model_fixed_params", {}),
        "N": config["N"],
        "db_version": config["db_version"],
        "baseline_cycle": config["baseline_cycle"],
        "feature_subset": config["feature_subset"],
        "preprocessing": config.get("preprocessing", {}),
        "tuning": {
            "protocol": tuning_block.get("protocol"),
            "n_trials": tuning_block.get("n_trials"),
            "inner_cv_folds": tuning_block.get("inner_cv_folds"),
            "outer_cv_folds": tuning_block.get("outer_cv_folds"),
            "test_frac": tuning_block.get("test_frac"),
            "optimize_metric": tuning_block.get("optimize_metric"),
        },
        "hp_provenance": {
            "source": hp_prov_block.get("source"),
            "source_run_slug": hp_prov_block.get("source_run_slug"),
            "representative_strategy": hp_prov_block.get("representative_strategy"),
        },
        "seeds": config.get("seeds", []),
        "n_seeds": len(config.get("seeds", [])),
        "n_cells_labeled_trainable": int(n_cells_labeled_trainable),
        "n_cells_scored": int(n_cells_scored),
        "positive_class": "pass (good cell, survived past N cycles)",
        "negative_class": "bad (faded at or before N cycles)",
        "preprocess_source": (
            str(preprocess_manifest.get("slug")) if preprocess_manifest else None
        ),
        "preprocess_schema_version": (
            int(preprocess_manifest["schema_version"]) if preprocess_manifest else None
        ),
        "preprocess_column_roles_sha256": (
            preprocess_manifest.get("column_roles_sha256") if preprocess_manifest else None
        ),
        "shap_summary_scope": shap_summary_scope,
        "generated_at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "runtime_seconds": round(float(runtime_seconds), 1),
        "versions": versions,
    }
    out["resolved_config_sha256"] = hash_resolved_config(config_for_hash)
    return out


def write_manifest(out_dir: Path, manifest: dict[str, Any]) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / "manifest.json"
    path.write_text(json.dumps(manifest, indent=2, default=str) + "\n")
    return path


def read_manifest(out_dir: Path) -> dict[str, Any] | None:
    path = out_dir / "manifest.json"
    if not path.exists():
        return None
    return json.loads(path.read_text())

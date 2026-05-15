"""Manifest hash stability + write/read round-trip + fingerprint guard."""

import json
from pathlib import Path

from cell_classifier.utils.manifest import (
    _HASH_FIELDS, _HASH_FIELDS_FINGERPRINT, _compute_hash_fields_fingerprint,
    build_manifest, hash_resolved_config,
    hash_resolved_config_ignoring_versions, read_manifest, write_manifest,
)


def _config():
    return {
        "slug": "rf__N300__A2.2_b1__fs_cv",
        "mode": "validation",
        "model": "random_forest",
        "model_fixed_params": {"n_jobs": -1},
        "N": 300,
        "db_version": "A2.2",
        "baseline_cycle": 1,
        "feature_subset": "fs_cv",
        "preprocessing": {"imputer_strategy": "median"},
        "tuning": {
            "protocol": "nested_cv",
            "n_trials": 100,
            "inner_cv_folds": 5,
            "outer_cv_folds": 5,
            "test_frac": None,
            "optimize_metric": "f1",
        },
        "hp_provenance": {
            "source": None,
            "source_run_slug": None,
            "representative_strategy": None,
        },
        "seeds": [1, 2, 3],
        "out_root": "/tmp/x",
        "versions": {"python": "3.11.0"},  # pin versions for stable hash
    }


def test_hash_stable():
    c1 = _config()
    c2 = _config()
    assert hash_resolved_config(c1) == hash_resolved_config(c2)


def test_hash_changes_with_seed():
    c1 = _config()
    c2 = {**c1, "seeds": [1, 2, 4]}
    assert hash_resolved_config(c1) != hash_resolved_config(c2)


def test_hash_changes_with_tuning_protocol():
    c1 = _config()
    c2 = {**c1, "tuning": {**c1["tuning"], "protocol": "tune_inner_cv"}}
    assert hash_resolved_config(c1) != hash_resolved_config(c2)


def test_hash_changes_with_hp_provenance():
    c1 = _config()
    c2 = {**c1, "hp_provenance": {**c1["hp_provenance"], "source": "retune"}}
    assert hash_resolved_config(c1) != hash_resolved_config(c2)


def test_hash_ignore_versions():
    c1 = _config()
    c2 = {**c1, "versions": {"python": "3.12.0"}}
    assert hash_resolved_config(c1) != hash_resolved_config(c2)
    assert hash_resolved_config_ignoring_versions(c1) == hash_resolved_config_ignoring_versions(c2)


def test_write_read_round_trip(tmp_path: Path):
    c = _config()
    manifest = build_manifest(
        config=c, runtime_seconds=1.0,
        n_cells_labeled_trainable=100, n_cells_scored=100,
    )
    write_manifest(tmp_path, manifest)
    loaded = read_manifest(tmp_path)
    assert loaded is not None
    assert loaded["slug"] == c["slug"]
    assert loaded["schema_version"] == 2
    assert loaded["tuning"]["protocol"] == "nested_cv"
    assert loaded["tuning"]["outer_cv_folds"] == 5
    assert loaded["hp_provenance"]["source"] is None
    assert "resolved_config_sha256" in loaded


def test_no_flat_legacy_fields_in_manifest(tmp_path: Path):
    """Schema-v2 manifest must NOT carry the v1 flat tuning/provenance fields."""
    c = _config()
    manifest = build_manifest(
        config=c, runtime_seconds=1.0,
        n_cells_labeled_trainable=100, n_cells_scored=100,
    )
    for forbidden in ("tuning_protocol", "outer_k", "params_source",
                       "params_source_run", "split", "tune"):
        assert forbidden not in manifest, (
            f"v1 flat field {forbidden!r} leaked into v2 manifest"
        )


def test_hash_fields_fingerprint_matches():
    """The frozen fingerprint must equal a fresh computation."""
    assert _compute_hash_fields_fingerprint(_HASH_FIELDS) == _HASH_FIELDS_FINGERPRINT


def test_hash_fields_fingerprint_catches_reorder():
    """Reordering _HASH_FIELDS must produce a different fingerprint."""
    reordered = tuple(reversed(_HASH_FIELDS))
    assert _compute_hash_fields_fingerprint(reordered) != _HASH_FIELDS_FINGERPRINT

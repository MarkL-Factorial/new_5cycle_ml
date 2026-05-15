"""Discover existing runs by globbing manifest.json under results/runs/{mode}/.

Filters apply on manifest keys: mode, model, N, db_version, baseline_cycle,
feature_subset. None means "don't filter". Symlinks (the ``{slug}`` →
``{slug}__{timestamp}`` pointers) are skipped so each real run is yielded
exactly once.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Optional


def _default_out_root() -> Path:
    """Project-local results/ if running from cell_classifier/ root; else cwd/results."""
    # cell_classifier/src/cell_classifier/utils/discover.py
    return Path(__file__).resolve().parents[3] / "results"


def find_runs(
    out_root: Optional[Path] = None,
    *,
    mode: Optional[str] = None,
    model: Optional[str] = None,
    N: Optional[int] = None,
    db_version: Optional[str] = None,
    baseline_cycle: Optional[int] = None,
    feature_subset: Optional[str] = None,
) -> list[dict[str, Any]]:
    """Glob manifest.json under out_root/runs/{mode?}/{slug}/.

    Returns a list of dicts, each = {"path": Path, **manifest_keys}.
    """
    root = Path(out_root) if out_root is not None else _default_out_root()
    base = root / "runs"
    if not base.exists():
        return []

    if mode is not None:
        scopes = [base / mode]
    else:
        scopes = [d for d in base.iterdir() if d.is_dir()]

    out: list[dict[str, Any]] = []
    for scope in scopes:
        if not scope.exists():
            continue
        for run_dir in sorted(scope.iterdir()):
            # Skip {slug} symlinks — the real timestamped folder shows up
            # separately and we don't want duplicates.
            if run_dir.is_symlink():
                continue
            manifest_path = run_dir / "manifest.json"
            if not manifest_path.exists():
                continue
            manifest = json.loads(manifest_path.read_text())
            if _matches(manifest, model, N, db_version, baseline_cycle, feature_subset):
                out.append({"path": run_dir, **manifest})
    return out


def _matches(manifest, model, N, db_version, baseline_cycle, feature_subset) -> bool:
    if model is not None and manifest.get("model") != model:
        return False
    if N is not None and manifest.get("N") != N:
        return False
    if db_version is not None and manifest.get("db_version") != db_version:
        return False
    if baseline_cycle is not None and manifest.get("baseline_cycle") != baseline_cycle:
        return False
    if feature_subset is not None and manifest.get("feature_subset") != feature_subset:
        return False
    return True

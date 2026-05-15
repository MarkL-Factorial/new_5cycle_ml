"""Shared helpers for labels.py and features.py.

Owns: env-derived paths, cohort assignment, annotation-JSON iteration,
output writing, and the v3 dataset-bundle / manifest scheme.

v3 layout: every output bundle is keyed on (db_version, baseline_cycle)
and lives at ``datasets/{db_version}_b{baseline_cycle}/``. Each bundle
carries a ``manifest.json`` with provenance (db_version, baseline_cycle,
annot_dir, generated_at, stages_populated, column_roles_sha256, …).

Author: Mark Liao (Sheng-Lun Liao)
"""
from __future__ import annotations

import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterator

import polars as pl

DEFAULT_ANNOT_DIR = "/mnt/data/mliao/battery-ml-workbench/data/A2.2/annotations"
ANNOT_DIR = Path(os.getenv("BAT_ANNOT_DIR", DEFAULT_ANNOT_DIR))
DATASETS_DIR = Path(__file__).resolve().parent / "datasets"


def db_version_from_path(annot_dir: Path = ANNOT_DIR) -> str:
    """Parse the DB version tag from the annotation directory path.

    Convention: ``/…/<db_version>/annotations``. For the default
    ``/mnt/data/mliao/battery-ml-workbench/data/A2.2/annotations``,
    this returns ``'A2.2'``.

    Raises ValueError when the path does not end in '.../annotations',
    so the caller knows to pass ``--db-version`` explicitly.
    """
    if annot_dir.name != "annotations":
        raise ValueError(
            f"expected ANNOT_DIR to end in '.../annotations', got {annot_dir}; "
            f"pass --db-version explicitly"
        )
    return annot_dir.parent.name


def dataset_dir_for(db_version: str, baseline_cycle: int) -> Path:
    """Return ``datasets/{db_version}_b{baseline_cycle}/`` (created on demand)."""
    sub = DATASETS_DIR / f"{db_version}_b{baseline_cycle}"
    sub.mkdir(parents=True, exist_ok=True)
    return sub


def _cohort(cell_name: str) -> str:
    return "0MC" if cell_name.startswith("0MC") else "AR"


def iter_annotations() -> Iterator[tuple[Path, dict]]:
    """Yield (path, parsed_json) for every annotation JSON in ANNOT_DIR.

    Malformed JSON is logged to stderr and skipped (matches preprocess.py's
    prior behavior). Caller iterates; missing dir is a hard fail.
    """
    if not ANNOT_DIR.exists():
        print(f"ERROR: annotations dir not found: {ANNOT_DIR}", file=sys.stderr)
        sys.exit(1)
    for path in sorted(ANNOT_DIR.glob("*.annotations.json")):
        try:
            d = json.loads(path.read_text())
        except json.JSONDecodeError as exc:
            print(f"WARNING: skipping malformed JSON {path.name}: {exc}", file=sys.stderr)
            continue
        yield path, d


def iter_regulars(annot: dict) -> list[dict]:
    """Return regular_cd events sorted by regular_cycle (ascending).

    Filters to event_kind == 'regular_cd' with non-null regular_cycle and
    non-null capacity_discharge_ah. Caller may further filter / slice.
    """
    return sorted(
        (
            e for e in annot.get("cd_events", [])
            if e.get("event_kind") == "regular_cd"
            and e.get("regular_cycle") is not None
            and e.get("capacity_discharge_ah") is not None
        ),
        key=lambda e: e["regular_cycle"],
    )


def write_outputs(
    df: pl.DataFrame,
    basename: str,
    out_dir: Path,
) -> tuple[Path, Path]:
    """Write df as parquet + csv pair into out_dir. Returns (parquet, csv).

    ``out_dir`` is REQUIRED in v3 (no implicit default) — callers pass
    ``dataset_dir_for(db_version, baseline_cycle)``.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    parquet_path = out_dir / f"{basename}.parquet"
    csv_path = out_dir / f"{basename}.csv"
    df.write_parquet(parquet_path)
    df.write_csv(csv_path)
    return parquet_path, csv_path


def write_manifest(out_dir: Path, fragment: dict[str, Any]) -> Path:
    """Merge ``fragment`` into ``out_dir/manifest.json`` (create if absent).

    Lets labels.main() and features.main() each contribute provenance
    keys without clobbering the other's. ``generated_at`` is always
    rewritten to the current UTC time, and ``stages_populated`` is the
    union (sorted) of any existing stages and any stages in the fragment.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / "manifest.json"
    existing: dict[str, Any] = (
        json.loads(path.read_text()) if path.exists() else {}
    )
    stages_existing = existing.get("stages_populated", []) or []
    stages_fragment = fragment.get("stages_populated", []) or []
    stages_merged = sorted(set(stages_existing) | set(stages_fragment))
    merged: dict[str, Any] = {**existing, **fragment}
    merged["stages_populated"] = stages_merged
    merged["generated_at"] = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    path.write_text(json.dumps(merged, indent=2) + "\n")
    return path

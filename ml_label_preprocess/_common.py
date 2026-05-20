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

# Set once per Python process. labels.main() + features.main() invoked
# in the same process (e.g. via preprocess.py --all) share one snapshot;
# standalone runs each get their own. Local time matches the
# cell_lifetime/results/run/ convention; manifest.json still records
# generated_at in UTC.
_PROCESS_TIMESTAMP = datetime.now().strftime("%Y%m%d_%H%M")


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
    """Return the per-process timestamped snapshot dir for this bundle.

    Layout: ``datasets/{db_version}_b{baseline_cycle}/{db_version}_b{baseline_cycle}_{ts}/``
    where ``ts`` is the YYYYMMDD_HHMM time at which this Python process
    first imported _common. Calls from the same process always resolve
    to the same snapshot dir, so labels + features run together (via
    preprocess.py) share one snapshot.

    Use ``promote_to_latest(snapshot_dir)`` to update the
    ``{bundle}_latest`` symlink after writing all stage outputs.
    """
    bundle = DATASETS_DIR / f"{db_version}_b{baseline_cycle}"
    snapshot = bundle / f"{db_version}_b{baseline_cycle}_{_PROCESS_TIMESTAMP}"
    snapshot.mkdir(parents=True, exist_ok=True)
    return snapshot


def promote_to_latest(snapshot_dir: Path) -> Path:
    """Update ``{bundle}_latest`` symlink to point at ``snapshot_dir``.

    The symlink target is *relative* (just the snapshot dir name, not
    its absolute path) so the dataset tree is portable across machines
    and mountpoints. Existing symlink or file at the target is removed
    first; this is atomic enough for our single-writer workflow.

    Returns the symlink path.
    """
    bundle_dir = snapshot_dir.parent
    latest_link = bundle_dir / f"{bundle_dir.name}_latest"
    if latest_link.is_symlink() or latest_link.exists():
        latest_link.unlink()
    latest_link.symlink_to(snapshot_dir.name, target_is_directory=True)
    return latest_link


def migrate_legacy_bundle(bundle_dir: Path) -> Path | None:
    """One-shot helper: convert a legacy flat bundle into the snapshot layout.

    If ``bundle_dir`` still has the pre-snapshot layout (parquet/csv/
    manifest at root, no timestamped subdirs), move every regular file
    into ``{bundle_dir}/{bundle_dir.name}_{ts}_legacy/`` where ``ts`` is
    derived from the existing ``manifest.json`` ``generated_at`` field.
    Then set ``{bundle_dir}_latest`` to point at the new snapshot.

    Returns the new snapshot path, or None if there was nothing to
    migrate (bundle empty or already migrated).
    """
    if not bundle_dir.exists() or not bundle_dir.is_dir():
        return None

    manifest_path = bundle_dir / "manifest.json"
    flat_files = [p for p in bundle_dir.iterdir() if p.is_file()]
    if not flat_files:
        return None  # Already migrated (only subdirs / symlinks present).

    # Derive a snapshot timestamp from the manifest's generated_at (UTC),
    # converted to local-time YYYYMMDD_HHMM to match the new convention.
    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text())
        gen_at = manifest.get("generated_at")  # e.g. "2026-05-20T17:52:09Z"
        if gen_at and gen_at.endswith("Z"):
            dt_utc = datetime.strptime(gen_at, "%Y-%m-%dT%H:%M:%SZ").replace(
                tzinfo=timezone.utc
            )
            ts = dt_utc.astimezone().strftime("%Y%m%d_%H%M")
        else:
            # No usable timestamp — use file mtime instead.
            ts = datetime.fromtimestamp(manifest_path.stat().st_mtime).strftime(
                "%Y%m%d_%H%M"
            )
    else:
        # Fall back to the newest file's mtime.
        newest = max(flat_files, key=lambda p: p.stat().st_mtime)
        ts = datetime.fromtimestamp(newest.stat().st_mtime).strftime("%Y%m%d_%H%M")

    snapshot = bundle_dir / f"{bundle_dir.name}_{ts}_legacy"
    snapshot.mkdir(parents=True, exist_ok=False)
    for f in flat_files:
        f.rename(snapshot / f.name)

    promote_to_latest(snapshot)
    return snapshot


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

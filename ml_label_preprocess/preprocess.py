"""CLI dispatcher for ML label + feature preprocessing (v3).

Two pipelines, both keyed on the per-cell `regular_cycle` index emitted
by the annotation toolkit:

  labels    → datasets/{db_version}_b{baseline_cycle}/cell_labels.{parquet,csv}
              (always 1 row per JSON)
  features  → datasets/{db_version}_b{baseline_cycle}/cell_features.{parquet,csv}
              (1 row per cell with n_regular >= 5)

The features pipeline additionally writes
``datasets/{db_version}_b{baseline_cycle}/cell_features_status.csv``
(per-cycle KWW fit success/error log). Each bundle dir also gets a
``manifest.json`` recording provenance (db_version, baseline_cycle,
annot_dir, generated_at, stages_populated, column_roles_sha256).

v3 changes (vs v2):
  - Output bundles live under ``datasets/{slug}/`` instead of
    ``out/`` / ``out/baseline_{N}/`` (no back-compat carve-out).
  - DB version is auto-parsed from ANNOT_DIR (e.g.
    ``/.../A2.2/annotations`` → ``A2.2``). Override with --db-version.
  - Each bundle carries a manifest.json with provenance keys.

Usage:
    python preprocess.py                          # labels @ auto db + baseline 1
    python preprocess.py --labels                 # explicit labels
    python preprocess.py --features               # features only
    python preprocess.py --all                    # both pipelines
    python preprocess.py --all --baseline-cycle 3 # both, baseline 3
    python preprocess.py --all --db-version FOO   # both, override DB tag
    python preprocess.py --selftest               # run all selftests
    python preprocess.py --features --cells AR-3420 0MC2-251022-001
                                                  # features subset (debug)

Column-role manifest at ``ml_label_preprocess_v3/column_roles.yaml`` is
the source of truth for which output column is meta / label / feature /
quality. Downstream ML training scripts MUST consult it to prevent data
leakage.

Author: Mark Liao (Sheng-Lun Liao)
"""
from __future__ import annotations

import argparse
import sys


def _parse_args(argv: list[str]) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        prog="preprocess.py",
        description="Run the labels and/or features pipelines.",
    )
    p.add_argument("--labels", action="store_true",
                   help="run the labels pipeline (default if no flag given)")
    p.add_argument("--features", action="store_true",
                   help="run the features pipeline")
    p.add_argument("--all", action="store_true",
                   help="run both labels and features")
    p.add_argument("--selftest", action="store_true",
                   help="run all selftests; exit non-zero on any failure")
    p.add_argument("--cells", nargs="+", default=None,
                   help="restrict the features pipeline to these cell names "
                        "(debugging; ignored by labels)")
    p.add_argument("--baseline-cycle", type=int, default=1, choices=[1, 2, 3, 4],
                   help="regular_cycle ordinal used as the retention "
                        "denominator (default 1). Limited to 1..4 because the "
                        "post-baseline window must have >= 2 cycles for "
                        "Tier-B std. Output bundle path is "
                        "datasets/{db_version}_b{N}/.")
    p.add_argument("--db-version", default=None,
                   help="DB version tag for output dirname (e.g. 'A2.2'). "
                        "Default: auto-parsed from ANNOT_DIR — for "
                        "'/.../A2.2/annotations' this is 'A2.2'. Override when "
                        "ANNOT_DIR doesn't follow that convention.")
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv if argv is not None else sys.argv[1:])

    if args.selftest:
        import labels
        import features
        import _common
        fail = 0
        fail += _selftest_common()
        fail += labels.selftest()
        fail += features.selftest()
        if fail:
            print(f"\nTotal selftest failures: {fail}", file=sys.stderr)
            return 2
        return 0

    from _common import db_version_from_path
    db_version = args.db_version or db_version_from_path()

    run_labels = args.labels or args.all or not (args.labels or args.features or args.all)
    run_features = args.features or args.all

    if run_labels:
        import labels
        labels.main(baseline_cycle=args.baseline_cycle, db_version=db_version)

    if run_features:
        import features
        features.main(
            cells=args.cells,
            baseline_cycle=args.baseline_cycle,
            db_version=db_version,
        )

    return 0


def _selftest_common() -> int:
    """Selftests for the v3 path / manifest helpers in _common.py."""
    from pathlib import Path
    import json
    import tempfile

    from _common import (
        dataset_dir_for,
        db_version_from_path,
        write_manifest,
    )

    print("Self-test (common helpers):")
    fail = 0

    # db_version_from_path happy path
    got = db_version_from_path(Path("/x/y/A2.2/annotations"))
    if got != "A2.2":
        print(f"  [FAIL] db_version_from_path happy: got {got!r} expected 'A2.2'")
        fail += 1

    got = db_version_from_path(Path("/some/where/A9.99/annotations"))
    if got != "A9.99":
        print(f"  [FAIL] db_version_from_path two-digit minor: got {got!r}")
        fail += 1

    # db_version_from_path error path
    raised = False
    try:
        db_version_from_path(Path("/x/y/annotations/foo"))
    except ValueError:
        raised = True
    if not raised:
        print("  [FAIL] db_version_from_path: missing 'annotations' tail did not raise")
        fail += 1

    # dataset_dir_for creates the dir with the expected slug
    target = dataset_dir_for("A9.9", 4)
    if target.name != "A9.9_b4":
        print(f"  [FAIL] dataset_dir_for slug: got {target.name!r}")
        fail += 1
    if not target.is_dir():
        print(f"  [FAIL] dataset_dir_for did not create dir: {target}")
        fail += 1

    # write_manifest merge round-trip in a sandbox dir
    with tempfile.TemporaryDirectory() as tmpdir:
        sandbox = Path(tmpdir)
        write_manifest(sandbox, {
            "schema_version": 1,
            "db_version": "TESTDB",
            "baseline_cycle": 1,
            "stages_populated": ["labels"],
        })
        first = json.loads((sandbox / "manifest.json").read_text())
        if first.get("stages_populated") != ["labels"]:
            print(f"  [FAIL] write_manifest first stages: {first.get('stages_populated')}")
            fail += 1

        write_manifest(sandbox, {
            "n_cells_features": 42,
            "stages_populated": ["features"],
        })
        second = json.loads((sandbox / "manifest.json").read_text())
        if second.get("stages_populated") != ["features", "labels"]:
            print(f"  [FAIL] write_manifest merged stages: {second.get('stages_populated')}")
            fail += 1
        if second.get("db_version") != "TESTDB":
            print("  [FAIL] write_manifest dropped first fragment's db_version")
            fail += 1
        if second.get("n_cells_features") != 42:
            print("  [FAIL] write_manifest dropped second fragment's n_cells_features")
            fail += 1
        if second.get("schema_version") != 1:
            print("  [FAIL] write_manifest dropped schema_version")
            fail += 1

    # Clean up the A9.9_b4 sandbox dir we made above
    try:
        target.rmdir()
        target.parent.rmdir()  # remove datasets/ if now empty
    except OSError:
        pass  # not empty (real run output sitting alongside) — fine

    if fail:
        print(f"\n{fail} common-helper self-test cases FAILED")
    else:
        print("All common-helper self-test cases PASSED")
    return fail


if __name__ == "__main__":
    sys.exit(main())

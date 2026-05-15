"""find_runs() filters."""

import json
from pathlib import Path

from cell_classifier.utils.discover import find_runs


def _make_run(root: Path, mode: str, slug: str, manifest: dict) -> None:
    d = root / "runs" / mode / slug
    d.mkdir(parents=True, exist_ok=True)
    (d / "manifest.json").write_text(json.dumps({"slug": slug, **manifest}))


def test_find_runs_filters(tmp_path: Path):
    _make_run(tmp_path, "validation", "rf__N300__A2.2_b1__fs_cv",
              {"mode": "validation", "model": "random_forest", "N": 300,
               "db_version": "A2.2", "baseline_cycle": 1, "feature_subset": "fs_cv"})
    _make_run(tmp_path, "validation", "rf__N200__A2.2_b1__fs_cv",
              {"mode": "validation", "model": "random_forest", "N": 200,
               "db_version": "A2.2", "baseline_cycle": 1, "feature_subset": "fs_cv"})
    _make_run(tmp_path, "production", "rf__N300__A2.2_b1__fs_cv",
              {"mode": "production", "model": "random_forest", "N": 300,
               "db_version": "A2.2", "baseline_cycle": 1, "feature_subset": "fs_cv"})

    all_runs = find_runs(out_root=tmp_path)
    assert len(all_runs) == 3

    val = find_runs(out_root=tmp_path, mode="validation")
    assert len(val) == 2

    n300 = find_runs(out_root=tmp_path, N=300)
    assert len(n300) == 2

    n300_val = find_runs(out_root=tmp_path, mode="validation", N=300)
    assert len(n300_val) == 1


def test_find_runs_empty(tmp_path: Path):
    assert find_runs(out_root=tmp_path) == []

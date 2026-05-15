"""Slug round-trip + run_dir shape."""

from pathlib import Path

import pytest

from cell_classifier.utils.paths import parse_slug, run_dir, run_slug


@pytest.mark.parametrize(
    "model,N,db,bc,fs,expected",
    [
        ("random_forest", 300, "A2.2", 1, "fs_cv", "rf__N300__A2.2_b1__fs_cv"),
        ("random_forest", 200, "A2.2", 3, "fs_cv", "rf__N200__A2.2_b3__fs_cv"),
        ("random_forest", 400, "A2.3", 4, "fs_all", "rf__N400__A2.3_b4__fs_all"),
        ("ebm", 300, "A2.2", 1, "fs_cv", "ebm__N300__A2.2_b1__fs_cv"),
    ],
)
def test_run_slug(model, N, db, bc, fs, expected):
    assert run_slug(model, N, db, bc, fs) == expected


def test_round_trip():
    slug = run_slug("random_forest", 300, "A2.2", 1, "fs_cv")
    parsed = parse_slug(slug)
    assert parsed["model"] == "random_forest"
    assert parsed["N"] == 300
    assert parsed["db_version"] == "A2.2"
    assert parsed["baseline_cycle"] == 1
    assert parsed["feature_subset"] == "fs_cv"


def test_parse_slug_rejects_bad():
    with pytest.raises(ValueError):
        parse_slug("not_a_slug")
    with pytest.raises(ValueError):
        parse_slug("rf__N300")


def test_run_dir():
    p = run_dir(Path("/tmp/x"), "validation", "rf__N300__A2.2_b1__fs_cv")
    assert p == Path("/tmp/x/runs/validation/rf__N300__A2.2_b1__fs_cv")
    with pytest.raises(ValueError):
        run_dir(Path("/tmp/x"), "bogus", "slug")

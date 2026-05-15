"""Data-leakage canary: no fs_cv feature is named like a cohort aggregate."""

import re

import yaml

from cell_classifier.data.loader import column_roles_path


_COHORT_AGG_PATTERN = re.compile(r"(by_cohort|cohort_)", flags=re.IGNORECASE)


def test_no_cohort_aggregates_in_fs_cv():
    manifest = yaml.safe_load(column_roles_path().read_text())
    fs_cv = manifest["subsets"]["fs_cv"]["members"]
    leaks = [f for f in fs_cv if _COHORT_AGG_PATTERN.search(f)]
    assert leaks == [], f"cohort-aggregate-looking features in fs_cv: {leaks}"

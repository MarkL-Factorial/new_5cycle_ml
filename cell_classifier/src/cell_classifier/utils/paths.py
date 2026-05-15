"""Run-directory slug helpers.

The slug encodes the five axes that determine an ML run:
  model, N, db_version, baseline_cycle, feature_subset

Example: ``rf__N300__A2.2_b1__fs_cv``

On disk, each invocation lands in a timestamped folder
``{slug}__{YYYYMMDD_HHMMSS}/``. A ``{slug}`` symlink alongside it points at
the most recently completed run, so all readers (idempotency check, sweep
aggregation, ``from_validation_run`` production lookup) keep using the
unchanged ``{out_root}/runs/{mode}/{slug}/`` path and silently follow the
symlink.

All slug + path logic lives here — no other module constructs run paths
from scratch.
"""

from __future__ import annotations

import os
import re
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional, TypedDict


class RunAxes(TypedDict):
    model: str
    N: int
    db_version: str
    baseline_cycle: int
    feature_subset: str


_MODEL_SHORT = {
    "random_forest": "rf",
    "ebm": "ebm",
    "bart": "bart",
}
_SHORT_TO_MODEL = {v: k for k, v in _MODEL_SHORT.items()}

_SLUG_RE = re.compile(
    r"^(?P<model_short>[a-z]+)"
    r"__N(?P<N>\d+)"
    r"__(?P<db_version>[A-Za-z0-9.]+)_b(?P<baseline_cycle>\d+)"
    r"__(?P<feature_subset>[A-Za-z0-9_]+)$"
)


def run_slug(model: str, N: int, db_version: str, baseline_cycle: int,
             feature_subset: str) -> str:
    short = _MODEL_SHORT.get(model, model)
    return f"{short}__N{N}__{db_version}_b{baseline_cycle}__{feature_subset}"


def parse_slug(slug: str) -> RunAxes:
    m = _SLUG_RE.match(slug)
    if m is None:
        raise ValueError(f"slug does not match expected pattern: {slug!r}")
    model_short = m.group("model_short")
    model = _SHORT_TO_MODEL.get(model_short, model_short)
    return RunAxes(
        model=model,
        N=int(m.group("N")),
        db_version=m.group("db_version"),
        baseline_cycle=int(m.group("baseline_cycle")),
        feature_subset=m.group("feature_subset"),
    )


def run_dir(out_root: Path, mode: str, slug: str) -> Path:
    """{out_root}/runs/{mode}/{slug}/  (not created).

    This is the stable **lookup path**. On disk it is normally a symlink
    into the latest ``{slug}__{timestamp}/`` folder; readers do not need
    to know about the timestamp.
    """
    if mode not in ("validation", "production"):
        raise ValueError(f"unknown mode {mode!r}")
    return Path(out_root) / "runs" / mode / slug


_TIMESTAMP_FMT = "%Y%m%d_%H%M%S"


def make_run_dir(
    out_root: Path, mode: str, slug: str, *, now: Optional[datetime] = None,
) -> Path:
    """Create a fresh ``{slug}__{timestamp}/`` and return its path.

    The actual artifacts of a run are written here. The accompanying
    ``{slug}`` symlink is updated separately by ``update_latest_symlink``
    once the run completes — so an interrupted run never silently
    replaces a prior good one.

    If two runs land in the same second (back-to-back ``--force``
    re-runs in tests, mostly), the timestamp is bumped forward by 1s
    until an unused name is found, so no run ever clobbers another.
    """
    if mode not in ("validation", "production"):
        raise ValueError(f"unknown mode {mode!r}")
    base_now = now or datetime.now()
    for attempt in range(60):
        ts = (base_now + timedelta(seconds=attempt)).strftime(_TIMESTAMP_FMT)
        p = Path(out_root) / "runs" / mode / f"{slug}__{ts}"
        try:
            p.mkdir(parents=True, exist_ok=False)
            return p
        except FileExistsError:
            continue
    raise RuntimeError(
        f"could not allocate a unique timestamped run dir under "
        f"{Path(out_root) / 'runs' / mode} for slug {slug!r}"
    )


def update_latest_symlink(timestamped_dir: Path, slug: str) -> Path:
    """Atomically point ``{slug}`` at ``timestamped_dir`` (relative link).

    Safe to call repeatedly: replaces an existing symlink, but refuses
    to overwrite a non-symlink (e.g. a legacy slug-only directory) so
    we never silently nuke real data.
    """
    parent = timestamped_dir.parent
    link = parent / slug
    if link.exists() and not link.is_symlink():
        # Legacy slug-only folder (pre-timestamp era). Leave it alone.
        return link
    tmp = parent / f".{slug}.tmp"
    if tmp.is_symlink() or tmp.exists():
        tmp.unlink()
    tmp.symlink_to(timestamped_dir.name)
    os.replace(tmp, link)
    return link

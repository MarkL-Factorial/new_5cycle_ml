# `feature_candidates/` — curated feature sets, between exploration and production

This folder is the **middle tier** of the three-tier feature pipeline:

| Tier | Location | Status | Refresh |
|---|---|---|---|
| 1. Exploration | `investigations/` | Messy, may be abandoned. Timestamped snapshots stack up under each `out/{ts}/`. | Each run writes a new snapshot. |
| 2. **Candidates** | `feature_candidates/` (← here) | **Curated.** Proven enough to belong in downstream classifier experiments, not yet ready for production. | Manual copy from a chosen investigation snapshot. |
| 3. Production | `features.py` | Sacred. Generates the production feature set for `datasets/A2.2_b1_latest/`. | Only changes when a feature is fully proven. |

A candidate folder is **self-contained**: it holds the artifact parquet
plus a reference copy of the scripts that generated it. There is no
import dependency on `investigations/` — the candidate is frozen the
moment it's promoted, and future investigation edits don't affect it.

## Candidates index

| Name | Source investigation | Promoted at | Feature columns | Used by |
|---|---|---|---|---|
| [dqdv_v1](dqdv_v1/README.md) | `investigations/dqdv_features` | 2026-05-22 | 4 cols (peak-shape: peak V, c1→c5 shift, charge↔discharge hysteresis, cosine similarity) | _(none yet)_ |
| [dqdv_v2](dqdv_v2/README.md) | `investigations/dqdv_features` | 2026-05-22 | 4 cols (Severson ΔQ(V) stats: var_log10, min, mean, skew) | _(none yet)_ |
| [dop_peak_theta](dop_peak_theta/README.md) | `investigations/drt_dop_features` | 2026-05-22 | 6 cols (DOP peak θ at c1/c5 × chg/dis + 2 signed shifts) | _(none yet)_ |

When you promote a candidate, add a row here and link to its README:
`[dqdv_v1](dqdv_v1/README.md)`.

## Folder layout (per candidate)

```
feature_candidates/{candidate_name}/
├── README.md           one paragraph: what this candidate is, why promoted
├── provenance.json     machine-readable source pointer (schema below)
├── features.parquet    the artifact — single canonical file
├── features.csv        CSV mirror
└── scripts/            reference copy of the investigation workflow
    ├── extraction.py
    ├── runner.py
    └── selftest.py
```

The parquet sits at the candidate folder root (not under `out/{ts}/`)
because there's only ever **one canonical version** per candidate.
Refresh overwrites in place; the provenance.json is the audit trail.

## How to promote an investigation snapshot

7-step manual procedure. Run from `ml_label_preprocess/`.

```bash
# 1. Pick the investigation snapshot you want to freeze.
SRC=investigations/dqdv_features/out/20260522_1047
NAME=dqdv_v1

# 2. Create the candidate folder.
mkdir -p feature_candidates/$NAME/scripts

# 3. Copy the artifact (rename to canonical `features.*` for
#    consistency across candidates).
cp $SRC/cell_dqdv_features_v1.parquet feature_candidates/$NAME/features.parquet
cp $SRC/cell_dqdv_features_v1.csv     feature_candidates/$NAME/features.csv

# 4. Copy the workflow scripts (reference copy, frozen at promotion).
cp investigations/dqdv_features/dqdv_features.py     feature_candidates/$NAME/scripts/extraction.py
cp investigations/dqdv_features/run_investigation.py feature_candidates/$NAME/scripts/runner.py
cp investigations/dqdv_features/selftest.py          feature_candidates/$NAME/scripts/selftest.py

# 5. Write provenance.json (see schema below). One-shot Python:
python -c "
import json, datetime as dt, polars as pl, sys
from pathlib import Path
src = Path('$SRC')
manifest = json.loads((src / 'manifest.json').read_text())
parquet = pl.read_parquet(src / 'cell_dqdv_features_v1.parquet')
feature_cols = [c for c in parquet.columns if c not in ('cell_name', 'cohort')]
prov = {
    'name': '$NAME',
    'promoted_at': dt.datetime.now(dt.timezone.utc).isoformat(timespec='seconds'),
    'promoted_by': 'Mark Liao',
    'source': {
        'investigation': 'investigations/dqdv_features',
        'snapshot': str(src.relative_to(Path('.'))),
        'parquet_filename': 'cell_dqdv_features_v1.parquet',
        'manifest': {
            'db_version': manifest.get('db_version'),
            'schema_version': manifest.get('schema_version'),
            'generated_at': manifest.get('generated_at'),
        },
    },
    'feature_columns': feature_cols,
    'notes': 'Promoted because <fill in>',
}
Path('feature_candidates/$NAME/provenance.json').write_text(
    json.dumps(prov, indent=2) + '\n'
)
print('wrote feature_candidates/$NAME/provenance.json')
"

# 6. Write feature_candidates/$NAME/README.md — one paragraph describing
#    the features and why this candidate was promoted.

# 7. Add a row to the Candidates index in feature_candidates/README.md
#    (this file).
```

## How to refresh an existing candidate

```bash
# Default path: copy a fresh parquet from a new investigation snapshot.
NAME=dqdv_v1
NEW_TS=investigations/dqdv_features/out/<new_ts>
cp $NEW_TS/cell_dqdv_features_v1.parquet feature_candidates/$NAME/features.parquet
cp $NEW_TS/cell_dqdv_features_v1.csv     feature_candidates/$NAME/features.csv

# Update provenance.json's source.snapshot and source.manifest fields.
# The scripts/ copy stays as-is unless the extraction logic changed.
# If extraction logic changed, also re-copy the scripts to keep the
# reference copy in sync.
```

When the extraction logic itself changes (not just new data), treat it
as a fresh promotion: bump the candidate name (e.g. `dqdv_v1_rev2/`)
rather than overwriting, so downstream experiments don't silently
switch feature definitions.

## How to add NEW cells to an existing candidate

When a handful of new cells land in the annotation registry and you
don't want to recompute the entire full sweep (~3 hours for DOP), use
the investigation runner's `--cells X Y Z` flag, then merge the
partial output into the candidate. The merged rows take precedence
on `cell_name` collisions (`unique(keep='last')`).

```bash
CELLS="0MC6-260514-R002 0MC6-260514-R003 0MC6-260514-R004"

# 1. Run the extraction for just the new cells.
cd investigations/dqdv_features   # or drt_dop_features
python run_investigation.py --cells $CELLS --no-plots
# Captures: out/{new_ts}/cell_*.parquet  (one row per cell; gitignored)

# 2. Merge into the candidate (polars concat + dedupe + sort).
python -c "
import polars as pl
old = pl.read_parquet('feature_candidates/dqdv_v1/features.parquet')
new = pl.read_parquet('investigations/dqdv_features/out/{new_ts}/cell_dqdv_features_v1.parquet')
merged = pl.concat([old, new]).unique(subset='cell_name', keep='last').sort('cell_name')
merged.write_parquet('feature_candidates/dqdv_v1/features.parquet')
merged.write_csv('feature_candidates/dqdv_v1/features.csv')
"

# 3. Append a merged_in entry to provenance.json (schema below).
```

### `provenance.json:merged_in` schema

```json
"merged_in": [
  {
    "merged_at": "2026-05-26T12:05:00+00:00",
    "snapshot": "investigations/dqdv_features/out/20260526_1204",
    "parquet_filename": "cell_dqdv_features_v1.parquet",
    "cells": ["0MC6-260514-R002", "0MC6-260514-R003", "..."],
    "manifest": {
      "db_version": "A2.2",
      "schema_version": 2,
      "generated_at": "2026-05-26T12:04:00+00:00",
      "mode": "subset",
      "n_cells_attempted": 5,
      "n_cells_full": 5
    }
  }
]
```

**Total row count of `features.parquet`** is
`source.manifest.n_cells_attempted` + union of all
`merged_in[*].cells`.

**When NOT to use incremental merge**: if the investigation's
`extraction.py` changed between the candidate's existing rows and now,
the merge silently mixes two algorithms in one parquet. In that case,
re-run the full sweep instead and replace via the standard refresh
above.

## `provenance.json` schema

```json
{
  "name": "dqdv_v1",
  "promoted_at": "2026-05-22T20:00:00+00:00",
  "promoted_by": "Mark Liao",
  "source": {
    "investigation": "investigations/dqdv_features",
    "snapshot": "investigations/dqdv_features/out/20260522_1047",
    "parquet_filename": "cell_dqdv_features_v1.parquet",
    "manifest": {
      "db_version": "A2.2",
      "schema_version": 2,
      "generated_at": "2026-05-22T14:47:30+00:00"
    }
  },
  "feature_columns": [
    "dqdv_peak_v_c5_dis",
    "dqdv_peak_v_shift_c1c5_dis",
    "dqdv_charge_discharge_hysteresis_c5",
    "dqdv_cosine_sim_c1c5_dis"
  ],
  "notes": "v1 peak-shape features. Promoted because exp_o classifier shows..."
}
```

The `source.manifest` block copies the relevant keys from the
investigation snapshot's `manifest.json` at promotion time — so the
provenance survives even if the investigation folder is later
reshaped or its snapshot deleted.

## Downstream pinning convention

New downstream experiments (under `cell_lifetime/experiments/`) pin to
a candidate's stable artifact path:

```python
FEATURES_PATH = Path(
    "ml_label_preprocess/feature_candidates/dqdv_v1/features.parquet"
)
```

No timestamps in the pin. When the candidate is refreshed, downstream
re-runs pick up the new parquet on the next read. Each downstream
experiment's own log/manifest should record the resolved
`provenance.json` it consumed so historical runs stay reproducible.

`cell_lifetime/experiments/exp_o`, `exp_p`, `exp_q` are **not migrated**
to this convention — they keep their frozen investigation pins. The
candidate-tier pin applies only to **new** experiments going forward.

## Conventions

- **Parquet filename inside a candidate is always `features.parquet`**
  (canonical, predictable for downstream pinning). The source filename
  from the investigation is preserved in `provenance.json:source.parquet_filename`.
- **Candidate folder names use the feature family**, not the
  investigation name. Examples: `dqdv_v1`, `dqdv_v2`, `dop_peak_theta`.
- **One candidate per feature group.** If an investigation emits two
  parquets (like dqdv's v1 + v2 split), it becomes two candidate
  folders — one per group.
- **No `_latest` symlink** inside a candidate. There's only ever one
  parquet (the canonical one); refresh overwrites in place.
- **Refresh is always deliberate.** Never automate. The friction is
  the feature, not a bug — it forces a human to confirm the data is
  still right.

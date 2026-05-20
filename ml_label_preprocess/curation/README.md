# Cohort curation pipeline

Identifies cells whose retention curves need human judgement before they
enter ML training, and captures those judgements in a git-tracked
artifact (`decisions.json`) that `../labels.py` consumes.

This package graduated from `../investigations/{outlier_detection,
sustained_step, manual_validation}/` once the pipeline stabilised.
History: see [../investigations/jump_detection/](../investigations/jump_detection/)
for the original sustained-step exploration (now diagnostic-only).

## Pipeline order

```
   ┌──────────────────────────────┐
   │ outlier_detection.py         │ → outlier_sidecar.json
   │ (flags 1–3-cycle glitches)   │   (GIT-TRACKED — consumed by labels.py)
   └────────────┬─────────────────┘
                │
                ▼
   ┌──────────────────────────────┐
   │ sustained_step.py            │ → reports/sustained_step_report.csv
   │ (strict, masked, rate_changed-│   plots/sustained_step/*.png
   │  excluded)                    │
   └────────────┬─────────────────┘
                │
                ▼
   ┌──────────────────────────────────────┐
   │ validation.py sync                   │ → pending/ (gitignored)
   │ Surfaces cells flagged by any of:    │
   │   • sustained_step.py                │
   │   • outliers in last 5 cycles        │
   │     (tail-outlier criterion)         │
   │   • stale decisions                  │
   │     (n_regular_at_review drift)      │
   └────────────┬─────────────────────────┘
                │
                │   ← human edits decisions.json
                ▼
       decisions.json
       (GIT-TRACKED — consumed by labels.py)
                │
                ▼
   ┌──────────────────────────────┐
   │ ../labels.py                 │ → datasets/{db}_b{N0}/cell_labels.parquet
   │ (reads decisions.json +      │
   │  outlier_sidecar.json)       │
   └──────────────────────────────┘
```

## Usage

```bash
source /home/mliao/miniconda3/etc/profile.d/conda.sh && conda activate eis

# 1. Run outlier detection (when annotation data or params change)
python -m curation.outlier_detection

# 2. Run sustained-step detection (after outlier_detection)
python -m curation.sustained_step

# 3. Surface cells needing review
python -m curation.validation sync
#    (or: python -m curation.validation refresh — re-runs steps 1+2)

# 4. Edit decisions.json with judgements; re-run sync to validate.

# 5. Once `Pending review: 0`, run labels:
python -c "from labels import main; main()"
# (or run the full preprocess pipeline)
python preprocess.py --all
```

Each module has a `--help` flag. The pure-algorithm modules
(`jump_detection`, `outlier_detector`) also have `--selftest` for the
synthetic-case tests.

For workspaces whose `decisions.json` predates the
`n_regular_at_review` schema field, run the one-shot migration before
the first `sync` (see "One-shot migration" below):

```bash
python -m curation.validation migrate-snapshot
```

## File layout

```
curation/
├── __init__.py
├── README.md                       this file
├── .gitignore                      pending/ reports/ plots/outliers/ plots/sustained_step/
│
├── jump_detection.py               pure algorithm — detect_jumps, DetectorParams, …
├── outlier_detector.py             pure algorithm — detect_outliers, OutlierParams, …
├── outlier_detection.py            CLI — runs detect_outliers over the cohort
├── sustained_step.py               CLI — runs detect_jumps with strict params
├── validation.py                   CLI — refresh / sync subcommands
│
├── decisions.json                  GIT-TRACKED — human-curated judgements
├── outlier_sidecar.json            GIT-TRACKED — flagged outlier cycles per cell
│
├── plots/
│   ├── outliers/                   (gitignored — diagnostic)
│   │   ├── with_outliers/
│   │   ├── known_glitches_audit/
│   │   └── no_outliers_audit/
│   ├── sustained_step/             (gitignored — feeds validation pending/)
│   └── validated/                  GIT-TRACKED — snapshot at review time
│
├── pending/                        (gitignored — regenerated each sync)
│   ├── cell_list.txt
│   ├── template.json
│   └── plots/<cell>.png
└── reports/                        (gitignored)
    ├── outlier_report.csv
    ├── outlier_summary.txt
    ├── sustained_step_report.csv
    └── sustained_step_summary.txt
```

## decisions.json schema

```json
{
  "AR4313": {
    "exclude_from_ml": false,
    "last_available_cycle": 265,
    "event_type": "event",
    "reason": "protocol change after cycle 268; truncate at 265",
    "validated_at": "2026-05-20",
    "n_regular_at_review": 472
  },
  "AR4269": {
    "exclude_from_ml": true,
    "last_available_cycle": null,
    "event_type": null,
    "reason": "early-life instability; no usable ML signal",
    "validated_at": "2026-05-20",
    "n_regular_at_review": 23
  }
}
```

| Field | Type | Required | Meaning |
|---|---|---|---|
| `exclude_from_ml` | bool | always | If true, drop cell from cohort entirely. |
| `last_available_cycle` | int or null | always | Cycles strictly after this are unreliable and ignored. Null = use full cell (or N/A when excluded). |
| `event_type` | `"censor"` / `"event"` / null | always | `event` = cell faded (crossed 0.85, no recovery) by `last_available_cycle`. `censor` = cell still healthy at that cycle. Null when excluded. |
| `reason` | string | always | Free-text audit note. |
| `validated_at` | ISO date string | always | When the entry was added. |
| `n_regular_at_review` | int | always | Cell's `n_regular` (lifetime cycle count) at the time the decision was recorded. `sync` compares this against current state to flag stale decisions when upstream data extends/shortens. |

Schema rules enforced by `validation.py sync`:
- All six fields must be present; no extras allowed.
- `exclude_from_ml` must be bool; `last_available_cycle` int-or-null;
  `event_type` one of {"censor", "event", null}; `n_regular_at_review`
  int.
- **When `exclude_from_ml=false`, `event_type` cannot be null** — if
  the cell is kept, the reviewer must commit to censor vs event.
- **When `event_type="event"`, `last_available_cycle` cannot be null** —
  it becomes the asserted `last_fade_cycle` in cell_labels.parquet, so
  a value is required.

## outlier_sidecar.json schema

Generated by `outlier_detection.py`; committed for reproducibility.

```json
{
  "<cell_name>": {
    "n_outliers": <int>,
    "outliers": [
      {
        "cycle": <int>,
        "list_index": <int>,
        "retention": <float>,
        "predicted": <float>,
        "residual": <float>,
        "z_score": <float>,
        "pre_post_disagreement": <float or NaN>,
        "pre_n_points": <int>,
        "post_n_points": <int>
      },
      ...
    ]
  },
  ...
}
```

`labels.py` only reads the `cycle` field; the rest is for audit.

## How `validation.py sync` surfaces cells for review

A cell appears in `pending/cell_list.txt` (and `pending/plots/`) when
**any** of three criteria fire:

1. **Sustained-step criterion** — the cell has at least one row in
   `reports/sustained_step_report.csv`. This is the original criterion:
   strict sustained-step detection on outlier-masked retentions.

2. **Tail-outlier criterion** — the cell has at least one cycle in
   `outlier_sidecar.json[cell].outliers` that falls within the last
   `--tail-window` regular cycles (default **5**).

3. **Stale-decision criterion** — the cell is in `decisions.json` but
   its recorded `n_regular_at_review` no longer matches the cohort's
   current `n_regular` (cell extended, shortened, or vanished since
   the prior review).

Cells already present in `decisions.json` are skipped on criteria 1+2
(once decided, they don't reappear). Criterion 3 specifically
re-surfaces decided cells whose data has drifted.

### Why tail outliers need their own review trigger

The outlier detector can mask cycles that are actually real fade
events when those events happen near the end of life. The motivating
case was **AR4195**: cycles 32–35 of a 35-cycle cell were flagged as
a burst, hiding the cell's real fade at cycle 35 from `labels.py`.

Auto-surfacing any cell with outlier flags in `[n_regular − 4,
n_regular]` ensures these cases reach the reviewer. The reviewer then
decides:

- **The cell really did fade** → write `event_type: "event"`,
  `last_available_cycle: <fade_cycle>` (mirrors AR4195's decision).
- **The tail is genuinely noisy / cell wasn't faded** → write
  `event_type: "censor"`, optionally lower `last_available_cycle` to
  the pre-noise cycle.

### Why stale-decision needs its own review trigger

`decisions.json` is authoritative — `labels.py` applies whatever the
human asserted. That's the right default, but it has a hidden failure
mode: the data the reviewer saw can change. Cycler keeps running,
annotations refresh, outliers re-detect. The decision can quietly
become wrong.

Recording `n_regular_at_review` in every entry lets `sync` compare
"current state" vs "state at review time" and re-surface cells whose
data has drifted. The reviewer then:

- **Ratifies** (data still consistent with the decision) → just bump
  `n_regular_at_review` to the current value, leave other fields.
- **Revises** (data changed materially) → update `event_type` /
  `last_available_cycle` / `reason` / `validated_at` AND
  `n_regular_at_review`.
- **Drops** (decision no longer applicable) → delete the entry; cell
  falls back to the algorithm path.

Stale entries are listed with their detail embedded in
`review_reason`, e.g.:
```
review_reason = stale_decision(was:113,now:163)
review_reason = stale_decision(was:113,now:missing)     ← orphan
```

### One-shot migration

The first time you add `n_regular_at_review` to a workspace whose
`decisions.json` predates this field, run:

```bash
python -m curation.validation migrate-snapshot
```

It fills in `n_regular_at_review = current n_regular` for every entry
that lacks the field. Subsequent decisions get the field from the
template stub.

### CLI overrides

```bash
# Default: surface cells with outliers in last 5 cycles + stale decisions
python -m curation.validation sync

# Widen the window (catches more borderline cases for one-off audits)
python -m curation.validation sync --tail-window 10
```

### `cell_list.txt` columns

```
cell  review_reason  sustained_cycle  delta  persist  tail_outlier_cycles  n_outliers_masked  n_regulars
```

- `review_reason`: comma-separated list of one or more of
  `sustained_step`, `tail_outlier`, `stale_decision(was:X,now:Y)`.
- For cells flagged only by the sustained criterion,
  `tail_outlier_cycles` is `-`.
- For cells flagged only by the tail / stale criterion,
  `sustained_cycle`, `delta`, `persist` are `-`.

## How labels.py interprets decisions.json

**decisions.json is authoritative.** When a cell has an entry, the
output row in `cell_labels.parquet` is derived **directly** from that
entry — the outlier mask and fade detector are bypassed for that cell.
This guarantees that manual review wins over algorithmic detection in
every case.

The mapping from one entry to the resulting label row:

| Decision | Resulting columns |
|---|---|
| `exclude_from_ml: true` | `status="excluded"`, `exclusion_reason="human_review"` |
| `event_type: "event"`, `last_available_cycle: X` | `status="faded"`, `last_fade_cycle=X`, `n_regular=X`, `truncation_cycle=X`, `n_outliers_masked=0` |
| `event_type: "censor"`, `last_available_cycle: X` (int) | `status="in_testing"`, `last_fade_cycle=null`, `n_regular=X`, `truncation_cycle=X`, `n_outliers_masked=0` |
| `event_type: "censor"`, `last_available_cycle: null` | `status="in_testing"`, `last_fade_cycle=null`, `n_regular=full_lifetime`, `truncation_cycle=null` |

`baseline_dis_ah` and `final_retention` are still computed from the
actual cell data (they are derived facts that don't conflict with the
human's assertion). `final_retention` is taken at
`last_available_cycle` (or the nearest preceding regular cycle that
exists, in case the asserted cycle is between regulars) — for `event`
entries it's the retention at the asserted fade cycle.

**Cells without a `decisions.json` entry**: the existing algorithm
runs. The retention curve is built excluding cycles flagged in
`outlier_sidecar.json`, and `_last_crossing_into_bad` runs on the
cleaned curve. `n_outliers_masked` counts the cycles dropped from the
fade-detection window.

Missing decisions.json / outlier_sidecar.json files degrade gracefully:
`labels.py` falls back to its original (pre-curation) behavior.

## What's NOT here

- A separate Pattern B handler. `cycling_consistency == "rate_changed"`
  exclusion is in `../labels.py`'s rate_changed early-return.
- Algorithm-driven decisions. Every `decisions.json` entry is
  human-authored. The pipeline only surfaces candidates.
- UI / browser tool for validation. Workflow is "open plot in IDE,
  edit JSON in IDE".

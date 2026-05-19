# `ml_label_preprocess_v3`

Per-cell **labels** and **features** for the battery dataset's first-5-cycle
ML task. v3 is the current working pipeline; v1 / v2 are frozen as
reference snapshots.

## v3 in one paragraph

Every output is keyed on two axes — the upstream annotation **DB version**
(e.g. `A2.2`, auto-parsed from `BAT_ANNOT_DIR`) and the **baseline cycle**
N0 (1..4) used as the retention denominator. Each `(db_version, baseline_cycle)`
combination produces a self-contained **dataset bundle** at
`datasets/{db_version}_b{baseline_cycle}/` containing parquets + CSVs +
a `manifest.json` recording provenance. There is no implicit "default"
bundle; both axes always appear in the path.

## Output layout

```
ml_label_preprocess_v3/
├── datasets/
│   ├── A2.2_b1/                    # DB A2.2, baseline cycle 1
│   │   ├── cell_labels.parquet
│   │   ├── cell_labels.csv
│   │   ├── cell_features.parquet
│   │   ├── cell_features.csv
│   │   ├── cell_features_status.csv
│   │   └── manifest.json
│   ├── A2.2_b3/                    # same DB, baseline cycle 3
│   │   └── ...
│   └── A2.3_b1/                    # future DB (next regeneration)
│       └── ...
├── preprocess.py
├── _common.py
├── labels.py
├── features.py
├── column_roles.yaml
└── README.md
```

The slug `{db_version}_b{baseline_cycle}` is the only place these two axes
appear; filenames inside the bundle stay fixed. `manifest.json` looks like:

```json
{
  "schema_version": 2,
  "db_version": "A2.2",
  "baseline_cycle": 3,
  "annot_dir": "/mnt/data/mliao/battery-ml-workbench/data/A2.2/annotations",
  "generated_at": "2026-05-14T12:30:00Z",
  "n_cells_labels": 455,
  "n_cells_features": 360,
  "column_roles_sha256": "abcd1234...",
  "stages_populated": ["features", "labels"]
}
```

`stages_populated` is the union of what's been run — running `--labels`
then `--features` produces the same final manifest as `--all`.

## v3 vs v2 (back-compat NOT preserved)

- v2 had two output rules: `baseline=1` → `out/`, others → `out/baseline_{N}/`.
- v3 always routes to `datasets/{db_version}_b{baseline_cycle}/`. No special
  case for baseline 1; the path always encodes both axes.
- The processing algorithms (Tier A/B/C math, fade detection, omission rule)
  are identical to v2. `cell_features.parquet` content for `(A2.2, b=1)` is
  the same as v2's `out/cell_features.parquet`; same for `(A2.2, b=3)` vs
  v2's `out/baseline_3/cell_features.parquet`.
- `ml_label_preprocess/` (v1) and `ml_label_preprocess_v2/` remain on disk
  as frozen reference snapshots. Downstream `ml_classification_v2` has been
  updated to read from v3's `datasets/{...}/` paths.

## Pipelines

Two independent passes over the annotation JSONs at `$BAT_ANNOT_DIR`
(default `/mnt/data/mliao/battery-ml-workbench/data/A2.2/annotations`),
sharing one CLI dispatcher.

| Pipeline | Entry | Files in bundle | Workbench-app needed? |
|---|---|---|---|
| labels   | `python preprocess.py` (default) or `--labels`   | `cell_labels.{parquet,csv}` (12 status cols + 6 per-N classification cols) | no |
| features | `python preprocess.py --features` | `cell_features.{parquet,csv}` + `cell_features_status.csv` | yes |

Every run writes into `datasets/{db_version}_b{baseline_cycle}/` and
merges its provenance into that bundle's `manifest.json`.

Flags:
- `--all` — run both pipelines
- `--selftest` — common helpers + labels + features selftests
- `--cells CELL [CELL ...]` — restrict features to a subset (debug)
- `--baseline-cycle N` — N0 ∈ {1, 2, 3, 4}, default 1; see "Baseline cycle" below
- `--db-version TAG` — override the DB tag (default: auto-parsed from `ANNOT_DIR`)

## Baseline cycle (N0)

`--baseline-cycle N` controls the regular_cycle ordinal used as the
**retention denominator** for every Tier A and Tier B feature, AND for
the fade-detection logic in `labels.py`. Default is 1 (v1 behavior).

- **Tier A**: `discharge_capacity_retention_final = cap_dis(c5) /
  cap_dis(N0)`. `charge_capacity_retention_min` is taken over the
  post-baseline window `[N0, 5]` divided by `cap_chg(N0)`.
- **Tier B**: nominal voltage retention features use cycle N0's
  nominal voltage as the denominator and aggregate over `[N0, 5]`.
  With N0=3 the std is computed over only 3 points (vs 5 at N0=1) —
  expect noisier `discharge_nominal_voltage_retention_std`.
- **Tier C**: KWW fits on cycles 3, 4, 5 are **unchanged** by N0 —
  they describe CV-phase current decay, not retention.
- **Labels**: `status` (faded / in_testing), `last_fade_cycle`, and
  every `label_n{N}` / `trainable_n{N}` shift according to the new
  retention curve. A cell that just barely passes N=300 under cycle-1
  baseline may flip to bad under cycle-3 baseline (or vice versa) —
  this is by design; feature retention and label retention must share
  the same N0 for the experiment to be coherent.
- **Omission rule** is unchanged: a cell must still have cycles 1..5
  with all Tier-A inputs present. This keeps the cell pool identical
  across baselines so per-baseline outputs are directly comparable.
- **Schema** is identical across baselines (41 columns in
  `cell_features`, 17 columns in `cell_labels`). The manifest at
  `column_roles.yaml` does not change.

Valid range: 1..4. (N0=5 would leave a single-point window for Tier B,
so std is undefined.)

## Files in this directory

```
preprocess.py            CLI dispatcher (thin) — --baseline-cycle, --db-version, --selftest
labels.py                Fade-status + per-N classification labels
features.py              41-column per-cell feature extraction (Tiers A/B/C complete)
_common.py               Shared: paths, _cohort, annotation iteration, dataset_dir_for, write_manifest
column_roles.yaml        Schema manifest (schema_version: 2)
datasets/                Output bundles, one per (db_version, baseline_cycle)
```

## ⚠ Cycle indexing: this pipeline DIFFERS from the legacy ML pipeline by design

If you compare our `cell_features.csv` against the legacy reference
[`experiment_cv_features/data/BOL_with_cv_features.csv`](../experiment_cv_features/data/BOL_with_cv_features.csv)
on shared cells, the Tier-A retention / CE columns **will not match
exactly**. That is intentional, not a bug.

**Why:** the legacy pipeline groups data by `tester_cycle` (the cycler's
own cycle counter, which co-mingles formation, rate-test, rebalance,
and regular cycling within a single ordinal). Our pipeline keys on
the annotation toolkit's renumbered `regular_cycle` ordinal, which
isolates the "regular cycling" segment of each cell's life.

Concrete example — cell `AR4389`'s first 8 cd_events:

| cd_index | event_kind   | regular_cycle | tester_cycle |
|---:|---|---:|---:|
| 0 | formation     | — | 0 |
| 1 | formation     | — | 0 |
| 2 | rate_test_cd  | — | 0 |
| 3 | formation     | — | 0 |
| 4 | rebalance_cd  | — | 0 |
| 5 | regular_cd    | 1 | 0 |
| 6 | regular_cd    | 2 | 1 |
| 7 | regular_cd    | 3 | 2 |

The legacy pipeline's "first cycle" for AR4389 is `tester_cycle == 0`,
which is an aggregate of formation + rate-test + rebalance + the first
regular cycle. Its baseline values therefore include formation-cycle
data. Our pipeline's "first cycle" is `regular_cycle == 1` (cd_index 5
in this example), which is the first clean regular charge-discharge
event after the cell's setup phase is done.

**Consequence:** retention ratios, CE values, and any per-cycle
aggregations computed by this pipeline are **cleaner**, but **not
drop-in compatible** with models trained on the legacy `BOL_*` CSV
features. A model trained against our outputs must be re-trained; you
cannot validate by spot-checking against the legacy numbers.

The annotation toolkit + workbench-app exist specifically because the
old `tester_cycle` indexing produced this kind of mis-aligned ML
input. This is the intended new baseline.

## Column-role manifest

[`column_roles.yaml`](column_roles.yaml) is the single source of truth
for which output column belongs to which of three roles:

- **meta** — known at prediction time but not trained on (identifiers,
  cohort, protocol classification, cycle-1 baseline). Conservative
  default: we don't train on these to avoid cohort / protocol shortcuts.
- **label** — outcome / target. Requires observing the cell past cycle 5
  to compute. Hard-fail if used as a feature.
- **feature** — model input. Strictly derived from cycles 1–5.

Plus one auxiliary role for diagnostics:

- **quality** — fit-success counts and error logs. Use to filter rows
  pre-training, never as features themselves.

Downstream ML training scripts MUST filter columns by role to prevent
data leakage. Example:

```python
import yaml
m = yaml.safe_load(open("ml_label_preprocess_v3/column_roles.yaml"))
feature_cols = [c["name"] for c in m["datasets"]["cell_features"]["columns"]
                if c["role"] == "feature"]
label_cols   = [c["name"] for c in m["datasets"]["cell_labels"]["columns"]
                if c["role"] == "label"]
assert set(feature_cols).isdisjoint(label_cols), "leakage!"
```

`features.py` runs a manifest-consistency check at startup that aborts
if the YAML and the in-code `SCHEMA` drift.

The 12 FS_CV features used in
`experiment_cv_features/report_M2_vs_lean8_vs_CV_20260501` are listed
under `subsets.fs_cv` in the manifest — they are a strict subset of
the `feature`-role columns.

## N-threshold classification labels

`cell_labels.parquet` carries six additional columns — two per threshold
N ∈ {200, 300, 400} — for downstream binary classification of "will the
cell survive past N cycles with discharge retention ≥ 0.85?".

| Column | Type | Values |
|---|---|---|
| `label_n200`     | string | `pass` / `bad` / `censor` / `excluded` |
| `trainable_n200` | bool   | `True` iff `label_n200 ∈ {pass, bad}` |
| `label_n300`, `trainable_n300` | same | same |
| `label_n400`, `trainable_n400` | same | same |

### Decision table

Computed by [`labels.py::_classification_label_at`](labels.py):

| Status from fade rule | Condition | Label | Trainable |
|---|---|---|---|
| `excluded`   | (any)                          | `excluded` | False |
| `faded`      | `last_fade_cycle > N`          | `pass`     | True  |
| `faded`      | `last_fade_cycle <= N`         | `bad`      | True  |
| `in_testing` | `n_regular >= N`               | `pass`     | True  |
| `in_testing` | `n_regular < N`                | `censor`   | False |

**Edge cases (boundary behaviour):**
- Cell fades at *exactly* cycle N → `bad` (did not exceed N healthy cycles).
- Cell still in testing with `n_regular == N` → `pass` (observed N healthy cycles).

**Censor semantics:** a `censor` cell has not yet been observed to fade
*and* has not yet reached N cycles. It is **unknown** whether it will
eventually pass or fail at N — therefore it must be **excluded from
training** (`trainable_n{N} == False`). Including censored cells would
inject pseudo-`pass` labels (because the cell hasn't failed *yet*) and
bias the model towards optimism.

### Distributions on the current A2.2 dataset (455 cells)

| Label | N=200 | N=300 | N=400 |
|---|---:|---:|---:|
| `pass`     | 223 | 157 | 118 |
| `bad`      |  68 |  91 | 115 |
| `censor`   | 147 | 190 | 205 |
| `excluded` |  17 |  17 |  17 |
| **trainable** | **291 (64%)** | **248 (55%)** | **233 (51%)** |

As N grows, more cells fall into `censor` (haven't reached N yet) and
`bad` shifts to include later-faded cells; `pass` therefore shrinks.
The N=400 row is approximately balanced (118 pass vs 115 bad) — good
for binary classification — but heavily 0MC-censored: at N=400 the
0MC cohort contributes only 14 pass + 13 bad, while AR contributes
104 + 102. **A model trained at N=400 will be dominated by AR
chemistry.** For 0MC-focused work, use N=200 (more 0MC cells have
reached the threshold).

### Usage in training scripts

```python
import polars as pl
import yaml

bundle = "ml_label_preprocess_v3/datasets/A2.2_b1"
labels = pl.read_parquet(f"{bundle}/cell_labels.parquet")
features = pl.read_parquet(f"{bundle}/cell_features.parquet")
df = labels.join(features, on="cell_name", how="inner")

# Read feature subset from manifest (avoids leakage)
m = yaml.safe_load(open("ml_label_preprocess_v3/column_roles.yaml"))
fs_cv = m["subsets"]["fs_cv"]["members"]                     # 12 FS_CV features
all_features = [c["name"] for c in m["datasets"]["cell_features"]["columns"]
                if c["role"] == "feature"]                   # all 41 feature cols

# Train at N=400 — binary pass/bad classification
N = 400
train = df.filter(pl.col(f"trainable_n{N}"))
X = train.select(fs_cv).to_numpy()
y = (train[f"label_n{N}"] == "pass").to_numpy().astype(int)
# fit your classifier on (X, y) ...
```

The `trainable_n{N}` flag is the single source of truth for which rows
to keep — it filters out both `censor` (unknown outcome) and `excluded`
(unusable cell) in one step. Do not train on rows where it is False.

## Rate-changed cells: predict-only admission (schema v2)

Cells whose annotation toolkit classification is
`cycling_consistency == 'rate_changed'` cannot be trained on — their
retention curve mixes capacities measured at different rates, so the
fade detector cannot run honestly. But if the **first** rate regime
spans cycles 1..5 entirely (i.e. `regime[0].n_regular_cd >= 5` in the
annotation JSON), the cell's 5-cycle feature window is rate-consistent
and the cell is admitted to `cell_features.parquet` for production-
inference scoring only.

These cells have, on the **labels** side:
- `status='excluded'`, `exclusion_reason='rate_changed'`
- `trainable_n{N}=False` for every N (training filters reject them)
- `last_fade_cycle=None`, `final_retention=None` (not meaningful)
- `n_regular` = lifetime regular count (same formula as single_rate
  cells — the cell really did run that many cycles, just at different
  rates). This lets the downstream asymmetric `n_regular >= 5` predict
  filter in `cell_lifetime` admit the row for scoring.
- `baseline_dis_ah` populated from cycle N0 (well-defined; cycles 1..N0
  are at the original rate)
- `n_regular_pre_rate_change = regime[0].n_regular_cd` — diagnostic
  count of cycles before the rate first changed

And on the **features** side, a normal Tier A/B/C row computed from
cycles 1..5 (all at the original rate).

For rate_changed cells whose first regime is shorter than 5 (would mean
the rate change happened on or before cycle 5), the cell stays fully
excluded: no feature row, `n_regular=0`, only
`n_regular_pre_rate_change` is populated as a diagnostic.

The diagnostic column on **all** cells:

| `cycling_consistency` | `n_regular_pre_rate_change` |
|---|---|
| `single_rate`   | `null` (no rate change ever happened) |
| `rate_changed` (admitted) | `regime[0].n_regular_cd` (>= 5) |
| `rate_changed` (excluded) | `regime[0].n_regular_cd` (< 5) |
| `no_regular`    | `null` (no regimes detected) |

Note: single_rate cells can still have many regimes in the annotation
JSON (the toolkit splits regimes at RPT-segment boundaries even when
rates are similar — so `regime[0].n_regular_cd` is not the cell's
lifetime count for those cells). The column intentionally returns null
for them to keep its "pre-rate-change" semantics unambiguous; use the
`n_regular` column for lifetime counts.

To recover the "fully featurizable, training-eligible" subset:

```python
predict_only = (labels["status"] == "excluded") & (labels["n_regular"] >= 5)
train_eligible = labels[f"trainable_n{N}"]
```

A2.2 ground truth (as of 2026-05-19): all 9 rate_changed cells in A2.2
satisfy the admission gate (`regime[0].n_regular_cd ∈ {5, 5, 5, 5, 5, 5,
5, 82, 437}`), so all 9 flow through to `cell_features.parquet`.

## Implementation stages

The features pipeline was built in three stages (see
[preprocess_extension_feasibility.md](preprocess_extension_feasibility.md)
for the full design). **All three are now landed:**

- **Stage A** ✅ — Tier A (3 features from annotation JSON scalars).
- **Stage B** ✅ — Tier B (3 nominal-voltage retention features). Reads
  raw `(t, V, I)` via `battery_workbench.core.data.annotations`,
  integrates `V·I·dt` and `I·dt` locally to get capacity-weighted
  mean voltage per phase per cycle.
- **Stage C** ✅ — Tier C (15 per-cycle KWW fit outputs + 15
  aggregated + 4 engineered A-ratio + 1 quality column). Imports
  `extract_cv_phase_by_cd` and `fit_kww_fast_exp` from
  `battery_workbench.core.analysis.cv_fitting`. See
  [KNOWN_ISSUES.md](KNOWN_ISSUES.md) for the small set of cells with
  pinned-to-bound fit failures.

Stage B and Stage C require workbench-app to be importable. In the
`eis` conda env this is wired via a `.pth` file at
`/home/mliao/miniconda3/envs/eis/lib/python3.11/site-packages/battery_workbench_src.pth`
pointing to `battery-workbench-app/src`. No `pip install` needed —
this avoids dragging in workbench-app's UI dependencies (Dash, Plotly,
kaleido) which we don't use here.

## Running

```bash
# from this directory, in the eis conda env
python preprocess.py                            # labels (default), auto db + baseline 1
python preprocess.py --features                 # features (requires workbench-app on path)
python preprocess.py --all                      # both
python preprocess.py --all --baseline-cycle 3   # both, baseline = cycle 3
python preprocess.py --all --db-version A2.3    # override DB tag
python preprocess.py --selftest                 # validate helpers + both pipelines

# debugging a subset (features only)
python preprocess.py --features --cells AR-3420 0MC2-251022-001
```

Default DB tag is auto-parsed from `BAT_ANNOT_DIR`: a path of
`/.../A2.2/annotations` → tag `A2.2`. Pass `--db-version` only when your
ANNOT_DIR doesn't follow that convention (rare).

Environment:
- `BAT_ANNOT_DIR` — annotation JSON directory (default: A2.2 location)
- `BAT_DATA_DIR` — raw parquet directory (used by workbench-app, needed Stage B+)

## Reading outputs from downstream code

```python
import json
from pathlib import Path
import polars as pl

bundle = Path("ml_label_preprocess_v3/datasets/A2.2_b1")
manifest = json.loads((bundle / "manifest.json").read_text())
assert manifest["schema_version"] == 2
labels   = pl.read_parquet(bundle / "cell_labels.parquet")
features = pl.read_parquet(bundle / "cell_features.parquet")
```

`ml_classification_v2/data.py::load_dataset(N, baseline_cycle=…, db_version=…)`
does this wrapping for the classifier and is the recommended entry point.

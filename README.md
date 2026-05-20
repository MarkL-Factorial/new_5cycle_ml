# new_5cycle_ml

First-5-regular-cycles ML stack for the battery dataset: from per-cell
annotation JSONs to a trained N-cycle survival classifier. Two stages,
each its own subproject; the classifier consumes the preprocess bundles
the labeling stage produces.

```
   data/A2.2/annotations/*.annotations.json          (upstream: annotation toolkit)
                          |
                          v
   ml_label_preprocess/   ── labels + features per cell
        datasets/{db_version}_b{baseline_cycle}/     (one bundle per (DB, N0))
                          |
                          v
   cell_classifier/       ── trains the survival classifier
        out/runs/{mode}/{slug}/                      (one run dir per config)
```

Author: Mark Liao (Sheng-Lun Liao). All work runs in the shared `eis`
conda env — `source /home/mliao/miniconda3/etc/profile.d/conda.sh &&
conda activate eis`.

---

## Layout

| Path                     | Role                                                  |
|--------------------------|-------------------------------------------------------|
| [ml_label_preprocess/](ml_label_preprocess/README.md)     | Labeling + feature extraction                     |
| [cell_classifier/](cell_classifier/README.md)             | N-cycle survival classifier (src/ layout, console entry) |

Two earlier subprojects — `ml_label_preprocess_v3/` and
`ml_classification/` — have been archived under `legacy/` and are
excluded from this repo via `.gitignore`. They are slated for removal;
ignore them for new work.

---

## Stage 1 — `ml_label_preprocess/` (labeling + features)

Iterates the annotation JSONs at `$BAT_ANNOT_DIR` (default
`/mnt/data/mliao/battery-ml-workbench/data/A2.2/annotations`) and emits
per-cell labels and features keyed on the toolkit's `regular_cycle`
ordinal — never on the cycler's raw `tester_cycle`. Two output axes,
encoded in the bundle path:

- **DB version** — auto-parsed from `BAT_ANNOT_DIR` (e.g. `A2.2`).
- **Baseline cycle N0 ∈ {1..4}** — `regular_cycle` ordinal used as the
  retention denominator for every Tier-A and Tier-B feature, and for
  the fade-rule status that drives the labels.

Each `(db_version, baseline_cycle)` pair gets a self-contained bundle at
`datasets/{db_version}_b{baseline_cycle}/` containing
`cell_labels.{parquet,csv}`, `cell_features.{parquet,csv}`,
`cell_features_status.csv`, and a `manifest.json` recording provenance.

**Labels**: 11 status columns (`status`, `last_fade_cycle`, `n_regular`,
`final_retention`, `exclusion_reason`, …) plus 6 per-threshold
classification columns — `label_n{N}` ∈ {`pass`, `bad`, `censor`,
`excluded`} and `trainable_n{N}` (boolean) for N ∈ {200, 300, 400}. A
cell is *trainable at N* iff its outcome at N is known
(`label_n{N} ∈ {pass, bad}`); `censor` and `excluded` rows must be left
out of training.

**Features**: 41 columns across three tiers (Tier A retention/CE from
annotation scalars; Tier B nominal-voltage retention from raw `(t, V, I)`;
Tier C per-cycle KWW fits on the CV-phase current decay). The 12-feature
`fs_cv` subset used by the current classifier configs is declared under
`subsets.fs_cv` in [`column_roles.yaml`](ml_label_preprocess/column_roles.yaml).
That manifest is the single source of truth for which output column is a
**meta**, **label**, **feature**, or **quality** column — downstream
training scripts MUST consult it to prevent leakage.

```bash
cd ml_label_preprocess/
python preprocess.py --all                            # labels + features, baseline 1
python preprocess.py --all --baseline-cycle 3         # same, baseline = cycle 3
python preprocess.py --all --db-version A2.3          # override the DB tag
python preprocess.py --selftest                       # validate helpers + both pipelines
```

See [ml_label_preprocess/README.md](ml_label_preprocess/README.md) for the
full feature catalog, label decision table, current A2.2 distributions,
and the rationale for `regular_cycle` indexing.

---

## Stage 2 — `cell_classifier/` (classification)

`pip install`-able package with a console entry `cell-classifier`.
Predicts, from the 12 `fs_cv` features of a cell's first 5 regular
cycles, whether the cell will retain ≥ 0.85 discharge capacity past N
cycles (binary `pass` vs `bad`). Two modes:

- **`--mode validation`** — held-out evaluation under a chosen tuning
  protocol: `tune_inner_cv` (stratified 80/20 + inner-CV tune) or
  `nested_cv` (K outer folds, inner-CV tune per fold; every cell scored
  exactly once per seed). Emits per-seed metrics, per-cohort AUC,
  permutation importance, SHAP, and Optuna trial history.
- **`--mode production`** — train on all trainable cells, predict for
  every cell with features (including censored ones). No metrics; HPs
  are reused from the matching validation run via
  `--production-params-source from_validation_run`.

```bash
cd cell_classifier/
pip install -e .                                       # core install
cell-classifier run \
    --mode validation --model-config configs/rf.yaml \
    --N 300 --db-version A2.2 --baseline-cycle 1 --feature-subset fs_cv \
    --tuning-protocol nested_cv --outer-k 5
cell-classifier sweep --sweep configs/sweeps/rf_n_x_baseline.yaml
```

Each invocation writes to
`out/runs/{mode}/{model}__N{N}__{db}_b{baseline}__{feature_subset}/`
with a `manifest.json` (resolved config + SHA-256 for idempotency).
RandomForest is the only fully-wired model today; EBM (`interpret`) and
BART (`pymc-bart`) are scaffolded behind optional extras and slated for
v0.2 — see [cell_classifier/ROADMAP.md](cell_classifier/ROADMAP.md).

The classifier resolves preprocess bundles from
`../ml_label_preprocess/datasets/{db}_b{N}/` by default; override with
`BCC_PREPROCESS_ROOT` or `data.preprocess_root` in the config YAML.

See [cell_classifier/README.md](cell_classifier/README.md) for install
extras, sweep YAML schema, and notebook usage.

---

## Conventions shared by both stages

- **Activate `eis` before anything**:
  `source /home/mliao/miniconda3/etc/profile.d/conda.sh && conda activate eis`.
- **Paths via env vars, not hardcodes**: `BAT_ANNOT_DIR` (labeling),
  `BCC_PREPROCESS_ROOT` (classifier override).
- **`regular_cycle` indexing only** — never `tester_cycle`. The labeling
  README explains why this matters and why outputs are intentionally
  not drop-in compatible with the legacy `BOL_with_cv_features.csv`.
- **One bundle per `(db_version, baseline_cycle)`**. Every classifier
  run is parameterised by both axes plus `N` and `feature_subset`; the
  four together form the slug used for output paths and idempotency.
- **No data committed**. Annotation JSONs, preprocess bundles, and run
  outputs all live under gitignored directories.

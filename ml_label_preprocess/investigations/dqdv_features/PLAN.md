# Plan — dQ/dV feature extraction (investigation)

## Context

The 5-cycle ML pipeline ([ml_label_preprocess/features.py](../../../mnt/data/mliao/battery-ml-workbench/new_5cycle_ml/ml_label_preprocess/features.py)) currently emits 41 columns (Tier A retention + Tier B nominal-voltage + Tier C KWW CV-phase fits). We want to add **CC-phase dQ/dV-derived features** because dQ/dV captures phase-transition physics that the CV-decay fits don't see.

This work is **investigation-stage**: extract features into a *separate* folder, evaluate distributions and correlations, then later promote into `features.py` if they look useful. User explicitly asked to not pollute the current `datasets/` bundles.

Final feature set (4 columns, all dimensionless — Severson et al. 2019
ΔQ(V) statistics adapted to a 5-cycle baseline):

1. `dqv_norm_c5_c1_var_log10` — `log10(var(ΔQ_norm))`
2. `dqv_norm_c5_c1_min`       — `min(ΔQ_norm)`
3. `dqv_norm_c5_c1_mean`      — `mean(ΔQ_norm)`
4. `dqv_norm_c5_c1_skew`      — sample skewness of ΔQ_norm

Where `ΔQ_norm(V) = (|Q_c5(V)| − |Q_c1(V)|) / |c1_total_discharge|`.
The c1 capacity denominator is computed *inline* from the cell's own
cycle-1 discharge half-cycle (NOT looked up from
`cell_labels.parquet`) so the investigation has no external runtime
dependency.

### Feature-set history

- 8-feature draft (peak + Ah-scale ΔQ): 4 absolute-Ah features
  collapsed onto cohort capacity (47× ratio between 0MC ≈ 6.65 Ah and
  AR ≈ 0.14 Ah). Dropped.
- 4-feature voltage/dimensionless draft (peak V, peak shift,
  hysteresis, cosine sim): scale-free, but empirically didn't help
  the downstream classifier. Replaced.
- Current 4-feature draft (this file): narrowed to the Severson ΔQ(V)
  statistics with c1-capacity normalization baked in.

## Folder layout

New folder following the existing `investigations/jump_detection/` precedent:

```
ml_label_preprocess/investigations/dqdv_features/
├── PLAN.md                     # copy of this plan (first action after exit-plan-mode)
├── README.md                   # short how-to-run
├── dqdv_features.py            # pure feature functions (importable, unit-tested)
├── run_investigation.py        # CLI: iterates cells → writes out/{ts}/
├── selftest.py                 # synthetic-case tests
├── .gitignore                  # ignores out/, __pycache__/
└── out/                        # gitignored — all run outputs land here
    └── {YYYYMMDD_HHMM}/        # one subfolder per run, never overwritten
        ├── cell_dqdv_features.parquet
        ├── cell_dqdv_features.csv
        ├── cell_dqdv_features_status.csv
        ├── manifest.json
        └── plots/
            ├── overlay_<cell>.png      # dQ/dV c1+c3+c5 overlay, peaks marked
            └── deltaQ_<cell>.png       # Q_c5(V) − Q_c1(V) curve
```

Every output stays inside `investigations/dqdv_features/out/{timestamp}/`. The main `datasets/` tree is untouched.

## Reused primitives (do not reinvent)

From [battery-workbench-app/src/battery_workbench/core/analysis/dqdv.py](../../../mnt/data/mliao/battery-ml-workbench/battery-workbench-app/src/battery_workbench/core/analysis/dqdv.py):
- `extract_cc_voltage_capacity(cell_name, cd_index, direction="both")` — returns `{"charge": (V, Q), "discharge": (V, Q)}` with charge CC-only (CV plateau trimmed via `detect_cv_start`)
- `compute_dqdv(V, Q, method="savgol")` — defaults: 1 mV grid, savgol window=51, polyorder=3

From [ml_label_preprocess/_common.py](../../../mnt/data/mliao/battery-ml-workbench/new_5cycle_ml/ml_label_preprocess/_common.py):
- `iter_annotations()` — yields `(path, annot_json)` for every annotation JSON under `BAT_ANNOT_DIR`
- `iter_regulars(annot)` — sorted list of regular_cd events with regular_cycle 1, 2, 3, …
- `_cohort(cell_name)` — '0MC' / 'AR' labelling

We **do not** need `load_raw_tagged` — `extract_cc_voltage_capacity` does its own raw load via `get_cd_charge_raw` / `get_cd_discharge_raw`.

External: `numpy`, `scipy.signal.find_peaks`, `polars`, `matplotlib` (plots only).

## Feature-function design (in `dqdv_features.py`)

Pure functions, one per logical feature group. Each takes numpy arrays and returns floats or `None`/`nan`.

```python
def common_voltage_grid(V_c1, V_c5, step_v=0.005) -> np.ndarray:
    """Intersection of c1 and c5 voltage ranges, sampled at step_v."""

def delta_q_v_statistics(V_c1, Q_c1, V_c5, Q_c5, step_v=0.005) -> dict[str, float]:
    """{'dqv_c5_c1_var', '_min', '_mean'}. Interp Q onto common grid,
    take differences, return statistics. Returns NaN dict if either
    curve is too short or grids don't overlap."""

def find_dominant_peak(V, dqdv, *, direction) -> tuple[float, float] | None:
    """Return (V_peak, height) of the highest-prominence peak; None if
    no peak meets the prominence threshold. For 'discharge', operate
    on |dqdv| since discharge dqdv is negative."""

def cosine_similarity(V_c1, dqdv_c1, V_c5, dqdv_c5, step_v=0.005) -> float:
    """Interp both onto common grid, return cos(θ) = u·v/(|u||v|).
    NaN if either curve has zero norm or grids don't overlap."""

def featurize_cell(cell_name, annot) -> tuple[dict, dict]:
    """Returns (feature_row, status_row). Pulls cd_indices for
    regular_cycle 1..5 from annot, computes everything, gracefully
    degrades to NaN per feature on individual failures."""
```

Peak-detection params (defaults, tunable on CLI later):
- `find_peaks(prominence=peak_height_max * 0.05, distance=20)` on the savgol 1 mV grid
- Dominant peak = highest prominence

Common-grid step: 5 mV for ΔQ(V) statistics and cosine sim. Coarser than the savgol 1 mV grid (less noise; small numerical cost).

## CLI (in `run_investigation.py`)

```bash
# default: all A2.2 cells, output to out/{ts}/
python -m ml_label_preprocess.investigations.dqdv_features.run_investigation

# pilot: 5 cells per cohort + diagnostic plots
python ... --pilot

# subset for debug
python ... --cells AR-3420 0MC2-251022-001

# skip plots (faster)
python ... --no-plots
```

Status CSV columns (parallel to `cell_features_status.csv`):
- `cell_name`, `cohort`
- `has_c1_dis`, `has_c5_dis`, `has_c1_chg`, `has_c5_chg` (raw data presence)
- `n_peaks_c1_dis`, `n_peaks_c5_dis`, `n_peaks_c5_chg`
- `peak_detection_ok_c5_dis` (bool)
- `dqdv_n_success` (count of the 8 features that came out non-NaN)
- `error_msg` (str, if anything raised)

Manifest:
```json
{
  "schema_version": 1,
  "db_version": "A2.2",
  "annot_dir": "...",
  "generated_at": "...",
  "n_cells_attempted": ...,
  "n_cells_full": ...,        # all 8 features non-NaN
  "smoothing": {"method": "savgol", "window_length": 51, "polyorder": 3},
  "common_grid_step_v": 0.005,
  "peak_detection": {"prominence_frac": 0.05, "min_distance_samples": 20}
}
```

## Selftests (`selftest.py`)

Synthetic-case unit tests (run first, gate the CLI):
- **Gaussian peak**: synthetic `dqdv = Gaussian(V; mu=3.7, sigma=0.05)`. `find_dominant_peak` returns `V_peak` within 2 mV of 3.7.
- **Identical curves**: `cosine_similarity(curve, curve) ≈ 1.0`.
- **Orthogonal curves**: `cosine_similarity(sine, cosine) ≈ 0`.
- **Zero ΔQ**: identical Q(V) inputs → all three ΔQ statistics ≈ 0.
- **Shifted peak**: c5 peak shifted by +10 mV vs c1 → `dqdv_peak_v_shift_c1c5_dis ≈ +0.010` within tolerance.
- **Empty / short inputs**: every feature function returns NaN, not a crash.
- **Non-overlapping voltage ranges**: returns NaN, not garbage.

## Run plan

1. Create the folder + skeleton (`dqdv_features.py`, `run_investigation.py`, `selftest.py`, `README.md`, `.gitignore`). Copy this PLAN.md in.
2. Implement and run `selftest.py`. All pass before touching real data.
3. **Pilot run**: 5 0MC + 5 AR cells, plots on. Eyeball the overlay plots — do detected peaks sit on actual peaks? Do ΔQ(V) curves look smooth and non-degenerate?
4. **Full run**: all 458 featurizable A2.2 cells. Target <10% NaN-rate on any single feature.
5. **Distribution report** (text + maybe one summary plot): per-cohort mean/std of each feature; histogram of `dqdv_n_success`; the **per-cohort peak-voltage distribution** specifically — this is the empirical answer to "do 0MC and AR have different peak voltages?" (the thing I claimed at 50 mV without evidence).
6. Stop. Discuss results with user before any integration into `features.py` / `column_roles.yaml`.

## Critical files

- New: `ml_label_preprocess/investigations/dqdv_features/{PLAN.md, README.md, dqdv_features.py, run_investigation.py, selftest.py, .gitignore}`
- Read-only references:
  - [battery-workbench-app/src/battery_workbench/core/analysis/dqdv.py](../../../mnt/data/mliao/battery-ml-workbench/battery-workbench-app/src/battery_workbench/core/analysis/dqdv.py) — primitives
  - [ml_label_preprocess/_common.py](../../../mnt/data/mliao/battery-ml-workbench/new_5cycle_ml/ml_label_preprocess/_common.py) — `iter_annotations`, `iter_regulars`, `_cohort`
  - [ml_label_preprocess/features.py](../../../mnt/data/mliao/battery-ml-workbench/new_5cycle_ml/ml_label_preprocess/features.py) — style/pattern reference (do not modify in this phase)

## Verification

End-to-end check, in order:

1. `python selftest.py` — all cases pass.
2. `python run_investigation.py --pilot` — produces `out/{ts}/` with 10 rows + 10 overlay plots + 10 ΔQ plots. Manually inspect at least 3 plots: do detected peaks visually land on dQ/dV maxima? Is the c1 vs c5 overlay sensible (i.e. not horizontally translated by an artifact)?
3. `python run_investigation.py` (full A2.2) — completes in <15 min, status CSV shows >90% of cells with `dqdv_n_success == 8`.
4. Quick `polars` sanity on the output:
   ```python
   df = pl.read_parquet("out/{ts}/cell_dqdv_features.parquet")
   df.group_by("cohort").agg([pl.col("dqdv_peak_v_c5_dis").mean(),
                              pl.col("dqdv_peak_v_c5_dis").std()])
   df.select(pl.corr("dqv_c5_c1_var",
                     <discharge_capacity_retention_final from cell_features>))
   ```
   — peak-voltage cohort means quantify the actual 0MC vs AR shift.
   — `dqv_c5_c1_var` should correlate with the Tier-A retention column (sanity), but ideally carry independent signal too.
5. Report results back to user; no `features.py` / manifest changes in this phase.

## Out of scope (explicitly)

- Integration into `column_roles.yaml` or `features.py`.
- Re-running `preprocess.py --all`.
- Touching `datasets/A2.2_b1/` or `_b3/`.
- Additional features beyond the agreed 8.
- A b3 baseline variant. (dQ/dV is computed from raw V/Q on cycles 1 and 5; baseline-cycle N0 is not a parameter for any of these 8 features.)

# DOP-peak feature investigation

Standalone exploration of **Distribution-of-Polarization (DOP)** peak
positions extracted from the first 10 min of charge and first 5 min of
discharge on cycles 1 and 5 of every cell. **Not** integrated into
`../../features.py` / `column_roles.yaml` — outputs land here, parallel
to the main pipeline, so the main `datasets/` bundles are untouched.

Mirrors `../dqdv_features/` in shape so the two investigations are
easy to compare. The hypothesis: the θ shift of the dominant DOP peak
between cycle 1 and cycle 5 carries fade information that's orthogonal
to what dQ/dV captures.

## Features (6 columns)

Each fit produces a ρ(θ) curve. The "dominant peak" is the peak with
the largest `rho_max` after the wrapper's built-in θ ≤ 5° (Ohmic) filter.

| Column | Description |
|---|---|
| `dop_peak_theta_c1_chg`            | θ (deg) of dominant DOP peak, cycle 1 charge (first 10 min) |
| `dop_peak_theta_c5_chg`            | θ (deg) of dominant DOP peak, cycle 5 charge (first 10 min) |
| `dop_peak_theta_c1_dis`            | θ (deg) of dominant DOP peak, cycle 1 discharge (first 5 min) |
| `dop_peak_theta_c5_dis`            | θ (deg) of dominant DOP peak, cycle 5 discharge (first 5 min) |
| `dop_peak_theta_shift_chg_c1c5`    | Signed `θ_c5_chg − θ_c1_chg` |
| `dop_peak_theta_shift_dis_c1c5`    | Signed `θ_c5_dis − θ_c1_dis` |

Failure modes (any → NaN for the affected column, propagating to the
shift):
- raw extract too short (< 4 samples after CV trim + window crop)
- hybrid-drt `fit_chrono` raises
- DOP fit produces zero peaks (after the θ ≤ 5° filter)

## Dependencies

Pure Python: `numpy`, `polars`, `matplotlib`.

External:
- **`battery_workbench.core.analysis.drt` / `drt_wrapper`** — provides
  `extract_cd_transient` and `DRTAnalyzer.fit_transient`.
  `battery-workbench-app` is wired into the `eis` env via .pth.
- **`hybrid-drt`** — the actual DRT solver. Local-path install required:
  ```bash
  pip install -e /mnt/data/mliao/battery-ml-workbench/ChronoAnalytics/external/hybrid-drt
  ```
  Smoke-test before running:
  ```bash
  python -c "from battery_workbench.core.analysis.drt_wrapper import DRTAnalyzer; print('drt OK')"
  ```

## Running

```bash
source /home/mliao/miniconda3/etc/profile.d/conda.sh && conda activate eis
cd /mnt/data/mliao/battery-ml-workbench/new_5cycle_ml/ml_label_preprocess/investigations/drt_dop_features

# 1. selftest — pure-helper unit tests (no hybrid-drt fits, fast).
python selftest.py

# 2. pilot: 5 0MC + 5 AR cells with diagnostic plots. Watch stdout for
#    per-cell fit_time_s_total — pilot exists to time the full sweep.
python run_investigation.py --pilot

# 3. full A2.2 run (~458 cells × 4 fits each)
python run_investigation.py

# 4. debugging a specific cell
python run_investigation.py --cells AR-3420 0MC2-251022-001 --plots
```

Every run writes to `out/{YYYYMMDD_HHMM}/`. Nothing is overwritten.

## Output layout per run

```
out/{ts}/
├── cell_dop_features.parquet         # 1 row per cell, 8 cols (cell + cohort + 6 features)
├── cell_dop_features.csv
├── cell_dop_features_status.csv      # per-cell QC (extract flags, DOP convergence, timing)
├── manifest.json                     # provenance: db_version, params, counts
└── plots/                            # only if --plots or --pilot
    └── overlay_<cell>.png            # 4 panels: c1/c5 × chg/dis ρ(θ) with dominant peak marked
```

### Adopting a fresh run downstream

No downstream consumers yet. If the pilot results look promising, the
follow-up is to create `cell_lifetime/experiments/exp_*_dop_classifier/`
mirroring `exp_o_dqdv_classifier` and pin its `DOP_PATH` constant at
the parquet path here.

## Not in scope for this folder

- Modifying `../../features.py` or `../../column_roles.yaml`
- Re-running `../../preprocess.py`
- Writing to `../../datasets/`
- DRT features (the same `fit_transient` call returns them too — clear
  future extension; see plan history)
- Peak height (`rho_max`), peak width, or the fit's `r_inf`/`r_pol`/
  `r_total` — also free per fit but deferred to v1

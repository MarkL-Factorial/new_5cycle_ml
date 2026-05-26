# dQ/dV feature investigation

Standalone exploration of two dQ/dV feature designs (v1 + v2) extracted
from the first 5 regular cycles of every cell. **Not** integrated into
`../../features.py` / `column_roles.yaml` — outputs land here, parallel
to the main pipeline, so the main `datasets/` bundles are untouched.

See [PLAN.md](PLAN.md) for the full design and history.

## Features (8 columns: v1 + v2)

Each run of `run_investigation.py` emits both feature sets into two
separate parquets in the same timestamped folder. Downstream consumers
(`cell_lifetime/experiments/exp_o`, `exp_p`, `exp_q`) join whichever
set they need.

### v1 — physically-interpretable peak/shape features (4 columns)

Computed on the smoothed dQ/dV curves (savgol on 1 mV grid). Sign
conventions are inherited from the frozen v1 parquet at
`out/20260521_1406/`.

| Column | Description |
|---|---|
| `dqdv_peak_v_c5_dis`                  | V of the dominant dQ/dV peak at cycle 5 discharge |
| `dqdv_peak_v_shift_c1c5_dis`          | Signed `V_peak_c5 − V_peak_c1` (V); negative = peak shifts to lower V |
| `dqdv_charge_discharge_hysteresis_c5` | Signed `V_peak_charge − V_peak_discharge` (V) at c5; almost always positive |
| `dqdv_cosine_sim_c1c5_dis`            | Cosine similarity of c1 and c5 discharge dQ/dV(V) on the common voltage grid |

### v2 — Severson ΔQ(V) statistics (4 columns, all dimensionless)

`ΔQ_norm(V) = ( |Q_c5(V)| − |Q_c1(V)| ) / |c1_total_discharge_capacity|`

The denominator is computed inline from the cell's own cycle-1
discharge half-cycle (NOT from `cell_labels.parquet`) so the
investigation stays self-contained. Normalizing by the cell's own c1
capacity removes the 47× cohort scale (0MC ≈ 6.65 Ah baseline vs AR ≈
0.14 Ah) so the features measure fractional capacity drift, not cell
size.

| Column | Description |
|---|---|
| `dqv_norm_c5_c1_var_log10` | `log10(var(ΔQ_norm))` — Severson's headline cycle-life feature |
| `dqv_norm_c5_c1_min`       | `min(ΔQ_norm)` — largest single-voltage capacity loss as a fraction of c1 |
| `dqv_norm_c5_c1_mean`      | `mean(ΔQ_norm)` — average fractional capacity drift |
| `dqv_norm_c5_c1_skew`      | `skew(ΔQ_norm)` — shape of the loss profile |

### History

- **8-feature draft** (peaks + Severson Ah-scale): 4 of the 8 features
  were in absolute Ah and got dominated by the 47× cohort capacity
  ratio. Dropped.
- **v1**: peak V, peak shift, hysteresis, cosine sim — all
  dimensionless / V. Empirically didn't help downstream classification.
- **v2**: Severson ΔQ(V) statistics with c1-capacity normalization.
  Replaced v1, which was then deleted from the module.
- **Current revision** (this file): both v1 and v2 are emitted from one
  pass so each can be regenerated on fresh annotation snapshots without
  re-deriving the v1 code. Downstream experiments
  (`cell_lifetime/experiments/exp_o`, `exp_p`, `exp_q`) join whichever
  set they need.

## Running

```bash
source /home/mliao/miniconda3/etc/profile.d/conda.sh && conda activate eis
cd /mnt/data/mliao/battery-ml-workbench/new_5cycle_ml/ml_label_preprocess/investigations/dqdv_features

# 1. selftest — synthetic-case unit tests; gates the CLI
python selftest.py

# 2. pilot: 5 0MC + 5 AR cells with diagnostic plots
python run_investigation.py --pilot

# 3. full A2.2 run (~458 cells)
python run_investigation.py

# 4. debugging a specific cell
python run_investigation.py --cells AR-3420 0MC2-251022-001 --plots
```

Every run writes to `out/{YYYYMMDD_HHMM}/`. Nothing is overwritten.

## Output layout per run

```
out/{ts}/
├── cell_dqdv_features_v1.parquet    # 1 row per cell, 6 cols (cell_name + cohort + 4 v1)
├── cell_dqdv_features_v1.csv
├── cell_dqdv_features_v2.parquet    # 1 row per cell, 6 cols (cell_name + cohort + 4 v2)
├── cell_dqdv_features_v2.csv
├── cell_dqdv_features_status.csv    # per-cell QC for both sets (success counts split v1 / v2)
├── manifest.json                    # provenance: db_version, params, counts (v1 + v2)
└── plots/                           # only if --plots or --pilot
    └── overlay_<cell>.png           # dQ/dV c1+c3+c5 overlay with peaks marked
```

### Adopting a fresh run downstream

`cell_lifetime/experiments/exp_o`, `exp_p`, and `exp_q` each pin a
`DQDV_PATH` to a frozen snapshot. When adopting a new run, update the
constant to point at the corresponding parquet in this folder
(`cell_dqdv_features_v1.parquet` for exp_o, `_v2.parquet` for exp_p,
both for exp_q). The column names are stable across runs.

## Dependencies

Uses `extract_cc_voltage_capacity` + `compute_dqdv` from
`battery_workbench.core.analysis.dqdv` (the workbench-app .pth wire-up
is already in the `eis` env — same setup that `../../features.py` uses
for Tier B/C). Also `numpy`, `scipy.signal.find_peaks`, `polars`,
`matplotlib`.

## Not in scope for this folder

- Modifying `../../features.py` or `../../column_roles.yaml`
- Re-running `../../preprocess.py`
- Writing to `../../datasets/`

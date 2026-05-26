# dqdv_v1 — dQ/dV peak-shape features (4 columns)

## What this candidate is

Four physically-interpretable peak/shape descriptors of the discharge
dQ/dV curve, computed on cycles 1 and 5 of every cell from the smoothed
(Savitzky-Golay) curves on a 1 mV grid. Captures where the dominant
dQ/dV peak sits, how it moves between c1 and c5, the charge↔discharge
hysteresis at c5, and the c1-vs-c5 curve similarity. Source:
[`../../investigations/dqdv_features`](../../investigations/dqdv_features).

## Feature columns

| Column | Description |
|---|---|
| `dqdv_peak_v_c5_dis` | V of the dominant dQ/dV peak at cycle 5 discharge |
| `dqdv_peak_v_shift_c1c5_dis` | Signed `V_peak_c5 − V_peak_c1` (V); negative = peak shifts to lower V |
| `dqdv_charge_discharge_hysteresis_c5` | Signed `V_peak_charge − V_peak_discharge` (V) at c5; almost always positive |
| `dqdv_cosine_sim_c1c5_dis` | Cosine similarity of c1 and c5 discharge dQ/dV(V) on the common voltage grid |

## Provenance

See [`provenance.json`](provenance.json). Source:
`investigations/dqdv_features/out/20260522_1047/cell_dqdv_features_v1.parquet` —
461/470 cells full, db_version `A2.2`, schema_version 2.

## Refresh

To refresh from a newer investigation snapshot, follow the procedure
in [`../README.md`](../README.md#how-to-refresh-an-existing-candidate).
TL;DR:

1. Copy the fresh `cell_dqdv_features_v1.parquet` over `features.parquet`
   (and the matching `.csv`).
2. Update `provenance.json:source.snapshot` and the `source.manifest` block
   (`db_version`, `schema_version`, `generated_at`, `mode`,
   `n_cells_attempted`, `n_cells_full`).

If the investigation's extraction logic changed (not just new data),
also re-copy the files under `scripts/`. If the column set changes,
bump to a new candidate folder (`dqdv_v1_rev2/`) rather than overwriting.

## Downstream pinning

New downstream experiments pin to this candidate's stable artifact path:

```python
FEATURES_PATH = Path(
    "ml_label_preprocess/feature_candidates/dqdv_v1/features.parquet"
)
```

Read [`provenance.json`](provenance.json) in your downstream experiment
and log the resolved snapshot for reproducibility.

`cell_lifetime/experiments/exp_o_dqdv_classifier` is **not migrated**
to this convention — it stays frozen against its original investigation
snapshot.

## Notes on the copied `scripts/`

The investigation's [`scripts/extraction.py`](scripts/extraction.py)
(a copy of `investigations/dqdv_features/dqdv_features.py`) emits
**both** v1 and v2 feature columns in a single pass. Only the 4 v1
columns above are present in this candidate's `features.parquet`; the
v2 columns live in the sibling [`../dqdv_v2/`](../dqdv_v2/) candidate.

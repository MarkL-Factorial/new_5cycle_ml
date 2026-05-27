# dqdv_v2 — Severson ΔQ(V) statistics (4 columns)

## What this candidate is

Four dimensionless statistical features of the Severson-style ΔQ(V)
curve between cycle 1 and cycle 5. The capacity curve `Q(V)` is
re-sampled onto a common voltage grid; the residual
`ΔQ_norm(V) = (Q_c5(V) − Q_c1(V)) / c1_total_discharge_capacity`
removes the 47× capacity ratio between cohorts (0MC ≈ 6.65 Ah vs
AR ≈ 0.14 Ah) so the features measure fractional drift, not cell
size. Captures the headline `var(log10 ΔQ_norm)` cycle-life feature
from Severson et al. plus three shape descriptors. Source:
[`../../investigations/dqdv_features`](../../investigations/dqdv_features).

## Feature columns

| Column | Description |
|---|---|
| `dqv_norm_c5_c1_var_log10` | `log10(var(ΔQ_norm))` — Severson's headline cycle-life feature |
| `dqv_norm_c5_c1_min` | `min(ΔQ_norm)` — largest single-voltage capacity loss as a fraction of c1 |
| `dqv_norm_c5_c1_mean` | `mean(ΔQ_norm)` — average fractional capacity drift |
| `dqv_norm_c5_c1_skew` | `skew(ΔQ_norm)` — shape of the loss profile |

## Provenance

See [`provenance.json`](provenance.json). Primary source:
`investigations/dqdv_features/out/20260522_1047/cell_dqdv_features_v2.parquet` —
470 attempted, 462 with all 4 features non-NaN, db_version `A2.2`,
schema_version 2, mode `full`.

Plus 5 cells merged in from a `--cells` partial run; see
`provenance.json:merged_in[*]` for the structured audit trail.

Total `features.parquet` rows = `source.manifest.n_cells_attempted` (470)
+ union of `merged_in[*].cells` (5) = **475**.

## Refresh

To refresh from a newer investigation snapshot, follow the procedure
in [`../README.md`](../README.md#how-to-refresh-an-existing-candidate).
TL;DR:

1. Copy the fresh `cell_dqdv_features_v2.parquet` over `features.parquet`
   (and the matching `.csv`).
2. Update `provenance.json:source.snapshot` and the `source.manifest` block.

If the investigation's extraction logic changed, also re-copy the
files under `scripts/`. If the column set changes, bump to a new
candidate folder (`dqdv_v2_rev2/`) rather than overwriting.

## Downstream pinning

```python
FEATURES_PATH = Path(
    "ml_label_preprocess/feature_candidates/dqdv_v2/features.parquet"
)
```

Read [`provenance.json`](provenance.json) in your downstream experiment
and log the resolved snapshot for reproducibility.

`cell_lifetime/experiments/exp_p_dqdv_v2_classifier` is **not migrated**
to this convention — it stays frozen against its original investigation
snapshot.

## Notes on the copied `scripts/`

The investigation's [`scripts/extraction.py`](scripts/extraction.py)
(a copy of `investigations/dqdv_features/dqdv_features.py`) emits
**both** v1 and v2 feature columns in a single pass. Only the 4 v2
columns above are present in this candidate's `features.parquet`; the
v1 columns live in the sibling [`../dqdv_v1/`](../dqdv_v1/) candidate.

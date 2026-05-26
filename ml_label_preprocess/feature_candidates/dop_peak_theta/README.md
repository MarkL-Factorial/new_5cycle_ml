# dop_peak_theta — DOP peak θ features (6 columns)

## What this candidate is

Six features extracted from the Distribution of Polarization (DOP)
distribution `ρ(θ)` produced by hybrid-drt's `fit_chrono` on short
chronoamperometry windows: the first **10 minutes of charge** and the
first **5 minutes of discharge** at cycles 1 and 5. Each fit yields a
ρ(θ) curve; we keep only the θ (degrees) of the dominant peak (largest
ρ_max, post Ohmic filter at θ ≤ 5°). Two derived signed c5−c1 shifts
capture the cycle-to-cycle drift that's hypothesized to track cell
fade. Source:
[`../../investigations/drt_dop_features`](../../investigations/drt_dop_features).

## Feature columns

| Column | Description |
|---|---|
| `dop_peak_theta_c1_chg` | θ (deg) of dominant DOP peak, cycle 1 charge (first 10 min) |
| `dop_peak_theta_c5_chg` | θ (deg) of dominant DOP peak, cycle 5 charge (first 10 min) |
| `dop_peak_theta_c1_dis` | θ (deg) of dominant DOP peak, cycle 1 discharge (first 5 min) |
| `dop_peak_theta_c5_dis` | θ (deg) of dominant DOP peak, cycle 5 discharge (first 5 min) |
| `dop_peak_theta_shift_chg_c1c5` | Signed `θ_c5_chg − θ_c1_chg` |
| `dop_peak_theta_shift_dis_c1c5` | Signed `θ_c5_dis − θ_c1_dis` |

## Provenance

See [`provenance.json`](provenance.json). Primary source:
`investigations/drt_dop_features/out/20260522_1532/cell_dop_features.parquet` —
470 attempted, 429 with all 6 features non-NaN, db_version `A2.2`,
schema_version 1, mode `full`.

Plus 5 cells merged in from a `--cells` partial run; see
`provenance.json:merged_in[*]` for the structured audit trail.

Total `features.parquet` rows = `source.manifest.n_cells_attempted` (470)
+ union of `merged_in[*].cells` (5) = **475**.

## Refresh

To refresh from a newer full-sweep investigation snapshot, follow the
procedure in
[`../README.md`](../README.md#how-to-refresh-an-existing-candidate).
TL;DR:

1. Copy the fresh `cell_dop_features.parquet` over `features.parquet`
   (and the matching `.csv`).
2. Update `provenance.json:source.snapshot` and `source.manifest`.

To add **only a few new cells** without recomputing the full ~3-hour
sweep, follow
[`../README.md` § How to add NEW cells](../README.md#how-to-add-new-cells-to-an-existing-candidate)
— `--cells X Y Z` partial run, then merge into `features.parquet`,
then append a `merged_in` entry to `provenance.json`.

If the investigation's extraction logic changed (not just new data),
also re-copy the files under `scripts/`. If the column set changes,
bump to a new candidate folder (`dop_peak_theta_rev2/`).

## Downstream pinning

```python
FEATURES_PATH = Path(
    "ml_label_preprocess/feature_candidates/dop_peak_theta/features.parquet"
)
```

Read [`provenance.json`](provenance.json) in your downstream experiment
and log both the `source.snapshot` and any `merged_in` snapshots for
reproducibility.

No `cell_lifetime/experiments/exp_*` consumer exists for DOP yet.

## Notes on the copied `scripts/`

[`scripts/extraction.py`](scripts/extraction.py) (a copy of
`investigations/drt_dop_features/dop_features.py`) emits exactly these
6 columns. Hard dependency on `hybrid-drt`
(`ChronoAnalytics/external/hybrid-drt`) — see the investigation README
for the install line.

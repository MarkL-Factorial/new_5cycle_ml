#!/usr/bin/env python
"""Post-hoc verification: confirm all 3 models per seed share the SAME
20% held-out test set, and report the across-seed fingerprints so any
two-seed reshuffles are visible.

Why this exists: in a fair RSF vs regressor comparison the test set
must be byte-identical for all three models. The driver computes a
single `test_idx` and uses it for all predictions, so by construction
this holds — but if anyone refactors run.py later, this script will
catch a regression.

Reads:
  - runs/seed_*/predictions.csv  → one cell_name column shared by 3 model preds
  - runs/seed_*/results.json     → recorded test_cell_fingerprint

Prints a small per-seed summary plus the across-seed table.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pandas as pd


HERE = Path(__file__).resolve().parent


def fingerprint(cells: list[str]) -> str:
    return hashlib.sha256(",".join(sorted(cells)).encode()).hexdigest()[:16]


def main() -> int:
    rows = []
    for seed_dir in sorted((HERE / "runs").glob("seed_*")):
        pred_path = seed_dir / "predictions.csv"
        res_path = seed_dir / "results.json"
        if not pred_path.exists() or not res_path.exists():
            continue
        df = pd.read_csv(pred_path)
        results = json.loads(res_path.read_text())

        # (1) Inside a single seed there's exactly one cell_name column → all
        # 3 models score the same cells by construction. We assert that the
        # csv has columns for all 3 models and no row-wise nan in cell_name.
        for col in ("cell_name", "rsf_pred", "xgb_pred", "ebm_pred", "y_true"):
            assert col in df.columns, f"{seed_dir.name}: missing column {col}"
        assert df["cell_name"].notna().all(), f"{seed_dir.name}: NaN cell_name"
        assert len(df) == len(df["cell_name"].unique()), f"{seed_dir.name}: duplicates"

        # (2) Fingerprint matches what the driver recorded.
        cell_list = df["cell_name"].astype(str).tolist()
        fp_now = fingerprint(cell_list)
        fp_recorded = results.get("test_cell_fingerprint", "—")
        if fp_recorded != "—":
            assert fp_now == fp_recorded, (
                f"{seed_dir.name}: fingerprint drift "
                f"(csv→{fp_now}, results→{fp_recorded})"
            )

        rows.append({
            "seed": int(seed_dir.name.split("_")[1]),
            "n_test": len(df),
            "fingerprint": fp_now,
            "first_3_cells": ", ".join(sorted(cell_list)[:3]),
        })

    if not rows:
        print("No per-seed predictions found.")
        return 1

    df = pd.DataFrame(rows).sort_values("seed")
    print("All 3 models share the same test cells within each seed:")
    print(df.to_string(index=False))

    if df["fingerprint"].nunique() == len(df):
        print(
            "\nAcross seeds: every fingerprint distinct (test sets differ "
            "by seed — expected, that's how multi-seed CV works)."
        )
    else:
        print(
            f"\nAcross seeds: {df['fingerprint'].nunique()} distinct "
            f"fingerprints among {len(df)} seeds — investigate if not "
            "deliberate."
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

#!/usr/bin/env python
"""Verify Exp I's per-seed test cells are byte-identical to Exp H's.

Why this matters: the censored-data ablation only isolates the
censored contribution if both variants are scored against the EXACT
same 38 faded cells per seed — which means matching Exp H's split
exactly. The driver computes the split via train_test_split(faded_idx,
random_state=seed, shuffle=True), so if the loader's row ordering is
the same and `faded_idx` is the same, the test cells match.

This script reads Exp H's recorded fingerprints and Exp I's current
fingerprints and confirms they match per-seed.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pandas as pd


HERE = Path(__file__).resolve().parent
EXP_H = HERE.parent / "exp_h_rsf_vs_regressors_fair"


def _exp_h_fingerprint(seed: int) -> str | None:
    """Read Exp H's seed fingerprint; fall back to recomputing it from
    predictions.csv if results.json doesn't store the field (the original
    Exp H run didn't record it)."""
    res_path = EXP_H / "runs" / f"seed_{seed}" / "results.json"
    if res_path.exists():
        fp = json.loads(res_path.read_text()).get("test_cell_fingerprint")
        if fp is not None:
            return fp
    pred_path = EXP_H / "runs" / f"seed_{seed}" / "predictions.csv"
    if not pred_path.exists():
        return None
    cells = pd.read_csv(pred_path)["cell_name"].astype(str).tolist()
    return hashlib.sha256(",".join(sorted(cells)).encode()).hexdigest()[:16]


def main() -> int:
    rows = []
    mismatches = 0
    for seed_dir in sorted((HERE / "runs").glob("seed_*")):
        seed = int(seed_dir.name.split("_")[1])
        i_res_path = seed_dir / "results.json"
        if not i_res_path.exists():
            print(f"seed={seed}: no Exp I results.json yet")
            continue
        i_fp = json.loads(i_res_path.read_text()).get("test_cell_fingerprint")
        h_fp = _exp_h_fingerprint(seed)
        if h_fp is None:
            print(f"seed={seed}: no Exp H data to compare")
            continue
        match = (i_fp == h_fp)
        if not match:
            mismatches += 1
        rows.append({
            "seed": seed,
            "exp_h_fingerprint": h_fp,
            "exp_i_fingerprint": i_fp,
            "match": match,
        })

    if not rows:
        print("No seed results found.")
        return 1

    df = pd.DataFrame(rows)
    print(df.to_string(index=False))
    if mismatches:
        print(f"\nFAIL: {mismatches} seed(s) have different test cells than Exp H.")
        return 1
    print(f"\nOK: all {len(df)} seeds match Exp H's test cells.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

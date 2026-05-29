"""Score cells with persisted EBM + RSF ensembles.

Inverse of `run_production`: instead of training, this loads the joblib'd
EBM (3 horizons × K seeds) + RSF (K seeds) from a run dir's
`models_<timestamp>/` folder and scores every cell in `cell_features.parquet`
(plus the dqdv_v1 + dop_peak_theta joins).

Defaults to the parquet paths recorded in `predict_manifest.json`. CLI
overrides take precedence for batch scoring against an updated
features parquet (e.g. with newly-arrived cells appended).

Output CSV schema is a subset of production's `predictions_*.csv`:
`prob_pass_n{200,300,400}`, `pred_pass_n{200,300,400}`,
`prob_pass_n{N}_std`, `rsf_median_cycle`, `rsf_median_cycle_std`. Columns
that depend on labels (`true_pass_n*`, `in_training_set_n*`, `event`,
`last_fade_cycle`, `time`) are not emitted — new cells are
inference-only.
"""

from __future__ import annotations

import json
import logging
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
import polars as pl

from cell_lifetime.pipelines.production import (
    DOP_COLS,
    DQDV_V1_COLS,
    _join_block,
    _load_block,
    median_survival_from_sf,
)


def _setup_logging(log_path: Path) -> logging.Logger:
    root = logging.getLogger()
    root.setLevel(logging.INFO)
    for h in list(root.handlers):
        root.removeHandler(h)
    fmt = logging.Formatter(
        "%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%Y-%m-%dT%H:%M:%S",
    )
    fh = logging.FileHandler(log_path, mode="w")
    fh.setFormatter(fmt)
    sh = logging.StreamHandler(sys.stdout)
    sh.setFormatter(fmt)
    root.addHandler(fh)
    root.addHandler(sh)
    return logging.getLogger(__name__)


def _resolve_models_dir(run_dir: Path) -> Path:
    """Find the single timestamped models_*/ folder inside run_dir."""
    candidates = sorted(c for c in run_dir.glob("models_*") if c.is_dir())
    if not candidates:
        raise FileNotFoundError(
            f"No models_*/ folder in {run_dir}. The production run that "
            f"produced this directory predates the model-persistence patch; "
            f"re-run `cell-lifetime production` to populate it."
        )
    if len(candidates) > 1:
        raise RuntimeError(
            f"Found multiple models_*/ folders in {run_dir}: "
            f"{[c.name for c in candidates]}. Point --run-dir at a specific "
            f"run, or remove the extras."
        )
    return candidates[0]


def run_predict(
    *,
    run_dir: Path,
    out_csv: Path | None = None,
    cell_features_path: Path | None = None,
    dqdv_path: Path | None = None,
    dop_path: Path | None = None,
) -> dict[str, Any]:
    """Load persisted ensembles from `run_dir`, score cells from input parquets.

    Returns a dict with `out_csv`, `n_cells`, `ensemble_seeds`, `horizons`,
    and per-horizon `pass_rates`.
    """
    run_dir = Path(run_dir).resolve()
    if not run_dir.is_dir():
        raise FileNotFoundError(f"run_dir {run_dir} does not exist")
    models_dir = _resolve_models_dir(run_dir)
    manifest_path = models_dir / "predict_manifest.json"
    if not manifest_path.is_file():
        raise FileNotFoundError(f"predict_manifest.json not found in {models_dir}")
    manifest = json.loads(manifest_path.read_text())

    cell_features_path = Path(
        cell_features_path or manifest["inputs"]["cell_features_path"]
    )
    dqdv_path = Path(dqdv_path or manifest["inputs"]["dqdv_v1_path"])
    dop_path = Path(dop_path or manifest["inputs"]["dop_path"])

    timestamp = datetime.now().strftime("%Y%m%d_%H%M")
    if out_csv is None:
        out_csv = run_dir / f"predict_{timestamp}.csv"
    out_csv = Path(out_csv)
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    log_path = out_csv.with_suffix(".log")
    log = _setup_logging(log_path)

    log.info(f"Predict from {run_dir}")
    log.info(f"  models_dir: {models_dir.name}")
    log.info(f"  cell_features: {cell_features_path}")
    log.info(f"  dqdv_v1: {dqdv_path}")
    log.info(f"  dop_peak_theta: {dop_path}")

    feature_columns: list[str] = manifest["feature_columns"]
    horizons: list[int] = list(manifest["horizons"])
    rsf_t_cap = float(manifest["rsf_t_cap"])
    K = int(manifest["ensemble_seeds"])
    extra_blocks = set(manifest.get("extra_feature_blocks", []))
    have_v1 = "dqdv_v1" in extra_blocks
    have_dop = "dop_peak_theta" in extra_blocks

    # The base columns are whatever's in feature_columns that isn't in the
    # v1 or dop blocks, preserving original order.
    v1_set = set(DQDV_V1_COLS) if have_v1 else set()
    dop_set = set(DOP_COLS) if have_dop else set()
    base_cols = [c for c in feature_columns if c not in v1_set and c not in dop_set]
    log.info(
        f"Manifest: {len(feature_columns)} feature cols "
        f"(base={len(base_cols)}, v1={'on' if have_v1 else 'off'}, "
        f"dop={'on' if have_dop else 'off'}); K={K}; horizons={horizons}; "
        f"rsf_t_cap={rsf_t_cap:.1f}"
    )

    cf = pl.read_parquet(cell_features_path)
    missing = [c for c in base_cols if c not in cf.columns]
    if missing:
        raise RuntimeError(
            f"cell_features.parquet at {cell_features_path} is missing required "
            f"base columns: {missing}"
        )
    base_df = cf.select(["cell_name", *base_cols]).to_pandas()
    cell_names = base_df["cell_name"].to_numpy()
    X = base_df[base_cols].copy()
    log.info(f"Loaded {len(X)} cells × {len(base_cols)} base columns")

    if have_v1:
        v1_block = _load_block(dqdv_path, DQDV_V1_COLS)
        X, miss_v1 = _join_block(X, cell_names, v1_block, DQDV_V1_COLS)
        log.info(
            f"Joined dqdv_v1 ({len(DQDV_V1_COLS)} cols); "
            f"cells missing from parquet: {miss_v1}/{len(X)}"
        )
    if have_dop:
        dop_block = _load_block(dop_path, DOP_COLS)
        X, miss_dop = _join_block(X, cell_names, dop_block, DOP_COLS)
        log.info(
            f"Joined dop_peak_theta ({len(DOP_COLS)} cols); "
            f"cells missing from parquet: {miss_dop}/{len(X)}"
        )

    if list(X.columns) != feature_columns:
        log.info(
            f"Column order differs from manifest; reordering to match training."
        )
        X = X[feature_columns]
    log.info(f"Final feature matrix: {X.shape}")

    n_total = len(cell_names)
    out_rows: dict[str, np.ndarray] = {"cell_name": cell_names}
    for N in horizons:
        member_probs: list[np.ndarray] = []
        for k in range(K):
            model = joblib.load(models_dir / f"ebm_classifier_n{N}_seed{k}.joblib")
            member_probs.append(model.predict_proba(X)[:, 1])
        stack = np.stack(member_probs, axis=0)
        prob_mean = stack.mean(axis=0)
        prob_std = stack.std(axis=0, ddof=0) if K > 1 else np.zeros(n_total)
        pred = (prob_mean >= 0.5).astype(np.int8)
        out_rows[f"prob_pass_n{N}"] = prob_mean
        out_rows[f"prob_pass_n{N}_std"] = prob_std
        out_rows[f"pred_pass_n{N}"] = pred
        log.info(
            f"  N={N}: pass rate={pred.mean():.3f}, mean prob={prob_mean.mean():.3f}, "
            f"mean std={prob_std.mean():.4f}"
        )

    rsf_medians: list[np.ndarray] = []
    for k in range(K):
        rsf = joblib.load(models_dir / f"rsf_seed{k}.joblib")
        sfs = rsf.predict_survival_curve(X)
        med = np.array(
            [median_survival_from_sf(sf, rsf_t_cap) for sf in sfs], dtype=float,
        )
        rsf_medians.append(med)
    rsf_stack = np.stack(rsf_medians, axis=0)
    out_rows["rsf_median_cycle"] = rsf_stack.mean(axis=0)
    out_rows["rsf_median_cycle_std"] = (
        rsf_stack.std(axis=0, ddof=0) if K > 1 else np.zeros(n_total)
    )
    log.info(
        f"  rsf median range "
        f"[{out_rows['rsf_median_cycle'].min():.0f}, "
        f"{out_rows['rsf_median_cycle'].max():.0f}], "
        f"mean {out_rows['rsf_median_cycle'].mean():.1f}, "
        f"mean std {out_rows['rsf_median_cycle_std'].mean():.1f}"
    )

    df = pd.DataFrame(out_rows)
    df.to_csv(out_csv, index=False)
    log.info(f"Wrote {out_csv} ({len(df)} rows × {len(df.columns)} cols)")

    summary = {
        "out_csv": str(out_csv),
        "log_path": str(log_path),
        "n_cells": int(len(df)),
        "ensemble_seeds": K,
        "horizons": horizons,
        "pass_rates": {N: float(df[f"pred_pass_n{N}"].mean()) for N in horizons},
    }
    log.info(f"Done. {summary}")
    return summary

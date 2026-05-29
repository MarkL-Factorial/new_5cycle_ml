"""Production orchestrator: 3 classifiers (N=200/300/400) + 1 RSF.

Called by the `cell-lifetime production` CLI subcommand. Trains the
canonical battery-of-models on the trainable_n{N} cohort (per
CONVENTIONS.md), runs inference on every cell admitted by the loader
(459 in the A2.2_b1 May-19 bundle: 444 single_rate with n_reg≥5 plus
15 rate_changed cells held out of training but scored for predict-only
output), and writes a wide `predictions.csv` plus best-params JSON, log,
and 3 plots to a timestamped run directory.

All four models share the same 13-col feature vector:
`fs_a_only` (3 cols from the bundle) + `dqdv_v1` (4 cols from
feature_candidates) + `dop_peak_theta` (6 cols from the full-sweep
investigations snapshot). The merge happens in-pipeline below, mirroring
the established `_join_block` pattern from `experiments/exp_v_*`.

Default seed strategy is K=5 INDEPENDENT ensembling: each ensemble
member runs its own Optuna study (different inner-CV partition →
different best params), then refits with that seed's params. Per-cell
predictions are the mean across the K models; `_std` columns capture
ensemble dispersion.

Single-fit fallback (K=1) is supported for smoke/debug.

Importable as `run_production(...)`. The CLI builds the kwargs and
calls in.
"""

from __future__ import annotations

import json
import logging
import shutil
import subprocess
import sys
import time as _time
from datetime import datetime
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import optuna
import pandas as pd
import polars as pl
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import KFold
from sksurv.metrics import concordance_index_censored

from cell_classifier.data.loader import (
    _resolve_bundle_dir,
    _resolve_preprocess_root,
    column_roles_path,
)
from cell_lifetime.data.loader import load_dataset
from cell_lifetime.models.ebm_classifier import EBMClassifierModel
from cell_lifetime.models.rsf import RSFModel
from cell_lifetime.pipelines import production_plots


NS = (200, 300, 400)


# ---------- Candidate-feature paths (v1 + dop) ------------------------------
# When feature extraction becomes an automated pipeline these paths will move
# to the new filer; update only these two constants.
DQDV_V1_PATH = Path(
    "/mnt/data/mliao/battery-ml-workbench/new_5cycle_ml/ml_label_preprocess/"
    "feature_candidates/dqdv_v1/features.parquet"
)
DQDV_V1_COLS = [
    "dqdv_peak_v_c5_dis",
    "dqdv_peak_v_shift_c1c5_dis",
    "dqdv_charge_discharge_hysteresis_c5",
    "dqdv_cosine_sim_c1c5_dis",
]
# Both v1 and dop now live under feature_candidates/ (dop was refreshed
# from pilot to full mode 2026-05-26; n_cells_full=429). When feature
# extraction becomes an automated pipeline these two paths will move to
# the new filer; update only these two constants.
DOP_PATH = Path(
    "/mnt/data/mliao/battery-ml-workbench/new_5cycle_ml/ml_label_preprocess/"
    "feature_candidates/dop_peak_theta/features.parquet"
)
DOP_COLS = [
    "dop_peak_theta_c1_chg",
    "dop_peak_theta_c5_chg",
    "dop_peak_theta_c1_dis",
    "dop_peak_theta_c5_dis",
    "dop_peak_theta_shift_chg_c1c5",
    "dop_peak_theta_shift_dis_c1c5",
]


def _load_block(path: Path, cols: list[str]) -> pd.DataFrame:
    """Load a feature-candidate block parquet, return cell_name-indexed DF.

    Used by run_production (defaults to DQDV_V1_PATH / DOP_PATH) and by
    run_predict (paths from predict_manifest.json or CLI overrides).
    """
    return (
        pl.read_parquet(path)
        .select(["cell_name", *cols])
        .to_pandas()
        .set_index("cell_name")
    )


def _load_dqdv() -> pd.DataFrame:
    return _load_block(DQDV_V1_PATH, DQDV_V1_COLS)


def _load_dop() -> pd.DataFrame:
    return _load_block(DOP_PATH, DOP_COLS)


def _join_block(
    X_base: pd.DataFrame, names: np.ndarray, block: pd.DataFrame,
    cols: list[str],
) -> tuple[pd.DataFrame, int]:
    """Left-join `block[cols]` onto X_base by cell name; NaN-fill misses.

    Returns (augmented_X, n_cells_missing_from_block). Existing model
    imputers handle the NaNs downstream.
    """
    data: dict[str, list[float]] = {c: [] for c in cols}
    missing = 0
    for n_ in names:
        if n_ in block.index:
            for c in cols:
                v = block.at[n_, c]
                data[c].append(float(v) if not pd.isna(v) else float("nan"))
        else:
            missing += 1
            for c in cols:
                data[c].append(float("nan"))
    out = X_base.copy()
    for c in cols:
        out[c] = data[c]
    return out, missing


# ---------- Reproducibility snapshot ---------------------------------------

def _git_state(repo_dir: Path) -> dict[str, Any]:
    """Capture branch + HEAD + dirty flag for the cell_lifetime checkout."""
    def _git(*args: str) -> str:
        return subprocess.check_output(
            ["git", "-C", str(repo_dir), *args], text=True,
        ).strip()
    try:
        return {
            "branch": _git("branch", "--show-current"),
            "head": _git("rev-parse", "HEAD"),
            "dirty": bool(_git("status", "-s")),
        }
    except Exception as e:
        return {"error": str(e)}


def _lib_versions() -> dict[str, str]:
    """Versions of libraries whose updates can change outputs."""
    out: dict[str, str] = {}
    for mod in ("xgboost", "interpret", "sksurv", "optuna",
                "polars", "pandas", "numpy", "sklearn"):
        try:
            m = __import__(mod)
            out[mod] = getattr(m, "__version__", "?")
        except ImportError:
            out[mod] = "missing"
    return out


def _build_regenerate_command(config: dict[str, Any]) -> str:
    """The exact CLI line a future operator would use to re-run."""
    parts = ["cell-lifetime", "production"]
    parts += ["--trials",         str(config["trials"])]
    parts += ["--inner-cv",       str(config["inner_cv"])]
    parts += ["--ensemble-seeds", str(config["ensemble_seeds"])]
    parts += ["--baseline-cycle", str(config["baseline_cycle"])]
    parts += ["--db-version",     config["db_version"]]
    parts += ["--classifier-feature-subset", config["classifier_feature_subset"]]
    parts += ["--rsf-feature-subset",        config["rsf_feature_subset"]]
    if not config.get("with_extra_features", True):
        parts += ["--no-extra-features"]
    if not config.get("make_plots", True):
        parts += ["--no-plots"]
    return " ".join(parts)


def _snapshot_run_inputs(
    out_dir: Path,
    *,
    config: dict[str, Any],
    feature_base: str,
    with_extra_features: bool,
    baseline_cycle: int,
    db_version: str,
    log: logging.Logger,
) -> Path:
    """Copy every input parquet/yaml into out_dir/inputs/ and write
    run_config.json. Called at the start of run_production, before any
    training, so a crashed run still leaves a triage-able snapshot.
    """
    inputs_dir = out_dir / "inputs"
    inputs_dir.mkdir(parents=True, exist_ok=True)

    # Resolve symlinks so we record the concrete snapshot dir name
    # (e.g. A2.2_b1_20260521_1658), not the floating "_latest" alias.
    bundle_dir = _resolve_bundle_dir(
        _resolve_preprocess_root(None), db_version, baseline_cycle, None,
    ).resolve()
    for fname, target in [
        ("cell_features.parquet", "cell_features.parquet"),
        ("cell_labels.parquet",   "cell_labels.parquet"),
        ("manifest.json",         "bundle_manifest.json"),
    ]:
        src = bundle_dir / fname
        if src.exists():
            shutil.copy2(src, inputs_dir / target)

    cr_path = column_roles_path(None)
    shutil.copy2(cr_path, inputs_dir / "column_roles.yaml")

    shutil.copy2(DQDV_V1_PATH, inputs_dir / "dqdv_v1.parquet")
    dqdv_prov = DQDV_V1_PATH.parent / "provenance.json"
    if dqdv_prov.exists():
        shutil.copy2(dqdv_prov, inputs_dir / "dqdv_v1_provenance.json")

    shutil.copy2(DOP_PATH, inputs_dir / "dop_peak_theta.parquet")
    # Sidecar may be `provenance.json` (feature_candidates convention) or
    # `manifest.json` (investigations convention); copy whichever exists.
    for sidecar_name, target in [
        ("provenance.json", "dop_peak_theta_provenance.json"),
        ("manifest.json",   "dop_peak_theta_manifest.json"),
    ]:
        src = DOP_PATH.parent / sidecar_name
        if src.exists():
            shutil.copy2(src, inputs_dir / target)

    repo_dir = Path(__file__).resolve().parents[3]  # cell_lifetime/
    snapshot_cfg = {
        "config": config,
        "feature_base": feature_base,
        "with_extra_features": with_extra_features,
        "inputs": {
            "bundle_dir":  str(bundle_dir),
            "bundle_name": bundle_dir.name,
            "column_roles_path": str(cr_path),
            "dqdv_v1_path": str(DQDV_V1_PATH),
            "dop_path":     str(DOP_PATH),
        },
        "git": _git_state(repo_dir),
        "library_versions": _lib_versions(),
        "regenerate_command": _build_regenerate_command(config),
        "snapshot_written_at": datetime.now().isoformat(),
    }
    (out_dir / "run_config.json").write_text(
        json.dumps(snapshot_cfg, indent=2, default=str)
    )
    git = snapshot_cfg["git"]
    log.info(
        f"Wrote inputs snapshot to {inputs_dir} "
        f"(bundle={bundle_dir.name}; "
        f"git={git.get('head', '?')[:8]}"
        f"{'+dirty' if git.get('dirty') else ''})"
    )
    return inputs_dir


# ---------- logging setup ---------------------------------------------------

def setup_logging(out_dir: Path) -> Path:
    log_path = out_dir / "run.log"
    root = logging.getLogger()
    root.setLevel(logging.INFO)
    for h in list(root.handlers):
        root.removeHandler(h)
    # Local wall-clock time (matches run-directory timestamp). Python's
    # logging.Formatter defaults to local time; the previous Z suffix
    # falsely advertised UTC.
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
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    return log_path


# ---------- RSF median-survival extraction ----------------------------------

def median_survival_from_sf(sf, t_cap: float) -> float:
    times = np.asarray(sf.x, dtype=float)
    surv = np.asarray(sf.y, dtype=float)
    below = np.where(surv <= 0.5)[0]
    if len(below) == 0:
        return float(t_cap)
    return float(times[below[0]])


# ---------- Classifier tune + OOF + refit ----------------------------------

def _ebm_inner_cv_auc(
    params: dict[str, Any], X: pd.DataFrame, y: np.ndarray,
    inner_cv: int, seed: int,
) -> tuple[float, list[float]]:
    kf = KFold(n_splits=inner_cv, shuffle=True, random_state=seed)
    fold_scores: list[float] = []
    for tr, va in kf.split(X):
        if len(np.unique(y[tr])) < 2 or len(np.unique(y[va])) < 2:
            continue
        model = EBMClassifierModel(params)
        model.fit(X.iloc[tr], y[tr])
        prob = model.predict_proba(X.iloc[va])[:, 1]
        fold_scores.append(float(roc_auc_score(y[va], prob)))
    if not fold_scores:
        return float("nan"), []
    return float(np.mean(fold_scores)), fold_scores


def _tune_ebm(
    X: pd.DataFrame, y: np.ndarray, trials: int, inner_cv: int, seed: int,
) -> tuple[dict[str, Any], float, list[float]]:
    sampler = optuna.samplers.TPESampler(seed=seed)
    study = optuna.create_study(direction="maximize", sampler=sampler)
    study.optimize(
        lambda trial: _ebm_inner_cv_auc(
            EBMClassifierModel.suggest_params(trial), X, y, inner_cv, seed,
        )[0],
        n_trials=trials, show_progress_bar=False,
    )
    best = dict(study.best_params)
    best_score, fold_scores = _ebm_inner_cv_auc(best, X, y, inner_cv, seed)
    return best, best_score, fold_scores


def _oof_probabilities(
    X: pd.DataFrame, y: np.ndarray, params: dict[str, Any],
    inner_cv: int, seed: int,
) -> np.ndarray:
    kf = KFold(n_splits=inner_cv, shuffle=True, random_state=seed)
    oof = np.full(len(X), np.nan, dtype=float)
    for tr, va in kf.split(X):
        model = EBMClassifierModel(params)
        model.fit(X.iloc[tr], y[tr])
        oof[va] = model.predict_proba(X.iloc[va])[:, 1]
    return oof


# ---------- RSF tune --------------------------------------------------------

def _rsf_inner_cv_cindex(
    params: dict[str, Any], X: pd.DataFrame,
    time_arr: np.ndarray, event_arr: np.ndarray,
    inner_cv: int, seed: int,
) -> tuple[float, list[float]]:
    kf = KFold(n_splits=inner_cv, shuffle=True, random_state=seed)
    fold_scores: list[float] = []
    for tr, va in kf.split(X):
        model = RSFModel(params)
        model.fit(X.iloc[tr], time=time_arr[tr], event=event_arr[tr])
        risk = model.predict(X.iloc[va])
        c, *_ = concordance_index_censored(event_arr[va].astype(bool), time_arr[va], risk)
        fold_scores.append(float(c))
    return float(np.mean(fold_scores)), fold_scores


def _tune_rsf(
    X: pd.DataFrame, time_arr: np.ndarray, event_arr: np.ndarray,
    trials: int, inner_cv: int, seed: int,
) -> tuple[dict[str, Any], float, list[float]]:
    sampler = optuna.samplers.TPESampler(seed=seed)
    study = optuna.create_study(direction="maximize", sampler=sampler)

    def objective(trial: optuna.Trial) -> float:
        params = RSFModel.suggest_params(trial)
        try:
            return _rsf_inner_cv_cindex(
                params, X, time_arr, event_arr, inner_cv, seed,
            )[0]
        except Exception as e:
            logging.warning(f"RSF trial failed: {e}")
            return float("nan")

    study.optimize(objective, n_trials=trials, show_progress_bar=False)
    best = dict(study.best_params)
    best_score, fold_scores = _rsf_inner_cv_cindex(
        best, X, time_arr, event_arr, inner_cv, seed,
    )
    return best, best_score, fold_scores


# ---------- Overfit diagnostics --------------------------------------------

def _log_overfit_diagnostics(log, df: pd.DataFrame,
                             best_params: dict[str, Any], K: int) -> None:
    """Emit overfit-spotting diagnostics to the log.

    Three signals:
      1. Spread of per-seed inner-CV scores per model — tight clusters
         indicate hyperparameter choice is robust to CV partition.
      2. Hyperparameter agreement across seeds — wildly different best
         params per seed = the optimum is poorly identified.
      3. Per-cell prediction std — high `_std` cells are uncertain.
    """
    if K == 1:
        log.info(
          "  K=1: no ensemble diagnostics (single deterministic fit)"
        )
        return

    # 1. Per-seed inner-CV score spread (classifiers + RSF)
    for N in NS:
        per_seed = best_params[f"ebm_classifier_n{N}"]["inner_cv_auc_per_seed"]
        log.info(
            f"  AUC per seed N={N}: mean={np.mean(per_seed):.4f} ± "
            f"std={np.std(per_seed, ddof=1) if K > 1 else 0:.4f} "
            f"(range {min(per_seed):.4f} - {max(per_seed):.4f})"
        )
    per_seed_c = best_params["rsf"]["inner_cv_cindex_per_seed"]
    log.info(
        f"  C-index per seed (rsf): mean={np.mean(per_seed_c):.4f} ± "
        f"std={np.std(per_seed_c, ddof=1) if K > 1 else 0:.4f} "
        f"(range {min(per_seed_c):.4f} - {max(per_seed_c):.4f})"
    )

    # 2. Hyperparameter spread per model (just key numeric params)
    for key in (f"ebm_classifier_n{N}" for N in NS):
        sets = best_params[key]["best_params_per_seed"]
        # Pick a couple of representative numeric hyperparams
        for pname in ("max_bins", "max_interaction_bins", "interactions",
                      "learning_rate", "max_leaves"):
            vals = [s.get(pname) for s in sets if s.get(pname) is not None]
            if not vals:
                continue
            if isinstance(vals[0], float):
                summary = (
                    f"{min(vals):.4g}-{max(vals):.4g} "
                    f"(med {np.median(vals):.4g})"
                )
            else:
                summary = f"{min(vals)}-{max(vals)} (med {int(np.median(vals))})"
            log.info(f"  {key} hyperparams[{pname}] across seeds: {summary}")
    # RSF hyperparams
    sets = best_params["rsf"]["best_params_per_seed"]
    for pname in ("n_estimators", "max_depth", "min_samples_split",
                  "min_samples_leaf"):
        vals = [s.get(pname) for s in sets if s.get(pname) is not None]
        if not vals:
            continue
        summary = f"{min(vals)}-{max(vals)} (med {int(np.median(vals))})"
        log.info(f"  rsf hyperparams[{pname}] across seeds: {summary}")
    mx = [s.get("max_features") for s in sets]
    log.info(f"  rsf hyperparams[max_features] across seeds: {mx}")

    # 3. Per-cell std distribution: how stable are individual predictions?
    for N in NS:
        col = f"prob_pass_n{N}_std"
        s = df[col].dropna()
        if len(s) == 0:
            continue
        n_high = int((s > 0.10).sum())
        log.info(
            f"  prob_pass_n{N}_std: mean={s.mean():.4f}, "
            f"p50={s.median():.4f}, p90={s.quantile(0.90):.4f}, "
            f"p99={s.quantile(0.99):.4f}; cells with std>0.10: {n_high}/{len(s)}"
        )
    s = df["rsf_median_cycle_std"]
    n_high = int((s > 50).sum())
    log.info(
        f"  rsf_median_cycle_std: mean={s.mean():.1f}, "
        f"p50={s.median():.1f}, p90={s.quantile(0.90):.1f}, "
        f"p99={s.quantile(0.99):.1f}; cells with std>50 cyc: {n_high}/{len(s)}"
    )

    log.info(
        "  Read: tighter inner-CV score spread + smaller per-cell std => "
        "ensemble is more robust to hyperparameter choice (lower overfit risk)."
    )


# ---------- Ground-truth pass labels ---------------------------------------

def _true_pass(status: str, event: bool, last_fade: float, n_reg: int, N: int) -> float:
    # Excluded cells (rate_changed admitted for inference only) have no
    # honest ground-truth label: their n_regular counts cycles across mixed
    # rate regimes, so n_reg >= N does NOT mean the cell healthily survived
    # to N. Upstream sets label_n{N}='excluded' for exactly this reason.
    if status == "excluded":
        return float("nan")
    if event:
        return 1.0 if last_fade >= N else 0.0
    if n_reg >= N:
        return 1.0
    return float("nan")


# ---------- Main entry point ----------------------------------------------

def run_production(
    *,
    out_dir: Path,
    trials: int = 50,
    inner_cv: int = 5,
    ensemble_seeds: int = 5,
    baseline_cycle: int = 1,
    db_version: str = "A2.2",
    classifier_feature_subset: str = "fs_a_only",
    rsf_feature_subset: str = "fs_a_only",
    with_extra_features: bool = True,
    make_plots: bool = True,
) -> dict[str, Any]:
    """Production fit: 3 classifiers + 1 RSF, K-seed independent ensemble.

    All four models share the same X built from
    `feature_subset=rsf_feature_subset` (default `fs_a_only`, 3 cols)
    optionally augmented with the v1 + dop_peak_theta candidate blocks
    (10 cols, gated by `with_extra_features=True`).

    The `classifier_feature_subset` kwarg is retained for back-compat; if
    it disagrees with `rsf_feature_subset` a warning is logged and the
    rsf value wins (the classifier and RSF must see the same X for the
    unified-feature pipeline to be coherent).

    Writes:
      - out_dir / "predictions.csv"
      - out_dir / "best_params.json"
      - out_dir / "run.log"
      - (if make_plots) 3 PNGs

    Returns a summary dict with run config + key metrics.
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    log_path = setup_logging(out_dir)
    log = logging.getLogger(__name__)
    log.info(
        f"Production fit. ensemble_seeds={ensemble_seeds}, trials={trials}, "
        f"inner_cv={inner_cv}, baseline_cycle={baseline_cycle}, "
        f"db_version={db_version}, feature_base={rsf_feature_subset}, "
        f"with_extra_features={with_extra_features}"
    )
    if classifier_feature_subset != rsf_feature_subset:
        log.warning(
            f"classifier_feature_subset={classifier_feature_subset!r} differs "
            f"from rsf_feature_subset={rsf_feature_subset!r}; the unified "
            f"production pipeline uses a single X for both, so the rsf value "
            f"will be used as the base for the classifier as well."
        )
    feature_base = rsf_feature_subset
    log.info(f"Out dir: {out_dir}")
    log.info(f"Log path: {log_path}")

    # Snapshot inputs + config FIRST so a crash mid-training still leaves
    # the reproducibility bundle on disk.
    _snapshot_run_inputs(
        out_dir,
        config={
            "trials": trials, "inner_cv": inner_cv,
            "ensemble_seeds": ensemble_seeds,
            "baseline_cycle": baseline_cycle, "db_version": db_version,
            "classifier_feature_subset": classifier_feature_subset,
            "rsf_feature_subset": rsf_feature_subset,
            "with_extra_features": with_extra_features,
            "make_plots": make_plots,
        },
        feature_base=feature_base,
        with_extra_features=with_extra_features,
        baseline_cycle=baseline_cycle,
        db_version=db_version,
        log=log,
    )

    K = max(int(ensemble_seeds), 1)
    seeds = list(range(K))

    # ---- Load base dataset once -------------------------------------------
    # Production load uses min_n_regular=5, drop_excluded=False so we PREDICT
    # on cells with n_regular≥5 INCLUDING status='excluded' cells (rate_changed
    # cells that upstream admitted to cell_features.parquet — see
    # CONVENTIONS.md). Training masks below AND with status!='excluded' so
    # those cells flow through inference only, never into training.
    log.info(
        f"Loading base dataset (feature_subset={feature_base}, "
        f"min_n_regular=5, drop_excluded=False)…"
    )
    ds_cv = load_dataset(
        N=300, feature_subset=feature_base,
        baseline_cycle=baseline_cycle, db_version=db_version,
        min_n_regular=5, drop_excluded=False,
    )
    n_total = len(ds_cv)
    n_faded = int(ds_cv.event.sum())
    n_censored = int((~ds_cv.event).sum())
    log.info(
        f"Dataset: {n_total} cells = {n_faded} faded + {n_censored} censored; "
        f"base features: {ds_cv.X.shape[1]}"
    )
    X_cv_all = ds_cv.X.reset_index(drop=True)
    event_all = ds_cv.event.astype(bool)
    time_all = ds_cv.time.astype(np.int64)
    y_cycle_all = ds_cv.y_cycle.astype(float)
    cell_names = ds_cv.cell_names
    n_regular_all = ds_cv.n_regular.astype(np.int64)
    status_all = ds_cv.status
    excl_all = ds_cv.exclusion_reason

    # ---- Merge candidate feature blocks (dqdv_v1 + dop_peak_theta) --------
    # See exp_v_n200_full_audit/run.py — same pattern, same parquets.
    # The merged X is reused for all 3 classifiers AND the RSF so every
    # model sees an identical 13-col feature vector.
    feature_block_labels: list[str] = [feature_base]
    if with_extra_features:
        log.info(f"Joining dqdv_v1 ({len(DQDV_V1_COLS)} cols) from {DQDV_V1_PATH}")
        X_cv_all, miss_v1 = _join_block(
            X_cv_all, cell_names, _load_dqdv(), DQDV_V1_COLS,
        )
        log.info(f"Joining dop_peak_theta ({len(DOP_COLS)} cols) from {DOP_PATH}")
        X_cv_all, miss_dop = _join_block(
            X_cv_all, cell_names, _load_dop(), DOP_COLS,
        )
        nan_per_col = {
            c: int(X_cv_all[c].isna().sum())
            for c in DQDV_V1_COLS + DOP_COLS
        }
        log.info(
            f"  v1 cells missing from parquet: {miss_v1}/{n_total}; "
            f"dop cells missing from parquet: {miss_dop}/{n_total}; "
            f"per-col NaN in merged X: {nan_per_col}"
        )
        feature_block_labels += ["dqdv_v1", "dop_peak_theta"]
    feature_block_label = "+".join(feature_block_labels)
    log.info(
        f"Unified feature vector: {X_cv_all.shape[1]} cols "
        f"({feature_block_label})"
    )

    # Dump the exact 13-col X fed to all 4 models, with cell metadata and
    # per-N labels side-by-side. Lets a future reader inspect/re-run from
    # the run dir alone.
    merged_df = X_cv_all.copy()
    merged_df.insert(0, "cell_name", cell_names)
    merged_df["status"] = status_all
    merged_df["exclusion_reason"] = excl_all
    merged_df["n_regular"] = n_regular_all
    labels_df = pl.read_parquet(
        _resolve_bundle_dir(
            _resolve_preprocess_root(None), db_version, baseline_cycle, None,
        ).resolve() / "cell_labels.parquet"
    ).select([
        "cell_name", "label_n200", "label_n300", "label_n400",
        "trainable_n200", "trainable_n300", "trainable_n400",
        "last_fade_cycle",
    ]).to_pandas()
    merged_df = merged_df.merge(labels_df, on="cell_name", how="left")
    merged_df.to_parquet(out_dir / "merged_features.parquet", index=False)
    log.info(
        f"Wrote merged_features.parquet ({len(merged_df)} rows × "
        f"{len(merged_df.columns)} cols)"
    )

    # Training mask: n_regular>=6 (canonical training cutoff) AND
    # status!='excluded' (rate_changed cells admitted upstream for inference
    # only). The status check is load-bearing — without it, rate_changed cells
    # with large lifetime n_regular (e.g. AR4142=639) would slip into RSF
    # training. Inference happens on the full n_total cells.
    train_eligible = (n_regular_all >= 6) & (status_all != "excluded")
    n_inference_only_low_nreg = int(
        ((n_regular_all < 6) & (status_all != "excluded")).sum()
    )
    n_inference_only_excluded = int((status_all == "excluded").sum())
    log.info(
        f"Training-eligible cells (n_regular>=6 & not excluded): "
        f"{int(train_eligible.sum())}/{n_total}; "
        f"inference-only n_regular=5: {n_inference_only_low_nreg}; "
        f"inference-only status='excluded' (rate_changed admitted): "
        f"{n_inference_only_excluded}"
    )

    # ---- Model-persistence dir (joblib dumps + predict_manifest.json) -----
    # Folder name carries the run timestamp so it's self-identifying even
    # if copied out of its parent run dir.
    models_dir = out_dir / f"models_{out_dir.name}"
    models_dir.mkdir(exist_ok=True)
    persisted_models: list[dict[str, Any]] = []

    # ---- Per-N classifier training (independent K-seed ensemble) ----------
    best_params: dict[str, Any] = {}
    classifier_predictions: dict[int, dict[str, np.ndarray]] = {}
    in_training_set: dict[int, np.ndarray] = {}

    for N in NS:
        log.info(f"=== ebm_classifier × {feature_block_label} × N={N} ===")
        # Only need ds_N for the per-N label_mask / y_class; the feature
        # matrix is the shared merged X_cv_all (same 13 cols for all 4 models).
        ds_N = load_dataset(
            N=N, feature_subset=feature_base,
            baseline_cycle=baseline_cycle, db_version=db_version,
            min_n_regular=5, drop_excluded=False,
        )
        assert (ds_N.cell_names == cell_names).all(), \
            f"N={N} loader returned different cell ordering than base load"
        assert (ds_N.event == event_all).all(), \
            f"N={N} loader returned different event array than base load"
        assert (ds_N.n_regular == n_regular_all).all(), \
            f"N={N} loader returned different n_regular than base load"
        assert (ds_N.status == status_all).all(), \
            f"N={N} loader returned different status array than base load"

        X_all_N = X_cv_all  # shared merged X — identical for all 4 models
        label_mask = ds_N.label_mask.astype(bool)
        y_class = ds_N.y_class.astype(np.int8)

        # Training requires BOTH trainable_n{N} (definitive label) AND
        # n_regular>=6 (stable feature signature). Cells failing either
        # only get predictions, not OOF.
        train_mask = label_mask & train_eligible
        train_idx = np.where(train_mask)[0]
        X_train = X_all_N.iloc[train_idx].reset_index(drop=True)
        y_train = y_class[train_idx]
        n_train = len(train_idx)
        n_train_faded = int((event_all[train_idx]).sum())
        n_train_censored = n_train - n_train_faded
        n_dropped_low_nreg = int(
            (label_mask & ~train_eligible).sum()
        )
        log.info(
            f"  trainable_n{N} ∩ n_reg≥6: {n_train} cells = {n_train_faded} faded + "
            f"{n_train_censored} censored (pass={int((y_train==1).sum())}, "
            f"fail={int((y_train==0).sum())}); "
            f"dropped {n_dropped_low_nreg} labeled cells with n_reg<6"
        )
        in_training_set[N] = train_mask

        # K ensemble members
        member_probs: list[np.ndarray] = []
        member_oofs: list[np.ndarray] = []
        member_bests: list[dict[str, Any]] = []
        member_aucs: list[float] = []
        for k in seeds:
            t0 = _time.time()
            log.info(f"  [k={k}] tune ({trials} trials × {inner_cv} CV)…")
            best, auc_best, fold_aucs = _tune_ebm(
                X_train, y_train, trials, inner_cv, seed=k,
            )
            log.info(
                f"  [k={k}] best AUC = {auc_best:.4f} "
                f"(folds: {[round(s, 4) for s in fold_aucs]}), "
                f"best params: {best}, t={_time.time()-t0:.1f}s"
            )

            oof_train = _oof_probabilities(X_train, y_train, best, inner_cv, seed=k)
            oof_full = np.full(n_total, np.nan, dtype=float)
            oof_full[train_idx] = oof_train
            member_oofs.append(oof_full)

            model = EBMClassifierModel(best)
            model.fit(X_train, y_train)
            prob = model.predict_proba(X_all_N)[:, 1]
            member_probs.append(prob)
            member_bests.append(best)
            member_aucs.append(auc_best)

            joblib_name = f"ebm_classifier_n{N}_seed{k}.joblib"
            joblib.dump(model, models_dir / joblib_name)
            persisted_models.append({
                "head": f"ebm_classifier_n{N}",
                "horizon": N,
                "seed": k,
                "path": joblib_name,
            })

        prob_stack = np.stack(member_probs, axis=0)  # (K, n_total)
        oof_stack = np.stack(member_oofs, axis=0)     # (K, n_total)
        prob_mean = prob_stack.mean(axis=0)
        prob_std = prob_stack.std(axis=0, ddof=0) if K > 1 else np.zeros(n_total)
        oof_mean = np.nanmean(oof_stack, axis=0)
        oof_std = (
            np.nanstd(oof_stack, axis=0, ddof=0) if K > 1
            else np.zeros(n_total)
        )
        pred = (prob_mean >= 0.5).astype(np.int8)

        log.info(
            f"  ensemble pass rate: {pred.mean():.4f}, mean prob: {prob_mean.mean():.4f}, "
            f"per-cell prob std mean: {np.nanmean(prob_std):.4f}"
        )

        classifier_predictions[N] = {
            "prob_mean": prob_mean,
            "prob_std": prob_std,
            "oof_mean": oof_mean,
            "oof_std": oof_std,
            "pred": pred,
        }
        best_params[f"ebm_classifier_n{N}"] = {
            "ensemble_seeds": K,
            "best_params_per_seed": member_bests,
            "inner_cv_auc_per_seed": member_aucs,
            "inner_cv_auc_mean": float(np.mean(member_aucs)),
            "n_training_cells": n_train,
            "n_training_faded": n_train_faded,
            "n_training_censored": n_train_censored,
            "n_features": int(X_train.shape[1]),
        }

    # ---- RSF training (independent K-seed ensemble) -----------------------
    # Trained on cells with n_regular≥6 only; predicts on all loaded cells.
    rsf_train_idx = np.where(train_eligible)[0]
    n_rsf_train = len(rsf_train_idx)
    X_rsf_train = X_cv_all.iloc[rsf_train_idx].reset_index(drop=True)
    time_rsf_train = time_all[rsf_train_idx]
    event_rsf_train = event_all[rsf_train_idx]
    log.info(
        f"=== rsf × {feature_block_label} (train on {n_rsf_train} n_reg≥6 cells; "
        f"predict on all {n_total}) ==="
    )
    rsf_medians: list[np.ndarray] = []
    rsf_bests: list[dict[str, Any]] = []
    rsf_cindices: list[float] = []
    for k in seeds:
        t0 = _time.time()
        log.info(f"  [k={k}] tune ({trials} trials × {inner_cv} CV)…")
        best, cindex_best, fold_cs = _tune_rsf(
            X_rsf_train, time_rsf_train, event_rsf_train, trials, inner_cv, seed=k,
        )
        log.info(
            f"  [k={k}] best C-index = {cindex_best:.4f} "
            f"(folds: {[round(s, 4) for s in fold_cs]}), "
            f"best params: {best}, t={_time.time()-t0:.1f}s"
        )

        rsf_model = RSFModel({**best, "low_memory": False, "random_state": k})
        rsf_model.fit(X_rsf_train, time=time_rsf_train, event=event_rsf_train)

        rsf_joblib_name = f"rsf_seed{k}.joblib"
        joblib.dump(rsf_model, models_dir / rsf_joblib_name)
        persisted_models.append({
            "head": "rsf",
            "horizon": None,
            "seed": k,
            "path": rsf_joblib_name,
        })

        sfs = rsf_model.predict_survival_curve(X_cv_all)
        t_cap = float(time_rsf_train.max())  # cap from TRAINING distribution
        median = np.array(
            [median_survival_from_sf(sf, t_cap) for sf in sfs],
            dtype=float,
        )
        rsf_medians.append(median)
        rsf_bests.append(best)
        rsf_cindices.append(cindex_best)

    rsf_stack = np.stack(rsf_medians, axis=0)  # (K, n_total)
    rsf_median_mean = rsf_stack.mean(axis=0)
    rsf_median_std = rsf_stack.std(axis=0, ddof=0) if K > 1 else np.zeros(n_total)
    log.info(
        f"  ensemble median range [{rsf_median_mean.min():.0f}, "
        f"{rsf_median_mean.max():.0f}], mean {rsf_median_mean.mean():.1f}, "
        f"per-cell std mean: {rsf_median_std.mean():.1f}"
    )

    best_params["rsf"] = {
        "ensemble_seeds": K,
        "best_params_per_seed": rsf_bests,
        "inner_cv_cindex_per_seed": rsf_cindices,
        "inner_cv_cindex_mean": float(np.mean(rsf_cindices)),
        "n_training_cells": n_rsf_train,
        "n_inference_cells": n_total,
        "n_features": int(X_cv_all.shape[1]),
        "t_cap": float(time_rsf_train.max()),
    }

    # ---- Write predict_manifest.json --------------------------------------
    # Captures everything `run_predict` needs to load the persisted ensemble
    # and score new cells: feature column order, source parquet paths, RSF
    # t_cap, and per-member joblib filenames.
    bundle_dir_resolved = _resolve_bundle_dir(
        _resolve_preprocess_root(None), db_version, baseline_cycle, None,
    ).resolve()
    manifest = {
        "schema_version": 1,
        "snapshot_written_at": datetime.now().isoformat(),
        "ensemble_seeds": K,
        "feature_columns": list(X_cv_all.columns),
        "feature_base": feature_base,
        "extra_feature_blocks": (
            ["dqdv_v1", "dop_peak_theta"] if with_extra_features else []
        ),
        "n_features": int(X_cv_all.shape[1]),
        "horizons": list(NS),
        "rsf_t_cap": float(time_rsf_train.max()),
        "inputs": {
            "bundle_name": bundle_dir_resolved.name,
            "cell_features_path": str(bundle_dir_resolved / "cell_features.parquet"),
            "dqdv_v1_path": str(DQDV_V1_PATH),
            "dop_path": str(DOP_PATH),
        },
        "models": persisted_models,
    }
    (models_dir / "predict_manifest.json").write_text(
        json.dumps(manifest, indent=2, default=str)
    )
    log.info(
        f"  Wrote {len(persisted_models)} .joblib files + predict_manifest.json "
        f"to {models_dir}"
    )

    # ---- Assemble predictions.csv -----------------------------------------
    log.info(f"=== Assembling predictions.csv ({n_total} rows) ===")
    true_pass = {
        N: np.array([
            _true_pass(
                str(status_all[i]),
                bool(event_all[i]),
                float(y_cycle_all[i]) if event_all[i] else float("nan"),
                int(n_regular_all[i]),  # true n_regular per cell (≠ time for faded)
                N,
            )
            for i in range(n_total)
        ])
        for N in NS
    }

    df = pd.DataFrame({
        "cell_name": cell_names,
        "status": status_all,
        "exclusion_reason": excl_all,
        "event": event_all,
        "last_fade_cycle": np.where(event_all, y_cycle_all, np.nan),
        "n_regular": n_regular_all,
        "time": time_all,
        "true_pass_n200": true_pass[200],
        "true_pass_n300": true_pass[300],
        "true_pass_n400": true_pass[400],
        "in_training_set_n200": in_training_set[200],
        "in_training_set_n300": in_training_set[300],
        "in_training_set_n400": in_training_set[400],
        "pred_pass_n200": classifier_predictions[200]["pred"],
        "pred_pass_n300": classifier_predictions[300]["pred"],
        "pred_pass_n400": classifier_predictions[400]["pred"],
        "prob_pass_n200": classifier_predictions[200]["prob_mean"],
        "prob_pass_n300": classifier_predictions[300]["prob_mean"],
        "prob_pass_n400": classifier_predictions[400]["prob_mean"],
        "prob_pass_n200_std": classifier_predictions[200]["prob_std"],
        "prob_pass_n300_std": classifier_predictions[300]["prob_std"],
        "prob_pass_n400_std": classifier_predictions[400]["prob_std"],
        "oof_prob_pass_n200": classifier_predictions[200]["oof_mean"],
        "oof_prob_pass_n300": classifier_predictions[300]["oof_mean"],
        "oof_prob_pass_n400": classifier_predictions[400]["oof_mean"],
        "oof_prob_pass_n200_std": classifier_predictions[200]["oof_std"],
        "oof_prob_pass_n300_std": classifier_predictions[300]["oof_std"],
        "oof_prob_pass_n400_std": classifier_predictions[400]["oof_std"],
        "rsf_median_cycle": rsf_median_mean,
        "rsf_median_cycle_std": rsf_median_std,
    })

    # Sanity
    pass_rates = {N: float(df[f"pred_pass_n{N}"].mean()) for N in NS}
    log.info(
        f"  Pass rates (predicted, ensemble mean@0.5): "
        f"N=200 → {pass_rates[200]:.3f}, "
        f"N=300 → {pass_rates[300]:.3f}, "
        f"N=400 → {pass_rates[400]:.3f}"
    )
    if not (pass_rates[200] >= pass_rates[300] >= pass_rates[400]):
        log.warning("  Pass rates are NOT monotone decreasing in N — investigate")

    # Filename includes the timestamp (= the run directory's name) so the
    # CSV is self-identifying even when copied out of its directory.
    timestamp = Path(out_dir).name
    out_csv = out_dir / f"predictions_{timestamp}.csv"
    df.to_csv(out_csv, index=False)
    log.info(f"  Wrote {out_csv} ({len(df)} rows × {len(df.columns)} cols)")

    out_json = out_dir / "best_params.json"
    with out_json.open("w") as f:
        json.dump({
            "ensemble_seeds": K,
            "feature_subsets": {
                "classifier": feature_block_label,
                "rsf": feature_block_label,
            },
            "feature_base": feature_base,
            "with_extra_features": with_extra_features,
            "extra_feature_blocks": (
                ["dqdv_v1", "dop_peak_theta"] if with_extra_features else []
            ),
            "n_features": int(X_cv_all.shape[1]),
            "baseline_cycle": baseline_cycle,
            "db_version": db_version,
            "trials": trials,
            "inner_cv": inner_cv,
            "n_total_cells": n_total,
            "n_faded": n_faded,
            "n_censored": n_censored,
            "models": best_params,
        }, f, indent=2, default=str)
    log.info(f"  Wrote {out_json}")

    if make_plots:
        log.info("Rendering plots…")
        paths = production_plots.render_all(df, out_dir)
        for p in paths:
            log.info(f"  Wrote {p}")

    # ---- Overfit diagnostics ----------------------------------------------
    log.info("=== Overfit diagnostics ===")
    _log_overfit_diagnostics(log, df, best_params, K)

    log.info("Production run complete.")
    return {
        "out_dir": str(out_dir),
        "n_cells": n_total,
        "ensemble_seeds": K,
        "classifier_auc_mean": {
            N: best_params[f"ebm_classifier_n{N}"]["inner_cv_auc_mean"]
            for N in NS
        },
        "rsf_cindex_mean": best_params["rsf"]["inner_cv_cindex_mean"],
        "pass_rates": pass_rates,
    }

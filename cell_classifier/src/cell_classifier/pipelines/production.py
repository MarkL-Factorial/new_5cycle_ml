"""Production-mode pipeline orchestrator.

Trains on all trainable (labeled) cells and predicts for every cell in the
features parquet. A SINGLE representative hyperparameter set is used across
all production seeds — seeds vary `random_state` only, providing model-level
variance reduction without coupling ensembling to HP noise.

HP sources (`config["hp_provenance"]["source"]`):
  - "from_validation_run": load the matching validation run's
    `best_params_per_seed.csv` or `best_params_per_fold.csv`, collapse via
    per-HP mode/median to a single representative set.
  - "retune": run a single Optuna study on the full labeled set; use the
    resulting HPs directly as the representative set.

Persists predictions.csv, predictions_per_seed.parquet (and optional
predictions_posterior.parquet for Bayesian models), feature_importance.csv,
shap_inference_per_cell.parquet, shap_inference_summary.csv (scope = labeled
subset), best_params_production.json, optuna_history.csv (populated under
retune, empty under from_validation_run), manifest.json, plots/.
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any

import numpy as np
import optuna
import pandas as pd

from cell_classifier.data.loader import column_roles_path, load_dataset
from cell_classifier.data.splits import split_production
from cell_classifier.evaluation.importance import compute_importance
from cell_classifier.evaluation.plots import plot_perm_importance, plot_shap_summary
from cell_classifier.evaluation.shap import aggregate_shap, compute_seed_shap
from cell_classifier.inference.predict import (
    ensemble_predictions, per_seed_long, per_seed_posterior_long,
)
from cell_classifier.models.registry import get_model_class
from cell_classifier.training.core import train
from cell_classifier.training.representative import (
    read_hp_csv, representative_hp_set,
)
from cell_classifier.training.tuning import tune
from cell_classifier.utils.manifest import build_manifest, write_manifest
from cell_classifier.utils.paths import run_dir
from cell_classifier.utils.reproducibility import (
    snapshot_inputs, write_resolved_config,
)


def run_production(config: dict[str, Any], *, out_dir: Path) -> dict[str, Any]:
    t0 = time.time()
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    plots_dir = out_dir / "plots"

    optuna.logging.set_verbosity(optuna.logging.WARNING)

    preprocess_root_override = config.get("data", {}).get("preprocess_root")
    ds = load_dataset(
        N=config["N"],
        feature_subset=config["feature_subset"],
        baseline_cycle=config["baseline_cycle"],
        db_version=config["db_version"],
        preprocess_root=preprocess_root_override,
    )
    snapshot_inputs(
        out_dir,
        source_bundle=ds.source_dir,
        column_roles_yaml=column_roles_path(preprocess_root_override),
    )
    write_resolved_config(out_dir, config)
    train_idx, inference_idx = split_production(ds.label_mask)
    X_train, y_train = ds.X.iloc[train_idx], ds.y[train_idx]
    X_inference = ds.X.iloc[inference_idx]
    cell_names_inf = ds.cell_names[inference_idx]
    print(
        f"[production] N={ds.N} db={ds.db_version} b={ds.baseline_cycle} "
        f"subset={config['feature_subset']} "
        f"train={len(train_idx)} inference={len(inference_idx)}"
    )

    ModelClass = get_model_class(config["model"])
    imputer_strategy = config["preprocessing"]["imputer_strategy"]
    tuning = config["tuning"]
    hp_provenance = config["hp_provenance"]
    source = hp_provenance["source"]

    # --- Determine the single representative HP set ---
    optuna_history_df: pd.DataFrame | None = None
    if source == "from_validation_run":
        representative_params = _load_representative_from_validation(
            out_root=config["out_root"], slug=config["slug"],
        )
    elif source == "retune":
        representative_params, optuna_history_df = _retune_on_full_labeled(
            ModelClass=ModelClass, X_train=X_train, y_train=y_train,
            n_trials=int(tuning["n_trials"]),
            inner_cv=int(tuning["inner_cv_folds"]),
            optimize=str(tuning["optimize_metric"]),
            imputer_strategy=imputer_strategy,
            seed=int(config["seeds"][0]),
        )
    else:
        raise ValueError(f"unknown hp_provenance.source {source!r}")

    # --- Train + predict per seed (same HPs, different random_state) ---
    per_seed_proba_pass: list[np.ndarray] = []
    per_seed_samples: list[np.ndarray] = []
    importance_frames: list[pd.DataFrame] = []
    shap_frames_inf: list[pd.DataFrame] = []

    seeds_list = [int(s) for s in config["seeds"]]
    for seed in seeds_list:
        model = train(
            ModelClass, representative_params, X_train, y_train, seed=seed,
            imputer_strategy=imputer_strategy,
        )

        proba = model.predict_proba(X_inference)[:, 1]
        per_seed_proba_pass.append(proba)

        samples = model.predict_proba_samples(X_inference)
        if samples is not None:
            per_seed_samples.append(np.asarray(samples))

        importance_frames.append(
            compute_importance(
                model, X_train, y_train, ds.feature_names, seed=seed,
            ).assign(seed=seed)
        )
        shap_frames_inf.append(
            compute_seed_shap(model, X_inference, cell_names_inf, seed=seed)
        )

    # --- Predictions ---
    preds = ensemble_predictions(cell_names_inf, per_seed_proba_pass)
    preds.to_csv(out_dir / "predictions.csv", index=False)
    per_seed_long(cell_names_inf, seeds_list, per_seed_proba_pass) \
        .to_parquet(out_dir / "predictions_per_seed.parquet", index=False)
    if per_seed_samples:
        per_seed_posterior_long(cell_names_inf, seeds_list, per_seed_samples) \
            .to_parquet(out_dir / "predictions_posterior.parquet", index=False)

    # --- Importance ---
    importance_all = pd.concat(importance_frames, ignore_index=True)
    importance_agg = (
        importance_all.groupby("feature").agg(
            native_mean=("native_importance", "mean"),
            native_std=("native_importance", "std"),
            perm_mean=("perm_importance_mean", "mean"),
            perm_std=("perm_importance_mean", "std"),
        ).reset_index().sort_values("perm_mean", ascending=False)
    )
    importance_agg.to_csv(out_dir / "feature_importance.csv", index=False)
    plot_perm_importance(
        importance_agg, plots_dir / "feature_importance.png",
        title=f"Feature importance (production) — N={ds.N}, {len(seeds_list)} seeds",
    )

    # --- SHAP: per-cell on full inference set; summary on labeled subset ---
    shap_long_inf = pd.concat(shap_frames_inf, ignore_index=True) if shap_frames_inf else pd.DataFrame()
    shap_emitted = not shap_long_inf.empty
    if shap_emitted:
        shap_long_inf.to_parquet(out_dir / "shap_inference_per_cell.parquet", index=False)
        labeled_cells = set(ds.cell_names[ds.label_mask].tolist())
        shap_long_labeled = shap_long_inf[shap_long_inf["cell_name"].isin(labeled_cells)]
        summary = aggregate_shap(shap_long_labeled, ds.feature_names)
        summary.to_csv(out_dir / "shap_inference_summary.csv", index=False)
        plot_shap_summary(
            summary, plots_dir / "shap_summary.png",
            title=f"Mean |SHAP| (labeled subset) — N={ds.N}, {len(seeds_list)} seeds",
        )

    # --- best_params_production.json + optuna_history.csv ---
    (out_dir / "best_params_production.json").write_text(
        json.dumps(representative_params, indent=2, default=str) + "\n"
    )
    if optuna_history_df is not None and not optuna_history_df.empty:
        optuna_history_df.to_csv(out_dir / "optuna_history.csv", index=False)
    else:
        (out_dir / "optuna_history.csv").write_text("")

    # --- Manifest ---
    preprocess_manifest = _load_preprocess_manifest(ds.source_dir)
    manifest = build_manifest(
        config=config,
        runtime_seconds=time.time() - t0,
        n_cells_labeled_trainable=int(ds.label_mask.sum()),
        n_cells_scored=int(len(inference_idx)),
        preprocess_manifest=preprocess_manifest,
        shap_summary_scope="labeled_subset",
    )
    write_manifest(out_dir, manifest)

    print(
        f"[production] DONE — predicted {len(preds)} cells "
        f"(mean P(pass) = {preds['mean_proba_pass'].mean():.3f}, "
        f"std = {preds['mean_proba_pass'].std():.3f}); "
        f"({manifest['runtime_seconds']}s)"
    )
    return {"status": "ok", "out_dir": str(out_dir)}


def _load_representative_from_validation(
    *, out_root: str, slug: str,
) -> dict[str, Any]:
    """Locate the matching validation run, dispatch on its tuning protocol,
    and collapse the per-seed/per-fold HP table into a single representative set."""
    val_dir = run_dir(out_root, "validation", slug)
    manifest_path = val_dir / "manifest.json"
    if not manifest_path.exists():
        raise FileNotFoundError(
            f"no matching validation run at {val_dir}. "
            f"Either run validation with the same axes first, or pass "
            f"--production-params-source retune."
        )
    val_manifest = json.loads(manifest_path.read_text())
    protocol = val_manifest.get("tuning", {}).get("protocol")
    if protocol == "tune_inner_cv":
        csv_path = val_dir / "best_params_per_seed.csv"
        index_columns = ["seed"]
    elif protocol == "nested_cv":
        csv_path = val_dir / "best_params_per_fold.csv"
        index_columns = ["seed", "fold"]
    else:
        raise ValueError(
            f"validation manifest at {manifest_path} has unrecognized "
            f"tuning.protocol {protocol!r}"
        )
    if not csv_path.exists():
        raise FileNotFoundError(
            f"validation run at {val_dir} is missing the expected HP "
            f"artifact {csv_path.name} (protocol={protocol!r}). "
            f"Re-run validation under the same protocol, or use "
            f"--production-params-source retune."
        )
    # Discover HP columns by reading the CSV header and stripping the index.
    header = pd.read_csv(csv_path, nrows=0).columns.tolist()
    hp_columns = [c for c in header if c not in index_columns]
    rows = read_hp_csv(
        csv_path, index_columns=index_columns, hp_columns=hp_columns,
    )
    template = {c: rows[0][c] for c in hp_columns}
    hp_only_rows = [{c: r[c] for c in hp_columns} for r in rows]
    representative, _ = representative_hp_set(hp_only_rows, template)
    return representative


def _retune_on_full_labeled(
    *,
    ModelClass: type,
    X_train: pd.DataFrame,
    y_train: np.ndarray,
    n_trials: int,
    inner_cv: int,
    optimize: str,
    imputer_strategy: str,
    seed: int,
) -> tuple[dict[str, Any], pd.DataFrame]:
    """Single Optuna study on the full labeled set. Returns (best_params,
    optuna_history_df). The seed is the first production seed for
    Optuna-sampler determinism."""
    best_params, study = tune(
        ModelClass, X_train, y_train,
        n_trials=n_trials, inner_cv=inner_cv, seed=seed,
        optimize=optimize, imputer_strategy=imputer_strategy,
    )
    history_df = pd.DataFrame([
        {"seed": seed, "fold": -1, "trial_number": t.number,
         "value": t.value, "state": t.state.name,
         **{f"param_{k}": v for k, v in t.params.items()}}
        for t in study.trials
    ])
    return best_params, history_df


def _load_preprocess_manifest(bundle_dir: Path) -> dict[str, Any] | None:
    p = bundle_dir / "manifest.json"
    if not p.exists():
        return None
    return json.loads(p.read_text())

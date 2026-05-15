"""SHAP analysis — per-seed TreeSHAP (or model-native equivalent).

Models that return None from `compute_shap` (e.g., EBM/BART stubs) yield
empty DataFrames; the pipeline then skips the SHAP files entirely.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from cell_classifier.models.base import BaseModel


def compute_seed_shap(
    model: BaseModel,
    X_test: pd.DataFrame,
    cell_names: np.ndarray,
    seed: int,
) -> pd.DataFrame:
    """Long-format SHAP for one seed. Columns: seed, cell_name, feature, shap_value."""
    shap_vals = model.compute_shap(X_test)
    if shap_vals is None:
        return pd.DataFrame(columns=["seed", "cell_name", "feature", "shap_value"])

    shap_vals = np.asarray(shap_vals)
    if shap_vals.shape != (len(X_test), X_test.shape[1]):
        raise ValueError(
            f"compute_shap returned shape {shap_vals.shape}; "
            f"expected ({len(X_test)}, {X_test.shape[1]})"
        )

    df = pd.DataFrame(shap_vals, columns=list(X_test.columns))
    df.insert(0, "cell_name", cell_names)
    df.insert(0, "seed", int(seed))
    return df.melt(
        id_vars=["seed", "cell_name"],
        var_name="feature",
        value_name="shap_value",
    )


def aggregate_shap(per_seed_long: pd.DataFrame, feature_names: list[str]) -> pd.DataFrame:
    """One row per feature: mean ± std of mean(|SHAP|) across seeds."""
    if per_seed_long.empty:
        return pd.DataFrame(
            columns=["feature", "mean_abs_shap_mean", "mean_abs_shap_std"]
        )

    per_seed_mean = (
        per_seed_long
        .assign(abs_shap=lambda d: d["shap_value"].abs())
        .groupby(["seed", "feature"], as_index=False)["abs_shap"].mean()
    )

    summary = (
        per_seed_mean.groupby("feature")["abs_shap"]
        .agg(mean_abs_shap_mean="mean", mean_abs_shap_std="std")
        .reset_index()
    )
    summary["mean_abs_shap_std"] = summary["mean_abs_shap_std"].fillna(0.0)

    feature_order = pd.CategoricalDtype(categories=feature_names, ordered=True)
    summary["feature"] = summary["feature"].astype(feature_order)
    return summary.sort_values("mean_abs_shap_mean", ascending=False).reset_index(drop=True)

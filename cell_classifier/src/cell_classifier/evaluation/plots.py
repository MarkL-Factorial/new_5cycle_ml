"""Plotting helpers — per-run diagnostics and cross-N summary figures.

Two flavors:

  * `plot_perm_importance(df, out_path)`  — horizontal bar chart of
    permutation importance (mean ± std across seeds). Drawn by the pipeline
    inside each `out/rf_n{N}/plots/` folder.
  * `plot_shap_summary(df, out_path)` — same shape, mean(|SHAP|).
  * `plot_cross_n_distribution(metric_long, out_path)` — boxplot of test
    metrics over seeds with one box per N. Drawn by `run_all.py`.

All figures use matplotlib only (no seaborn). Saved at 150 dpi.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def _ensure_parent(out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)


def plot_perm_importance(df: pd.DataFrame, out_path: Path, title: str = "") -> None:
    _ensure_parent(out_path)
    d = df.sort_values("perm_mean", ascending=True)
    fig, ax = plt.subplots(figsize=(7, max(3, 0.35 * len(d))))
    y = np.arange(len(d))
    ax.barh(y, d["perm_mean"], xerr=d["perm_std"], color="#4878CF",
            ecolor="#444444", capsize=2)
    ax.axvline(0, color="black", linewidth=0.6)
    ax.set_yticks(y)
    ax.set_yticklabels(d["feature"])
    ax.set_xlabel("Permutation importance (ROC-AUC drop)")
    if title:
        ax.set_title(title)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def plot_shap_summary(df: pd.DataFrame, out_path: Path, title: str = "") -> None:
    _ensure_parent(out_path)
    if df.empty:
        return
    d = df.sort_values("mean_abs_shap_mean", ascending=True)
    fig, ax = plt.subplots(figsize=(7, max(3, 0.35 * len(d))))
    y = np.arange(len(d))
    ax.barh(y, d["mean_abs_shap_mean"], xerr=d["mean_abs_shap_std"],
            color="#C44E52", ecolor="#444444", capsize=2)
    ax.set_yticks(y)
    ax.set_yticklabels(d["feature"])
    ax.set_xlabel("Mean |SHAP value| (class 1)")
    if title:
        ax.set_title(title)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def plot_cross_n_distribution(
    metric_long: pd.DataFrame,
    out_path: Path,
    metrics: tuple[str, ...] = ("test_f1", "test_roc_auc", "test_accuracy"),
) -> None:
    """Boxplot of metrics over seeds, faceted across (metric, N).

    `metric_long` must have columns: N, seed, plus one column per metric.
    """
    _ensure_parent(out_path)
    Ns = sorted(metric_long["N"].unique())
    fig, axes = plt.subplots(1, len(metrics), figsize=(4.5 * len(metrics), 4.5),
                             sharey=False)
    if len(metrics) == 1:
        axes = [axes]
    for ax, metric in zip(axes, metrics):
        data = [metric_long.loc[metric_long["N"] == N, metric].dropna().to_numpy()
                for N in Ns]
        bp = ax.boxplot(data, tick_labels=[f"N={N}" for N in Ns], showmeans=True,
                        meanprops={"marker": "D", "markerfacecolor": "white",
                                   "markeredgecolor": "black", "markersize": 6})
        for patch, _ in zip(bp["boxes"], data):
            patch.set_color("#1f77b4")
        ax.set_title(metric.replace("test_", "").upper())
        ax.set_ylabel(metric)
        ax.grid(axis="y", alpha=0.3)
    fig.suptitle(f"Test-set metrics across {len(metric_long['seed'].unique())} "
                 f"seeds, by threshold", y=1.02)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_paired_forest(
    paired_df: pd.DataFrame,
    out_path: Path,
) -> None:
    """Forest plot of paired-t deltas across N pairs.

    `paired_df` columns: pair (str like 'N=300 vs N=200'), metric, delta_pp,
    ci_lo_pp, ci_hi_pp, p_value.
    """
    _ensure_parent(out_path)
    metrics = list(paired_df["metric"].unique())
    fig, axes = plt.subplots(1, len(metrics), figsize=(4 * len(metrics), 3),
                             sharey=True)
    if len(metrics) == 1:
        axes = [axes]
    for ax, metric in zip(axes, metrics):
        d = paired_df[paired_df["metric"] == metric].reset_index(drop=True)
        y = np.arange(len(d))
        ax.errorbar(
            d["delta_pp"], y,
            xerr=[d["delta_pp"] - d["ci_lo_pp"], d["ci_hi_pp"] - d["delta_pp"]],
            fmt="o", color="black", capsize=4, ecolor="#666666",
        )
        ax.axvline(0, color="red", linestyle="--", linewidth=0.8)
        for i, row in d.iterrows():
            marker = "*" if row["p_value"] < 0.05 else ""
            ax.text(row["ci_hi_pp"] + 0.2, y[i],
                    f"p={row['p_value']:.3f}{marker}",
                    va="center", fontsize=8)
        ax.set_yticks(y)
        ax.set_yticklabels(d["pair"])
        ax.set_xlabel(f"Δ {metric.replace('test_', '')} (pp)")
        ax.set_title(metric.replace("test_", "").upper())
    fig.suptitle("Cross-threshold paired-t comparison", y=1.02)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)

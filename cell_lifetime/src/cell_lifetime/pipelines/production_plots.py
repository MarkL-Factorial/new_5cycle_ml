"""Plots emitted by `cell-lifetime production`.

Three figures, ported from `experiments/exp_j_production_predictions/`:

  - `actual_vs_predicted.png`        — RSF cycle life (faded vs censored)
  - `classifier_roc_confusion.png`   — 2×3 ROC + confusion matrix per N
  - `classifier_prob_vs_cycle.png`   — 1×3 OOF prob vs cycle, per N

Evaluation surfaces:
  - Classifier ROC/confusion/prob-vs-cycle plots evaluate on cells where
    `in_training_set_n{N}` is True (= trainable_n{N}). This matches the
    prior multi-seed experiments' reported F1/AUC.
  - Actual vs predicted plots distinguish faded (filled circles, known
    cycle life) from censored cells (right-pointing triangles, actual
    cycle ≥ observed n_regular).
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score, confusion_matrix, f1_score, roc_auc_score, roc_curve,
)


NS = (200, 300, 400)


def plot_actual_vs_predicted(df: pd.DataFrame, out_path: Path) -> Path:
    """RSF median-survival vs actual cycle life, faded vs censored markers."""
    faded = df[df["status"] == "faded"]
    censored = df[df["status"] == "in_testing"]

    x_faded = faded["last_fade_cycle"].to_numpy()
    y_faded = faded["rsf_median_cycle"].to_numpy()
    x_cens = censored["n_regular"].to_numpy()
    y_cens = censored["rsf_median_cycle"].to_numpy()

    err = y_faded - x_faded
    mae = float(np.mean(np.abs(err)))
    rmse = float(np.sqrt(np.mean(err ** 2)))
    mape = float(np.mean(np.abs(err) / x_faded)) * 100.0
    ss_res = float(np.sum(err ** 2))
    ss_tot = float(np.sum((x_faded - x_faded.mean()) ** 2))
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else float("nan")

    xmax = float(max(x_faded.max(), x_cens.max(), y_faded.max(), y_cens.max()))
    lim = (0.0, xmax * 1.05)

    fig, ax = plt.subplots(figsize=(7.5, 7.5))
    ax.plot(lim, lim, linestyle="--", color="gray", linewidth=1.0,
            label="y = x (perfect prediction)")
    ax.scatter(
        x_faded, y_faded,
        marker="o", facecolor="#1f77b4", edgecolor="white", linewidth=0.4,
        s=42, alpha=0.85,
        label=f"Faded cells (n={len(faded)}) — actual = last_fade_cycle",
    )
    ax.scatter(
        x_cens, y_cens,
        marker=">", facecolor="#ff7f0e", edgecolor="white", linewidth=0.4,
        s=48, alpha=0.85,
        label=f"Censored cells (n={len(censored)}) — actual ≥ n_regular",
    )

    ax.set_xlabel("Actual cycle life (cycles)")
    ax.set_ylabel("Predicted cycle life — RSF median survival (cycles)")
    ax.set_title(
        "Production — Actual vs Predicted Cycle Life\n"
        "RSF × fs_cv, trained on all cells (faded + censored)"
    )
    ax.set_xlim(lim)
    ax.set_ylim(lim)
    ax.set_aspect("equal", adjustable="box")
    ax.grid(True, linestyle=":", alpha=0.6)
    ax.legend(loc="upper left", frameon=True, fontsize=9)

    ax.text(
        0.97, 0.03,
        f"Faded-cell metrics:\n"
        f"  MAE  = {mae:.1f} cyc\n"
        f"  RMSE = {rmse:.1f} cyc\n"
        f"  MAPE = {mape:.1f}%\n"
        f"  R²   = {r2:.3f}",
        transform=ax.transAxes, ha="right", va="bottom", fontsize=9,
        family="monospace",
        bbox=dict(boxstyle="round,pad=0.5", facecolor="white", edgecolor="0.6"),
    )

    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    return out_path


def _feature_label_for_title(out_path: Path) -> str:
    """Read sibling best_params.json (if present) to render an accurate
    feature label in the suptitle. Falls back to a generic phrase.
    """
    sibling = out_path.parent / "best_params.json"
    if not sibling.exists():
        return "ebm_classifier"
    try:
        import json
        meta = json.loads(sibling.read_text())
        cls_fs = meta.get("feature_subsets", {}).get("classifier")
        n_feat = meta.get("n_features")
        if cls_fs and n_feat:
            return f"ebm_classifier × {cls_fs} ({n_feat} features)"
    except Exception:
        pass
    return "ebm_classifier"


def plot_classifier_roc_confusion(df: pd.DataFrame, out_path: Path) -> Path:
    """2×3 grid: ROC top, confusion matrix bottom, one column per N.

    Evaluates on cells inside `in_training_set_n{N}` (matches the
    trainable_n{N} surface used by prior multi-seed runs). Per-panel
    metrics (AUC, Acc, F1_pass, F1_fail) live in an in-panel text box
    so the title stays short and never overflows into siblings.
    """
    fig, axes = plt.subplots(2, 3, figsize=(13, 8.5))

    for col, N in enumerate(NS):
        sub = df[df[f"in_training_set_n{N}"]].copy()
        y_true = sub[f"true_pass_n{N}"].to_numpy().astype(int)
        y_prob = sub[f"oof_prob_pass_n{N}"].to_numpy()
        y_pred = (y_prob >= 0.5).astype(int)
        n_faded = int((sub["status"] == "faded").sum())
        n_censored = int((sub["status"] == "in_testing").sum())
        n_pass = int(y_true.sum())
        n_fail = len(y_true) - n_pass

        # ROC
        ax = axes[0, col]
        fpr, tpr, _ = roc_curve(y_true, y_prob)
        auc = roc_auc_score(y_true, y_prob)
        f1_pass = f1_score(y_true, y_pred, pos_label=1)
        f1_fail = f1_score(y_true, y_pred, pos_label=0)
        acc = accuracy_score(y_true, y_pred)
        ax.plot(fpr, tpr, color="#1f77b4", linewidth=2, label="ROC")
        ax.plot([0, 1], [0, 1], linestyle="--", color="gray", linewidth=1,
                label="random")
        ax.set_xlabel("False Positive Rate")
        ax.set_ylabel("True Positive Rate")
        ax.set_title(
            f"N={N}  ({len(y_true)} cells)\n"
            f"pass={n_pass} / fail={n_fail}  ·  "
            f"{n_faded} faded + {n_censored} cens.",
            fontsize=10,
        )
        ax.text(
            0.97, 0.03,
            f"AUC     = {auc:.3f}\n"
            f"Acc     = {acc:.3f}\n"
            f"F1_pass = {f1_pass:.3f}\n"
            f"F1_fail = {f1_fail:.3f}",
            transform=ax.transAxes, ha="right", va="bottom",
            fontsize=9, family="monospace",
            bbox=dict(boxstyle="round,pad=0.4",
                      facecolor="white", edgecolor="0.6"),
        )
        ax.set_xlim(-0.02, 1.02)
        ax.set_ylim(-0.02, 1.02)
        ax.grid(True, linestyle=":", alpha=0.6)
        ax.legend(loc="center right", fontsize=8)
        ax.set_aspect("equal")

        # Confusion matrix
        ax = axes[1, col]
        cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
        ax.imshow(cm, cmap="Blues", vmin=0, vmax=cm.max())
        ax.set_xticks([0, 1])
        ax.set_yticks([0, 1])
        ax.set_xticklabels(["pred fail", "pred pass"])
        ax.set_yticklabels(["true fail", "true pass"])
        ax.set_title(f"Confusion matrix — N={N} (threshold=0.5)")
        for i in range(2):
            for j in range(2):
                color = "white" if cm[i, j] > cm.max() * 0.5 else "black"
                ax.text(j, i, str(cm[i, j]), ha="center", va="center",
                        color=color, fontsize=13, fontweight="bold")

    fig.suptitle(
        f"Production — {_feature_label_for_title(out_path)}, "
        "OOF predictions on trainable_n{N} cells",
        fontsize=12,
    )
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return out_path


def plot_classifier_prob_vs_cycle(df: pd.DataFrame, out_path: Path) -> Path:
    """1×3 panels: OOF P(pass N) vs cycle life, by N.

    Faded cells plotted at (last_fade_cycle, OOF prob) with pass/fail
    markers; censored-in-training cells plotted at (n_regular, OOF prob)
    as right-pointing triangles (signal: actual cycle ≥ x).
    """
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.8), sharey=True)
    for col, N in enumerate(NS):
        ax = axes[col]
        sub = df[df[f"in_training_set_n{N}"]].copy()
        faded = sub[sub["status"] == "faded"]
        censored = sub[sub["status"] == "in_testing"]

        # Faded — by true pass/fail
        f_cyc = faded["last_fade_cycle"].to_numpy()
        f_prob = faded[f"oof_prob_pass_n{N}"].to_numpy()
        f_true_pass = (f_cyc >= N).astype(int)
        for label_val, color, name in (
            (1, "#2ca02c", "faded → true pass"),
            (0, "#d62728", "faded → true fail"),
        ):
            sel = (f_true_pass == label_val)
            marker = "o" if label_val == 1 else "X"
            ax.scatter(
                f_cyc[sel], f_prob[sel],
                marker=marker, facecolor=color, edgecolor="white",
                linewidth=0.4, s=42, alpha=0.85,
                label=f"{name} (n={int(sel.sum())})",
            )

        # Censored-in-training cells (all true pass)
        c_cyc = censored["n_regular"].to_numpy()
        c_prob = censored[f"oof_prob_pass_n{N}"].to_numpy()
        ax.scatter(
            c_cyc, c_prob,
            marker=">", facecolor="#1f77b4", edgecolor="white",
            linewidth=0.4, s=46, alpha=0.85,
            label=f"censored → true pass, actual ≥ x (n={len(c_cyc)})",
        )

        ax.axvline(N, color="gray", linestyle="--", linewidth=1, label=f"N={N}")
        ax.axhline(0.5, color="gray", linestyle=":", linewidth=1)
        ax.set_xlabel("Cycle life (faded: last_fade; censored: n_regular)")
        if col == 0:
            ax.set_ylabel("OOF P(pass N)")
        ax.set_title(f"N={N}")
        ax.set_ylim(-0.03, 1.03)
        ax.grid(True, linestyle=":", alpha=0.6)
        ax.legend(loc="center right", fontsize=7)

    fig.suptitle(
        "Production — OOF P(pass N) vs cycle life, by N (trainable_n{N} cells)",
        fontsize=12,
    )
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return out_path


def render_all(df: pd.DataFrame, out_dir: Path) -> list[Path]:
    """Emit all 3 production plots into `out_dir`. Returns the paths written."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    paths = [
        plot_actual_vs_predicted(df, out_dir / "actual_vs_predicted.png"),
        plot_classifier_roc_confusion(df, out_dir / "classifier_roc_confusion.png"),
        plot_classifier_prob_vs_cycle(df, out_dir / "classifier_prob_vs_cycle.png"),
    ]
    return paths

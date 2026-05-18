#!/usr/bin/env python
"""Visualize the 3 ebm_classifier × fs_a_only models at N=200, 300, 400.

Uses the **out-of-fold (OOF) probabilities** from `predictions.csv`.
The evaluation surface is **the cells inside `trainable_n{N}`** — i.e.
all cells with a definitive label at N (faded cells with known cycle
life + censored cells observed past N). This matches the eval surface
the prior multi-seed cell_lifetime experiments used for their reported
F1/AUC, so numbers here are directly comparable.

Two figures:

  1. `classifier_roc_confusion.png` — 2 rows × 3 cols:
       row 1: ROC curve per N, with AUC annotated (eval on trainable_n{N})
       row 2: confusion matrix at threshold 0.5

  2. `classifier_prob_vs_cycle.png` — 1 row × 3 cols:
       OOF P(pass N) vs cycle life on the x-axis. Faded cells use
       last_fade_cycle (filled circles); censored-in-training cells
       use n_regular (right triangles, signalling actual cycle ≥ x).
       Vertical dashed line at N marks the truth boundary.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import (
    confusion_matrix, roc_auc_score, roc_curve, f1_score, accuracy_score,
)

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


HERE = Path(__file__).resolve().parent
NS = (200, 300, 400)


def plot_roc_and_confusion(df: pd.DataFrame) -> Path:
    fig, axes = plt.subplots(2, 3, figsize=(13, 8.5))

    for col, N in enumerate(NS):
        # Eval on the full trainable_n{N} set (faded + censored-known-pass).
        # This matches the eval surface used by prior multi-seed runs.
        sub = df[df[f"in_training_set_n{N}"]].copy()
        y_true = sub[f"true_pass_n{N}"].to_numpy().astype(int)
        y_prob = sub[f"oof_prob_pass_n{N}"].to_numpy()
        y_pred = (y_prob >= 0.5).astype(int)
        n_faded = int((sub["status"] == "faded").sum())
        n_censored = int((sub["status"] == "censored").sum())

        # ----- ROC curve -----
        ax = axes[0, col]
        fpr, tpr, _ = roc_curve(y_true, y_prob)
        auc = roc_auc_score(y_true, y_prob)
        f1 = f1_score(y_true, y_pred)
        acc = accuracy_score(y_true, y_pred)

        ax.plot(fpr, tpr, color="#1f77b4", linewidth=2,
                label=f"AUC = {auc:.3f}")
        ax.plot([0, 1], [0, 1], linestyle="--", color="gray", linewidth=1,
                label="random")
        ax.set_xlabel("False Positive Rate")
        ax.set_ylabel("True Positive Rate")
        ax.set_title(
            f"ROC — N={N}\nF1={f1:.3f}, Acc={acc:.3f}, "
            f"pass={int(y_true.sum())}/{len(y_true)} "
            f"({n_faded} faded + {n_censored} cens.)"
        )
        ax.set_xlim(-0.02, 1.02)
        ax.set_ylim(-0.02, 1.02)
        ax.grid(True, linestyle=":", alpha=0.6)
        ax.legend(loc="lower right", fontsize=9)
        ax.set_aspect("equal")

        # ----- Confusion matrix -----
        ax = axes[1, col]
        cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
        im = ax.imshow(cm, cmap="Blues", vmin=0, vmax=cm.max())
        ax.set_xticks([0, 1])
        ax.set_yticks([0, 1])
        ax.set_xticklabels(["pred fail", "pred pass"])
        ax.set_yticklabels(["true fail", "true pass"])
        ax.set_title(f"Confusion matrix — N={N} (threshold=0.5)")
        # Annotate each cell with counts
        for i in range(2):
            for j in range(2):
                color = "white" if cm[i, j] > cm.max() * 0.5 else "black"
                ax.text(j, i, str(cm[i, j]), ha="center", va="center",
                        color=color, fontsize=13, fontweight="bold")
        # Per-row totals (recall) and per-column totals (precision)
        row_totals = cm.sum(axis=1)
        col_totals = cm.sum(axis=0)
        if row_totals[1] > 0:
            recall = cm[1, 1] / row_totals[1]
            ax.text(1.7, 1, f"recall\n{recall:.2f}",
                    ha="center", va="center", fontsize=8, color="0.3")
        if col_totals[1] > 0:
            precision = cm[1, 1] / col_totals[1]
            ax.text(1, -0.55, f"precision: {precision:.2f}",
                    ha="center", va="bottom", fontsize=8, color="0.3")

    fig.suptitle(
        "Exp J — ebm_classifier × fs_a_only (3 features), OOF predictions on trainable_n{N} cells",
        fontsize=12, y=0.995,
    )
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    out_png = HERE / "classifier_roc_confusion.png"
    fig.savefig(out_png, dpi=150)
    plt.close(fig)
    return out_png


def plot_prob_vs_cycle(df: pd.DataFrame) -> Path:
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.8), sharey=True)
    for col, N in enumerate(NS):
        ax = axes[col]
        sub = df[df[f"in_training_set_n{N}"]].copy()

        faded = sub[sub["status"] == "faded"]
        censored = sub[sub["status"] == "censored"]

        # Faded cells — actual cycle is last_fade_cycle. Two markers by truth.
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

        # Censored-in-training cells — all true pass; actual cycle ≥ n_regular.
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
        "Exp J — OOF P(pass N) vs cycle life, by N (trainable_n{N} cells)",
        fontsize=12, y=1.02,
    )
    fig.tight_layout()
    out_png = HERE / "classifier_prob_vs_cycle.png"
    fig.savefig(out_png, dpi=150)
    plt.close(fig)
    return out_png


def main() -> int:
    df = pd.read_csv(HERE / "predictions.csv")
    p1 = plot_roc_and_confusion(df)
    p2 = plot_prob_vs_cycle(df)
    print(f"Wrote {p1}")
    print(f"Wrote {p2}")
    # Print a small text summary — eval on trainable_n{N} (matches prior runs)
    for N in NS:
        sub = df[df[f"in_training_set_n{N}"]]
        y_true = sub[f"true_pass_n{N}"].to_numpy().astype(int)
        y_prob = sub[f"oof_prob_pass_n{N}"].to_numpy()
        y_pred = (y_prob >= 0.5).astype(int)
        auc = roc_auc_score(y_true, y_prob)
        f1 = f1_score(y_true, y_pred)
        acc = accuracy_score(y_true, y_pred)
        cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
        n_faded = int((sub["status"] == "faded").sum())
        n_censored = int((sub["status"] == "censored").sum())
        print(f"  N={N}: AUC={auc:.3f}, F1={f1:.3f}, Acc={acc:.3f}, "
              f"n={len(y_true)} ({n_faded} faded + {n_censored} cens.), "
              f"CM[TN,FP,FN,TP]={cm.flatten().tolist()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

#!/usr/bin/env python
"""Plot actual vs predicted cycle life from predictions.csv.

Faded (event) cells: solid blue circles at (last_fade_cycle, rsf_median_cycle).
Censored cells: orange right-pointing triangles at (n_regular, rsf_median_cycle).
The triangle shape and the rightward arrow on the legend signal that the
"actual" for censored cells is a lower bound — the true cycle life is at
least n_regular.

A y=x reference line is drawn. MAE/R²/MAPE on faded cells are annotated.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


HERE = Path(__file__).resolve().parent


def main() -> int:
    df = pd.read_csv(HERE / "predictions.csv")
    faded = df[df["status"] == "faded"]
    censored = df[df["status"] == "censored"]

    # Actual (x) — for faded use last_fade_cycle; for censored use n_regular
    # (observation length, which is the lower bound).
    x_faded = faded["last_fade_cycle"].to_numpy()
    y_faded = faded["rsf_median_cycle"].to_numpy()
    x_cens = censored["n_regular"].to_numpy()
    y_cens = censored["rsf_median_cycle"].to_numpy()

    # Faded-only metrics
    err = y_faded - x_faded
    mae = float(np.mean(np.abs(err)))
    rmse = float(np.sqrt(np.mean(err ** 2)))
    mape = float(np.mean(np.abs(err) / x_faded)) * 100.0
    ss_res = float(np.sum(err ** 2))
    ss_tot = float(np.sum((x_faded - x_faded.mean()) ** 2))
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else float("nan")

    # Axis extents
    xmax = max(x_faded.max(), x_cens.max(), y_faded.max(), y_cens.max())
    xmin = 0
    lim = (xmin, xmax * 1.05)

    fig, ax = plt.subplots(figsize=(7.5, 7.5))

    # y=x reference
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
        "Exp J — Actual vs Predicted Cycle Life\n"
        "RSF × fs_cv, trained on all 415 cells (187 faded + 228 censored)"
    )
    ax.set_xlim(lim)
    ax.set_ylim(lim)
    ax.set_aspect("equal", adjustable="box")
    ax.grid(True, linestyle=":", alpha=0.6)
    ax.legend(loc="upper left", frameon=True, fontsize=9)

    # Annotate metrics on faded cells
    txt = (
        "Faded-cell metrics:\n"
        f"  MAE  = {mae:.1f} cyc\n"
        f"  RMSE = {rmse:.1f} cyc\n"
        f"  MAPE = {mape:.1f}%\n"
        f"  R²   = {r2:.3f}"
    )
    ax.text(
        0.97, 0.03, txt,
        transform=ax.transAxes, ha="right", va="bottom", fontsize=9,
        family="monospace",
        bbox=dict(boxstyle="round,pad=0.5", facecolor="white", edgecolor="0.6"),
    )

    out_png = HERE / "actual_vs_predicted.png"
    fig.tight_layout()
    fig.savefig(out_png, dpi=150)
    plt.close(fig)
    print(f"Wrote {out_png}")
    print(
        f"  faded n={len(faded)}, censored n={len(censored)}; "
        f"MAE={mae:.1f}, MAPE={mape:.1f}%, R²={r2:.3f}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

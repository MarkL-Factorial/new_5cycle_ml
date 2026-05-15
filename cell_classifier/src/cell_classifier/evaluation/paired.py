"""Paired-t comparison across thresholds.

Each seed produces one test metric per N. Pairing on `seed` (the same model
initialization is used across N), this module asks: at the per-seed level, is
N=A's test metric systematically different from N=B's?

Caveat noted in the plan: the train/test cells differ across N (each N has its
own `trainable_n{N}` filter), so the comparison is *not* a like-for-like
"same cells, different threshold" test — it's a "same model-init randomness,
different cell pool" test. We surface this in the output column `note` so the
caller does not over-interpret.

Returns a tidy DataFrame: one row per (pair, metric).
"""

from __future__ import annotations

from itertools import combinations
from typing import Sequence

import numpy as np
import pandas as pd
from scipy import stats


def _bootstrap_ci(
    paired_delta: np.ndarray,
    n_resamples: int = 10000,
    alpha: float = 0.05,
    seed: int = 20260514,
) -> tuple[float, float]:
    if len(paired_delta) == 0:
        return float("nan"), float("nan")
    rng = np.random.default_rng(seed)
    boot_means = np.empty(n_resamples, dtype=float)
    n = len(paired_delta)
    for i in range(n_resamples):
        sample = rng.choice(paired_delta, size=n, replace=True)
        boot_means[i] = sample.mean()
    lo = float(np.quantile(boot_means, alpha / 2))
    hi = float(np.quantile(boot_means, 1 - alpha / 2))
    return lo, hi


def _cohens_d_paired(delta: np.ndarray) -> float:
    if len(delta) < 2 or float(delta.std(ddof=1)) == 0.0:
        return float("nan")
    return float(delta.mean() / delta.std(ddof=1))


def paired_compare(
    per_n_metrics: dict[int, pd.DataFrame],
    metrics: Sequence[str] = ("test_f1", "test_roc_auc", "test_accuracy"),
    pair_order: str = "descending_N",
) -> pd.DataFrame:
    """Compute paired-t across N for each given metric.

    `per_n_metrics` is {N: per_seed_metrics_df}, with `seed` as a column.

    `pair_order='descending_N'` means pairs are (larger_N − smaller_N), so a
    positive Δ favours the larger N. Use 'ascending_N' to flip.
    """
    Ns = sorted(per_n_metrics.keys())
    rows = []
    for a, b in combinations(Ns, 2):
        big, small = (b, a) if pair_order == "descending_N" else (a, b)
        df_big = per_n_metrics[big][["seed", *metrics]].rename(
            columns={m: f"{m}_big" for m in metrics}
        )
        df_small = per_n_metrics[small][["seed", *metrics]].rename(
            columns={m: f"{m}_small" for m in metrics}
        )
        paired = df_big.merge(df_small, on="seed", how="inner")

        if paired.empty:
            for metric in metrics:
                rows.append({
                    "pair": f"N={big} vs N={small}", "metric": metric,
                    "n_seeds": 0, "delta_pp": float("nan"),
                    "ci_lo_pp": float("nan"), "ci_hi_pp": float("nan"),
                    "p_value": float("nan"), "cohens_d": float("nan"),
                    "note": "no overlapping seeds",
                })
            continue

        for metric in metrics:
            big_vals = paired[f"{metric}_big"].to_numpy(dtype=float)
            small_vals = paired[f"{metric}_small"].to_numpy(dtype=float)
            mask = ~(np.isnan(big_vals) | np.isnan(small_vals))
            delta = (big_vals[mask] - small_vals[mask]) * 100.0  # in pp

            if len(delta) < 2:
                rows.append({
                    "pair": f"N={big} vs N={small}", "metric": metric,
                    "n_seeds": int(len(delta)),
                    "delta_pp": float("nan"),
                    "ci_lo_pp": float("nan"), "ci_hi_pp": float("nan"),
                    "p_value": float("nan"), "cohens_d": float("nan"),
                    "note": "fewer than 2 paired seeds",
                })
                continue

            t_res = stats.ttest_rel(big_vals[mask], small_vals[mask])
            ci_lo, ci_hi = _bootstrap_ci(delta)
            rows.append({
                "pair": f"N={big} vs N={small}",
                "metric": metric,
                "n_seeds": int(len(delta)),
                "delta_pp": float(delta.mean()),
                "ci_lo_pp": float(ci_lo),
                "ci_hi_pp": float(ci_hi),
                "p_value": float(t_res.pvalue),
                "cohens_d": _cohens_d_paired(delta),
                "note": "different cell pools per N (paired only on seed)",
            })

    return pd.DataFrame(rows)

"""Survival metrics: C-index + time-dependent AUC at the classification horizons.

Wraps scikit-survival's `concordance_index_censored` for the rank-based
C-index, and computes a hand-rolled time-dependent AUC at N ∈ {200, 300, 400}
by converting model predictions into a "predicted to pass N" boolean and
calling sklearn's roc_auc_score against the (observed-by-N) outcome.

The hand-rolled time-AUC avoids needing a full survival function and works
identically for both AFT (`risk_orientation='time_high'`) and RSF
(`risk_orientation='risk_high'`) callers, as long as the pipeline has
already normalised `risk_scores` so that higher = sooner failure.

Integrated Brier Score is intentionally omitted: it requires a fitted
survival function S(t|x), which AFT/RSF expose differently and which
isn't load-bearing for cross-model comparison at our N-horizons.
"""

from __future__ import annotations

import numpy as np
from sklearn.metrics import roc_auc_score
from sksurv.metrics import concordance_index_censored


def _c_index(event: np.ndarray, time: np.ndarray, risk_scores: np.ndarray) -> float:
    """Wraps sksurv's concordance_index_censored; returns just the C-index.

    `risk_scores`: higher = sooner failure (the standard sksurv convention).
    """
    event_bool = np.asarray(event, dtype=bool)
    if event_bool.sum() < 2:
        return float("nan")
    time_f = np.asarray(time, dtype=float)
    risk_f = np.asarray(risk_scores, dtype=float)
    try:
        c, _conc, _disc, _tied_r, _tied_t = concordance_index_censored(
            event_bool, time_f, risk_f
        )
        return float(c)
    except (ValueError, ZeroDivisionError):
        return float("nan")


def _auc_at_horizon(
    event: np.ndarray,
    time: np.ndarray,
    risk_scores: np.ndarray,
    horizon: int,
) -> float:
    """Time-dependent AUC at a fixed horizon N.

    Constructs the binary labels:
      - "bad by N": (event==1) AND (time <= N)  → label = 1
      - "passed N (or censored after N)": time > N  → label = 0
      - "censored before N": time <= N AND event==0 → DROP (unknown outcome)

    Then AUC against `risk_scores` (higher = sooner failure → more likely
    to be class 1).
    """
    event = np.asarray(event, dtype=bool)
    time = np.asarray(time, dtype=float)
    risk = np.asarray(risk_scores, dtype=float)
    # Usable: faded-by-N OR survived past N
    usable = (event & (time <= horizon)) | (time > horizon)
    if usable.sum() < 2:
        return float("nan")
    y_label = (event[usable] & (time[usable] <= horizon)).astype(int)
    if len(np.unique(y_label)) < 2:
        return float("nan")
    try:
        return float(roc_auc_score(y_label, risk[usable]))
    except ValueError:
        return float("nan")


def survival_metrics(
    event: np.ndarray,
    time: np.ndarray,
    risk_scores: np.ndarray,
    *,
    horizons: tuple[int, ...] = (200, 300, 400),
    cohorts: np.ndarray | None = None,
) -> dict[str, float]:
    """Compute C-index + per-cohort C-index + time-dep AUC at each horizon.

    `risk_scores` MUST be risk-positive (higher = sooner failure). The
    validation pipeline normalises this from each model's predict() based
    on the model's `risk_orientation` class attribute.
    """
    out: dict[str, float] = {
        "n": float(len(event)),
        "n_events": float(np.asarray(event, dtype=bool).sum()),
        "c_index": _c_index(event, time, risk_scores),
    }
    for N in horizons:
        out[f"auc_at_{N}"] = _auc_at_horizon(event, time, risk_scores, N)

    if cohorts is not None:
        cohorts = np.asarray(cohorts)
        for c in np.unique(cohorts):
            mask = cohorts == c
            event_c = np.asarray(event, dtype=bool)[mask]
            if event_c.sum() < 5:
                out[f"c_index_{c}"] = float("nan")
                continue
            out[f"c_index_{c}"] = _c_index(
                event_c, np.asarray(time)[mask], np.asarray(risk_scores)[mask]
            )
    return out


def prefix(d: dict[str, float], prefix_: str) -> dict[str, float]:
    return {prefix_ + k: v for k, v in d.items()}

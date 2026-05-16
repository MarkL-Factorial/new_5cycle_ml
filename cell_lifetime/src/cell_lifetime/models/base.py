"""CycleLifeModel — subclass of cell_classifier.BaseModel with a `task` axis.

The classification interface of cell_classifier.BaseModel is preserved
(fit, predict, predict_proba, feature_importance, compute_shap).
Regression and survival subclasses opt into looser semantics:

  - task="regression": predict_proba is allowed to raise NotImplementedError.
    predict(X) returns continuous (n,) values on the untransformed scale.
  - task="survival":   predict(X) returns a scalar (risk score for RSF,
    log-cycle-life for XGB-AFT). predict_proba may also raise.

This module deliberately does not redefine the abstract methods; it
subclasses BaseModel and adds the `task` class attribute. Concrete
model files (xgb_*, ebm_*, rsf) provide their own implementations.
"""

from __future__ import annotations

from typing import ClassVar

from cell_classifier.models.base import BaseModel


class CycleLifeModel(BaseModel):
    """Marker subclass adding a `task` axis + `risk_orientation` for survival.

    Subclasses MUST set `task` to one of "classification", "regression",
    "survival". The pipeline reads this attribute to pick targets +
    metrics + masks.

    For `task == "survival"`, subclasses also declare `risk_orientation`:
      - "risk_high"  → predict(X) returns risk scores (higher = sooner failure).
                       Feed directly to concordance_index_censored.
      - "time_high"  → predict(X) returns predicted time / log-time
                       (higher = later failure). The pipeline negates before
                       computing C-index. XGB-AFT uses this.

    Regression/classification subclasses ignore `risk_orientation`.
    """

    task: ClassVar[str] = "regression"  # default; subclasses override
    risk_orientation: ClassVar[str] = "time_high"  # only meaningful for task='survival'

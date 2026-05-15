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
    """Marker subclass adding a `task` axis.

    Subclasses MUST set `task` to one of "classification", "regression",
    "survival". The pipeline reads this attribute to pick targets +
    metrics + masks.
    """

    task: ClassVar[str] = "regression"  # default; subclasses override

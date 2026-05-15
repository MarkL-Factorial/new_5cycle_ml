"""Model registry — `get_model_class(name)` → BaseModel subclass.

Mirrors cell_classifier.models.registry's guarded-import pattern.
xgboost / interpret / scikit-survival are all optional extras; if any
of them isn't installed the corresponding model just isn't registered.
"""

from __future__ import annotations

from cell_classifier.models.base import BaseModel


_REGISTRY: dict[str, type[BaseModel]] = {}


def register(name: str, cls: type[BaseModel]) -> None:
    _REGISTRY[name] = cls


def get_model_class(name: str) -> type[BaseModel]:
    if name not in _REGISTRY:
        raise KeyError(
            f"unknown model {name!r}; registered: {sorted(_REGISTRY)}. "
            f"For xgb_*/ebm_*/rsf, ensure the corresponding extra is installed "
            f"(e.g. `pip install -e .[xgb,ebm,survival]`)."
        )
    return _REGISTRY[name]


def registered_models() -> list[str]:
    return sorted(_REGISTRY)


# --- Optional registrations (guarded) ---

try:
    from cell_lifetime.models.xgb_classifier import XGBClassifierModel
    _REGISTRY["xgb_classifier"] = XGBClassifierModel
except ImportError:
    pass

try:
    from cell_lifetime.models.xgb_regressor import XGBRegressorModel
    _REGISTRY["xgb_regressor"] = XGBRegressorModel
except ImportError:
    pass

try:
    from cell_lifetime.models.ebm_regressor import EBMRegressorModel
    _REGISTRY["ebm_regressor"] = EBMRegressorModel
except ImportError:
    pass

# Phase 2: xgb_aft — registered when the file lands
try:
    from cell_lifetime.models.xgb_aft import XGBAFTModel  # noqa: F401
    _REGISTRY["xgb_aft"] = XGBAFTModel
except ImportError:
    pass

# Phase 3: rsf — registered when the file lands AND sksurv is installed
try:
    from cell_lifetime.models.rsf import RSFModel  # noqa: F401
    _REGISTRY["rsf"] = RSFModel
except ImportError:
    pass

"""Model class registry — the only module that maps `model_name` → class.

`get_model_class("random_forest")` returns the `BaseModel` subclass to
instantiate. Training / evaluation / inference / pipelines code calls this
function instead of branching on the model name themselves.

EBM and BART are registered conditionally: if their backend libraries
(`interpret`, `pymc-bart`) aren't installed in the current environment,
the import fails silently and only RandomForestModel is available.
"""

from __future__ import annotations

from cell_classifier.models.base import BaseModel
from cell_classifier.models.random_forest import RandomForestModel

_REGISTRY: dict[str, type[BaseModel]] = {
    "random_forest": RandomForestModel,
}


def register(name: str, cls: type[BaseModel]) -> None:
    _REGISTRY[name] = cls


def get_model_class(name: str) -> type[BaseModel]:
    if name not in _REGISTRY:
        raise KeyError(
            f"unknown model {name!r}; registered: {sorted(_REGISTRY)}. "
            f"For ebm/bart, ensure the corresponding extra is installed "
            f"(e.g. `pip install -e .[ebm]`)."
        )
    return _REGISTRY[name]


def registered_models() -> list[str]:
    return sorted(_REGISTRY)


# --- Optional registrations (guarded) ---
try:
    from cell_classifier.models.ebm import EBMModel
    _REGISTRY["ebm"] = EBMModel
except ImportError:
    pass

try:
    from cell_classifier.models.bart import BARTModel
    _REGISTRY["bart"] = BARTModel
except ImportError:
    pass

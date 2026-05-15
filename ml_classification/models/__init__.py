"""Model registry — lookup by name from config files.

Adding a new model: implement a `ModelSpec` subclass in a sibling module and add
it here. Pipeline code never references concrete model classes — only the registry.
"""

from .base import ModelSpec
from .random_forest import RFModelSpec
from .ebm import EBMModelSpec
from .bart import BARTModelSpec

MODEL_REGISTRY: dict[str, type[ModelSpec]] = {
    "random_forest": RFModelSpec,
    "ebm": EBMModelSpec,
    "bart": BARTModelSpec,
}


def get_model_spec(name: str) -> ModelSpec:
    if name not in MODEL_REGISTRY:
        raise KeyError(
            f"unknown model {name!r}. Available: {sorted(MODEL_REGISTRY)}"
        )
    return MODEL_REGISTRY[name]()


__all__ = ["ModelSpec", "MODEL_REGISTRY", "get_model_spec"]

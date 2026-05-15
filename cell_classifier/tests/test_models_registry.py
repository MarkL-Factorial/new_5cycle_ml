"""Registry exposes RF; EBM/BART register only when extras are installed."""

import importlib.util

import pytest

from cell_classifier.models.registry import get_model_class, registered_models


def test_random_forest_always_registered():
    assert "random_forest" in registered_models()
    cls = get_model_class("random_forest")
    assert cls.name == "random_forest"


def test_unknown_model_raises():
    with pytest.raises(KeyError, match="unknown model"):
        get_model_class("not_a_model")


def test_ebm_registered_iff_interpret_installed():
    has_interpret = importlib.util.find_spec("interpret") is not None
    assert ("ebm" in registered_models()) == has_interpret


def test_bart_registered_iff_pymc_bart_installed():
    has_pymc_bart = importlib.util.find_spec("pymc_bart") is not None
    assert ("bart" in registered_models()) == has_pymc_bart

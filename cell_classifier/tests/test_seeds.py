"""Named seed presets + resolve_seeds mutual exclusion."""

import pytest

from cell_classifier.utils.seeds import SEEDS_PRESETS, resolve_seeds


def test_fresh_preset_is_50_unique_ints():
    fresh = SEEDS_PRESETS["fresh"]
    assert len(fresh) == 50
    assert len(set(fresh)) == 50
    assert all(isinstance(s, int) and s > 0 for s in fresh)


def test_resolve_preset():
    seeds = resolve_seeds(preset="fresh", literal=None)
    assert seeds == SEEDS_PRESETS["fresh"]


def test_resolve_literal():
    assert resolve_seeds(preset=None, literal=[1, 2, 3]) == [1, 2, 3]


def test_resolve_mutually_exclusive():
    with pytest.raises(ValueError):
        resolve_seeds(preset=None, literal=None)
    with pytest.raises(ValueError):
        resolve_seeds(preset="fresh", literal=[1, 2])


def test_unknown_preset():
    with pytest.raises(KeyError):
        resolve_seeds(preset="ghost", literal=None)

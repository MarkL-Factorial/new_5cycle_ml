"""Named seed presets.

`SEEDS_PRESETS["fresh"]` is the canonical 50-seed list seeded by the date the
classifier project was kicked off. Other presets can be added without churning
the canonical one (which would invalidate any published numbers).
"""

from __future__ import annotations

import numpy as np

_FRESH_GENERATOR_SEED = 20260514
_N_FRESH = 50


def _generate(generator_seed: int, n: int) -> list[int]:
    rng = np.random.default_rng(generator_seed)
    return [int(s) for s in rng.integers(low=1, high=2**31 - 1, size=n)]


SEEDS_PRESETS: dict[str, list[int]] = {
    "fresh": _generate(_FRESH_GENERATOR_SEED, _N_FRESH),
}


def resolve_seeds(preset: str | None, literal: list[int] | None) -> list[int]:
    """Pick exactly one of preset (named) or literal (a list of ints).

    Caller (CLI) is responsible for ensuring the two flags are mutually
    exclusive at the argparse level; this function asserts the same.
    """
    if (preset is None) == (literal is None):
        raise ValueError(
            "exactly one of seeds-preset or seeds (literal list) must be set"
        )
    if preset is not None:
        if preset not in SEEDS_PRESETS:
            raise KeyError(
                f"unknown seeds preset {preset!r}; available: {list(SEEDS_PRESETS)}"
            )
        return list(SEEDS_PRESETS[preset])
    return list(literal)

"""Ensemble aggregation + per-seed long format."""

import numpy as np

from cell_classifier.inference.predict import (
    ensemble_predictions, per_seed_long, per_seed_posterior_long,
)


def test_ensemble_basic():
    cells = np.array(["a", "b", "c"])
    probas = [
        np.array([0.9, 0.4, 0.1]),
        np.array([0.8, 0.5, 0.2]),
    ]
    out = ensemble_predictions(cells, probas)
    assert list(out["cell_name"]) == ["a", "b", "c"]
    np.testing.assert_allclose(out["mean_proba_pass"], [0.85, 0.45, 0.15])
    # Class threshold = 0.5: a -> 1, b -> 0 (mean 0.45), c -> 0
    assert list(out["predicted_class"]) == [1, 0, 0]


def test_per_seed_long_shape():
    cells = np.array(["a", "b"])
    df = per_seed_long(cells, [1, 2], [np.array([0.9, 0.1]), np.array([0.8, 0.2])])
    assert len(df) == 4
    assert set(df.columns) == {"seed", "cell_name", "proba_pass"}


def test_posterior_long_shape():
    cells = np.array(["a", "b"])
    samples = np.zeros((10, 2, 2))
    samples[:, :, 1] = 0.7   # all draws say class-1 prob = 0.7
    df = per_seed_posterior_long(cells, [1], [samples])
    assert len(df) == 10 * 2
    assert (df["proba_pass"] == 0.7).all()

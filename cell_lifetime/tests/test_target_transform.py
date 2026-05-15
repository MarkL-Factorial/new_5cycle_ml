"""Target-transform fit/inverse round-trip tests."""

import numpy as np
import pytest

from cell_lifetime.preprocessing.target_transform import TargetTransform


@pytest.mark.parametrize("kind", ["none", "log", "sqrt", "boxcox"])
def test_round_trip(kind):
    rng = np.random.default_rng(0)
    y = rng.lognormal(mean=5.5, sigma=0.7, size=300).astype(float) + 1.0
    t = TargetTransform(kind=kind).fit(y)
    y_t = t.transform(y)
    y_back = t.inverse(y_t)
    np.testing.assert_allclose(y, y_back, rtol=1e-6, atol=1e-6)


def test_boxcox_lambda_in_range():
    rng = np.random.default_rng(0)
    y = rng.lognormal(mean=5.5, sigma=0.7, size=500)
    t = TargetTransform(kind="boxcox").fit(y)
    # For lognormal data, λ should be near 0 (close to log)
    assert -1.0 < t.lambda_ < 1.0, t.lambda_


def test_log_rejects_nonpositive():
    with pytest.raises(ValueError):
        TargetTransform(kind="log").fit(np.array([1.0, 2.0, 0.0]))


def test_transform_before_fit_raises():
    with pytest.raises(RuntimeError):
        TargetTransform(kind="log").transform(np.array([1.0]))

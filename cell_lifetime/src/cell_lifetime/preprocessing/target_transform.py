"""Target transformations for the cycle-life regression target.

`last_fade_cycle` is right-skewed (range 5–1052, median ~310 on A2.2).
Models fit on the transformed scale; metrics are reported on the
untransformed cycle-count scale. The transform stores its fit
parameters (Box-Cox λ, offset, etc.) so `inverse(transform(y)) == y`
within float tolerance.

Supported kinds:
  - "none":   identity
  - "log":    natural log (cycle lives are strictly positive, no offset needed)
  - "sqrt":   sqrt
  - "boxcox": scipy.stats.boxcox, λ found on .fit(); ML estimate
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

import numpy as np
from scipy.stats import boxcox, boxcox_normmax


Kind = Literal["none", "log", "sqrt", "boxcox"]


@dataclass
class TargetTransform:
    kind: Kind = "boxcox"
    lambda_: float | None = field(default=None, init=False)
    fitted_: bool = field(default=False, init=False)

    def fit(self, y: np.ndarray) -> "TargetTransform":
        y = np.asarray(y, dtype=float)
        if y.size == 0:
            raise ValueError("fit on empty array")
        if (y <= 0).any() and self.kind in ("log", "sqrt", "boxcox"):
            raise ValueError(
                f"transform {self.kind!r} requires strictly positive targets; "
                f"got min={y.min():.4g}"
            )
        if self.kind == "boxcox":
            # Find λ by maximum-likelihood; clamp to a reasonable range
            self.lambda_ = float(boxcox_normmax(y, brack=(-2.0, 2.0), method="mle"))
        else:
            self.lambda_ = None
        self.fitted_ = True
        return self

    def transform(self, y: np.ndarray) -> np.ndarray:
        if not self.fitted_:
            raise RuntimeError("call .fit() before .transform()")
        y = np.asarray(y, dtype=float)
        if self.kind == "none":
            return y.copy()
        if self.kind == "log":
            return np.log(y)
        if self.kind == "sqrt":
            return np.sqrt(y)
        if self.kind == "boxcox":
            assert self.lambda_ is not None
            return _boxcox_apply(y, self.lambda_)
        raise ValueError(f"unknown kind {self.kind!r}")

    def inverse(self, y_t: np.ndarray) -> np.ndarray:
        if not self.fitted_:
            raise RuntimeError("call .fit() before .inverse()")
        y_t = np.asarray(y_t, dtype=float)
        if self.kind == "none":
            return y_t.copy()
        if self.kind == "log":
            return np.exp(y_t)
        if self.kind == "sqrt":
            return np.square(y_t)
        if self.kind == "boxcox":
            assert self.lambda_ is not None
            return _boxcox_invert(y_t, self.lambda_)
        raise ValueError(f"unknown kind {self.kind!r}")

    def fit_transform(self, y: np.ndarray) -> np.ndarray:
        return self.fit(y).transform(y)


def _boxcox_apply(y: np.ndarray, lam: float) -> np.ndarray:
    if abs(lam) < 1e-12:
        return np.log(y)
    return (np.power(y, lam) - 1.0) / lam


def _boxcox_invert(y_t: np.ndarray, lam: float) -> np.ndarray:
    """Numerically-safe Box-Cox inverse.

    The naive formula `(lam*y_t + 1)**(1/lam)` produces NaN whenever
    `lam*y_t + 1 < 0` (fractional power of a negative number). This
    happens when an aggressive model predicts a transformed-y below
    `-1/lam` (≈ -1.87 for lam=0.534). We clip the inside of the power
    to a small positive epsilon so the inverse becomes a tiny-but-finite
    cycle life rather than NaN — which lets metrics (MAE/RMSE) keep
    working without dropping the trial.
    """
    if abs(lam) < 1e-12:
        return np.exp(np.clip(y_t, -50.0, 50.0))
    inside = lam * y_t + 1.0
    inside = np.maximum(inside, 1e-12)
    return np.power(inside, 1.0 / lam)

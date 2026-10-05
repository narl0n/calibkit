"""Post-hoc recalibration: temperature scaling for binary classifiers."""

from __future__ import annotations

import numpy as np
from scipy.optimize import minimize_scalar
from scipy.special import expit, logit

__all__ = ["TemperatureScaler"]

_EPS = 1e-12


def _to_logits(y_prob) -> np.ndarray:
    return logit(np.clip(np.asarray(y_prob, dtype=float), _EPS, 1 - _EPS))


class TemperatureScaler:
    """Rescale logits by a single learned temperature T > 0.

    Calibrated probability = sigmoid(logit(p) / T). T > 1 softens over-confident
    models, T < 1 sharpens under-confident ones. T is fitted by minimising the
    negative log-likelihood on a held-out validation set.
    """

    def __init__(self) -> None:
        self.temperature_: float | None = None

    def fit(self, y_true, y_prob) -> TemperatureScaler:
        y = np.asarray(y_true, dtype=float)
        z = _to_logits(y_prob)
        if y.shape != z.shape or y.ndim != 1 or y.size == 0:
            raise ValueError("y_true and y_prob must be non-empty 1-D arrays of equal length")
        if not np.isin(y, (0.0, 1.0)).all():
            raise ValueError("y_true must contain only 0 and 1")

        def nll(log_t: float) -> float:
            p = np.clip(expit(z / np.exp(log_t)), _EPS, 1 - _EPS)
            return float(-np.mean(y * np.log(p) + (1 - y) * np.log(1 - p)))

        # optimise log T so that T stays positive
        res = minimize_scalar(nll, bounds=(-4.0, 4.0), method="bounded")
        self.temperature_ = float(np.exp(res.x))
        return self

    def transform(self, y_prob) -> np.ndarray:
        if self.temperature_ is None:
            raise RuntimeError("TemperatureScaler is not fitted; call fit() first")
        return expit(_to_logits(y_prob) / self.temperature_)

    def fit_transform(self, y_true, y_prob) -> np.ndarray:
        return self.fit(y_true, y_prob).transform(y_prob)

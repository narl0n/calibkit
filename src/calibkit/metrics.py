"""Calibration metrics for binary classifiers.

A classifier is *calibrated* if, among all cases where it predicts probability p,
the observed fraction of positives is close to p. These functions quantify the gap.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

__all__ = ["ReliabilityBins", "brier_score", "expected_calibration_error",
           "max_calibration_error", "reliability_bins"]


@dataclass(frozen=True)
class ReliabilityBins:
    """Per-bin statistics used for ECE/MCE and reliability diagrams."""

    edges: np.ndarray        # shape (n_bins + 1,)
    counts: np.ndarray       # samples per bin
    mean_confidence: np.ndarray  # mean predicted probability per bin (nan if empty)
    accuracy: np.ndarray     # observed fraction of positives per bin (nan if empty)


def _validate(y_true, y_prob) -> tuple[np.ndarray, np.ndarray]:
    y_true = np.asarray(y_true)
    y_prob = np.asarray(y_prob, dtype=float)
    if y_true.ndim != 1 or y_prob.ndim != 1:
        raise ValueError("y_true and y_prob must be one-dimensional")
    if y_true.shape != y_prob.shape:
        raise ValueError("y_true and y_prob must have the same length")
    if y_true.size == 0:
        raise ValueError("inputs must not be empty")
    if not np.isin(y_true, (0, 1)).all():
        raise ValueError("y_true must contain only 0 and 1")
    if np.isnan(y_prob).any() or (y_prob < 0).any() or (y_prob > 1).any():
        raise ValueError("y_prob must lie in [0, 1] and contain no NaN")
    return y_true.astype(float), y_prob


def reliability_bins(y_true, y_prob, n_bins: int = 10) -> ReliabilityBins:
    """Split predictions into equal-width probability bins and summarise each."""
    if n_bins < 1:
        raise ValueError("n_bins must be >= 1")
    y_true, y_prob = _validate(y_true, y_prob)
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    # digitize against inner edges; p == 1.0 falls into the last bin
    idx = np.clip(np.digitize(y_prob, edges[1:-1], right=False), 0, n_bins - 1)
    counts = np.bincount(idx, minlength=n_bins)
    conf_sum = np.bincount(idx, weights=y_prob, minlength=n_bins)
    pos_sum = np.bincount(idx, weights=y_true, minlength=n_bins)
    with np.errstate(invalid="ignore", divide="ignore"):
        mean_conf = np.where(counts > 0, conf_sum / counts, np.nan)
        acc = np.where(counts > 0, pos_sum / counts, np.nan)
    return ReliabilityBins(edges, counts, mean_conf, acc)


def expected_calibration_error(y_true, y_prob, n_bins: int = 10) -> float:
    """Expected Calibration Error: bin-size-weighted mean |accuracy - confidence|."""
    bins = reliability_bins(y_true, y_prob, n_bins)
    mask = bins.counts > 0
    gaps = np.abs(bins.accuracy[mask] - bins.mean_confidence[mask])
    return float(np.sum(bins.counts[mask] / bins.counts.sum() * gaps))


def max_calibration_error(y_true, y_prob, n_bins: int = 10) -> float:
    """Maximum Calibration Error: the worst per-bin |accuracy - confidence| gap."""
    bins = reliability_bins(y_true, y_prob, n_bins)
    mask = bins.counts > 0
    return float(np.max(np.abs(bins.accuracy[mask] - bins.mean_confidence[mask])))


def brier_score(y_true, y_prob) -> float:
    """Brier score: mean squared error between probabilities and outcomes (lower is better)."""
    y_true, y_prob = _validate(y_true, y_prob)
    return float(np.mean((y_prob - y_true) ** 2))

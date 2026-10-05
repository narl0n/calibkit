"""Reliability diagram (requires the optional ``matplotlib`` dependency)."""

from __future__ import annotations

from .metrics import expected_calibration_error, reliability_bins

__all__ = ["plot_reliability_diagram"]


def plot_reliability_diagram(y_true, y_prob, n_bins: int = 10, ax=None, label: str = "model"):
    """Draw accuracy-vs-confidence per bin against the perfect-calibration diagonal."""
    try:
        import matplotlib.pyplot as plt
    except ImportError as exc:  # pragma: no cover - exercised only without matplotlib
        raise ImportError("Install plotting support: pip install 'calibkit[plot]'") from exc

    bins = reliability_bins(y_true, y_prob, n_bins)
    ece = expected_calibration_error(y_true, y_prob, n_bins)
    if ax is None:
        _, ax = plt.subplots(figsize=(4.5, 4.5))
    mask = bins.counts > 0
    ax.plot([0, 1], [0, 1], "--", color="gray", label="perfect calibration")
    ax.plot(bins.mean_confidence[mask], bins.accuracy[mask], "o-", label=f"{label} (ECE={ece:.3f})")
    ax.set_xlabel("Mean predicted probability")
    ax.set_ylabel("Observed fraction of positives")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.legend(loc="upper left")
    ax.set_title("Reliability diagram")
    return ax

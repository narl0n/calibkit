"""Demo: an over-confident synthetic classifier, before and after temperature scaling.

Run from the repository root:  python examples/demo.py
Saves examples/reliability.png.
"""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.special import expit

from calibkit import TemperatureScaler, brier_score, expected_calibration_error
from calibkit.plotting import plot_reliability_diagram

rng = np.random.default_rng(42)
z = rng.normal(0, 1.5, 5000)
y_true = (rng.random(5000) < expit(z)).astype(int)
y_prob = expit(3 * z)  # model is over-confident by a factor of ~3

scaler = TemperatureScaler().fit(y_true, y_prob)
y_cal = scaler.transform(y_prob)

print(f"Fitted temperature: {scaler.temperature_:.2f}")
print(f"ECE   before {expected_calibration_error(y_true, y_prob):.4f} "
      f"-> after {expected_calibration_error(y_true, y_cal):.4f}")
print(f"Brier before {brier_score(y_true, y_prob):.4f} -> after {brier_score(y_true, y_cal):.4f}")

fig, axes = plt.subplots(1, 2, figsize=(9, 4.5), sharey=True)
plot_reliability_diagram(y_true, y_prob, ax=axes[0], label="raw")
axes[0].set_title("Before temperature scaling")
plot_reliability_diagram(y_true, y_cal, ax=axes[1], label="calibrated")
axes[1].set_title("After temperature scaling")
fig.tight_layout()
out = Path(__file__).with_name("reliability.png")
fig.savefig(out, dpi=150)
print(f"Saved {out}")

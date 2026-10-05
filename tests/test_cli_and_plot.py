import matplotlib

matplotlib.use("Agg")

import numpy as np
import pytest
from scipy.special import expit

from calibkit.cli import main
from calibkit.plotting import plot_reliability_diagram


@pytest.fixture
def predictions_csv(tmp_path):
    rng = np.random.default_rng(3)
    z = rng.normal(0, 1.5, 2000)
    y = (rng.random(2000) < expit(z)).astype(int)
    p = expit(3 * z)
    path = tmp_path / "preds.csv"
    rows = "\n".join(f"{a},{b:.6f}" for a, b in zip(y, p, strict=True))
    path.write_text("y_true,y_prob\n" + rows)
    return path


def test_cli_report_prints_metrics(predictions_csv, capsys):
    assert main(["report", str(predictions_csv), "--bins", "10"]) == 0
    out = capsys.readouterr().out
    assert "samples: 2000" in out
    assert "temperature:" in out
    assert "ECE" in out and "Brier" in out


def test_cli_rejects_missing_columns(tmp_path):
    bad = tmp_path / "bad.csv"
    bad.write_text("a,b\n1,2\n")
    with pytest.raises(SystemExit):
        main(["report", str(bad)])


def test_reliability_diagram_returns_axes():
    ax = plot_reliability_diagram([0, 1, 1, 0, 1], [0.1, 0.8, 0.7, 0.3, 0.9], n_bins=5)
    assert ax.get_xlabel() == "Mean predicted probability"
    assert len(ax.lines) == 2  # diagonal + model curve

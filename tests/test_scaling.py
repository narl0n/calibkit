import numpy as np
import pytest
from scipy.special import expit, logit

from calibkit import TemperatureScaler, expected_calibration_error


def _overconfident_data(n=20000, temperature=3.0, seed=0):
    """True logits z; the model reports sigmoid(temperature * z), i.e. is over-confident."""
    rng = np.random.default_rng(seed)
    z = rng.normal(0, 1.5, n)
    y = (rng.random(n) < expit(z)).astype(int)
    return y, expit(temperature * z)


def test_recovers_known_temperature():
    y, p = _overconfident_data(temperature=3.0)
    scaler = TemperatureScaler().fit(y, p)
    assert scaler.temperature_ == pytest.approx(3.0, rel=0.1)


def test_scaling_reduces_ece():
    y, p = _overconfident_data()
    before = expected_calibration_error(y, p)
    after = expected_calibration_error(y, TemperatureScaler().fit_transform(y, p))
    assert after < before / 3


def test_temperature_one_for_calibrated_model():
    rng = np.random.default_rng(2)
    p = np.clip(rng.random(20000), 0.01, 0.99)
    y = (rng.random(20000) < p).astype(int)
    assert TemperatureScaler().fit(y, p).temperature_ == pytest.approx(1.0, abs=0.1)


def test_scaling_preserves_ranking():
    y, p = _overconfident_data(n=2000)
    q = TemperatureScaler().fit_transform(y, p)
    assert np.array_equal(np.argsort(p, kind="stable"), np.argsort(q, kind="stable"))


def test_handles_extreme_probabilities():
    q = TemperatureScaler().fit([1, 0, 1, 0], [1.0, 0.0, 0.9, 0.2]).transform([0.0, 1.0])
    assert np.isfinite(q).all()
    assert logit(np.clip(q, 1e-9, 1 - 1e-9)).shape == (2,)


def test_transform_before_fit_raises():
    with pytest.raises(RuntimeError):
        TemperatureScaler().transform([0.5])


def test_fit_rejects_bad_input():
    with pytest.raises(ValueError):
        TemperatureScaler().fit([1, 2], [0.5, 0.5])
    with pytest.raises(ValueError):
        TemperatureScaler().fit([1, 0], [0.5])

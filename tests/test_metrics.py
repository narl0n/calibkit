import numpy as np
import pytest

from calibkit import (
    brier_score,
    expected_calibration_error,
    max_calibration_error,
    reliability_bins,
)


def test_perfectly_calibrated_has_zero_ece():
    # in each bin the observed rate equals the stated probability
    y_prob = np.array([0.0] * 10 + [1.0] * 10)
    y_true = np.array([0] * 10 + [1] * 10)
    assert expected_calibration_error(y_true, y_prob) == pytest.approx(0.0)
    assert max_calibration_error(y_true, y_prob) == pytest.approx(0.0)


def test_overconfident_model_has_known_ece():
    # always says 0.9 but is right only half of the time -> |0.5 - 0.9| = 0.4
    y_prob = np.full(100, 0.9)
    y_true = np.array([1, 0] * 50)
    assert expected_calibration_error(y_true, y_prob) == pytest.approx(0.4)
    assert max_calibration_error(y_true, y_prob) == pytest.approx(0.4)


def test_brier_score_known_values():
    assert brier_score([1, 0], [1.0, 0.0]) == pytest.approx(0.0)
    assert brier_score([1, 0], [0.0, 1.0]) == pytest.approx(1.0)
    assert brier_score([1, 0], [0.5, 0.5]) == pytest.approx(0.25)


def test_probability_one_goes_into_last_bin():
    bins = reliability_bins([1, 1], [1.0, 1.0], n_bins=5)
    assert bins.counts.tolist() == [0, 0, 0, 0, 2]
    assert np.isnan(bins.accuracy[0])


def test_bin_counts_sum_to_n():
    rng = np.random.default_rng(0)
    p = rng.random(500)
    y = (rng.random(500) < p).astype(int)
    assert reliability_bins(y, p, n_bins=15).counts.sum() == 500


def test_well_calibrated_simulation_has_small_ece():
    rng = np.random.default_rng(1)
    p = rng.random(20000)
    y = (rng.random(20000) < p).astype(int)
    assert expected_calibration_error(y, p) < 0.02


@pytest.mark.parametrize(
    "y_true, y_prob",
    [
        ([1, 0], [0.5]),            # length mismatch
        ([1, 2], [0.5, 0.5]),       # non-binary label
        ([1, 0], [1.2, 0.1]),       # probability out of range
        ([1, 0], [np.nan, 0.1]),    # NaN
        ([], []),                   # empty
        ([[1, 0]], [[0.5, 0.5]]),   # not 1-D
    ],
)
def test_invalid_inputs_raise(y_true, y_prob):
    with pytest.raises(ValueError):
        brier_score(y_true, y_prob)


def test_invalid_bin_count_raises():
    with pytest.raises(ValueError):
        reliability_bins([1, 0], [0.5, 0.5], n_bins=0)

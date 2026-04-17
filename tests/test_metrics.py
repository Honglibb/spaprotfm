import numpy as np
import pytest

from spaprotfm.eval.metrics import (
    pearson_correlation,
    mean_squared_error,
    structural_similarity,
    frobenius_distance,
)


def test_pearson_perfect_correlation():
    x = np.linspace(0, 1, 100)
    y = 2 * x + 0.5
    assert pearson_correlation(x, y) == pytest.approx(1.0, abs=1e-6)


def test_pearson_anticorrelation():
    x = np.linspace(0, 1, 100)
    assert pearson_correlation(x, -x) == pytest.approx(-1.0, abs=1e-6)


def test_pearson_handles_constant_returns_zero():
    x = np.zeros(100)
    y = np.linspace(0, 1, 100)
    assert pearson_correlation(x, y) == 0.0


def test_mse_zero_when_identical(rng):
    x = rng.uniform(0, 1, 50).astype(np.float32)
    assert mean_squared_error(x, x) == 0.0


def test_mse_known_value():
    x = np.array([1.0, 2.0])
    y = np.array([2.0, 4.0])
    assert mean_squared_error(x, y) == pytest.approx((1 + 4) / 2)


def test_ssim_self_is_one(rng):
    img = rng.uniform(0, 1, (64, 64)).astype(np.float32)
    assert structural_similarity(img, img) == pytest.approx(1.0, abs=1e-4)


def test_ssim_decreases_with_noise(rng):
    img = rng.uniform(0, 1, (64, 64)).astype(np.float32)
    noisy = img + rng.normal(0, 0.5, img.shape)
    assert structural_similarity(img, noisy) < 0.9


def test_frobenius_distance_matches_norm():
    A = np.array([[1.0, 2.0], [3.0, 4.0]])
    B = np.zeros_like(A)
    expected = np.sqrt(1 + 4 + 9 + 16)
    assert frobenius_distance(A, B) == pytest.approx(expected)


def test_pearson_per_channel_returns_one_per_channel(rng):
    pred = rng.uniform(0, 1, (10, 10, 3)).astype(np.float32)
    target = pred.copy()
    target[..., 0] += rng.normal(0, 0.01, (10, 10))
    out = pearson_correlation(pred, target, per_channel=True)
    assert out.shape == (3,)
    assert out[1] == pytest.approx(1.0, abs=1e-4)
    assert out[2] == pytest.approx(1.0, abs=1e-4)

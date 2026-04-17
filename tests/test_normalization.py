import numpy as np
import pytest

from spaprotfm.data.normalization import (
    arcsinh_transform,
    percentile_clip,
    normalize_image,
)


def test_arcsinh_transform_with_default_cofactor():
    x = np.array([0.0, 5.0, 100.0])
    out = arcsinh_transform(x, cofactor=5.0)
    assert out[0] == 0.0
    assert np.isclose(out[1], np.arcsinh(1.0))
    assert np.isclose(out[2], np.arcsinh(20.0))


def test_arcsinh_handles_zeros_and_negatives():
    x = np.array([-1.0, 0.0, 1.0])
    out = arcsinh_transform(x, cofactor=1.0)
    assert out[1] == 0.0
    assert out[0] == -out[2]  # arcsinh is odd


def test_percentile_clip_bounds_values():
    x = np.array([1, 2, 3, 4, 100], dtype=np.float32)
    out = percentile_clip(x, lo=0, hi=80)
    assert out.min() == pytest.approx(1.0)
    assert out.max() == pytest.approx(np.percentile(x, 80))


def test_normalize_image_per_channel(rng):
    img = rng.uniform(0, 1000, size=(32, 32, 3)).astype(np.float32)
    out = normalize_image(img, cofactor=5.0, percentile=(0, 99))
    assert out.shape == img.shape
    assert out.dtype == np.float32
    assert out.min() >= 0
    # Each channel scaled to roughly [0, 1]
    for c in range(3):
        assert out[..., c].max() <= 1.0 + 1e-6


def test_normalize_image_constant_channel_does_not_nan():
    img = np.zeros((8, 8, 2), dtype=np.float32)
    img[..., 1] = 5.0
    out = normalize_image(img)
    assert not np.isnan(out).any()

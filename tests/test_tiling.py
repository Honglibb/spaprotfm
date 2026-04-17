import numpy as np
import pytest

from spaprotfm.data.tiling import tile_image, untile_image


def test_tile_image_no_overlap_exact_division(rng):
    img = rng.uniform(0, 1, size=(256, 512, 3)).astype(np.float32)
    tiles, coords = tile_image(img, patch_size=256, stride=256)
    assert tiles.shape == (2, 256, 256, 3)
    assert coords == [(0, 0), (0, 256)]


def test_tile_image_with_overlap(rng):
    img = rng.uniform(0, 1, size=(256, 384, 1)).astype(np.float32)
    tiles, coords = tile_image(img, patch_size=256, stride=128)
    # Two tiles at x=0 and x=128 (x=256 would be edge-aligned)
    assert tiles.shape[0] == 2
    assert (0, 0) in coords
    assert (0, 128) in coords


def test_tile_image_pads_when_dimensions_not_divisible(rng):
    img = rng.uniform(0, 1, size=(300, 300, 2)).astype(np.float32)
    tiles, coords = tile_image(img, patch_size=256, stride=256, pad=True)
    # 2x2 grid: (0,0), (0,256), (256,0), (256,256) — last ones padded
    assert tiles.shape == (4, 256, 256, 2)


def test_untile_image_reverses_no_overlap(rng):
    img = rng.uniform(0, 1, size=(256, 512, 3)).astype(np.float32)
    tiles, coords = tile_image(img, patch_size=256, stride=256)
    out = untile_image(tiles, coords, output_shape=(256, 512, 3))
    np.testing.assert_allclose(out, img)


def test_untile_image_averages_overlap(rng):
    img = rng.uniform(0, 1, size=(256, 384, 1)).astype(np.float32)
    tiles, coords = tile_image(img, patch_size=256, stride=128)
    out = untile_image(tiles, coords, output_shape=img.shape)
    np.testing.assert_allclose(out, img, atol=1e-5)

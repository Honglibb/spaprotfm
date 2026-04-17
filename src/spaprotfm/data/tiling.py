"""Tile (H, W, C) images into fixed-size patches and stitch back."""

from __future__ import annotations

import numpy as np


def tile_image(
    img: np.ndarray,
    patch_size: int = 256,
    stride: int | None = None,
    pad: bool = True,
) -> tuple[np.ndarray, list[tuple[int, int]]]:
    """Split image into patches.

    Args:
        img: (H, W, C) array.
        patch_size: side of square patches.
        stride: step between patches. Defaults to patch_size (no overlap).
        pad: if True, zero-pad so all patches are full-size.

    Returns:
        tiles: (N, patch_size, patch_size, C) array.
        coords: list of (y, x) top-left corners.
    """
    if stride is None:
        stride = patch_size
    H, W, C = img.shape
    if pad:
        pad_h = (-(H - patch_size) % stride) if H >= patch_size else (patch_size - H)
        pad_w = (-(W - patch_size) % stride) if W >= patch_size else (patch_size - W)
        img = np.pad(img, ((0, pad_h), (0, pad_w), (0, 0)))
        H, W, C = img.shape

    coords: list[tuple[int, int]] = []
    tiles: list[np.ndarray] = []
    for y in range(0, H - patch_size + 1, stride):
        for x in range(0, W - patch_size + 1, stride):
            tiles.append(img[y : y + patch_size, x : x + patch_size, :])
            coords.append((y, x))

    return np.stack(tiles, axis=0), coords


def untile_image(
    tiles: np.ndarray,
    coords: list[tuple[int, int]],
    output_shape: tuple[int, int, int],
) -> np.ndarray:
    """Stitch patches back into an image, averaging overlapping regions."""
    H, W, C = output_shape
    out = np.zeros((H, W, C), dtype=np.float32)
    cnt = np.zeros((H, W, 1), dtype=np.float32)
    p = tiles.shape[1]
    for tile, (y, x) in zip(tiles, coords):
        h = min(p, H - y)
        w = min(p, W - x)
        out[y : y + h, x : x + w, :] += tile[:h, :w, :]
        cnt[y : y + h, x : x + w, :] += 1.0
    cnt[cnt == 0] = 1.0
    return out / cnt

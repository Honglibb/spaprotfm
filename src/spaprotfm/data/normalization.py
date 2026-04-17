"""Image normalization for multiplex tissue images."""

from __future__ import annotations

import numpy as np

ArrayLike = np.ndarray


def arcsinh_transform(x: ArrayLike, cofactor: float = 5.0) -> np.ndarray:
    """Apply arcsinh(x / cofactor); standard CyTOF/CODEX preprocessing."""
    return np.arcsinh(np.asarray(x, dtype=np.float32) / cofactor)


def percentile_clip(x: ArrayLike, lo: float = 0.0, hi: float = 99.0) -> np.ndarray:
    """Clip values to [lo-percentile, hi-percentile] of x."""
    arr = np.asarray(x, dtype=np.float32)
    lo_v = np.percentile(arr, lo) if lo > 0 else arr.min()
    hi_v = np.percentile(arr, hi)
    return np.clip(arr, lo_v, hi_v)


def normalize_image(
    img: ArrayLike,
    cofactor: float = 5.0,
    percentile: tuple[float, float] = (0.0, 99.0),
) -> np.ndarray:
    """Per-channel arcsinh + percentile clipping + min-max scaling to [0, 1].

    Args:
        img: (H, W, C) array.
        cofactor: arcsinh cofactor.
        percentile: (lo, hi) percentiles for clipping.

    Returns:
        Float32 array in [0, 1] with same shape as input.
    """
    arr = np.asarray(img, dtype=np.float32)
    if arr.ndim != 3:
        raise ValueError(f"Expected (H, W, C) image, got shape {arr.shape}")

    out = np.empty_like(arr)
    for c in range(arr.shape[-1]):
        ch = arcsinh_transform(arr[..., c], cofactor=cofactor)
        ch = percentile_clip(ch, lo=percentile[0], hi=percentile[1])
        ch_min, ch_max = ch.min(), ch.max()
        if ch_max - ch_min < 1e-8:
            out[..., c] = 0.0
        else:
            out[..., c] = (ch - ch_min) / (ch_max - ch_min)
    return out

"""Evaluation metrics for spatial proteomics imputation."""

from __future__ import annotations

import numpy as np
from skimage.metrics import structural_similarity as _ssim


def pearson_correlation(
    x: np.ndarray, y: np.ndarray, per_channel: bool = False
) -> float | np.ndarray:
    """PCC between flattened arrays, or per-channel for (H, W, C) inputs.

    Returns 0.0 (or zeros) if either input has zero variance.
    """
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)

    if per_channel:
        if x.ndim != 3 or y.ndim != 3:
            raise ValueError("per_channel=True requires (H, W, C) inputs")
        C = x.shape[-1]
        return np.array(
            [pearson_correlation(x[..., c], y[..., c]) for c in range(C)]
        )

    xf, yf = x.ravel(), y.ravel()
    if xf.std() < 1e-12 or yf.std() < 1e-12:
        return 0.0
    return float(np.corrcoef(xf, yf)[0, 1])


def mean_squared_error(x: np.ndarray, y: np.ndarray) -> float:
    return float(np.mean((np.asarray(x, np.float64) - np.asarray(y, np.float64)) ** 2))


def structural_similarity(x: np.ndarray, y: np.ndarray) -> float:
    """SSIM for 2D or 3D (per-channel averaged) arrays in [0, 1]."""
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    data_range = max(x.max(), y.max()) - min(x.min(), y.min()) or 1.0
    if x.ndim == 2:
        return float(_ssim(x, y, data_range=data_range))
    if x.ndim == 3:
        return float(
            _ssim(x, y, data_range=data_range, channel_axis=-1)
        )
    raise ValueError(f"Expected 2D or 3D array, got {x.ndim}D")


def frobenius_distance(A: np.ndarray, B: np.ndarray) -> float:
    return float(np.linalg.norm(np.asarray(A, np.float64) - np.asarray(B, np.float64)))

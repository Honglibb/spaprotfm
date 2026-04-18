"""Synthesize pseudo-H&E RGB from IMC tiles.

Hematoxylin (purple-blue, nucleus) channel is derived from the DNA channels
(e.g. DNA1/DNA2 in Bodenmiller IMC); eosin (pink, cytoplasm/stroma) from the
mean of remaining biological channels. We approximate Beer-Lambert absorbance
inversion with simple linear mixing, clip to [0,1], and resize to 224x224 so
the output is directly consumable by a ViT-B encoder such as Phikon-v2.
"""

from __future__ import annotations

from typing import Iterable

import numpy as np
import torch
import torch.nn.functional as F

DEFAULT_OUTPUT_SIZE = 224


def _as_tensor(tile: np.ndarray | torch.Tensor) -> torch.Tensor:
    if isinstance(tile, np.ndarray):
        return torch.from_numpy(tile.astype(np.float32))
    return tile.float()


def _percentile_scale(x: torch.Tensor, low: float = 1.0, high: float = 99.0) -> torch.Tensor:
    lo = torch.quantile(x, low / 100.0)
    hi = torch.quantile(x, high / 100.0)
    if float(hi - lo) < 1e-8:
        return torch.zeros_like(x)
    return torch.clamp((x - lo) / (hi - lo), 0.0, 1.0)


def synthesize_pseudo_he(
    tile: np.ndarray | torch.Tensor,
    dna_idx: Iterable[int],
    bio_idx: Iterable[int] | None = None,
    output_size: int = DEFAULT_OUTPUT_SIZE,
) -> torch.Tensor:
    """Make a pseudo-H&E RGB tile from an IMC tile.

    Args:
        tile: (C, H, W) or (H, W, C). Values assumed in [0, 1] (per-channel
            normalized) but robust to arbitrary non-negative ranges via
            percentile scaling.
        dna_idx: channel indices that will form the hematoxylin (nuclear) stain.
            Must be non-empty.
        bio_idx: channel indices used for the eosin (cytoplasm) stain. If None,
            uses all channels not in dna_idx.
        output_size: spatial side of the returned RGB. Default 224 for ViT-B.

    Returns:
        RGB tensor of shape (3, output_size, output_size) with float32 values
        in [0, 1].
    """
    t = _as_tensor(tile)
    if t.ndim != 3:
        raise ValueError(f"expected 3D tile, got shape {tuple(t.shape)}")
    if t.shape[0] < t.shape[-1] and t.shape[0] <= 128:
        chw = t
    else:
        chw = t.permute(2, 0, 1).contiguous()

    C = chw.shape[0]
    dna_list = [int(i) for i in dna_idx]
    if not dna_list:
        raise ValueError("dna_idx must be non-empty")
    if any(i < 0 or i >= C for i in dna_list):
        raise ValueError(f"dna_idx out of range for C={C}: {dna_list}")

    if bio_idx is None:
        bio_list = [i for i in range(C) if i not in set(dna_list)]
    else:
        bio_list = [int(i) for i in bio_idx]
        if any(i < 0 or i >= C for i in bio_list):
            raise ValueError(f"bio_idx out of range for C={C}: {bio_list}")

    H = _percentile_scale(chw[dna_list].amax(dim=0))
    if bio_list:
        E = _percentile_scale(chw[bio_list].mean(dim=0))
    else:
        E = torch.zeros_like(H)

    # Beer-Lambert style inversion: more stain -> less light transmitted.
    # Coefficients chosen so that pure hematoxylin ≈ purple-blue and pure
    # eosin ≈ pink; mixed region becomes magenta-grey.
    r = 1.0 - 0.5 * H - 0.3 * E
    g = 1.0 - 0.8 * H - 0.2 * E
    b = 1.0 - 0.7 * H - 0.05 * E
    rgb = torch.stack([r, g, b], dim=0).clamp(0.0, 1.0)

    rgb = F.interpolate(
        rgb.unsqueeze(0),
        size=(output_size, output_size),
        mode="bilinear",
        align_corners=False,
    ).squeeze(0)
    return rgb.contiguous()


def synthesize_pseudo_he_batch(
    tiles: np.ndarray | torch.Tensor,
    dna_idx: Iterable[int],
    bio_idx: Iterable[int] | None = None,
    output_size: int = DEFAULT_OUTPUT_SIZE,
) -> torch.Tensor:
    """Apply synthesize_pseudo_he over a batch.

    Args:
        tiles: (B, C, H, W) tensor or array.

    Returns:
        (B, 3, output_size, output_size) float32 tensor.
    """
    t = _as_tensor(tiles)
    if t.ndim != 4:
        raise ValueError(f"expected 4D batch, got shape {tuple(t.shape)}")
    out = []
    for b in range(t.shape[0]):
        out.append(synthesize_pseudo_he(t[b], dna_idx, bio_idx, output_size))
    return torch.stack(out, dim=0)

"""Tests for pseudo-H&E synthesis."""

from __future__ import annotations

import numpy as np
import pytest
import torch

from spaprotfm.condition.pseudo_he import (
    synthesize_pseudo_he,
    synthesize_pseudo_he_batch,
)


def _random_tile(C: int = 8, H: int = 128, W: int = 128, seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return rng.random(size=(C, H, W), dtype=np.float32)


def test_output_shape_and_range():
    tile = _random_tile()
    rgb = synthesize_pseudo_he(tile, dna_idx=[0, 1])
    assert rgb.shape == (3, 224, 224)
    assert rgb.dtype == torch.float32
    assert float(rgb.min()) >= 0.0
    assert float(rgb.max()) <= 1.0


def test_output_size_override():
    tile = _random_tile()
    rgb = synthesize_pseudo_he(tile, dna_idx=[0, 1], output_size=64)
    assert rgb.shape == (3, 64, 64)


def test_deterministic():
    tile = _random_tile(seed=42)
    a = synthesize_pseudo_he(tile, dna_idx=[0, 1])
    b = synthesize_pseudo_he(tile, dna_idx=[0, 1])
    assert torch.allclose(a, b)


def test_accepts_hwc_layout():
    tile_chw = _random_tile(C=8, H=64, W=64)
    tile_hwc = np.moveaxis(tile_chw, 0, -1)
    a = synthesize_pseudo_he(tile_chw, dna_idx=[0, 1])
    b = synthesize_pseudo_he(tile_hwc, dna_idx=[0, 1])
    assert torch.allclose(a, b)


def test_accepts_torch_tensor():
    tile = torch.from_numpy(_random_tile())
    rgb = synthesize_pseudo_he(tile, dna_idx=[0, 1])
    assert rgb.shape == (3, 224, 224)


def test_explicit_bio_idx():
    tile = _random_tile(C=6)
    rgb_all = synthesize_pseudo_he(tile, dna_idx=[0])
    rgb_sub = synthesize_pseudo_he(tile, dna_idx=[0], bio_idx=[1, 2])
    # Different bio_idx should generally yield different eosin -> different RGB.
    assert not torch.allclose(rgb_all, rgb_sub)


def test_rejects_empty_dna_idx():
    tile = _random_tile()
    with pytest.raises(ValueError, match="dna_idx"):
        synthesize_pseudo_he(tile, dna_idx=[])


def test_rejects_out_of_range():
    tile = _random_tile(C=4)
    with pytest.raises(ValueError, match="out of range"):
        synthesize_pseudo_he(tile, dna_idx=[0, 99])


def test_batch_shape():
    B = 4
    batch = np.stack([_random_tile(seed=i) for i in range(B)], axis=0)
    rgb = synthesize_pseudo_he_batch(batch, dna_idx=[0, 1])
    assert rgb.shape == (B, 3, 224, 224)
    assert float(rgb.min()) >= 0.0
    assert float(rgb.max()) <= 1.0


def test_batch_matches_loop():
    B = 3
    batch = np.stack([_random_tile(seed=i) for i in range(B)], axis=0)
    bulk = synthesize_pseudo_he_batch(batch, dna_idx=[0, 1])
    loop = torch.stack(
        [synthesize_pseudo_he(batch[i], dna_idx=[0, 1]) for i in range(B)], dim=0
    )
    assert torch.allclose(bulk, loop)


def test_nuclei_affect_red_green_more_than_blue():
    """With a DNA mask occupying half the pixels and zero eosin, the nuclei
    region should be darker in G than in B (G has the heaviest hematoxylin
    coefficient). Smokes out sign / coefficient regressions."""
    C, H, W = 4, 32, 32
    tile = np.zeros((C, H, W), dtype=np.float32)
    # DNA concentrated on left half so percentile scaling produces a real gradient
    tile[0, :, : W // 2] = 1.0
    rgb = synthesize_pseudo_he(tile, dna_idx=[0], bio_idx=[1, 2, 3])
    # Left half (nuclei) should have G darker than B
    left_g = float(rgb[1, :, : 224 // 2].mean())
    left_b = float(rgb[2, :, : 224 // 2].mean())
    assert left_b > left_g

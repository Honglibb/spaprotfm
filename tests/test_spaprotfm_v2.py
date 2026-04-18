"""Tests for MaskedUNetV2 (v1 + H&E bottleneck conditioning)."""

from __future__ import annotations

import pytest
import torch

from spaprotfm.models.spaprotfm_v0 import build_masked_input, fixed_mask
from spaprotfm.models.spaprotfm_v2 import MaskedUNetV2


@pytest.fixture
def model() -> MaskedUNetV2:
    return MaskedUNetV2(n_channels=8, base=16, cond_in=32, cond_dim=16)


def _masked_input(C: int = 8, B: int = 2, H: int = 64, W: int = 64) -> torch.Tensor:
    full = torch.rand(B, C, H, W)
    m = fixed_mask(C, [0, 1], B, device="cpu")
    return build_masked_input(full, m)


def test_forward_with_condition(model: MaskedUNetV2):
    x = _masked_input()
    cond = torch.rand(2, 32, 14, 14)
    y = model(x, cond)
    assert y.shape == (2, 8, 64, 64)


def test_forward_without_condition(model: MaskedUNetV2):
    x = _masked_input()
    y = model(x, cond=None)
    assert y.shape == (2, 8, 64, 64)


def test_cond_changes_output(model: MaskedUNetV2):
    x = _masked_input()
    cond = torch.rand(2, 32, 14, 14)
    y1 = model(x, cond)
    y2 = model(x, cond=None)
    assert not torch.allclose(y1, y2)


def test_rejects_wrong_cond_channels(model: MaskedUNetV2):
    x = _masked_input()
    bad_cond = torch.rand(2, 64, 14, 14)
    with pytest.raises(ValueError, match="cond channel"):
        model(x, bad_cond)


def test_grad_flows_only_through_unet_and_cond_proj():
    m = MaskedUNetV2(n_channels=4, base=16, cond_in=16, cond_dim=8)
    x = _masked_input(C=4, B=1, H=32, W=32)
    cond = torch.rand(1, 16, 14, 14, requires_grad=False)
    y = m(x, cond)
    loss = y.mean()
    loss.backward()
    # cond_proj and cond_fuse must have gradients
    assert m.cond_proj.weight.grad is not None
    assert m.cond_fuse.weight.grad is not None
    # value_bias and U-Net layers too
    assert m.value_bias.grad is not None
    assert m.d1[0].weight.grad is not None


def test_cond_grid_upsample_to_bottleneck(model: MaskedUNetV2):
    """Cond grid 14x14 should be upsampled to match 8x8 bottleneck of 64-px input."""
    x = _masked_input(H=64, W=64)
    # 64 input -> 3 pools -> 8x8 bottleneck, not 16x16 (that needs 128-px input)
    cond = torch.rand(2, 32, 14, 14)
    y = model(x, cond)
    assert y.shape[-2:] == (64, 64)


def test_v2_at_128_bottleneck_16():
    """Full-size 128x128 input -> 16x16 bottleneck (matches Phikon 14x14 → 16x16 upsample)."""
    m = MaskedUNetV2(n_channels=4, base=16, cond_in=32, cond_dim=16)
    x = _masked_input(C=4, B=1, H=128, W=128)
    cond = torch.rand(1, 32, 14, 14)
    y = m(x, cond)
    assert y.shape == (1, 4, 128, 128)

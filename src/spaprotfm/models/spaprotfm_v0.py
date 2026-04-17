"""SpaProtFM v0: U-Net with random marker masking (MAE-style self-supervision)."""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn

from spaprotfm.baselines._vanilla_unet import _double_conv


class MaskedUNet(nn.Module):
    """U-Net taking (B, 2C, H, W) — value + mask per channel — predicting (B, C, H, W).

    Args:
        n_channels: number of marker channels C.
        base: base channel width for U-Net.
    """

    def __init__(self, n_channels: int, base: int = 32):
        super().__init__()
        self.n_channels = n_channels
        in_ch = 2 * n_channels  # values concat with mask
        self.d1 = _double_conv(in_ch, base)
        self.d2 = _double_conv(base, base * 2)
        self.d3 = _double_conv(base * 2, base * 4)
        self.d4 = _double_conv(base * 4, base * 8)
        self.pool = nn.MaxPool2d(2)
        self.up3 = nn.ConvTranspose2d(base * 8, base * 4, 2, stride=2)
        self.u3 = _double_conv(base * 8, base * 4)
        self.up2 = nn.ConvTranspose2d(base * 4, base * 2, 2, stride=2)
        self.u2 = _double_conv(base * 4, base * 2)
        self.up1 = nn.ConvTranspose2d(base * 2, base, 2, stride=2)
        self.u1 = _double_conv(base * 2, base)
        self.out = nn.Conv2d(base, n_channels, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        c1 = self.d1(x)
        c2 = self.d2(self.pool(c1))
        c3 = self.d3(self.pool(c2))
        c4 = self.d4(self.pool(c3))
        u3 = self.u3(torch.cat([self.up3(c4), c3], dim=1))
        u2 = self.u2(torch.cat([self.up2(u3), c2], dim=1))
        u1 = self.u1(torch.cat([self.up1(u2), c1], dim=1))
        return self.out(u1)


def build_masked_input(
    full: torch.Tensor, observed_idx: torch.Tensor
) -> torch.Tensor:
    """Build (B, 2C, H, W) masked input from (B, C, H, W) full and (B, C) bool/0-1 mask."""
    B, C, H, W = full.shape
    assert observed_idx.shape == (B, C), f"mask shape {observed_idx.shape} != {(B, C)}"
    mask = observed_idx.to(full.dtype).view(B, C, 1, 1).expand(B, C, H, W)
    values = full * mask
    # Interleave channels: [v_0, m_0, v_1, m_1, ...] → (B, 2C, H, W)
    out = torch.empty(B, 2 * C, H, W, device=full.device, dtype=full.dtype)
    out[:, 0::2] = values
    out[:, 1::2] = mask
    return out


def random_mask(n_channels: int, batch_size: int, k_min: int = 3, k_max: int | None = None,
                device: str | torch.device = "cpu") -> torch.Tensor:
    """Sample (B, C) binary mask: each row has k=Uniform(k_min, k_max) ones at random positions."""
    if k_max is None:
        k_max = n_channels - 1
    mask = torch.zeros(batch_size, n_channels, device=device)
    for b in range(batch_size):
        k = int(torch.randint(k_min, k_max + 1, (1,)).item())
        idx = torch.randperm(n_channels)[:k]
        mask[b, idx] = 1.0
    return mask


def fixed_mask(n_channels: int, observed_indices: list[int], batch_size: int,
               device: str | torch.device = "cpu") -> torch.Tensor:
    """Build a (B, C) mask with the SAME observed_indices for every batch row."""
    mask = torch.zeros(batch_size, n_channels, device=device)
    mask[:, observed_indices] = 1.0
    return mask

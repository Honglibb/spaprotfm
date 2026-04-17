"""SpaProtFM v1: MaskedUNet + per-channel learnable bias + larger capacity."""

from __future__ import annotations

import torch
import torch.nn as nn

from spaprotfm.baselines._vanilla_unet import _double_conv


class MaskedUNetV1(nn.Module):
    """Same I/O as MaskedUNet but with:
       (a) per-channel learnable bias added to the masked-value planes (helps model condition on channel identity)
       (b) larger default base (48)
    """

    def __init__(self, n_channels: int, base: int = 48):
        super().__init__()
        self.n_channels = n_channels
        in_ch = 2 * n_channels

        # (C,) learnable bias added to each value plane, broadcast over H/W
        self.value_bias = nn.Parameter(torch.zeros(n_channels))

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
        # x is (B, 2C, H, W) where channels [0::2] are values and [1::2] are masks.
        # Add per-channel bias to the value planes only.
        B, _, H, W = x.shape
        bias = self.value_bias.view(1, -1, 1, 1)  # (1, C, 1, 1)
        x = x.clone()
        x[:, 0::2] = x[:, 0::2] + bias  # add same bias to all H/W of each value plane

        c1 = self.d1(x)
        c2 = self.d2(self.pool(c1))
        c3 = self.d3(self.pool(c2))
        c4 = self.d4(self.pool(c3))
        u3 = self.u3(torch.cat([self.up3(c4), c3], dim=1))
        u2 = self.u2(torch.cat([self.up2(u3), c2], dim=1))
        u1 = self.u1(torch.cat([self.up1(u2), c1], dim=1))
        return self.out(u1)


def random_mask_with_always_observed(
    n_channels: int,
    batch_size: int,
    always_observed: list[int] | None,
    k_min: int = 3,
    k_max: int | None = None,
    device: str | torch.device = "cpu",
) -> torch.Tensor:
    """Like v0's random_mask but with channels in `always_observed` forced to 1.

    The k_min/k_max counts apply to channels OUTSIDE always_observed (the "samplable" pool).
    Final mask = always_observed ∪ (k samples from pool of size n_channels - len(always_observed)).
    """
    if always_observed is None:
        always_observed = []
    always_set = set(always_observed)
    pool = [i for i in range(n_channels) if i not in always_set]
    if k_max is None:
        k_max = max(k_min, len(pool) - 1)
    k_max = min(k_max, len(pool))

    mask = torch.zeros(batch_size, n_channels, device=device)
    if always_observed:
        mask[:, list(always_observed)] = 1.0

    pool_t = torch.tensor(pool, dtype=torch.long)
    for b in range(batch_size):
        if k_max <= 0:
            continue
        k = int(torch.randint(k_min, k_max + 1, (1,)).item())
        if len(pool) > 0:
            picks = pool_t[torch.randperm(len(pool))[:k]]
            mask[b, picks] = 1.0
    return mask

"""SpaProtFM v2: MaskedUNetV1 + H&E foundation-model conditioning.

Extends v1 with a condition branch that fuses pretrained pathology
foundation-model features (e.g. Phikon-v2) at the U-Net bottleneck:

    Phikon (frozen) -> (B, cond_in, 14, 14) features
                    -> upsample 16x16
                    -> 1x1 conv to cond_dim
                    -> concat with bottleneck (B, 8*base, 16, 16)
                    -> 1x1 conv back to 8*base
                    -> unchanged v1 decoder

Phikon weights are expected to be frozen by the caller; this module only
introduces trainable projection + fusion layers.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from spaprotfm.baselines._vanilla_unet import _double_conv

DEFAULT_COND_IN = 1024  # Phikon-v2 hidden size
DEFAULT_COND_DIM = 192  # ~4 * v1 base; keeps fusion conv small


class MaskedUNetV2(nn.Module):
    """MaskedUNetV1 with an optional H&E-feature conditioning branch.

    Args:
        n_channels: number of IMC protein channels.
        base: U-Net base width (v1 default is 48).
        cond_in: channel dim of the H&E feature map passed in (Phikon-v2
            hidden size = 1024).
        cond_dim: projected conditioning width.
        cond_grid: spatial side of the incoming H&E feature map (Phikon-v2
            on 224px input -> 14x14). Upsampled bilinearly to 16x16 before
            fusion.
    """

    def __init__(
        self,
        n_channels: int,
        base: int = 48,
        cond_in: int = DEFAULT_COND_IN,
        cond_dim: int = DEFAULT_COND_DIM,
        cond_grid: int = 14,
    ):
        super().__init__()
        self.n_channels = n_channels
        self.base = base
        self.cond_in = cond_in
        self.cond_dim = cond_dim
        self.cond_grid = cond_grid
        in_ch = 2 * n_channels

        self.value_bias = nn.Parameter(torch.zeros(n_channels))

        self.d1 = _double_conv(in_ch, base)
        self.d2 = _double_conv(base, base * 2)
        self.d3 = _double_conv(base * 2, base * 4)
        self.d4 = _double_conv(base * 4, base * 8)
        self.pool = nn.MaxPool2d(2)

        # Conditioning branch: Phikon feats -> upsample -> 1x1 -> concat
        self.cond_proj = nn.Conv2d(cond_in, cond_dim, kernel_size=1)
        self.cond_fuse = nn.Conv2d(base * 8 + cond_dim, base * 8, kernel_size=1)

        self.up3 = nn.ConvTranspose2d(base * 8, base * 4, 2, stride=2)
        self.u3 = _double_conv(base * 8, base * 4)
        self.up2 = nn.ConvTranspose2d(base * 4, base * 2, 2, stride=2)
        self.u2 = _double_conv(base * 4, base * 2)
        self.up1 = nn.ConvTranspose2d(base * 2, base, 2, stride=2)
        self.u1 = _double_conv(base * 2, base)
        self.out = nn.Conv2d(base, n_channels, 1)

    def _apply_value_bias(self, x: torch.Tensor) -> torch.Tensor:
        bias = self.value_bias.view(1, -1, 1, 1)
        x = x.clone()
        x[:, 0::2] = x[:, 0::2] + bias
        return x

    def _fuse_condition(
        self, bottleneck: torch.Tensor, cond: torch.Tensor | None
    ) -> torch.Tensor:
        """Inject conditioning features at the bottleneck.

        If ``cond`` is None, the conditioning path is skipped and the model
        behaves like v1 (+ an unused cond_fuse layer; cond_fuse(concat(b, 0))
        is avoided to keep v2 numerically identical to v1 for the None case).
        """
        if cond is None:
            return bottleneck
        if cond.ndim != 4:
            raise ValueError(f"expected (B, C, H, W) cond, got {tuple(cond.shape)}")
        if cond.shape[1] != self.cond_in:
            raise ValueError(
                f"cond channel mismatch: got {cond.shape[1]}, expected {self.cond_in}"
            )

        B, _, Hc, Wc = cond.shape
        Hb, Wb = bottleneck.shape[-2:]
        c = self.cond_proj(cond)  # (B, cond_dim, Hc, Wc)
        if (Hc, Wc) != (Hb, Wb):
            c = F.interpolate(c, size=(Hb, Wb), mode="bilinear", align_corners=False)
        return self.cond_fuse(torch.cat([bottleneck, c], dim=1))

    def forward(
        self, x: torch.Tensor, cond: torch.Tensor | None = None
    ) -> torch.Tensor:
        """Args:
            x: (B, 2C, H, W) masked input (values interleaved with mask planes).
            cond: (B, cond_in, cond_grid, cond_grid) H&E features, or None.
        """
        x = self._apply_value_bias(x)
        c1 = self.d1(x)
        c2 = self.d2(self.pool(c1))
        c3 = self.d3(self.pool(c2))
        c4 = self.d4(self.pool(c3))
        c4 = self._fuse_condition(c4, cond)
        u3 = self.u3(torch.cat([self.up3(c4), c3], dim=1))
        u2 = self.u2(torch.cat([self.up2(u3), c2], dim=1))
        u1 = self.u1(torch.cat([self.up1(u2), c1], dim=1))
        return self.out(u1)

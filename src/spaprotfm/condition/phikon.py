"""Phikon-v2 encoder wrapper (Owkin pathology foundation model).

Loads `owkin/phikon-v2` from HuggingFace (open access, no gated download),
freezes weights, and exposes an ``encode`` method that returns a patch-feature
map of shape ``(B, 768, 14, 14)`` given ``(B, 3, 224, 224)`` pseudo-H&E input.

The model is cached under ``data/hf_cache/`` so the download is
reproducible and sits on the largest disk (/code). Callers can override with
``cache_dir`` or the HF_HOME env var.
"""

from __future__ import annotations

import os
from pathlib import Path

import torch
import torch.nn as nn
from transformers import AutoModel

DEFAULT_MODEL_ID = "owkin/phikon-v2"
DEFAULT_CACHE_DIR = Path("/code/zkgy/hongliyin_computer/data/hf_cache")
DEFAULT_HF_ENDPOINT = "https://hf-mirror.com"
# Phikon-v2 is a DINOv2 ViT-Large: 16x16 patches, 14x14 tokens on 224 input,
# hidden dim 1024.
PHIKON_PATCH_SIZE = 16
PHIKON_HIDDEN_DIM = 1024
PHIKON_NUM_TOKENS = 14
# Phikon-v2 inherits the ImageNet-normalized input convention from DINOv2.
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


class PhikonEncoder(nn.Module):
    """Frozen Phikon-v2 feature extractor for pseudo-H&E tiles.

    Forward expects RGB float tensors in [0,1] with shape (B, 3, 224, 224).
    Returns (B, 768, 14, 14) patch-feature maps. The CLS token is discarded;
    if you need it use ``encode(..., return_cls=True)``.

    Weights are frozen and the module is forced into eval mode; autograd is
    disabled in the encode path to save memory during training.
    """

    def __init__(
        self,
        model_id: str = DEFAULT_MODEL_ID,
        cache_dir: str | os.PathLike | None = None,
        hf_endpoint: str | None = DEFAULT_HF_ENDPOINT,
    ):
        super().__init__()
        if hf_endpoint and not os.environ.get("HF_ENDPOINT"):
            os.environ["HF_ENDPOINT"] = hf_endpoint
        cache = Path(cache_dir) if cache_dir is not None else DEFAULT_CACHE_DIR
        cache.mkdir(parents=True, exist_ok=True)
        self.model_id = model_id
        self.cache_dir = str(cache)
        self.model = AutoModel.from_pretrained(model_id, cache_dir=self.cache_dir)

        for p in self.model.parameters():
            p.requires_grad = False
        self.model.eval()

        self.hidden_dim = PHIKON_HIDDEN_DIM
        self.grid_size = PHIKON_NUM_TOKENS

        mean = torch.tensor(IMAGENET_MEAN).view(1, 3, 1, 1)
        std = torch.tensor(IMAGENET_STD).view(1, 3, 1, 1)
        self.register_buffer("pixel_mean", mean, persistent=False)
        self.register_buffer("pixel_std", std, persistent=False)

    def train(self, mode: bool = True):  # type: ignore[override]
        # Keep the backbone frozen regardless of parent mode.
        super().train(mode)
        self.model.eval()
        return self

    @torch.no_grad()
    def encode(self, rgb: torch.Tensor, return_cls: bool = False) -> torch.Tensor:
        """Encode an RGB batch to a patch-feature map.

        Args:
            rgb: float tensor of shape (B, 3, 224, 224), values in [0, 1].
            return_cls: if True, also return the CLS token as a second output.

        Returns:
            feats: (B, 768, 14, 14) float tensor (dtype matches model weights).
            cls (optional): (B, 768) CLS embedding if return_cls is True.
        """
        if rgb.ndim != 4 or rgb.shape[1] != 3:
            raise ValueError(f"expected (B, 3, H, W) RGB, got {tuple(rgb.shape)}")
        if rgb.shape[-1] != 224 or rgb.shape[-2] != 224:
            raise ValueError(f"expected 224x224 tiles, got {tuple(rgb.shape[-2:])}")

        x = (rgb - self.pixel_mean.to(rgb.device)) / self.pixel_std.to(rgb.device)
        out = self.model(pixel_values=x)
        # last_hidden_state: (B, 1 + 196, 768)
        h = out.last_hidden_state
        cls = h[:, 0]
        patches = h[:, 1:]
        B, N, D = patches.shape
        assert N == self.grid_size * self.grid_size, (
            f"expected {self.grid_size**2} patches, got {N}"
        )
        feats = patches.transpose(1, 2).reshape(B, D, self.grid_size, self.grid_size)
        if return_cls:
            return feats, cls
        return feats

    def forward(self, rgb: torch.Tensor) -> torch.Tensor:
        return self.encode(rgb)

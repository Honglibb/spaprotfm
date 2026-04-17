import torch

from spaprotfm.models.spaprotfm_v0 import (
    MaskedUNet, build_masked_input, random_mask, fixed_mask,
)


def test_masked_unet_forward_shape():
    m = MaskedUNet(n_channels=10, base=8)
    x = torch.randn(2, 20, 128, 128)  # 2*C channels
    out = m(x)
    assert out.shape == (2, 10, 128, 128)


def test_build_masked_input_zeros_unobserved():
    full = torch.ones(1, 3, 4, 4)
    mask = torch.tensor([[1.0, 0.0, 1.0]])
    x = build_masked_input(full, mask)
    assert x.shape == (1, 6, 4, 4)
    # Channel 0 (value of marker 0): all 1
    assert (x[:, 0] == 1.0).all()
    # Channel 1 (mask of marker 0): all 1
    assert (x[:, 1] == 1.0).all()
    # Channel 2 (value of marker 1, masked): all 0
    assert (x[:, 2] == 0.0).all()
    # Channel 3 (mask of marker 1): all 0
    assert (x[:, 3] == 0.0).all()


def test_random_mask_has_at_least_k_min_observed():
    mask = random_mask(n_channels=10, batch_size=4, k_min=2, k_max=5)
    assert mask.shape == (4, 10)
    assert (mask.sum(dim=1) >= 2).all()
    assert (mask.sum(dim=1) <= 5).all()


def test_fixed_mask_obs_indices():
    mask = fixed_mask(n_channels=10, observed_indices=[0, 3, 5], batch_size=2)
    assert mask.shape == (2, 10)
    assert (mask[:, [0, 3, 5]] == 1.0).all()
    assert (mask[:, [1, 2, 4, 6, 7, 8, 9]] == 0.0).all()
